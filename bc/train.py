"""Behavior cloning : régression supervisée des actions de l'expert.

    python -m bc.train --arch deepsets      # architecture retenue
    python -m bc.train --arch mlp           # ablation : MLP sur le vecteur aplati

Choix d'entraînement :
* Perte MSE : l'expert est déterministe, on régresse sa moyenne (une gaussienne à
  variance fixe donnerait la même solution).
* Séparation train/validation **par épisode** et non par transition : deux pas
  consécutifs d'un même épisode sont presque identiques, les mélanger gonflerait
  artificiellement le score de validation.
* On ne garde que les démonstrations réussies de l'expert : imiter ses 3 % d'échecs
  (hésitations jusqu'au timeout) n'apporte rien.
* Adam + décroissance cosinus du pas d'apprentissage, sélection du meilleur epoch
  sur la perte de validation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bc.data import collect, load_dataset, save_dataset
from env.config import EnvConfig
from env.seeds import DEMO_SEEDS, VALIDATION_SEEDS, seed_range
from eval.metrics import evaluate_policy, format_summary
from expert import VFHExpert
from models import ActorPolicy, count_parameters, make_actor, save_actor


def train_actor(
    actor: nn.Module,
    data: dict,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    val_fraction: float = 0.1,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    episodes = np.unique(data["episode"])
    val_eps = rng.choice(episodes, size=max(1, int(val_fraction * len(episodes))), replace=False)
    is_val = np.isin(data["episode"], val_eps)

    x_tr = torch.as_tensor(data["obs"][~is_val])
    y_tr = torch.as_tensor(data["act"][~is_val])
    x_va = torch.as_tensor(data["obs"][is_val])
    y_va = torch.as_tensor(data["act"][is_val])

    opt = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = int(np.ceil(len(x_tr) / batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    history = {"train_loss": [], "val_loss": []}
    best, best_state = float("inf"), None

    for epoch in range(epochs):
        actor.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(x_tr), batch_size):
            idx = perm[i : i + batch_size]
            loss = nn.functional.mse_loss(actor(x_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        actor.eval()
        with torch.no_grad():
            val = nn.functional.mse_loss(actor(x_va), y_va).item()
        history["train_loss"].append(total / len(x_tr))
        history["val_loss"].append(val)
        if val < best:
            best, best_state = val, copy.deepcopy(actor.state_dict())
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train {history['train_loss'][-1]:.5f}  val {val:.5f}")

    actor.load_state_dict(best_state)
    actor.eval()
    history["best_val_loss"] = best
    history["n_train"], history["n_val"] = len(x_tr), len(x_va)
    return history


def plot_losses(histories: dict[str, dict], path):
    import matplotlib.pyplot as plt

    from env.render import GRID, INK, INK_SECONDARY, MUTED, SURFACE, style_for

    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    labels = {"bc": "DeepSets", "bc_mlp": "MLP aplati"}
    for name, h in histories.items():
        color, _ = style_for(name)
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        label = labels.get(name, name)
        ax.plot(epochs, h["train_loss"], color=color, lw=2, ls="-", label=f"{label} — train")
        ax.plot(epochs, h["val_loss"], color=color, lw=2, ls="--", label=f"{label} — validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch", color=INK_SECONDARY)
    ax.set_ylabel("MSE sur les actions", color=INK_SECONDARY)
    ax.set_title("Behavior cloning : courbes de perte", color=INK, loc="left", fontsize=10)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def get_demonstrations(n_episodes: int, cfg: EnvConfig, data_dir=Path("data")) -> dict:
    path = data_dir / f"demos_{n_episodes}.npz"
    if path.exists():
        return load_dataset(path)
    expert = VFHExpert(cfg)
    print(f"Collecte de {n_episodes} épisodes de démonstration…")
    data = collect(expert, expert, seed_range(DEMO_SEEDS, n_episodes), cfg, only_success=True)
    save_dataset(path, data)
    print(f"  {len(data['obs'])} transitions, succès de l'expert {float(data['behavior_success_rate']):.1%}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["deepsets", "mlp"], default="deepsets")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    cfg = EnvConfig()
    data = get_demonstrations(args.episodes, cfg)
    actor = make_actor(cfg, encoder=args.arch)
    print(f"BC [{args.arch}] : {count_parameters(actor)} paramètres, {len(data['obs'])} transitions")
    history = train_actor(actor, data, epochs=args.epochs, seed=args.seed)

    summary, _ = evaluate_policy(ActorPolicy(actor), seed_range(VALIDATION_SEEDS, args.eval_episodes), cfg)
    print(format_summary(f"bc_{args.arch}", summary))

    name = "bc" if args.arch == "deepsets" else f"bc_{args.arch}"
    save_actor(
        args.out / f"{name}.pt",
        actor,
        cfg,
        encoder=args.arch,
        meta={"algo": "bc", "episodes": args.episodes, "val_success": summary["success_rate"]},
    )
    results = Path("results")
    results.mkdir(exist_ok=True)
    history["validation"] = {k: v for k, v in summary.items() if k != "success_ci95"}
    (results / f"{name}_history.json").write_text(json.dumps(history, indent=1))
    histories = {
        n: json.loads(p.read_text()) for n in ("bc", "bc_mlp") if (p := results / f"{n}_history.json").exists()
    }
    plot_losses(histories, results / "bc_loss.png")


if __name__ == "__main__":
    main()
