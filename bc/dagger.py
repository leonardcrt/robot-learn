"""DAgger (Ross, Gordon & Bagnell, 2011) : corrige l'erreur composée du behavior cloning.

Le BC n'apprend que sur les états visités par l'expert. Dès que l'élève dévie un peu,
il atteint des états jamais vus, se trompe davantage, dévie encore plus : c'est le
« covariate shift », dont l'erreur croît en O(T²) avec l'horizon T. DAgger fait rouler
la politique apprise, demande à l'expert l'action correcte dans chaque état *réellement
visité*, agrège ces nouvelles paires au dataset et réentraîne (erreur en O(T)).

C'est possible ici parce que l'expert est un programme qu'on peut interroger dans
n'importe quel état — ce qui n'est pas le cas avec des démonstrations humaines.

    python -m bc.dagger --iterations 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bc.data import collect, merge
from bc.train import get_demonstrations, train_actor
from env.config import EnvConfig
from env.seeds import DAGGER_SEEDS, VALIDATION_SEEDS, seed_range
from eval.metrics import evaluate_policy, format_summary
from expert import VFHExpert
from models import ActorPolicy, load_actor, save_actor


class MixturePolicy:
    """À chaque pas : action de l'expert avec probabilité beta, sinon celle de l'élève."""

    def __init__(self, learner, expert, beta: float, seed: int = 0):
        self.learner, self.expert, self.beta = learner, expert, beta
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.expert(obs) if self.rng.random() < self.beta else self.learner(obs)


def plot_dagger(history: list[dict], expert: dict, path):
    import matplotlib.pyplot as plt

    from env.render import GRID, INK, INK_SECONDARY, MUTED, SURFACE, style_for

    it = [h["iteration"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    panels = [("success_rate", "succès (%)", (0, 100)), ("collision_rate", "collisions (%)", (0, 10))]
    for ax, (key, label, ylim) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        ax.plot(
            it, [100 * h[key] for h in history], color=style_for("dagger")[0], lw=2, marker="o", ms=8, label="DAgger"
        )
        ax.axhline(100 * expert[key], color=style_for("expert")[0], lw=2, ls="--", label="expert")
        ax.set_ylim(*ylim)
        ax.set_xticks(it)
        ax.set_xticklabels(["BC"] + [str(i) for i in it[1:]])
        ax.set_xlabel("itération DAgger", color=INK_SECONDARY)
        ax.set_title(label, color=INK, loc="left", fontsize=10)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower right")
    sizes = ", ".join(f"{h['dataset_size'] // 1000}k" for h in history)
    fig.suptitle(
        f"DAgger en validation ({history[0]['episodes']} cartes) — taille du dataset : {sizes}",
        fontsize=9,
        color=MUTED,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, default=Path("checkpoints/bc.pt"))
    parser.add_argument("--demo-episodes", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--episodes-per-iter", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--eval-episodes", type=int, default=200)
    args = parser.parse_args()

    cfg = EnvConfig()
    torch.manual_seed(0)
    expert = VFHExpert(cfg)
    val_seeds = seed_range(VALIDATION_SEEDS, args.eval_episodes)
    expert_summary, _ = evaluate_policy(expert, val_seeds, cfg)

    data = get_demonstrations(args.demo_episodes, cfg)
    actor, ckpt = load_actor(args.init, cfg)
    summary, _ = evaluate_policy(ActorPolicy(actor), val_seeds, cfg)
    print(format_summary("BC (init)", summary))
    history = [{"iteration": 0, "dataset_size": len(data["obs"]), **_scalars(summary)}]

    for i in range(1, args.iterations + 1):
        beta = 0.5**i
        behavior = MixturePolicy(ActorPolicy(actor), expert, beta, seed=i)
        seeds = seed_range(DAGGER_SEEDS + 1000 * i, args.episodes_per_iter)
        new = collect(behavior, expert, seeds, cfg)
        data = merge(data, new)
        train_actor(actor, data, epochs=args.epochs, lr=5e-4, seed=i, verbose=False)
        summary, _ = evaluate_policy(ActorPolicy(actor), val_seeds, cfg)
        print(format_summary(f"DAgger it {i}", summary) + f"  β={beta:.3f}  dataset {len(data['obs'])}")
        history.append({"iteration": i, "beta": beta, "dataset_size": len(data["obs"]), **_scalars(summary)})

    save_actor(
        "checkpoints/dagger.pt",
        actor,
        cfg,
        encoder=ckpt["encoder"],
        meta={"algo": "dagger", "iterations": args.iterations, "val_success": history[-1]["success_rate"]},
    )
    results = Path("results")
    expert = _scalars(expert_summary)
    (results / "dagger_history.json").write_text(json.dumps({"expert": expert, "iterations": history}, indent=1))
    plot_dagger(history, expert, results / "dagger.png")


def _scalars(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if isinstance(v, (int, float))}


if __name__ == "__main__":
    main()
