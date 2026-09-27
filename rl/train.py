"""PPO (Stable-Baselines3) avec le `NavEncoder` comme extracteur de features.

    python -m rl.train --timesteps 2000000

PPO n'a jamais accès à l'expert : il n'apprend que par la récompense.

Conception de la récompense (voir `env/config.py::RewardConfig`) :
* **Progrès** +1 par mètre gagné vers la cible. C'est un shaping par potentiel
  (Φ = -distance) : il densifie le signal sans modifier la politique optimale
  (Ng et al., 1999). Sans lui, la seule récompense serait l'arrivée, trop rare au
  début de l'apprentissage sur des trajets de 5 à 12 m.
* **Succès** +10, **collision** -10, **sortie de zone** -10 : terminaux, du même
  ordre que le gain de progrès maximal pour que se jeter sur un obstacle proche de
  la cible ne soit jamais rentable.
* **Temps** -0,01 par pas : départage deux trajets sûrs au profit du plus rapide.
* **Proximité** jusqu'à -0,1 par pas sous 30 cm d'un obstacle : apprend une marge
  de sécurité avant de découvrir la collision elle-même.

Choix d'hyperparamètres (valeurs usuelles pour le contrôle continu, ajustées au CPU) :
* 8 environnements en parallèle × 512 pas = 4096 transitions par mise à jour.
* Acteur et critique ont chacun leur encodeur (`share_features_extractor=False`) : les
  gradients de la fonction de valeur, d'échelle très différente, ne perturbent pas
  les features de la politique ; le coût est négligeable avec ~34 k paramètres.
* `log_std_init=-0.5` (σ ≈ 0,6) : avec σ = 1 dans un espace d'action [-1, 1], la
  plupart des actions seraient écrêtées au début et le gradient peu informatif.
* Normalisation des récompenses (VecNormalize) mais pas des observations, déjà
  normalisées par construction dans `env/observation.py`.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env.config import EnvConfig
from env.robot_env import NavigationEnv
from env.seeds import VALIDATION_SEEDS, seed_range
from eval.metrics import evaluate_policy, format_summary
from rl.export import export_ppo
from rl.features import NavFeaturesExtractor

SNAPSHOT_STEPS = [0, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000, 800_000]


class SB3Policy:
    def __init__(self, model: PPO):
        self.model = model

    def __call__(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class ValidationCallback(BaseCallback):
    """Évalue la politique déterministe sur des cartes de validation fixes et garde la meilleure."""

    def __init__(
        self,
        cfg: EnvConfig,
        every: int,
        n_episodes: int,
        log_path: Path,
        ckpt_dir: Path,
        snapshots: list[int] | None = None,
    ):
        super().__init__()
        self.cfg, self.every = cfg, every
        # Instantanés de la politique à des étapes espacées de façon logarithmique : c'est au début
        # que le comportement change le plus (voir eval/learning_progress.py et eval/play.py).
        self.snapshots = sorted(snapshots or [])
        self.snapshot_dir = ckpt_dir / "ppo_snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.seeds = seed_range(VALIDATION_SEEDS, n_episodes)
        self.log_path, self.ckpt_dir = log_path, ckpt_dir
        self.best = -1.0
        self.next_eval = 0
        self.t0 = time.time()

    def _on_training_start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "timesteps",
                    "success_rate",
                    "collision_rate",
                    "timeout_rate",
                    "train_return",
                    "train_ep_len",
                    "wall_s",
                ]
            )

    def _on_step(self) -> bool:
        snapshot_due = self.snapshots and self.num_timesteps >= self.snapshots[0]
        if self.num_timesteps < self.next_eval and not snapshot_due:
            return True
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
        summary, _ = evaluate_policy(SB3Policy(self.model), self.seeds, self.cfg)
        if snapshot_due:
            step = self.snapshots.pop(0)
            meta = {
                "timesteps": step,
                "val_success": summary["success_rate"],
                "val_collision": summary["collision_rate"],
            }
            export_ppo(self.model, self.cfg, self.snapshot_dir / f"ppo_{step:07d}.pt", meta=meta)
        infos = list(self.model.ep_info_buffer)
        ret = float(np.mean([e["r"] for e in infos])) if infos else float("nan")
        length = float(np.mean([e["l"] for e in infos])) if infos else float("nan")
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.num_timesteps,
                    summary["success_rate"],
                    summary["collision_rate"],
                    summary["timeout_rate"],
                    ret,
                    length,
                    round(time.time() - self.t0),
                ]
            )
        print(format_summary(f"{self.num_timesteps / 1e6:.2f}M", summary) + f"  ret_train {ret:.1f}", flush=True)
        if summary["success_rate"] > self.best:
            self.best = summary["success_rate"]
            self.model.save(self.ckpt_dir / "ppo_sb3.zip")
        return True


def plot_learning_curve(csv_path: Path, out_path: Path):
    import matplotlib.pyplot as plt

    from env.render import GRID, INK, INK_SECONDARY, MUTED, SURFACE, style_for

    rows = list(csv.DictReader(open(csv_path)))
    t = np.array([float(r["timesteps"]) for r in rows]) / 1e6
    color = style_for("ppo")[0]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5), dpi=130, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    series = [
        ("success_rate", 100, "succès en validation (%)"),
        ("train_return", 1, "retour moyen (entraînement)"),
    ]
    for ax, (key, scale, label) in zip(axes, series, strict=True):
        ax.set_facecolor(SURFACE)
        ax.plot(t, [scale * float(r[key]) for r in rows], color=color, lw=2)
        ax.set_ylabel(label, color=INK_SECONDARY, fontsize=9)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_title("PPO : courbe d'apprentissage", color=INK, loc="left", fontsize=10)
    axes[1].set_xlabel("pas d'environnement (millions)", color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--subproc", action="store_true", help="un processus par environnement")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    cfg = EnvConfig()
    ckpt_dir, results = Path("checkpoints"), Path("results")
    ckpt_dir.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    venv = make_vec_env(
        NavigationEnv, n_envs=args.n_envs, seed=args.seed, env_kwargs={"config": cfg}, vec_env_cls=vec_cls
    )
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "features_extractor_class": NavFeaturesExtractor,
            "features_extractor_kwargs": {"k_nearest": cfg.k_nearest, "hidden": 64, "features_dim": 128},
            "share_features_extractor": False,
            "net_arch": {"pi": [64], "vf": [64]},
            "activation_fn": torch.nn.ReLU,
            "log_std_init": -0.5,
        },
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    snapshots = [s for s in SNAPSHOT_STEPS if s < args.timesteps] + [args.timesteps]
    callback = ValidationCallback(
        cfg, args.eval_every, args.eval_episodes, results / "ppo_training.csv", ckpt_dir, snapshots
    )
    model.learn(total_timesteps=args.timesteps, callback=callback)
    callback._on_step()  # évaluation finale

    best = PPO.load(ckpt_dir / "ppo_sb3.zip", device="cpu")
    export_ppo(best, cfg, ckpt_dir / "ppo.pt", meta={"timesteps": args.timesteps, "val_success": callback.best})
    plot_learning_curve(results / "ppo_training.csv", results / "ppo_learning_curve.png")
    print(f"Meilleur succès en validation : {callback.best:.1%} -> checkpoints/ppo.pt")


if __name__ == "__main__":
    main()
