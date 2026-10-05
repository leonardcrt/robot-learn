"""PPO dans MuJoCoNavEnv, avec le même NavEncoder et les mêmes hyperparamètres que rl/train.py.

    python -m sim_mujoco.train_ppo --subproc                       # run complet
    python -m sim_mujoco.train_ppo --timesteps 300000 --eval-every 100000 --eval-episodes 20 \\
        --subproc --ckpt-dir $env:TEMP\\mj_test --results-dir $env:TEMP\\mj_test        # test de durée

Seule la dynamique change par rapport à rl/train.py : l'environnement est MuJoCoNavEnv. Le callback de
validation de rl/train.py évalue dans l'environnement cinématique ; on le sous-classe donc pour évaluer
dans MuJoCo (sans modifier rl/train.py).
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env.config import EnvConfig
from eval.metrics import evaluate_policy, format_summary
from rl.export import export_ppo
from rl.features import NavFeaturesExtractor
from rl.train import SB3Policy, ValidationCallback, plot_learning_curve
from sim_mujoco.mujoco_env import MuJoCoNavEnv


class MuJoCoValidationCallback(ValidationCallback):
    """Comme ValidationCallback, mais évalue dans MuJoCoNavEnv et sans instantanés."""

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.every
        # 1. Validation de la politique déterministe sur les cartes de validation, dans MuJoCo.
        summary, _ = evaluate_policy(SB3Policy(self.model), self.seeds, self.cfg, env_factory=MuJoCoNavEnv)
        infos = list(self.model.ep_info_buffer)
        ret = float(np.mean([e["r"] for e in infos])) if infos else float("nan")
        length = float(np.mean([e["l"] for e in infos])) if infos else float("nan")
        # 2. Une ligne dans le journal CSV (mêmes colonnes que rl/train.py).
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
        # 3. On garde le meilleur modèle selon le succès en validation.
        if summary["success_rate"] > self.best:
            self.best = summary["success_rate"]
            self.model.save(self.ckpt_dir / "ppo_mujoco_sb3.zip")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--subproc", action="store_true", help="un processus par environnement")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/mujoco"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/mujoco"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    cfg = EnvConfig()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Huit environnements MuJoCo en parallèle ; rewards normalisées, observations déjà normalisées.
    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    venv = make_vec_env(
        MuJoCoNavEnv, n_envs=args.n_envs, seed=args.seed, env_kwargs={"config": cfg}, vec_env_cls=vec_cls
    )
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)

    # 2. Mêmes hyperparamètres et même réseau que rl/train.py.
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

    # 3. Entraînement avec validation dans MuJoCo.
    csv_path = args.results_dir / "ppo_mujoco_training.csv"
    callback = MuJoCoValidationCallback(cfg, args.eval_every, args.eval_episodes, csv_path, args.ckpt_dir)
    callback.snapshot_dir.rmdir()  # dossier créé par la classe parente, inutile ici
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    callback._on_step()  # évaluation finale
    minutes = (time.time() - t0) / 60

    # 4. Export du meilleur modèle au format commun et courbe d'apprentissage.
    best = PPO.load(args.ckpt_dir / "ppo_mujoco_sb3.zip", device="cpu")
    meta = {"timesteps": args.timesteps, "val_success": callback.best, "simulator": "mujoco"}
    export_ppo(best, cfg, args.ckpt_dir / "ppo_mujoco.pt", meta=meta)
    plot_learning_curve(csv_path, args.results_dir / "ppo_mujoco_learning_curve.png")
    print(f"Durée : {minutes:.1f} min ({args.timesteps / (minutes * 60):.0f} pas/s)")
    print(f"Meilleur succès en validation : {callback.best:.1%} -> {args.ckpt_dir / 'ppo_mujoco.pt'}")


if __name__ == "__main__":
    main()
