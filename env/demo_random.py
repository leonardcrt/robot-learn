"""Critère S1 : une politique aléatoire produit un épisode filmable (GIF) et un tracé de trajectoire.

python -m env.demo_random --seed 3
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from env import EnvConfig, NavigationEnv
from env.render import animate_trajectories, plot_trajectories
from env.rollout import run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = EnvConfig()
    env = NavigationEnv(cfg)
    rng = np.random.default_rng(args.seed)
    ep = run_episode(env, lambda obs: rng.uniform(-1, 1, size=2), seed=args.seed)
    print(f"Politique aléatoire : {ep.outcome} après {ep.steps} pas, récompense {ep.total_reward:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    trajs = {"random": ep.trajectory}
    animate_trajectories(ep.scenario, trajs, cfg, args.out / "random_policy.gif", title="Politique aléatoire")
    plot_trajectories(ep.scenario, trajs, cfg, title=f"Politique aléatoire — {ep.outcome}")
    plt.savefig(args.out / "random_policy.png", bbox_inches="tight")
    print(f"Écrit : {args.out / 'random_policy.gif'} et {args.out / 'random_policy.png'}")


if __name__ == "__main__":
    main()
