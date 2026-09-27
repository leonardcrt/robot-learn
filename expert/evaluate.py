"""Critère S2 : taux de succès de l'expert >= 90 % sur N scénarios aléatoires.

python -m expert.evaluate --n 500 --gif
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from env import EnvConfig, NavigationEnv
from env.render import animate_trajectories, plot_trajectories
from env.rollout import run_episode
from expert import EXPERTS, make_expert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", choices=list(EXPERTS), default="vfh")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--gif", action="store_true", help="écrit un GIF de démonstration et un tracé des échecs")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = EnvConfig()
    env = NavigationEnv(cfg)
    expert = make_expert(args.expert, cfg)
    episodes = [run_episode(env, expert, seed=args.seed_offset + i) for i in range(args.n)]

    outcomes = Counter(ep.outcome for ep in episodes)
    success = np.mean([ep.success for ep in episodes])
    steps = [ep.steps for ep in episodes if ep.success]
    print(f"Expert {args.expert} sur {args.n} scénarios : succès {success:.1%}  {dict(outcomes)}")
    print(f"Durée moyenne (succès) : {np.mean(steps) * cfg.dt:.1f} s")
    failures = [args.seed_offset + i for i, ep in enumerate(episodes) if not ep.success]
    print(f"Graines en échec : {failures[:30]}")

    if args.gif:
        args.out.mkdir(parents=True, exist_ok=True)
        ep = episodes[0]
        animate_trajectories(ep.scenario, {"expert": ep.trajectory}, cfg, args.out / "expert.gif", title="Expert")
        if failures:
            n = min(6, len(failures))
            fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
            for ax, seed in zip(axes[0], failures[:n], strict=False):
                fail = episodes[seed - args.seed_offset]
                plot_trajectories(
                    fail.scenario, {"expert": fail.trajectory}, cfg, ax=ax, title=f"{seed} {fail.outcome}"
                )
            fig.savefig(args.out / "expert_failures.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
