"""GIF et figures de comparaison : les politiques rejouent les mêmes cartes de test.

    python -m eval.visualize

Produit dans results/ :
* comparison.gif   — toutes les politiques superposées sur la même carte
* side_by_side.gif — un panneau par politique, même carte, même horloge
* bc_vs_dagger.gif — une carte où le BC échoue et DAgger réussit (erreur composée)
* trajectories.png — six cartes de test, trajectoires superposées
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from env.config import EnvConfig
from env.render import (
    INK,
    SURFACE,
    animate_trajectories,
    draw_robot,
    draw_scenario,
    plot_trajectories,
    save_gif,
    style_for,
)
from env.robot_env import NavigationEnv
from env.rollout import Episode, run_episode
from env.seeds import TEST_SEEDS
from eval.policies import LABELS, load_policy


def run_all(names: list[str], seed: int, cfg: EnvConfig) -> dict[str, Episode]:
    env = NavigationEnv(cfg)
    return {n: run_episode(env, load_policy(n, cfg), seed=seed) for n in names}


def legend_labels(eps: dict[str, Episode], cfg: EnvConfig) -> dict[str, str]:
    out = {}
    for n, ep in eps.items():
        detail = f"{ep.steps * cfg.dt:.1f} s" if ep.success else ep.outcome
        out[n] = f"{LABELS.get(n, n)} — {detail}"
    return out


def trajectories_of(eps: dict[str, Episode]) -> dict[str, np.ndarray]:
    return {n: ep.trajectory for n, ep in eps.items()}


def find_seed(names, cfg, predicate, start=TEST_SEEDS, max_tries=300) -> tuple[int, dict[str, Episode]]:
    for seed in range(start, start + max_tries):
        eps = run_all(names, seed, cfg)
        if predicate(eps):
            return seed, eps
    raise RuntimeError("aucune carte ne satisfait le critère")


def side_by_side_gif(sc, eps: dict[str, Episode], cfg: EnvConfig, path: Path, stride: int = 2):
    n = len(eps)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4), dpi=90)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.86, wspace=0.08)
    horizon = max(len(ep.trajectory) for ep in eps.values())
    frames = []
    for t in list(range(0, horizon, stride)) + [horizon - 1]:
        for ax, (name, ep) in zip(axes, eps.items(), strict=True):
            ax.clear()
            draw_scenario(ax, sc, cfg)
            ax.set_xticks([])
            ax.set_yticks([])
            traj = ep.trajectory[: min(t, len(ep.trajectory) - 1) + 1]
            color, ls = style_for(name)
            ax.plot(traj[:, 0], traj[:, 1], color=color, ls=ls, lw=2)
            draw_robot(ax, traj[-1], cfg, color)
            done = t >= len(ep.trajectory) - 1
            status = f" — {ep.outcome}" if done else ""
            ax.set_title(f"{LABELS.get(name, name)}{status}", fontsize=9, color=INK, loc="left")
        fig.suptitle(f"t = {t * cfg.dt:4.1f} s", fontsize=9, color=INK, x=0.02, ha="left")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    save_gif(frames, path, fps=int(round(1 / (cfg.dt * stride))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="*", default=["expert", "bc", "dagger", "ppo"])
    parser.add_argument("--seed", type=int, default=None, help="carte à utiliser pour les GIF (défaut : automatique)")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = EnvConfig()
    names = args.policies
    args.out.mkdir(exist_ok=True)

    def all_succeed_with_detour(eps):
        expert = eps["expert"]
        straight = np.linalg.norm(expert.scenario.goal - expert.scenario.start)
        return all(ep.success for ep in eps.values()) and expert.path_length > 1.15 * straight

    if args.seed is None:
        seed, eps = find_seed(names, cfg, all_succeed_with_detour)
    else:
        seed, eps = args.seed, run_all(names, args.seed, cfg)
    sc = eps[names[0]].scenario
    print(f"Carte {seed} : " + ", ".join(f"{n}={ep.outcome}" for n, ep in eps.items()))
    animate_trajectories(
        sc, trajectories_of(eps), cfg, args.out / "comparison.gif", labels=legend_labels(eps, cfg), stride=2
    )
    side_by_side_gif(sc, eps, cfg, args.out / "side_by_side.gif")

    if "bc" in names and "dagger" in names:
        pair = ["expert", "bc", "dagger"]
        seed_f, eps_f = find_seed(
            pair, cfg, lambda e: e["bc"].collision and e["dagger"].success and e["expert"].success
        )
        print(f"Carte {seed_f} (BC échoue, DAgger réussit)")
        animate_trajectories(
            eps_f["bc"].scenario,
            trajectories_of(eps_f),
            cfg,
            args.out / "bc_vs_dagger.gif",
            labels=legend_labels(eps_f, cfg),
            stride=2,
        )

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    for ax, s in zip(axes.flat, range(TEST_SEEDS + 1, TEST_SEEDS + 7), strict=True):
        e = run_all(names, s, cfg)
        plot_trajectories(e[names[0]].scenario, trajectories_of(e), cfg, ax=ax, labels=legend_labels(e, cfg))
        ax.set_title(f"carte de test {s - TEST_SEEDS}", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(args.out / "trajectories.png")
    print(f"Écrit dans {args.out}/ : comparison.gif, side_by_side.gif, bc_vs_dagger.gif, trajectories.png")


if __name__ == "__main__":
    main()
