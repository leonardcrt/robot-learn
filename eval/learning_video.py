"""Vidéos MP4 de l'apprentissage de PPO (1280×720, H.264).

    python -m eval.learning_video               # carte choisie automatiquement
    python -m eval.learning_video --maps 3      # enchaîne 3 cartes de test
    python -m eval.learning_video --seed 1000042

Produit dans results/videos/ :
* ppo_evolution.mp4               — les stades d'entraînement se succèdent sur la même carte ;
                                    à droite, la courbe d'apprentissage indique où en est le réseau
* ppo_evolution_simultaneous.mp4  — tous les stades roulent en même temps sur la même carte
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from env.config import EnvConfig
from env.render import GRID, INK, INK_SECONDARY, MUTED, SURFACE, draw_scenario
from env.robot_env import NavigationEnv
from env.rollout import Episode, run_episode
from env.video import VideoWriter, figure_frame
from eval.learning_progress import color_for, load_snapshots, pick_maps, steps_label

FPS = 20
HOLD = 25  # images de pause sur l'issue de chaque stade


def read_curve(path=Path("results/ppo_training.csv")):
    rows = list(csv.DictReader(open(path)))
    return np.array([float(r["timesteps"]) for r in rows]), np.array([100 * float(r["success_rate"]) for r in rows])


def outcome_text(ep: Episode, cfg: EnvConfig) -> str:
    return f"succès en {ep.steps * cfg.dt:.1f} s" if ep.success else ep.outcome


class EvolutionFigure:
    """Carte à gauche, courbe d'apprentissage et légende à droite."""

    def __init__(self, cfg: EnvConfig, snapshots, curve):
        self.cfg, self.snapshots = cfg, snapshots
        self.fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        self.fig.patch.set_facecolor(SURFACE)
        self.ax = self.fig.add_axes([0.03, 0.05, 0.5, 0.86])
        self.ax_curve = self.fig.add_axes([0.61, 0.5, 0.36, 0.36])
        self.header = self.fig.text(0.03, 0.94, "", fontsize=17, color=INK, weight="bold")
        self.sub = self.fig.text(0.61, 0.38, "", fontsize=15, color=INK)
        self.sub2 = self.fig.text(0.61, 0.32, "", fontsize=12, color=INK_SECONDARY)
        self.sub3 = self.fig.text(0.61, 0.26, "", fontsize=12, color=INK_SECONDARY)
        self.fig.text(
            0.61,
            0.08,
            "Même réseau, même carte de test (jamais vue à l'entraînement).\n"
            "PPO n'a jamais vu l'expert : il apprend uniquement par la récompense.",
            fontsize=10,
            color=MUTED,
        )
        self._draw_curve(curve)

    def _draw_curve(self, curve):
        ax = self.ax_curve
        ax.set_facecolor(SURFACE)
        steps, success = curve
        ax.plot(np.maximum(steps, 1), success, color="#c3c2b7", lw=2, zorder=1)
        xs = [max(s["steps"], 1) for s in self.snapshots]
        ys = [100 * s["val_success"] for s in self.snapshots]
        n = len(self.snapshots)
        for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
            ax.plot(x, y, "o", ms=7, color=color_for(i, n), mec=SURFACE, mew=1.5, zorder=2)
        (self.marker,) = ax.plot([], [], "o", ms=15, mfc="none", mec=INK, mew=2, zorder=3)
        self.vline = ax.axvline(1, color=INK, lw=1, ls=":", zorder=0)
        ax.set_xscale("symlog", linthresh=10_000)
        ax.set_xlim(0, max(steps.max(), xs[-1]) * 1.1)
        ax.set_ylim(-3, 103)
        ax.set_xticks([0, 10_000, 100_000, 1_000_000])
        ax.set_xticklabels(["0", "10 k", "100 k", "1 M"])
        ax.set_title("Succès en validation pendant l'entraînement (%)", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("pas d'entraînement (échelle log)", color=INK_SECONDARY, fontsize=9)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)

    def set_stage(self, i: int):
        snap = self.snapshots[i]
        x = max(snap["steps"], 1)
        self.marker.set_data([x], [100 * snap["val_success"]])
        self.vline.set_xdata([x, x])
        self.sub.set_text(f"Stade {i + 1}/{len(self.snapshots)} : {steps_label(snap['steps'])}")
        self.sub2.set_text(f"Succès sur 100 cartes de validation : {snap['val_success']:.0%}")

    def draw_map(self, sc, ghosts, trajectory, color, robots=None):
        ax, cfg = self.ax, self.cfg
        ax.clear()
        draw_scenario(ax, sc, cfg)
        for g in ghosts:
            ax.plot(g[:, 0], g[:, 1], color=MUTED, lw=1, alpha=0.45, zorder=3)
        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, lw=3, zorder=4)
            robots = [(trajectory[-1], color)]
        for pose, c in robots or []:
            x, y, th = pose
            ax.add_patch(Circle((x, y), cfg.robot_radius, facecolor=c, edgecolor=SURFACE, lw=1.5, zorder=6))
            tip = 1.6 * cfg.robot_radius
            ax.plot([x, x + tip * np.cos(th)], [y, y + tip * np.sin(th)], color=INK, lw=1.8, zorder=7)

    def close(self):
        plt.close(self.fig)


def sequential_video(cfg, snapshots, maps, curve, path: Path):
    fig = EvolutionFigure(cfg, snapshots, curve)
    n = len(snapshots)
    with VideoWriter(path, fps=FPS) as video:
        for m, (_seed, episodes) in enumerate(maps):
            sc = episodes[0].scenario
            for i, ep in enumerate(episodes):
                color = color_for(i, n)
                fig.set_stage(i)
                stride = max(1, len(ep.trajectory) // 120)  # les timeouts (30 s) sont accélérés
                ts = list(range(0, len(ep.trajectory), stride)) + [len(ep.trajectory) - 1]
                ghosts = [e.trajectory for e in episodes[:i]]
                map_label = f"carte {m + 1}/{len(maps)} · " if len(maps) > 1 else ""
                for k, t in enumerate(ts + [ts[-1]] * HOLD):
                    done = k >= len(ts) - 1
                    fig.draw_map(sc, ghosts, ep.trajectory[: t + 1], color)
                    fig.header.set_text(f"PPO apprend à naviguer — {map_label}{steps_label(snapshots[i]['steps'])}")
                    fig.sub3.set_text(f"Sur cette carte : {outcome_text(ep, cfg)}" if done else "Sur cette carte : …")
                    video.append(figure_frame(fig.fig))
    fig.close()
    return video.frames


def simultaneous_video(cfg, snapshots, seed, episodes, curve, path: Path):
    fig = EvolutionFigure(cfg, snapshots, curve)
    n = len(snapshots)
    sc = episodes[0].scenario
    fig.marker.set_visible(False)
    fig.vline.set_visible(False)
    fig.sub.set_text("Tous les stades en même temps")
    fig.sub2.set_text("du plus clair (avant entraînement) au plus foncé (fin)")
    horizon = max(len(e.trajectory) for e in episodes)
    with VideoWriter(path, fps=FPS) as video:
        for t in list(range(horizon)) + [horizon - 1] * (2 * HOLD):
            fig.ax.clear()
            draw_scenario(fig.ax, sc, cfg)
            robots = []
            for i, ep in enumerate(episodes):
                traj = ep.trajectory[: min(t, len(ep.trajectory) - 1) + 1]
                label = f"{steps_label(snapshots[i]['steps'])}"
                if t >= len(ep.trajectory) - 1:
                    label += f" — {outcome_text(ep, cfg)}"
                fig.ax.plot(traj[:, 0], traj[:, 1], color=color_for(i, n), lw=2.5, label=label, zorder=4 + i / n)
                robots.append((traj[-1], color_for(i, n)))
            for pose, c in robots:
                x, y, th = pose
                fig.ax.add_patch(Circle((x, y), cfg.robot_radius, facecolor=c, edgecolor=SURFACE, lw=1.2, zorder=6))
            leg = fig.ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92, labelcolor=INK_SECONDARY)
            leg.get_frame().set_edgecolor(GRID)
            fig.header.set_text(f"PPO : tous les stades d'apprentissage sur la même carte — t = {t * cfg.dt:4.1f} s")
            video.append(figure_frame(fig.fig))
    fig.close()
    return video.frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="carte de test (défaut : choisie automatiquement)")
    parser.add_argument("--maps", type=int, default=1, help="nombre de cartes enchaînées dans ppo_evolution.mp4")
    parser.add_argument("--out", type=Path, default=Path("results/videos"))
    args = parser.parse_args()

    cfg = EnvConfig()
    snapshots = load_snapshots(cfg)
    curve = read_curve()
    if args.seed is not None:
        env = NavigationEnv(cfg)
        maps = [(args.seed, [run_episode(env, s["policy"], seed=args.seed) for s in snapshots])]
    else:
        maps = pick_maps(snapshots, cfg, args.maps)
    print("Cartes :", [seed for seed, _ in maps])

    n = sequential_video(cfg, snapshots, maps, curve, args.out / "ppo_evolution.mp4")
    print(f"Écrit : {args.out / 'ppo_evolution.mp4'} ({n} images, {n / FPS:.0f} s)")
    seed, episodes = maps[0]
    n = simultaneous_video(cfg, snapshots, seed, episodes, curve, args.out / "ppo_evolution_simultaneous.mp4")
    print(f"Écrit : {args.out / 'ppo_evolution_simultaneous.mp4'} ({n} images, {n / FPS:.0f} s)")


if __name__ == "__main__":
    main()
