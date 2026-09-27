"""Vidéo de présentation courte (12 s, 3:2, en boucle) pour un portfolio.

    python -m eval.showcase_video

Trois plans :
1. PPO apprend par essai-erreur : stades d'entraînement successifs sur la même carte.
2. Expert, imitation (DAgger) et PPO sur une même carte de test, en temps accéléré.
3. Chiffres clés du benchmark.

Style calqué sur le portfolio (fond crème, encre sombre, accent bleu marine).
Produit results/videos/showcase.mp4 et results/videos/showcase_poster.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image

from env.config import EnvConfig
from env.robot_env import NavigationEnv
from env.rollout import run_episode
from env.video import VideoWriter, figure_frame
from eval.learning_progress import load_snapshots
from eval.policies import load_policy

W, H, FPS = 1440, 960, 30
BG = "#f5f3ee"
INK = "#16181d"
INK2 = "#5b6070"
MUTED = "#9a9ea8"
NAVY = "#23395b"
OBST_FACE, OBST_EDGE = "#dfdbd1", "#c9c4b8"
GRID = "#e9e6de"
SANS, MONO = "Segoe UI", "Cascadia Mono"

LEARNING_SEED = 1_000_042  # carte où la progression de PPO est la plus lisible
RACE_SEED = 1_000_006  # carte où les trois approches prennent des chemins différents
STAGES = [0, 10_000, 25_000, 50_000, 1_500_000]
STAGE_COLORS = ["#c3cad6", "#a3afc2", "#7f90ab", "#566b8f", NAVY]
RACE = [
    ("expert", "Hand-written expert", MUTED, (0, (4, 3))),
    ("dagger", "Imitation (DAgger)", "#7f90ab", "-"),
    ("ppo", "Reinforcement (PPO)", NAVY, "-"),
]


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def steps_text(s: int) -> str:
    return "untrained" if s == 0 else (f"{s / 1e6:.1f}M steps" if s >= 1e6 else f"{s // 1000}k steps")


def outcome_text(ep) -> str:
    return {
        "succès": f"reaches the goal · {ep.steps / 10:.1f} s",
        "collision": "hits an obstacle",
        "sortie": "drives off the map",
        "timeout": "circles until timeout",
    }[ep.outcome]


class Canvas:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        self.fig.patch.set_facecolor(BG)
        # carte carrée à gauche, colonne de texte à droite
        self.ax = self.fig.add_axes([0.035, 0.06, 0.58, 0.87])

    def clear(self):
        for t in list(self.fig.texts):
            t.remove()
        self.fig.patches.clear()
        self.fig.lines.clear()
        self.ax.clear()

    def map(self, sc, alpha: float = 1.0):
        ax, cfg = self.ax, self.cfg
        m = 1.0  # marge d'une case : toutes les cases de la grille ont la même taille
        ax.set_xlim(-m, cfg.arena_size + m)
        ax.set_ylim(-m, cfg.arena_size + m)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(
            Rectangle(
                (-m, -m),
                cfg.arena_size + 2 * m,
                cfg.arena_size + 2 * m,
                facecolor="#fbfaf7",
                edgecolor=GRID,
                lw=1.2,
                zorder=0,
            )
        )
        for v in range(0, int(cfg.arena_size) + 1):
            ax.plot([v, v], [-m, cfg.arena_size + m], color=GRID, lw=0.6, zorder=0.5)
            ax.plot([-m, cfg.arena_size + m], [v, v], color=GRID, lw=0.6, zorder=0.5)
        for x, y, r in sc.obstacles:
            ax.add_patch(Circle((x, y), r, facecolor=OBST_FACE, edgecolor=OBST_EDGE, lw=1.2, alpha=alpha, zorder=2))
        ax.add_patch(
            Circle(sc.goal, cfg.goal_radius + 0.05, facecolor="none", edgecolor=INK, lw=2, alpha=alpha, zorder=5)
        )
        ax.add_patch(Circle(sc.goal, 0.09, color=INK, alpha=alpha, zorder=5))
        ax.add_patch(Circle(sc.start, 0.12, facecolor=BG, edgecolor=INK, lw=1.8, alpha=alpha, zorder=5))

    def path(self, traj, color, lw=3.2, alpha=1.0, ls="-", robot=True):
        ax, r = self.ax, self.cfg.robot_radius
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=lw, alpha=alpha, ls=ls, solid_capstyle="round", zorder=4)
        if robot:
            x, y, th = traj[-1]
            ax.add_patch(Circle((x, y), r * 1.15, facecolor=color, edgecolor=BG, lw=2, alpha=alpha, zorder=6))
            ax.plot(
                [x, x + 1.7 * r * np.cos(th)],
                [y, y + 1.7 * r * np.sin(th)],
                color=BG if color == NAVY else INK,
                lw=2.2,
                alpha=alpha,
                zorder=7,
            )

    def text(self, x, y, s, size=16, color=INK, font=SANS, weight="normal", alpha=1.0, **kw):
        return self.fig.text(x, y, s, fontsize=size, color=color, family=font, weight=weight, alpha=alpha, **kw)

    def header(self, label: str, title: str, alpha: float = 1.0):
        self.text(0.655, 0.865, label, size=13, color=INK2, font=MONO, alpha=alpha)
        self.text(0.655, 0.775, title, size=29, weight="semibold", alpha=alpha, linespacing=1.15, va="top")

    def frame(self) -> np.ndarray:
        return figure_frame(self.fig)


def partial(traj, frac):
    n = max(1, int(round(frac * (len(traj) - 1))) + 1)
    return traj[:n]


def shot_learning(cv: Canvas, snapshots, n_frames: int):
    """Plan 1 : chaque stade d'entraînement rejoue la même carte ; les précédents restent en fantôme."""
    cfg = cv.cfg
    env = NavigationEnv(cfg)
    stages = [s for s in snapshots if s["steps"] in STAGES]
    eps = [run_episode(env, s["policy"], seed=LEARNING_SEED) for s in stages]
    sc = eps[0].scenario
    per = n_frames // len(stages)
    frames = []
    for f in range(n_frames):
        i = min(f // per, len(stages) - 1)
        local = (f - i * per) / (per * 0.78)  # 78 % du créneau pour tracer, le reste en pause
        cv.clear()
        cv.map(sc)
        for j in range(i):
            cv.path(eps[j].trajectory, STAGE_COLORS[j], lw=2.2, alpha=0.55, robot=False)
        cv.path(partial(eps[i].trajectory, ease(local)), STAGE_COLORS[i], lw=3.6)
        cv.header("01 · REINFORCEMENT LEARNING", "The robot learns\nby trial and error")
        # frise des stades d'entraînement
        y0 = 0.53
        for j, s in enumerate(stages):
            active = j == i
            done = j < i
            y = y0 - j * 0.062
            filled = active or done
            dot = plt.Line2D(
                [0.667],
                [y + 0.009],
                transform=cv.fig.transFigure,
                figure=cv.fig,
                marker="o",
                ms=15 if active else 12,
                mfc=STAGE_COLORS[j] if filled else BG,
                mec=STAGE_COLORS[j] if filled else MUTED,
                mew=1.5,
                ls="none",
            )
            cv.fig.lines.append(dot)
            cv.text(
                0.685,
                y,
                steps_text(s["steps"]),
                size=17 if active else 15,
                color=INK if active else (INK2 if done else MUTED),
                weight="semibold" if active else "normal",
            )
            if active or done:
                show = done or local >= 1.0
                if show:
                    cv.text(
                        0.815, y + 0.002, outcome_text(eps[j]), size=13, color=INK2 if not active else INK, font=SANS
                    )
        cv.text(
            0.655,
            0.155,
            "Same network, same unseen test map.\nPPO never sees the expert: it learns from reward only.",
            size=13,
            color=MUTED,
            linespacing=1.5,
        )
        frames.append(cv.frame())
    return frames


def shot_race(cv: Canvas, n_frames: int, hold: int = 18):
    """Plan 2 : expert, imitation et PPO en temps accéléré sur la même carte."""
    cfg = cv.cfg
    env = NavigationEnv(cfg)
    eps = {k: run_episode(env, load_policy(k, cfg), seed=RACE_SEED) for k, *_ in RACE}
    sc = eps["expert"].scenario
    horizon = max(len(e.trajectory) for e in eps.values())
    anim = n_frames - hold
    speed = (horizon - 1) * cfg.dt / (anim / FPS)
    frames = []
    for f in range(n_frames):
        t = min(f / (anim - 1), 1.0) * (horizon - 1)
        cv.clear()
        cv.map(sc)
        for k, _, color, ls in RACE:
            traj = eps[k].trajectory[: min(int(t), len(eps[k].trajectory) - 1) + 1]
            cv.path(traj, color, lw=3.4, ls=ls)
        cv.header("02 · EXPERT  vs  IMITATION  vs  RL", "Same map,\nthree ways to learn")
        for j, (k, name, color, ls) in enumerate(RACE):
            y = 0.52 - j * 0.085
            cv.fig.lines.append(
                plt.Line2D(
                    [0.657, 0.695],
                    [y + 0.009] * 2,
                    transform=cv.fig.transFigure,
                    color=color,
                    lw=3.4,
                    ls=ls,
                    figure=cv.fig,
                )
            )
            cv.text(0.71, y, name, size=16, color=INK)
            ep = eps[k]
            finished = t >= len(ep.trajectory) - 1
            status = f"{ep.steps * cfg.dt:.1f} s" if finished else f"{min(t, ep.steps) * cfg.dt:.1f} s"
            cv.text(0.955, y, status, size=16, color=INK if finished else MUTED, ha="right", font=MONO)
        cv.text(
            0.655,
            0.155,
            f"Playback ×{speed:.0f}. PPO matches the expert's success rate\n"
            "and reaches the goal 34% faster on average.",
            size=13,
            color=MUTED,
            linespacing=1.5,
        )
        frames.append(cv.frame())
    return frames, sc, eps


def shot_results(cv: Canvas, sc, eps, n_frames: int):
    """Plan 3 : chiffres clés du benchmark sur 500 cartes de test."""
    stats = [
        ("97%", "success rate of PPO on 500 unseen test maps"),
        ("34%", "faster than the hand-written expert"),
        ("93% → 96%", "imitation learning, improved with DAgger"),
    ]
    frames = []
    for f in range(n_frames):
        cv.clear()
        cv.map(sc, alpha=0.45)
        for k, _, color, ls in RACE:
            cv.path(eps[k].trajectory, color, lw=2.6, alpha=0.35, ls=ls, robot=False)
        cv.header("03 · RESULTS", "Expert, imitation and\nreinforcement learning")
        for j, (big, small) in enumerate(stats):
            a = ease((f - 6 - 10 * j) / 12)
            y = 0.5 - j * 0.12
            cv.text(0.655, y, big, size=36, weight="semibold", color=NAVY, alpha=a)
            cv.text(0.657, y - 0.04, small, size=13.5, color=INK2, alpha=a)
        cv.text(
            0.655,
            0.12,
            "PyTorch · Stable-Baselines3 · ROS 2",
            size=13,
            color=MUTED,
            font=MONO,
            alpha=ease((f - 36) / 12),
        )
        frames.append(cv.frame())
    return frames


def crossfade(a: list[np.ndarray], b: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Fondu enchaîné : les n dernières images de a se mélangent aux n premières de b."""
    out = a[:-n]
    for k in range(n):
        w = ease((k + 1) / (n + 1))
        out.append(((1 - w) * a[len(a) - n + k] + w * b[k]).astype(np.uint8))
    return out + b[n:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/videos"))
    args = parser.parse_args()

    import torch

    torch.set_num_threads(1)
    cfg = EnvConfig()
    snapshots = load_snapshots(cfg)
    cv = Canvas(cfg)
    xf = 9
    s1 = shot_learning(cv, snapshots, n_frames=160)
    s2, sc, eps = shot_race(cv, n_frames=138)
    s3 = shot_results(cv, sc, eps, n_frames=84)
    frames = crossfade(crossfade(s1, s2, xf), s3, xf)
    # la fin se fond dans le début pour une boucle sans à-coup
    loop = [
        ((1 - ease((k + 1) / (xf + 1))) * frames[-xf + k] + ease((k + 1) / (xf + 1)) * frames[0]).astype(np.uint8)
        for k in range(xf)
    ]
    frames = frames[:-xf] + loop
    plt.close(cv.fig)

    args.out.mkdir(parents=True, exist_ok=True)
    with VideoWriter(args.out / "showcase.mp4", fps=FPS) as video:
        for fr in frames:
            video.append(fr)
    Image.fromarray(s2[len(s2) - 10]).save(args.out / "showcase_poster.png")
    print(f"Écrit : {args.out / 'showcase.mp4'} ({len(frames)} images, {len(frames) / FPS:.1f} s, {W}×{H})")


if __name__ == "__main__":
    main()
