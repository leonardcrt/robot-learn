"""Rendu vue du dessus (matplotlib) : images RGB, GIF animés et tracés de trajectoires."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from PIL import Image  # noqa: E402

from env.config import EnvConfig  # noqa: E402
from env.scenario import Scenario  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
OBSTACLE = "#c3c2b7"

# Couleur fixe par politique (jamais par rang) + style de trait comme encodage secondaire.
POLICY_STYLE = {
    "expert": ("#2a78d6", "-"),
    "ppo": ("#eb6834", "-"),
    "bc": ("#1baf7a", "--"),
    "bc_mlp": ("#e87ba4", "--"),
    "dagger": ("#4a3aa7", "-."),
    "random": (MUTED, ":"),
    "policy": ("#2a78d6", "-"),
}


def style_for(name: str):
    return POLICY_STYLE.get(name, (INK_SECONDARY, "-"))


def draw_scenario(ax, sc: Scenario, cfg: EnvConfig):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, cfg.arena_size)
    ax.set_ylim(0, cfg.arena_size)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, int(cfg.arena_size) + 1, 2))
    ax.set_yticks(range(0, int(cfg.arena_size) + 1, 2))
    ax.tick_params(colors=MUTED, labelsize=7, length=0)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(OBSTACLE)
    for x, y, r in sc.obstacles:
        ax.add_patch(Circle((x, y), r, facecolor=OBSTACLE, edgecolor=MUTED, linewidth=0.8))
    ax.add_patch(Circle(sc.goal, cfg.goal_radius, facecolor="none", edgecolor=INK, linewidth=1.2))
    ax.plot(*sc.goal, marker="*", color=INK, markersize=12, zorder=5)
    ax.plot(*sc.start, marker="o", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=7, zorder=5)


def draw_robot(ax, pose, cfg: EnvConfig, color):
    x, y, th = pose
    ax.add_patch(Circle((x, y), cfg.robot_radius, facecolor=color, edgecolor=SURFACE, linewidth=1.5, zorder=6))
    tip = np.array([x, y]) + 1.6 * cfg.robot_radius * np.array([np.cos(th), np.sin(th)])
    ax.plot([x, tip[0]], [y, tip[1]], color=INK, linewidth=1.5, zorder=7)


def draw_trajectories(ax, trajectories: dict[str, np.ndarray], cfg: EnvConfig, labels=None, robots=True):
    for name, traj in trajectories.items():
        color, ls = style_for(name)
        label = (labels or {}).get(name, name)
        ax.plot(traj[:, 0], traj[:, 1], color=color, linestyle=ls, linewidth=2, label=label, zorder=4)
        if robots:
            draw_robot(ax, traj[-1], cfg, color)
    if len(trajectories) >= 2:
        leg = ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.9, labelcolor=INK_SECONDARY)
        leg.get_frame().set_edgecolor(GRID)


class Renderer:
    """Réutilise une seule figure pour produire des images rapidement."""

    def __init__(self, cfg: EnvConfig, size_px: int = 480):
        self.cfg = cfg
        dpi = 100
        self.fig, self.ax = plt.subplots(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
        self.fig.patch.set_facecolor(SURFACE)
        self.fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.93)

    def frame(self, sc: Scenario, trajectories: dict[str, np.ndarray], title=None, labels=None) -> np.ndarray:
        self.ax.clear()
        draw_scenario(self.ax, sc, self.cfg)
        draw_trajectories(self.ax, trajectories, self.cfg, labels=labels)
        if title:
            self.ax.set_title(title, fontsize=9, color=INK, loc="left")
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        plt.close(self.fig)


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 10, hold_last: int = 15):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = list(frames) + [frames[-1]] * hold_last
    images = [Image.fromarray(f).quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=True)


def animate_trajectories(
    sc: Scenario, trajectories: dict[str, np.ndarray], cfg: EnvConfig, path, labels=None, title=None, stride=1
):
    """GIF où toutes les politiques avancent en même temps sur la même carte."""
    renderer = Renderer(cfg)
    horizon = max(len(t) for t in trajectories.values())
    frames = []
    for t in list(range(0, horizon, stride)) + [horizon - 1]:
        partial = {k: v[: min(t, len(v) - 1) + 1] for k, v in trajectories.items()}
        frames.append(renderer.frame(sc, partial, title=f"{title or ''}  t = {t * cfg.dt:4.1f} s", labels=labels))
    renderer.close()
    save_gif(frames, path, fps=int(round(1 / (cfg.dt * stride))))


def plot_trajectories(
    sc: Scenario, trajectories: dict[str, np.ndarray], cfg: EnvConfig, ax=None, title=None, labels=None
):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        fig.patch.set_facecolor(SURFACE)
    draw_scenario(ax, sc, cfg)
    draw_trajectories(ax, trajectories, cfg, labels=labels)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    return ax
