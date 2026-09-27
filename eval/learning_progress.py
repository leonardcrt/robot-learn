"""Évolution de l'apprentissage de PPO : la même carte, rejouée à chaque stade d'entraînement.

    python -m eval.learning_progress

Utilise les instantanés sauvegardés par `rl/train.py` dans checkpoints/ppo_snapshots/.
Produit dans results/ :
* ppo_evolution.gif — les stades s'enchaînent sur une même carte ; les essais précédents restent en fantôme
* ppo_evolution.png — un panneau par stade, avec l'issue et le taux de succès en validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from env.config import EnvConfig
from env.render import INK, MUTED, SURFACE, draw_robot, draw_scenario, save_gif
from env.robot_env import NavigationEnv
from env.rollout import Episode, run_episode
from env.seeds import TEST_SEEDS
from expert import VFHExpert
from models import ActorPolicy, load_actor

SNAPSHOT_DIR = Path("checkpoints/ppo_snapshots")

# Rampe séquentielle (une teinte, clair -> foncé) : la couleur encode l'avancement de l'entraînement.
RAMP = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281"]


def load_snapshots(cfg: EnvConfig, directory: Path = SNAPSHOT_DIR) -> list[dict]:
    """Liste triée de {'steps', 'val_success', 'policy', 'path'}."""
    out = []
    for path in sorted(directory.glob("ppo_*.pt")):
        actor, ckpt = load_actor(path, cfg)
        meta = ckpt["meta"]
        out.append(
            {
                "steps": int(meta["timesteps"]),
                "val_success": float(meta.get("val_success", float("nan"))),
                "policy": ActorPolicy(actor),
                "path": path,
            }
        )
    if not out:
        raise FileNotFoundError(f"aucun instantané dans {directory} : lancer `python -m rl.train`")
    return sorted(out, key=lambda s: s["steps"])


def steps_label(steps: int) -> str:
    if steps == 0:
        return "avant entraînement"
    if steps >= 1_000_000:
        return f"{steps / 1e6:.1f} M pas".replace(".0 ", " ")
    return f"{steps // 1000} k pas"


def color_for(i: int, n: int) -> str:
    return RAMP[round(i * (len(RAMP) - 1) / max(1, n - 1))]


def pick_map(snapshots, cfg, n_candidates: int = 200) -> tuple[int, list[Episode]]:
    return pick_maps(snapshots, cfg, 1, n_candidates)[0]


def pick_maps(snapshots, cfg, n_maps: int, n_candidates: int = 200) -> list[tuple[int, list[Episode]]]:
    """Choisit une carte de test qui raconte bien l'apprentissage.

    Critères : départ dégagé (sinon les premiers stades s'écrasent en 3 pas et on ne voit rien),
    contournement nécessaire, premiers stades qui errent visiblement, puis une fois le succès
    atteint il ne se perd plus. Parmi ces cartes, on préfère celles où le premier succès arrive vers
    le milieu de l'entraînement et où les échecs sont variés (collision, sortie, timeout).
    """
    env = NavigationEnv(cfg)
    expert = VFHExpert(cfg)
    n = len(snapshots)
    ranked = []
    for seed in range(TEST_SEEDS, TEST_SEEDS + n_candidates):
        ref = run_episode(env, expert, seed=seed)
        sc = ref.scenario
        straight = np.linalg.norm(sc.goal - sc.start)
        start_clearance = np.min(np.linalg.norm(sc.obstacles[:, :2] - sc.start, axis=1) - sc.obstacles[:, 2])
        if not ref.success or ref.path_length < 1.15 * straight or start_clearance < 1.2:
            continue
        eps = [run_episode(env, s["policy"], seed=seed) for s in snapshots]
        success = [e.success for e in eps]
        if success[0] or not success[-1] or eps[0].path_length < 2.0:
            continue
        first = success.index(True)
        if not all(success[first:]) or first < 2:
            continue
        variety = len({e.outcome for e in eps[:first]})
        wander = np.mean([e.path_length for e in eps[:first]])
        score = -abs(first - n / 2) + variety + 0.1 * wander
        ranked.append((score, seed, eps))
    if not ranked:
        raise RuntimeError("aucune carte ne montre une progression nette")
    ranked.sort(key=lambda r: -r[0])
    return [(seed, eps) for _, seed, eps in ranked[:n_maps]]


def title_for(snap: dict, ep: Episode, cfg: EnvConfig) -> str:
    issue = f"succès en {ep.steps * cfg.dt:.1f} s" if ep.success else ep.outcome
    return f"PPO — {steps_label(snap['steps'])} : {issue}"


def evolution_gif(snapshots, episodes, cfg: EnvConfig, path: Path, stride: int = 2):
    fig, ax = plt.subplots(figsize=(5.2, 5.6), dpi=90)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.05, top=0.88)
    sc = episodes[0].scenario
    frames = []
    n = len(snapshots)
    for i, (snap, ep) in enumerate(zip(snapshots, episodes, strict=True)):
        color = color_for(i, n)
        ts = list(range(0, len(ep.trajectory), stride)) + [len(ep.trajectory) - 1]
        for k, t in enumerate(ts + [ts[-1]] * 12):  # on marque une pause sur l'issue
            ax.clear()
            draw_scenario(ax, sc, cfg)
            for prev in episodes[:i]:
                ax.plot(prev.trajectory[:, 0], prev.trajectory[:, 1], color=MUTED, lw=1, alpha=0.45, zorder=3)
            traj = ep.trajectory[: t + 1]
            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=2.5, zorder=4)
            draw_robot(ax, traj[-1], cfg, color)
            done = k >= len(ts) - 1
            head = title_for(snap, ep, cfg) if done else f"PPO — {steps_label(snap['steps'])}"
            ax.set_title(head, fontsize=10, color=INK, loc="left")
            ax.text(
                0.0,
                1.075,
                f"stade {i + 1}/{n} · succès en validation : {snap['val_success']:.0%}",
                transform=ax.transAxes,
                fontsize=8,
                color=MUTED,
            )
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    save_gif(frames, path, fps=int(round(1 / (cfg.dt * stride))), hold_last=20)


def evolution_grid(snapshots, episodes, cfg: EnvConfig, path: Path):
    n = len(snapshots)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.3 * rows), dpi=110, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for i, (ax, snap, ep) in enumerate(zip(axes.flat, snapshots, episodes, strict=False)):
        color = color_for(i, n)
        draw_scenario(ax, ep.scenario, cfg)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(ep.trajectory[:, 0], ep.trajectory[:, 1], color=color, lw=2.2)
        draw_robot(ax, ep.trajectory[-1], cfg, color)
        issue = "succès" if ep.success else ep.outcome
        ax.set_title(f"{steps_label(snap['steps'])} — {issue}", fontsize=9, color=INK, loc="left")
        ax.text(0.02, 0.02, f"validation : {snap['val_success']:.0%}", transform=ax.transAxes, fontsize=7, color=MUTED)
    fig.suptitle(
        "PPO apprend par essai-erreur : même carte de test, stades d'entraînement successifs",
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(h_pad=2.0)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="carte de test (défaut : choisie automatiquement)")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = EnvConfig()
    snapshots = load_snapshots(cfg)
    if args.seed is None:
        seed, episodes = pick_map(snapshots, cfg)
    else:
        env = NavigationEnv(cfg)
        seed, episodes = args.seed, [run_episode(env, s["policy"], seed=args.seed) for s in snapshots]
    for snap, ep in zip(snapshots, episodes, strict=True):
        print(f"{steps_label(snap['steps']):>20}  validation {snap['val_success']:5.0%}  carte {seed} : {ep.outcome}")
    args.out.mkdir(exist_ok=True)
    evolution_grid(snapshots, episodes, cfg, args.out / "ppo_evolution.png")
    evolution_gif(snapshots, episodes, cfg, args.out / "ppo_evolution.gif")
    print(f"Écrit : {args.out / 'ppo_evolution.gif'}, {args.out / 'ppo_evolution.png'}")


if __name__ == "__main__":
    main()
