"""Benchmark final : toutes les politiques sur les mêmes cartes de test, jamais vues à l'entraînement.

    python -m eval.benchmark --episodes 500

Protocole :
* 500 scénarios tirés des graines de test (`env/seeds.py::TEST_SEEDS`), identiques pour
  toutes les politiques : chaque méthode affronte exactement les mêmes cartes.
* Politiques apprises évaluées en mode déterministe (pas de bruit d'exploration).
* Métriques : taux de succès (+ IC 95 % de Wilson), collisions, timeouts, SPL (succès
  pondéré par la longueur du chemin), durée moyenne des succès, distance minimale aux
  obstacles, temps d'inférence par pas.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from env.config import EnvConfig
from env.seeds import TEST_SEEDS, seed_range
from eval.metrics import evaluate_policy, format_summary
from eval.policies import LABELS, available_policies, load_policy


def _evaluate(name: str, n: int) -> tuple[str, dict, list[bool]]:
    import torch

    torch.set_num_threads(1)
    cfg = EnvConfig()
    summary, episodes = evaluate_policy(load_policy(name, cfg), seed_range(TEST_SEEDS, n), cfg)
    return name, summary, [ep.success for ep in episodes]


def markdown_table(results: dict[str, dict]) -> str:
    lines = [
        "| Politique | Succès | IC 95 % | Collision | Timeout | Sortie | SPL | Durée (s) | Marge min (m) | ms/pas |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in results.items():
        lo, hi = s["success_ci95"]
        lines.append(
            f"| {LABELS.get(name, name)} | **{s['success_rate']:.1%}** | {lo:.0%}–{hi:.0%} | {s['collision_rate']:.1%} "
            f"| {s['timeout_rate']:.1%} | {s['out_of_bounds_rate']:.1%} | {s['spl']:.3f} | {s['time_to_goal_s']:.1f} "
            f"| {s['min_clearance_m']:.2f} | {s['ms_per_step']:.2f} |"
        )
    return "\n".join(lines)


def plot_outcomes(results: dict[str, dict], path: Path):
    import matplotlib.pyplot as plt

    from env.render import GRID, INK, INK_SECONDARY, MUTED, SURFACE

    # Issues d'épisode = états : palette de statut, toujours accompagnée d'un libellé.
    outcomes = [
        ("success_rate", "succès", "#0ca30c"),
        ("collision_rate", "collision", "#d03b3b"),
        ("timeout_rate", "timeout", "#fab219"),
        ("out_of_bounds_rate", "sortie", "#ec835a"),
    ]
    names = list(results)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(names) + 1.4), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    left = np.zeros(len(names))
    for key, label, color in outcomes:
        vals = np.array([100 * results[n][key] for n in names])
        ax.barh(names, vals, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.6, label=label)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7, color=INK)
        left += vals
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS.get(n, n) for n in names], color=INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% des épisodes de test", color=INK_SECONDARY)
    ax.set_title(
        f"Issue des épisodes ({results[names[0]]['episodes']} cartes de test)", color=INK, loc="left", fontsize=10
    )
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    names = args.policies or available_policies()
    with ProcessPoolExecutor(max_workers=len(names)) as pool:
        outputs = list(pool.map(_evaluate, names, [args.episodes] * len(names)))

    results = {name: summary for name, summary, _ in outputs}
    for name, s in results.items():
        print(format_summary(name, s))

    args.out.mkdir(exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps(results, indent=1))
    per_episode = {name: successes for name, _, successes in outputs}
    (args.out / "benchmark_episodes.json").write_text(json.dumps(per_episode))
    table = markdown_table(results)
    (args.out / "benchmark.md").write_text(table + "\n", encoding="utf-8")
    plot_outcomes(results, args.out / "benchmark.png")
    print("\n" + table)


if __name__ == "__main__":
    main()
