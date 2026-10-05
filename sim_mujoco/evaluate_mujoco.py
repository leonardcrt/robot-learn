"""Évalue les politiques dans MuJoCo, sans réentraînement, sur les mêmes cartes de test.

    python -m sim_mujoco.evaluate_mujoco --episodes 500

Écrit dans results/mujoco/ :
* benchmark_mujoco.md  : même format que results/benchmark.md (qui n'est pas modifié)
* comparaison_simulateurs.md : les mêmes politiques dans le simulateur cinématique et dans MuJoCo
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch

from env.config import EnvConfig
from env.seeds import TEST_SEEDS, seed_range
from eval.benchmark import markdown_table
from eval.metrics import evaluate_policy, format_summary
from eval.policies import CHECKPOINTS, LABELS, load_policy
from sim_mujoco.mujoco_env import MuJoCoNavEnv
from sim_mujoco.robot_model import RobotParams

MUJOCO_CKPT = Path("checkpoints/mujoco")
BASE_POLICIES = ["expert", "bc", "dagger", "ppo"]
LABELS["ppo_mujoco"] = "PPO (entraîné dans MuJoCo)"  # en mémoire seulement : policies.py n'est pas modifié


def _evaluate(name: str, n: int, params: RobotParams) -> tuple[str, dict]:
    torch.set_num_threads(1)
    cfg = EnvConfig()
    # 1. Les checkpoints existants sont rechargés tels quels ; ppo_mujoco vient de son propre dossier.
    policy = load_policy(name, cfg, MUJOCO_CKPT if name == "ppo_mujoco" else CHECKPOINTS)
    # 2. Mêmes cartes de test que results/benchmark.md, mais dans MuJoCoNavEnv.
    factory = partial(MuJoCoNavEnv, params=params)
    summary, _ = evaluate_policy(policy, seed_range(TEST_SEEDS, n), cfg, env_factory=factory)
    return name, summary


def gap_table(mujoco: dict[str, dict], kinematic: dict[str, dict]) -> str:
    """Tableau cinématique contre MuJoCo pour les politiques évaluées dans les deux."""
    lines = [
        "| Politique | Succès cinématique | Succès MuJoCo | Collisions cinématique | Collisions MuJoCo "
        "| Durée cinématique (s) | Durée MuJoCo (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in mujoco.items():
        k = kinematic.get(name)
        # Une politique absente du benchmark cinématique (ppo_mujoco) n'a pas de chiffres cinématiques.
        succes_k = f"{k['success_rate']:.1%}" if k else "—"
        collision_k = f"{k['collision_rate']:.1%}" if k else "—"
        duree_k = f"{k['time_to_goal_s']:.1f}" if k else "—"
        lines.append(
            f"| {LABELS.get(name, name)} | {succes_k} | {m['success_rate']:.1%} | {collision_k} "
            f"| {m['collision_rate']:.1%} | {duree_k} | {m['time_to_goal_s']:.1f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/mujoco"))
    # Test de sensibilité : un robot plus lent (gains plus faibles = constantes de temps plus longues).
    parser.add_argument("--k-v", type=float, default=RobotParams().k_v, help="gain de vitesse avant (N·s/m)")
    parser.add_argument("--k-w", type=float, default=RobotParams().k_w, help="gain de rotation (N·m·s/rad)")
    args = parser.parse_args()

    params = replace(RobotParams(), k_v=args.k_v, k_w=args.k_w)
    names = args.policies or BASE_POLICIES + (["ppo_mujoco"] if (MUJOCO_CKPT / "ppo_mujoco.pt").exists() else [])
    with ProcessPoolExecutor(max_workers=len(names)) as pool:
        results = dict(pool.map(_evaluate, names, [args.episodes] * len(names), [params] * len(names)))
    for name, summary in results.items():
        print(format_summary(name, summary))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark_mujoco.json").write_text(json.dumps(results, indent=1))
    (args.out / "benchmark_mujoco.md").write_text(markdown_table(results) + "\n", encoding="utf-8")
    kinematic = json.loads(Path("results/benchmark.json").read_text())  # lecture seule
    (args.out / "comparaison_simulateurs.md").write_text(gap_table(results, kinematic) + "\n", encoding="utf-8")
    print("\n" + markdown_table(results) + "\n\n" + gap_table(results, kinematic))


if __name__ == "__main__":
    main()
