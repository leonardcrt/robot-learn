"""Métriques d'évaluation communes à toutes les politiques."""

from __future__ import annotations

import math
import time

import numpy as np

from env.config import EnvConfig
from env.robot_env import NavigationEnv
from env.rollout import Episode, Policy, run_episode


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance à 95 % d'une proportion (plus fiable que ±1.96σ près de 0 ou 1)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize(episodes: list[Episode], cfg: EnvConfig) -> dict:
    n = len(episodes)
    succ = [ep for ep in episodes if ep.success]
    straight = np.array([np.linalg.norm(ep.scenario.goal - ep.scenario.start) for ep in episodes])
    lengths = np.array([ep.path_length for ep in episodes])
    success = np.array([ep.success for ep in episodes], dtype=float)
    # SPL (Anderson et al., 2018) : succès pondéré par l'efficacité du chemin.
    spl = float(np.mean(success * straight / np.maximum(lengths, straight)))
    lo, hi = wilson_interval(len(succ), n)
    return {
        "episodes": n,
        "success_rate": len(succ) / n,
        "success_ci95": (lo, hi),
        "collision_rate": float(np.mean([ep.collision for ep in episodes])),
        "timeout_rate": float(np.mean([ep.outcome == "timeout" for ep in episodes])),
        "out_of_bounds_rate": float(np.mean([ep.out_of_bounds for ep in episodes])),
        "spl": spl,
        "time_to_goal_s": float(np.mean([ep.steps for ep in succ]) * cfg.dt) if succ else float("nan"),
        "min_clearance_m": float(np.mean([ep.min_clearance for ep in episodes])),
        "mean_return": float(np.mean([ep.total_reward for ep in episodes])),
    }


def evaluate_policy(policy: Policy, seeds: list[int], cfg: EnvConfig | None = None) -> tuple[dict, list[Episode]]:
    cfg = cfg or EnvConfig()
    env = NavigationEnv(cfg)
    t0 = time.perf_counter()
    episodes = [run_episode(env, policy, seed=s) for s in seeds]
    elapsed = time.perf_counter() - t0
    summary = summarize(episodes, cfg)
    summary["ms_per_step"] = 1000 * elapsed / max(1, sum(ep.steps for ep in episodes))
    return summary, episodes


def format_summary(name: str, s: dict) -> str:
    lo, hi = s["success_ci95"]
    return (
        f"{name:<14} succès {s['success_rate']:6.1%} [{lo:.0%}–{hi:.0%}]  collision {s['collision_rate']:5.1%}  "
        f"timeout {s['timeout_rate']:5.1%}  SPL {s['spl']:.3f}  durée {s['time_to_goal_s']:.1f} s"
    )
