"""Exécution d'épisodes : une politique est simplement une fonction obs -> action."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from env.robot_env import NavigationEnv
from env.scenario import Scenario

Policy = Callable[[np.ndarray], np.ndarray]


@dataclass
class Episode:
    trajectory: np.ndarray  # (T+1, 3)
    observations: np.ndarray  # (T, obs_dim)
    actions: np.ndarray  # (T, 2)
    total_reward: float
    success: bool
    collision: bool
    out_of_bounds: bool
    min_clearance: float
    scenario: Scenario

    @property
    def steps(self) -> int:
        return len(self.actions)

    @property
    def outcome(self) -> str:
        if self.success:
            return "succès"
        if self.collision:
            return "collision"
        if self.out_of_bounds:
            return "sortie"
        return "timeout"

    @property
    def path_length(self) -> float:
        return float(np.linalg.norm(np.diff(self.trajectory[:, :2], axis=0), axis=1).sum())


def run_episode(
    env: NavigationEnv, policy: Policy, seed: int | None = None, scenario: Scenario | None = None
) -> Episode:
    options = {"scenario": scenario} if scenario is not None else None
    obs, info = env.reset(seed=seed, options=options)
    observations, actions = [], []
    total, min_clear = 0.0, info["min_clearance"]
    while True:
        action = np.asarray(policy(obs), dtype=np.float32)
        observations.append(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        min_clear = min(min_clear, info["min_clearance"])
        if terminated or truncated:
            break
    return Episode(
        trajectory=np.array(env.trajectory),
        observations=np.array(observations),
        actions=np.array(actions),
        total_reward=total,
        success=info["is_success"],
        collision=info["collision"],
        out_of_bounds=info["out_of_bounds"],
        min_clearance=min_clear,
        scenario=env.scenario,
    )
