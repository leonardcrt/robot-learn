"""Collecte de paires (état, action) étiquetées par l'expert."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from env.config import EnvConfig
from env.robot_env import NavigationEnv
from env.rollout import Policy, run_episode


def collect(
    behavior: Policy,
    labeler: Policy,
    seeds: list[int],
    cfg: EnvConfig,
    only_success: bool = False,
) -> dict[str, np.ndarray]:
    """Exécute `behavior` et étiquette chaque état visité avec `labeler`.

    * BC     : behavior = labeler = expert (on n'apprend que sur les états que visite l'expert).
    * DAgger : behavior = politique apprise (ou mélange), labeler = expert. C'est ce qui corrige
               le « covariate shift » : on obtient la bonne action dans les états où l'élève
               se retrouve réellement après ses propres erreurs.
    """
    env = NavigationEnv(cfg)
    all_obs, all_act, all_ep = [], [], []
    n_success = 0
    for seed in seeds:
        ep = run_episode(env, behavior, seed=seed)
        n_success += ep.success
        if only_success and not ep.success:
            continue
        labels = ep.actions if behavior is labeler else np.array([labeler(o) for o in ep.observations])
        all_obs.append(ep.observations)
        all_act.append(labels)
        all_ep.append(np.full(len(labels), seed))
    return {
        "obs": np.concatenate(all_obs).astype(np.float32),
        "act": np.concatenate(all_act).astype(np.float32),
        "episode": np.concatenate(all_ep),
        "behavior_success_rate": np.array(n_success / len(seeds)),
    }


def save_dataset(path, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_dataset(path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def merge(a: dict, b: dict) -> dict:
    return {k: np.concatenate([a[k], b[k]]) for k in ("obs", "act", "episode")}
