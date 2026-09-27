"""Génération de scénarios : position de départ, cible, obstacles circulaires.

Un scénario est entièrement déterminé par une graine, ce qui permet de rejouer
exactement la même carte pour l'expert, le BC et le RL lors de l'évaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.config import EnvConfig


@dataclass(frozen=True)
class Scenario:
    start: np.ndarray  # (2,) position initiale
    start_theta: float  # cap initial (rad)
    goal: np.ndarray  # (2,)
    obstacles: np.ndarray  # (N, 3) : x, y, rayon


def surface_distances(pos: np.ndarray, obstacles: np.ndarray, robot_radius: float) -> np.ndarray:
    """Distance entre le bord du robot et le bord de chaque obstacle (négative = collision)."""
    if len(obstacles) == 0:
        return np.empty(0)
    centers_dist = np.linalg.norm(obstacles[:, :2] - pos, axis=1)
    return centers_dist - obstacles[:, 2] - robot_radius


def _sample_obstacle_near_segment(rng, a, b, r_range):
    t = rng.uniform(0.25, 0.75)
    direction = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-direction[1], direction[0]])
    center = a + t * (b - a) + normal * rng.normal(0.0, 0.4)
    return np.array([*center, rng.uniform(*r_range)])


def _sample_obstacle_uniform(rng, size, r_range):
    return np.array([*rng.uniform(0.0, size, size=2), rng.uniform(*r_range)])


def is_feasible(sc: Scenario, cfg: EnvConfig, resolution: float = 0.1) -> bool:
    """Vérifie qu'un chemin existe du départ à la cible (flood-fill sur une grille).

    Les obstacles sont gonflés du rayon du robot + une petite marge : on ne garde
    que des scénarios qu'un contrôleur raisonnable peut résoudre. Sans ce filtre,
    une partie des échecs mesurés viendrait de cartes impossibles, pas des politiques.
    """
    n = int(round(cfg.arena_size / resolution))
    coords = (np.arange(n) + 0.5) * resolution
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    free = np.ones((n, n), dtype=bool)
    margin = cfg.robot_radius + 0.05
    for x, y, r in sc.obstacles:
        free &= (xx - x) ** 2 + (yy - y) ** 2 > (r + margin) ** 2

    def cell(p):
        return tuple(np.clip((p / resolution).astype(int), 0, n - 1))

    s, g = cell(sc.start), cell(sc.goal)
    if not free[s] or not free[g]:
        return False
    reached = np.zeros_like(free)
    reached[s] = True
    while True:
        grown = reached.copy()
        grown[1:, :] |= reached[:-1, :]
        grown[:-1, :] |= reached[1:, :]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= free
        if grown[g]:
            return True
        if (grown == reached).all():
            return False
        reached = grown


def sample_scenario(rng: np.random.Generator, cfg: EnvConfig) -> Scenario:
    """Tire un scénario faisable au hasard (rejection sampling)."""
    size = cfg.arena_size
    while True:
        start = rng.uniform(1.0, size - 1.0, size=2)
        goal = rng.uniform(1.0, size - 1.0, size=2)
        if np.linalg.norm(goal - start) < cfg.min_start_goal_dist:
            continue

        obstacles = []
        attempts = 0
        while len(obstacles) < cfg.n_obstacles and attempts < 200:
            attempts += 1
            if len(obstacles) < cfg.n_obstacles_on_path:
                obs = _sample_obstacle_near_segment(rng, start, goal, cfg.obstacle_radius_range)
            else:
                obs = _sample_obstacle_uniform(rng, size, cfg.obstacle_radius_range)
            free_start = np.linalg.norm(obs[:2] - start) - obs[2] > cfg.spawn_clearance
            free_goal = np.linalg.norm(obs[:2] - goal) - obs[2] > cfg.spawn_clearance
            if free_start and free_goal:
                obstacles.append(obs)

        sc = Scenario(
            start=start,
            start_theta=float(rng.uniform(-np.pi, np.pi)),
            goal=goal,
            obstacles=np.array(obstacles).reshape(-1, 3),
        )
        if is_feasible(sc, cfg):
            return sc
