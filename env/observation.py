"""Construction du vecteur d'observation, commune à l'environnement, à l'expert et au nœud ROS2.

Le vecteur est exprimé dans le repère du robot (égocentrique) : la politique n'a pas
besoin d'apprendre que "aller vers une cible au nord-est" ou "au sud-ouest" est le
même problème à une rotation près. Cette invariance réduit fortement la quantité
de données nécessaire.

Disposition (dimension = 3 + 5*K) :
    [0:3]   cible     : distance / taille_arène, cos(angle), sin(angle)
    [3: ]   K obstacles triés du plus proche au plus lointain, 5 valeurs chacun :
            x_rel / portée, y_rel / portée, distance_de_surface / portée, rayon, masque
            (masque = 1 si l'obstacle est détecté, 0 pour un emplacement vide ; toutes
            les valeurs d'un emplacement vide valent 0)
"""

from __future__ import annotations

import numpy as np

from env.config import EnvConfig

GOAL_DIM = 3
OBSTACLE_DIM = 5


def obs_dim(cfg: EnvConfig) -> int:
    return GOAL_DIM + OBSTACLE_DIM * cfg.k_nearest


def world_to_robot(vectors: np.ndarray, theta: float) -> np.ndarray:
    """Tourne des vecteurs (N, 2) du repère monde vers le repère robot."""
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, s], [-s, c]])
    return vectors @ rot.T


def build_observation(pose: np.ndarray, goal: np.ndarray, obstacles: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    """pose = (x, y, theta) ; goal = (x, y) ; obstacles = (N, 3) avec (x, y, rayon)."""
    pos, theta = pose[:2], pose[2]
    obs = np.zeros(obs_dim(cfg), dtype=np.float32)

    goal_rel = world_to_robot((goal - pos)[None], theta)[0]
    dist = np.linalg.norm(goal_rel)
    obs[0] = dist / cfg.arena_size
    if dist > 1e-9:
        obs[1:3] = goal_rel / dist
    else:
        obs[1] = 1.0

    if len(obstacles) == 0:
        return obs
    rel = world_to_robot(obstacles[:, :2] - pos, theta)
    surface = np.linalg.norm(rel, axis=1) - obstacles[:, 2] - cfg.robot_radius
    visible = np.flatnonzero(surface < cfg.sensing_range)
    nearest = visible[np.argsort(surface[visible])][: cfg.k_nearest]

    r = cfg.sensing_range
    for slot, i in enumerate(nearest):
        base = GOAL_DIM + slot * OBSTACLE_DIM
        obs[base : base + OBSTACLE_DIM] = (
            rel[i, 0] / r,
            rel[i, 1] / r,
            surface[i] / r,
            obstacles[i, 2],
            1.0,
        )
    return obs


def split_observation(obs: np.ndarray, cfg: EnvConfig):
    """Sépare le vecteur en (features cible (3,), obstacles (K, 5)). Fonctionne en batch."""
    goal = obs[..., :GOAL_DIM]
    obstacles = obs[..., GOAL_DIM:].reshape(*obs.shape[:-1], cfg.k_nearest, OBSTACLE_DIM)
    return goal, obstacles
