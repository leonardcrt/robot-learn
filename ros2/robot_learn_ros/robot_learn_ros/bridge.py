"""Logique indépendante de ROS : conversions et calcul de la commande.

Séparée des nœuds pour être testable sans installation ROS2. Le vecteur d'observation
est construit par `env.observation.build_observation`, la même fonction que pendant
l'entraînement : il n'existe qu'une seule définition de ce que « voit » la politique.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from env.config import EnvConfig
from env.observation import build_observation
from env.robot_env import action_to_velocity


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def load_policy(spec: str, cfg: EnvConfig):
    """`spec` = "expert" (contrôleur VFH) ou chemin vers un checkpoint `models` (.pt)."""
    if spec == "expert":
        from expert import VFHExpert

        return VFHExpert(cfg)
    from models import ActorPolicy

    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint introuvable : {path}")
    return ActorPolicy.load(path, cfg)


class PolicyController:
    def __init__(self, policy, cfg: EnvConfig | None = None, goal_tolerance: float | None = None):
        self.cfg = cfg or EnvConfig()
        self.policy = policy
        self.goal_tolerance = goal_tolerance if goal_tolerance is not None else self.cfg.goal_radius

    def command(self, pose: np.ndarray, goal: np.ndarray, obstacles: np.ndarray) -> tuple[float, float, bool]:
        """Renvoie (v, omega, cible_atteinte). pose = (x, y, yaw), obstacles = (N, 3)."""
        if np.linalg.norm(goal - pose[:2]) <= self.goal_tolerance:
            return 0.0, 0.0, True
        obs = build_observation(pose, goal, obstacles.reshape(-1, 3), self.cfg)
        v, omega = action_to_velocity(self.policy(obs), self.cfg)
        return float(v), float(omega), False
