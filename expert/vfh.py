"""Contrôleur expert principal : Vector Field Histogram simplifié (Borenstein & Koren, 1991).

Pourquoi pas les champs de potentiel seuls ? Ils ont été essayés en premier
(`expert/potential_field.py`) et plafonnent autour de 70 % de succès : devant un
amas d'obstacles concave, attraction et répulsion s'annulent et le robot reste
bloqué (minimum local). Le VFH raisonne plutôt sur les *directions* libres :

1. On teste N directions candidates autour du robot (tous les 5°).
2. Une direction est bloquée si le segment [robot, robot + L·u] passe à moins de
   (rayon obstacle + rayon robot + marge) du centre d'un obstacle visible.
3. Parmi les directions libres, on prend la plus proche de la direction de la
   cible (avec un léger biais vers le cap actuel pour départager les égalités).
4. Un contrôleur proportionnel sur l'erreur de cap donne omega ; la vitesse v
   diminue quand le robot est mal orienté, près d'un obstacle ou de la cible.

Comme le champ de potentiel, il ne lit que le vecteur d'observation : l'élève (BC)
dispose donc exactement de la même information que le professeur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.config import EnvConfig
from env.observation import split_observation
from env.robot_env import velocity_to_action


@dataclass(frozen=True)
class VFHConfig:
    n_directions: int = 72
    lookahead: float = 2.5  # longueur (m) du segment testé dans chaque direction
    safety_margin: float = 0.2  # marge ajoutée au gonflement des obstacles
    # Coût par radian d'écart au cap actuel. Assez fort pour que, pendant une rotation, l'option
    # choisie devienne de plus en plus favorable : sinon deux trouées symétriques font osciller le robot.
    turn_bias: float = 0.25
    k_heading: float = 3.0
    obstacle_slowdown: float = 0.6
    goal_slowdown: float = 1.0
    min_speed_factor: float = 0.25


class VFHExpert:
    def __init__(self, env_cfg: EnvConfig | None = None, cfg: VFHConfig | None = None):
        self.env_cfg = env_cfg or EnvConfig()
        self.cfg = cfg or VFHConfig()
        # Décalages angulaires par rapport à la direction de la cible. Ancrer la grille sur la cible
        # (et non sur le cap du robot) rend la discrétisation stable quand le robot pivote.
        self._offsets = np.linspace(-np.pi, np.pi, self.cfg.n_directions, endpoint=False)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.act(obs)

    def candidate_angles(self, obs: np.ndarray) -> np.ndarray:
        """Angles candidats dans le repère robot, dans (-pi, pi]."""
        goal_angle = np.arctan2(obs[2], obs[1])
        return (goal_angle + self._offsets + np.pi) % (2 * np.pi) - np.pi

    def blocked_directions(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Masque (N,) des directions bloquées, distance à la cible, distance de surface minimale."""
        ec, c = self.env_cfg, self.cfg
        goal, obstacles = split_observation(np.asarray(obs, dtype=np.float64), ec)
        goal_dist = goal[0] * ec.arena_size
        angles = self.candidate_angles(obs)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        blocked = np.zeros(len(angles), dtype=bool)

        valid = obstacles[obstacles[:, 4] > 0.5]
        if len(valid) == 0:
            return blocked, goal_dist, np.inf
        p = valid[:, :2] * ec.sensing_range  # centres dans le repère robot
        surface = valid[:, 2] * ec.sensing_range
        center_dist = np.linalg.norm(p, axis=1)
        inflated = center_dist - surface + c.safety_margin  # rayon obstacle + rayon robot + marge

        # On ne regarde pas plus loin que la cible : un obstacle derrière elle ne gêne pas.
        length = min(c.lookahead, goal_dist + ec.goal_radius)
        proj = dirs @ p.T  # (N, M) : abscisse du centre le long de chaque direction
        t = np.clip(proj, 0.0, length)
        dist2 = center_dist[None] ** 2 - 2 * t * proj + t**2
        hits = dist2 < inflated[None] ** 2

        # Obstacle déjà à l'intérieur de sa zone gonflée : on ne bloque que les directions qui s'en rapprochent.
        inside = center_dist < inflated
        if inside.any():
            hits[:, inside] = proj[:, inside] > 0.0
        blocked = hits.any(axis=1)
        return blocked, goal_dist, float(surface.min())

    def choose_heading(self, obs: np.ndarray) -> tuple[float, float, float]:
        blocked, goal_dist, min_surface = self.blocked_directions(obs)
        angles = self.candidate_angles(obs)
        cost = np.abs(self._offsets) + self.cfg.turn_bias * np.abs(angles)
        if blocked.all():
            # Encerclé : on pivote sur place vers la cible, la vitesse sera nulle.
            return float(np.arctan2(obs[2], obs[1])), goal_dist, 0.0
        cost[blocked] = np.inf
        return float(angles[np.argmin(cost)]), goal_dist, min_surface

    def act(self, obs: np.ndarray) -> np.ndarray:
        ec, c = self.env_cfg, self.cfg
        heading_error, goal_dist, min_surface = self.choose_heading(obs)
        omega = float(np.clip(c.k_heading * heading_error, -ec.omega_max, ec.omega_max))
        align = max(np.cos(heading_error), 0.0) ** 2
        near_obstacle = np.clip(min_surface / c.obstacle_slowdown, c.min_speed_factor, 1.0) if min_surface > 0 else 0.0
        near_goal = np.clip(goal_dist / c.goal_slowdown, c.min_speed_factor, 1.0)
        v = ec.v_max * align * near_obstacle * near_goal
        return velocity_to_action(v, omega, ec)
