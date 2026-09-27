"""Contrôleur expert écrit à la main : champs de potentiel + composante tourbillonnaire.

Choix de conception
-------------------
* L'expert ne lit QUE le vecteur d'observation (cible + K obstacles proches dans le
  repère robot), exactement ce que verront le BC et PPO. Un expert "privilégié" qui
  connaîtrait toute la carte prendrait des décisions que l'élève ne peut pas
  reproduire à partir de ses entrées : l'erreur du BC deviendrait irréductible et la
  comparaison serait biaisée.
* Champs de potentiel classiques (Khatib, 1986) : attraction vers la cible +
  répulsion des obstacles. Leur défaut connu : les minima locaux (obstacle pile
  entre le robot et la cible, les forces s'annulent). On ajoute donc un champ
  tangentiel (« vortex ») qui fait contourner l'obstacle par le côté où se trouve
  la cible ; toutes les composantes tangentielles partagent le même sens de
  rotation, sinon deux obstacles voisins peuvent créer des vortex opposés qui
  s'annulent dans l'interstice.
* La direction désirée est convertie en (v, omega) par un contrôleur proportionnel
  sur l'erreur de cap, avec ralentissement quand le robot est mal orienté, près
  d'un obstacle ou près de la cible (évite d'orbiter autour de la cible).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.config import EnvConfig
from env.observation import split_observation
from env.robot_env import velocity_to_action


@dataclass(frozen=True)
class ExpertConfig:
    influence: float = 1.2  # distance de surface (m) sous laquelle un obstacle agit
    k_rep: float = 0.35  # gain de répulsion
    k_tan: float = 0.8  # gain tangentiel (contournement)
    k_heading: float = 3.0  # gain proportionnel sur l'erreur de cap
    obstacle_slowdown: float = 0.6  # distance de surface sous laquelle on ralentit
    goal_slowdown: float = 1.0  # distance à la cible sous laquelle on ralentit
    min_speed_factor: float = 0.25


class PotentialFieldExpert:
    def __init__(self, env_cfg: EnvConfig | None = None, cfg: ExpertConfig | None = None):
        self.env_cfg = env_cfg or EnvConfig()
        self.cfg = cfg or ExpertConfig()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.act(obs)

    def desired_direction(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """Renvoie la direction désirée (repère robot) et la distance de surface minimale."""
        ec, c = self.env_cfg, self.cfg
        goal, obstacles = split_observation(np.asarray(obs, dtype=np.float64), ec)
        g_hat = goal[1:3]
        force = g_hat.copy()

        valid = obstacles[obstacles[:, 4] > 0.5]
        if len(valid) == 0:
            return force, np.inf
        rel = valid[:, :2] * ec.sensing_range
        surface = valid[:, 2] * ec.sensing_range
        dist = np.maximum(np.linalg.norm(rel, axis=1), 1e-6)
        p_hat = rel / dist[:, None]

        near = surface < c.influence
        strength = np.where(near, 1.0 / np.maximum(surface, 0.05) - 1.0 / c.influence, 0.0)
        blocking = np.clip(p_hat @ g_hat, 0.0, None)  # 1 si l'obstacle est pile vers la cible

        # Sens de contournement commun, décidé par l'obstacle le plus gênant :
        # passer du côté de la cible, et en cas d'égalité du côté vers lequel le robot pointe déjà.
        score = strength * blocking
        if score.max() > 0:
            i = int(np.argmax(score))
            cross = p_hat[i, 0] * g_hat[1] - p_hat[i, 1] * g_hat[0]
            side = cross - 0.3 * p_hat[i, 1]
            sign = 1.0 if side >= 0 else -1.0
        else:
            sign = 1.0
        tangent = sign * np.stack([-p_hat[:, 1], p_hat[:, 0]], axis=1)

        force += c.k_rep * (strength[:, None] * -p_hat).sum(axis=0)
        force += c.k_tan * ((strength * blocking)[:, None] * tangent).sum(axis=0)
        return force, float(surface.min())

    def act(self, obs: np.ndarray) -> np.ndarray:
        ec, c = self.env_cfg, self.cfg
        force, min_surface = self.desired_direction(obs)
        heading_error = float(np.arctan2(force[1], force[0]))
        goal_dist = float(obs[0]) * ec.arena_size

        omega = float(np.clip(c.k_heading * heading_error, -ec.omega_max, ec.omega_max))
        align = max(np.cos(heading_error), 0.0) ** 2
        near_obstacle = np.clip(min_surface / c.obstacle_slowdown, c.min_speed_factor, 1.0)
        near_goal = np.clip(goal_dist / c.goal_slowdown, c.min_speed_factor, 1.0)
        v = ec.v_max * align * near_obstacle * near_goal
        return velocity_to_action(v, omega, ec)
