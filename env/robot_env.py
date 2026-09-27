"""Environnement Gymnasium : robot différentiel qui doit rejoindre une cible sans collision.

Modèle cinématique (unicycle, le modèle standard d'un robot à deux roues motrices) :
    x' = v cos(theta),  y' = v sin(theta),  theta' = omega
intégré par la méthode du point milieu, plus précise qu'Euler pour les virages
à pas de temps constant.

Action normalisée a ∈ [-1, 1]^2 :
    v     = v_max * (a[0] + 1) / 2   ∈ [0, v_max]      (marche avant uniquement)
    omega = omega_max * a[1]         ∈ [-omega_max, omega_max]
Normaliser l'action dans [-1, 1] permet la même tête de sortie (tanh) pour le BC
et une distribution gaussienne bien calibrée pour PPO.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.config import EnvConfig
from env.observation import build_observation, obs_dim
from env.scenario import Scenario, sample_scenario, surface_distances


def action_to_velocity(action: np.ndarray, cfg: EnvConfig) -> tuple[float, float]:
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return cfg.v_max * (a[0] + 1.0) / 2.0, cfg.omega_max * a[1]


def velocity_to_action(v: float, omega: float, cfg: EnvConfig) -> np.ndarray:
    a = np.array([2.0 * v / cfg.v_max - 1.0, omega / cfg.omega_max], dtype=np.float32)
    return np.clip(a, -1.0, 1.0)


def integrate_unicycle(pose: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
    x, y, theta = pose
    mid = theta + 0.5 * omega * dt
    theta_new = (theta + omega * dt + np.pi) % (2 * np.pi) - np.pi
    return np.array([x + v * np.cos(mid) * dt, y + v * np.sin(mid) * dt, theta_new])


class NavigationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(obs_dim(self.cfg),), dtype=np.float32)
        self.scenario: Scenario | None = None
        self.pose = np.zeros(3)
        self.trajectory: list[np.ndarray] = []
        self._renderer = None

    # --- API Gymnasium -------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.scenario = options.get("scenario") or sample_scenario(self.np_random, self.cfg)
        self.pose = np.array([*self.scenario.start, self.scenario.start_theta], dtype=np.float64)
        self.steps = 0
        self.trajectory = [self.pose.copy()]
        self._prev_dist = self._goal_distance()
        return self._obs(), self._info()

    def step(self, action):
        cfg, rw = self.cfg, self.cfg.reward
        v, omega = action_to_velocity(action, cfg)
        self.pose = integrate_unicycle(self.pose, v, omega, cfg.dt)
        self.steps += 1
        self.trajectory.append(self.pose.copy())

        dist = self._goal_distance()
        clearance = self._min_clearance()
        collision = clearance <= 0.0
        lo, hi = -cfg.boundary_margin, cfg.arena_size + cfg.boundary_margin
        out_of_bounds = not (lo <= self.pose[0] <= hi and lo <= self.pose[1] <= hi)
        success = dist <= cfg.goal_radius and not collision

        # Shaping par potentiel (Ng et al., 1999) : r = Φ(s) - Φ(s') avec Φ = distance à la cible.
        # Il densifie le signal sans changer la politique optimale.
        reward = rw.progress * (self._prev_dist - dist) + rw.time
        if clearance < rw.proximity_margin:
            reward += rw.proximity * (1.0 - max(clearance, 0.0) / rw.proximity_margin)
        if success:
            reward += rw.success
        if collision:
            reward += rw.collision
        if out_of_bounds:
            reward += rw.out_of_bounds
        self._prev_dist = dist

        terminated = bool(success or collision or out_of_bounds)
        truncated = bool(not terminated and self.steps >= cfg.max_steps)
        info = self._info(collision=collision, out_of_bounds=out_of_bounds, success=success)
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        from env.render import Renderer

        if self._renderer is None:
            self._renderer = Renderer(self.cfg)
        return self._renderer.frame(self.scenario, {"policy": np.array(self.trajectory)})

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # --- utilitaires ---------------------------------------------------------------

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.scenario.goal - self.pose[:2]))

    def _min_clearance(self) -> float:
        d = surface_distances(self.pose[:2], self.scenario.obstacles, self.cfg.robot_radius)
        return float(d.min()) if len(d) else np.inf

    def _obs(self) -> np.ndarray:
        return build_observation(self.pose, self.scenario.goal, self.scenario.obstacles, self.cfg)

    def _info(self, collision=False, out_of_bounds=False, success=False) -> dict:
        return {
            "is_success": bool(success),
            "collision": bool(collision),
            "out_of_bounds": bool(out_of_bounds),
            "distance_to_goal": self._goal_distance(),
            "min_clearance": self._min_clearance(),
        }
