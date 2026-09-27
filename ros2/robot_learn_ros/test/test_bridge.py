"""Tests sans ROS : conversions et boucle fermée contrôleur + modèle cinématique."""

import math

import numpy as np
import pytest

from env.config import EnvConfig
from env.robot_env import NavigationEnv, integrate_unicycle
from robot_learn_ros.bridge import PolicyController, load_policy, quaternion_from_yaw, yaw_from_quaternion


@pytest.mark.parametrize("yaw", [0.0, 1.0, -2.5, math.pi - 1e-6])
def test_quaternion_roundtrip(yaw):
    assert yaw_from_quaternion(*quaternion_from_yaw(yaw)) == pytest.approx(yaw)


def test_controller_stops_at_goal():
    ctrl = PolicyController(load_policy("expert", EnvConfig()))
    v, w, reached = ctrl.command(np.array([1.0, 1.0, 0.0]), np.array([1.1, 1.0]), np.zeros((0, 3)))
    assert reached and v == 0.0 and w == 0.0


def test_closed_loop_matches_training_env():
    """Même boucle que sim_node + policy_node, sans ROS : l'expert doit atteindre la cible."""
    cfg = EnvConfig()
    env = NavigationEnv(cfg)
    env.reset(seed=1_000_000)
    sc = env.scenario
    ctrl = PolicyController(load_policy("expert", cfg), cfg)
    pose = np.array([*sc.start, sc.start_theta])
    for _ in range(cfg.max_steps):
        v, w, reached = ctrl.command(pose, sc.goal, sc.obstacles)
        if reached:
            break
        pose = integrate_unicycle(pose, v, w, cfg.dt)
    assert reached
