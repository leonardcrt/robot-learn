"""Environnement de navigation 2D pour un robot différentiel."""

from gymnasium.envs.registration import register

from env.config import EnvConfig, RewardConfig
from env.robot_env import NavigationEnv, action_to_velocity, velocity_to_action
from env.scenario import Scenario, sample_scenario

register(id="RobotNav-v0", entry_point="env.robot_env:NavigationEnv")

__all__ = [
    "EnvConfig",
    "RewardConfig",
    "NavigationEnv",
    "Scenario",
    "sample_scenario",
    "action_to_velocity",
    "velocity_to_action",
]
