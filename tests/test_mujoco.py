"""Tests de l'extension MuJoCo. Ignorés si MuJoCo n'est pas installé (pip install -e ".[mujoco]")."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from env import EnvConfig, NavigationEnv, Scenario  # noqa: E402
from env.robot_env import velocity_to_action  # noqa: E402
from env.rollout import run_episode  # noqa: E402
from expert import VFHExpert  # noqa: E402
from sim_mujoco.mujoco_env import MuJoCoNavEnv  # noqa: E402

CFG = EnvConfig()
VITESSE_MAX = velocity_to_action(1.0, 0.0, CFG)  # v = 1 m/s, omega = 0


def _carte_vide():
    return Scenario(np.array([2.0, 5.0]), 0.0, np.array([9.0, 5.0]), np.zeros((0, 3)))


def test_same_maps_as_kinematic_env():
    cinematique, mujoco = NavigationEnv(CFG), MuJoCoNavEnv(CFG)
    for seed in (0, 7, 1_000_000):
        cinematique.reset(seed=seed)
        mujoco.reset(seed=seed)
        np.testing.assert_array_equal(cinematique.scenario.obstacles, mujoco.scenario.obstacles)
        np.testing.assert_array_equal(cinematique.scenario.goal, mujoco.scenario.goal)


def test_velocity_step_response():
    """Le robot atteint 90 % de la vitesse demandée en 0,3 à 0,5 s, sans aucune vitesse latérale."""
    env = MuJoCoNavEnv(CFG)
    env.reset(options={"scenario": _carte_vide()})
    vitesses = []
    for _ in range(15):
        env.step(VITESSE_MAX)
        vitesses.append(env.data.qvel[0])
    t90 = (int(np.argmax(np.array(vitesses) >= 0.9)) + 1) * CFG.dt
    assert 0.3 <= t90 <= 0.5
    assert abs(env.data.qvel[1]) < 1e-3


def test_collision_comes_from_contact():
    obstacle = Scenario(np.array([2.0, 5.0]), 0.0, np.array([9.0, 5.0]), np.array([[4.0, 5.0, 0.5]]))
    env = MuJoCoNavEnv(CFG)
    env.reset(options={"scenario": obstacle})
    for _ in range(40):
        _, reward, terminated, _, info = env.step(VITESSE_MAX)
        if terminated:
            break
    assert info["collision"] and reward < -9
    assert env.pose[0] < 4.0 - 0.5  # arrêté avant le centre de l'obstacle


def test_success_reward_and_expert_in_empty_map():
    env = MuJoCoNavEnv(CFG)
    ep = run_episode(env, VFHExpert(CFG), scenario=_carte_vide())
    assert ep.success and ep.total_reward > 5


def test_evaluate_policy_accepts_env_factory():
    """Le paramètre optionnel env_factory de evaluate_policy fait évaluer dans MuJoCo."""
    from eval.metrics import evaluate_policy

    resume, episodes = evaluate_policy(VFHExpert(CFG), [10_000, 10_001], CFG, env_factory=MuJoCoNavEnv)
    assert resume["episodes"] == 2 and len(episodes) == 2


def test_expert_succeeds_on_most_mujoco_maps():
    env = MuJoCoNavEnv(CFG)
    expert = VFHExpert(CFG)
    reussites = [run_episode(env, expert, seed=10_000 + i).success for i in range(12)]
    assert np.mean(reussites) >= 0.7
