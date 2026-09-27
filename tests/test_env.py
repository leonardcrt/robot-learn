import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env import EnvConfig, NavigationEnv, velocity_to_action
from env.observation import build_observation, split_observation, world_to_robot
from env.robot_env import action_to_velocity, integrate_unicycle
from env.scenario import Scenario, is_feasible, sample_scenario


def test_gymnasium_api():
    check_env(NavigationEnv(), skip_render_check=True)


def test_reset_is_deterministic_with_seed():
    env = NavigationEnv()
    o1, _ = env.reset(seed=42)
    o2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(o1, o2)


def test_straight_line_kinematics():
    pose = integrate_unicycle(np.array([0.0, 0.0, 0.0]), v=1.0, omega=0.0, dt=0.1)
    np.testing.assert_allclose(pose, [0.1, 0.0, 0.0])


def test_turn_in_place():
    pose = integrate_unicycle(np.array([1.0, 1.0, 0.0]), v=0.0, omega=2.0, dt=0.1)
    np.testing.assert_allclose(pose, [1.0, 1.0, 0.2])


def test_action_mapping_roundtrip():
    cfg = EnvConfig()
    a = velocity_to_action(0.4, -1.0, cfg)
    v, w = action_to_velocity(a, cfg)
    assert v == pytest.approx(0.4) and w == pytest.approx(-1.0)


def test_world_to_robot_rotation():
    # Un point au nord d'un robot orienté vers le nord est "devant" (x_robot > 0).
    rel = world_to_robot(np.array([[0.0, 2.0]]), theta=np.pi / 2)
    np.testing.assert_allclose(rel, [[2.0, 0.0]], atol=1e-12)


def test_observation_sorted_and_masked():
    cfg = EnvConfig()
    obstacles = np.array([[4.0, 0.0, 0.5], [2.0, 0.0, 0.5], [50.0, 50.0, 0.5]])
    obs = build_observation(np.array([0.0, 0.0, 0.0]), np.array([3.0, 4.0]), obstacles, cfg)
    goal, obst = split_observation(obs, cfg)
    assert goal[0] == pytest.approx(5.0 / cfg.arena_size)
    np.testing.assert_allclose(goal[1:], [0.6, 0.8], atol=1e-6)
    assert obst[0, 4] == 1.0 and obst[1, 4] == 1.0
    assert obst[0, 0] < obst[1, 0]  # le plus proche en premier
    assert (obst[2:] == 0).all()  # obstacle hors de portée masqué


def test_collision_terminates_with_penalty():
    cfg = EnvConfig()
    sc = Scenario(np.array([2.0, 5.0]), 0.0, np.array([8.0, 5.0]), np.array([[3.0, 5.0, 0.5]]))
    env = NavigationEnv(cfg)
    env.reset(options={"scenario": sc})
    for _ in range(20):
        _, r, term, _, info = env.step(np.array([1.0, 0.0]))
        if term:
            break
    assert info["collision"] and r < -5


def test_success_when_reaching_goal():
    cfg = EnvConfig()
    sc = Scenario(np.array([2.0, 5.0]), 0.0, np.array([7.0, 5.0]), np.zeros((0, 3)))
    env = NavigationEnv(cfg)
    env.reset(options={"scenario": sc})
    for _ in range(100):
        _, r, term, trunc, info = env.step(np.array([1.0, 0.0]))
        if term:
            break
    assert info["is_success"] and r > 5


def test_sampled_scenarios_are_feasible_and_clear():
    cfg = EnvConfig()
    rng = np.random.default_rng(0)
    for _ in range(20):
        sc = sample_scenario(rng, cfg)
        assert is_feasible(sc, cfg)
        assert np.linalg.norm(sc.goal - sc.start) >= cfg.min_start_goal_dist
        d = np.linalg.norm(sc.obstacles[:, :2] - sc.start, axis=1) - sc.obstacles[:, 2]
        assert (d > cfg.spawn_clearance).all()


def test_blocked_scenario_is_infeasible():
    cfg = EnvConfig()
    wall = np.array([[5.0, y, 0.5] for y in np.arange(0.0, 10.5, 0.5)])
    sc = Scenario(np.array([2.0, 5.0]), 0.0, np.array([8.0, 5.0]), wall)
    assert not is_feasible(sc, cfg)
