import numpy as np

from env import EnvConfig, NavigationEnv, Scenario
from env.observation import build_observation
from env.rollout import run_episode
from expert import PotentialFieldExpert, VFHExpert


def test_vfh_expert_success_rate():
    cfg = EnvConfig()
    env = NavigationEnv(cfg)
    expert = VFHExpert(cfg)
    episodes = [run_episode(env, expert, seed=10_000 + i) for i in range(40)]
    assert np.mean([ep.success for ep in episodes]) >= 0.9
    assert not any(ep.collision for ep in episodes)


def test_vfh_goes_straight_without_obstacles():
    cfg = EnvConfig()
    obs = build_observation(np.array([1.0, 1.0, 0.0]), np.array([6.0, 1.0]), np.zeros((0, 3)), cfg)
    a = VFHExpert(cfg)(obs)
    np.testing.assert_allclose(a, [1.0, 0.0], atol=1e-6)


def test_vfh_avoids_obstacle_straight_ahead():
    cfg = EnvConfig()
    obstacles = np.array([[2.5, 1.0, 0.5]])
    obs = build_observation(np.array([1.0, 1.0, 0.0]), np.array([6.0, 1.0]), obstacles, cfg)
    heading, _, _ = VFHExpert(cfg).choose_heading(obs)
    assert abs(heading) > 0.3  # ne fonce pas droit dans l'obstacle


def test_vfh_does_not_chatter_when_goal_is_behind():
    cfg = EnvConfig()
    sc = Scenario(np.array([5.0, 5.0]), 0.0, np.array([1.0, 5.0]), np.array([[3.5, 5.0, 0.6]]))
    env = NavigationEnv(cfg)
    ep = run_episode(env, VFHExpert(cfg), scenario=sc)
    assert ep.success


def test_potential_field_expert_runs():
    cfg = EnvConfig()
    ep = run_episode(NavigationEnv(cfg), PotentialFieldExpert(cfg), seed=0)
    assert ep.steps > 0
