import numpy as np
import torch

from bc.data import collect
from bc.train import train_actor
from env import EnvConfig, NavigationEnv
from env.observation import GOAL_DIM, OBSTACLE_DIM, obs_dim
from eval.metrics import wilson_interval
from expert import VFHExpert
from models import ActorPolicy, NavEncoder, load_actor, make_actor, save_actor


def random_obs(cfg, batch=16, seed=0):
    env = NavigationEnv(cfg)
    obs = [env.reset(seed=seed + i)[0] for i in range(batch)]
    return torch.as_tensor(np.stack(obs))


def test_encoder_is_permutation_invariant():
    cfg = EnvConfig()
    enc = NavEncoder(k_nearest=cfg.k_nearest)
    obs = random_obs(cfg)
    perm = torch.randperm(cfg.k_nearest)
    shuffled = obs.clone()
    obstacles = obs[:, GOAL_DIM:].reshape(-1, cfg.k_nearest, OBSTACLE_DIM)
    shuffled[:, GOAL_DIM:] = obstacles[:, perm].reshape(len(obs), -1)
    torch.testing.assert_close(enc(obs), enc(shuffled))


def test_encoder_ignores_empty_slots():
    cfg = EnvConfig()
    enc = NavEncoder(k_nearest=cfg.k_nearest)
    obs = torch.zeros(2, obs_dim(cfg))
    obs[:, 1] = 1.0
    obs[1, GOAL_DIM + 4 * OBSTACLE_DIM : GOAL_DIM + 4 * OBSTACLE_DIM + 4] = 0.7  # valeurs parasites, masque = 0
    out = enc(obs)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0], out[1])


def test_checkpoint_roundtrip(tmp_path):
    cfg = EnvConfig()
    actor = make_actor(cfg, "deepsets")
    save_actor(tmp_path / "a.pt", actor, cfg, "deepsets")
    loaded, _ = load_actor(tmp_path / "a.pt", cfg)
    obs = random_obs(cfg)
    torch.testing.assert_close(actor.act(obs), loaded.act(obs))


def test_bc_learns_expert_on_small_dataset():
    cfg = EnvConfig()
    expert = VFHExpert(cfg)
    data = collect(expert, expert, list(range(30)), cfg, only_success=True)
    actor = make_actor(cfg, "deepsets")
    history = train_actor(actor, data, epochs=8, verbose=False)
    assert history["val_loss"][-1] < history["val_loss"][0]
    action = ActorPolicy(actor)(data["obs"][0])
    assert action.shape == (2,) and np.all(np.abs(action) <= 1)


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0 < hi < 0.35
    lo, hi = wilson_interval(95, 100)
    assert 0.88 < lo < 0.95 < hi <= 1.0
