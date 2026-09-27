import numpy as np
import torch
from stable_baselines3 import PPO

from env import EnvConfig, NavigationEnv
from rl.export import export_ppo
from rl.features import NavFeaturesExtractor
from tests.test_models import random_obs


def test_ppo_export_matches_sb3(tmp_path):
    """L'acteur exporté doit reproduire exactement les actions déterministes de SB3."""
    cfg = EnvConfig()
    model = PPO(
        "MlpPolicy",
        NavigationEnv(cfg),
        n_steps=64,
        batch_size=32,
        policy_kwargs={
            "features_extractor_class": NavFeaturesExtractor,
            "features_extractor_kwargs": {"k_nearest": cfg.k_nearest},
            "share_features_extractor": False,
            "net_arch": {"pi": [64], "vf": [64]},
            "activation_fn": torch.nn.ReLU,
        },
        device="cpu",
    )
    model.learn(64)
    actor = export_ppo(model, cfg, tmp_path / "ppo.pt")
    obs = random_obs(cfg).numpy()
    sb3_actions, _ = model.predict(obs, deterministic=True)
    np.testing.assert_allclose(actor.act(torch.as_tensor(obs)).numpy(), sb3_actions, atol=1e-5)
