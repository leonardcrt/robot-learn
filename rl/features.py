"""Branche le `NavEncoder` PyTorch dans Stable-Baselines3 comme extracteur de features.

PPO et le BC utilisent ainsi exactement la même architecture d'encodeur : l'écart de
performance mesuré vient de l'algorithme d'apprentissage (imitation vs essai-erreur),
pas d'une différence de réseau.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from models.networks import NavEncoder


class NavFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Box, k_nearest: int = 5, hidden: int = 64, features_dim: int = 128
    ):
        super().__init__(observation_space, features_dim)
        self.encoder = NavEncoder(k_nearest=k_nearest, hidden=hidden, features_dim=features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)
