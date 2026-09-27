"""Format de checkpoint unique pour toutes les politiques apprises (BC, DAgger, PPO exporté).

Un seul format -> un seul chargeur pour l'évaluation et pour le nœud ROS2, qui n'a
donc besoin ni de Stable-Baselines3 ni du code d'entraînement.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from env.config import EnvConfig
from env.observation import obs_dim
from models.networks import Actor, build_encoder


def make_actor(cfg: EnvConfig, encoder: str = "deepsets", encoder_kwargs=None, head_hidden=(64,)) -> Actor:
    enc = build_encoder(encoder, obs_dim(cfg), cfg.k_nearest, **(encoder_kwargs or {}))
    return Actor(enc, head_hidden=tuple(head_hidden))


def save_actor(path, actor: Actor, cfg: EnvConfig, encoder: str, encoder_kwargs=None, head_hidden=(64,), meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder,
            "encoder_kwargs": encoder_kwargs or {},
            "head_hidden": list(head_hidden),
            "k_nearest": cfg.k_nearest,
            "obs_dim": obs_dim(cfg),
            "state_dict": actor.state_dict(),
            "meta": meta or {},
            "env_config": {k: v for k, v in asdict(cfg).items() if k != "reward"},
        },
        path,
    )


def load_actor(path, cfg: EnvConfig | None = None) -> tuple[Actor, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = cfg or EnvConfig()
    if ckpt["obs_dim"] != obs_dim(cfg):
        raise ValueError(f"{path} attend une observation de dimension {ckpt['obs_dim']}, l'env donne {obs_dim(cfg)}")
    actor = make_actor(cfg, ckpt["encoder"], ckpt["encoder_kwargs"], ckpt["head_hidden"])
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()
    return actor, ckpt


class ActorPolicy:
    """Adapte un `Actor` PyTorch à l'interface « fonction obs -> action » de l'environnement."""

    def __init__(self, actor: Actor):
        self.actor = actor.eval()

    @classmethod
    def load(cls, path, cfg: EnvConfig | None = None) -> ActorPolicy:
        return cls(load_actor(path, cfg)[0])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return self.actor.act(x)[0].numpy()
