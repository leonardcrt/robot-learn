"""Exporte l'acteur déterministe d'un modèle PPO vers le format de checkpoint commun.

Après export, l'évaluation et le nœud ROS2 chargent PPO exactement comme le BC
(`models.load_actor`), sans dépendre de Stable-Baselines3.

    python -m rl.export --model checkpoints/ppo_sb3.zip --out checkpoints/ppo.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO

from env.config import EnvConfig
from models import make_actor, save_actor


def export_ppo(model: PPO, cfg: EnvConfig, out_path, meta=None):
    policy = model.policy
    features_kwargs = policy.features_extractor_kwargs
    encoder_kwargs = {k: v for k, v in features_kwargs.items() if k != "k_nearest"}
    head_hidden = policy.net_arch["pi"]
    actor = make_actor(cfg, "deepsets", encoder_kwargs, head_hidden)

    extractor = policy.pi_features_extractor if not policy.share_features_extractor else policy.features_extractor
    actor.encoder.load_state_dict(extractor.encoder.state_dict())
    # SB3 : policy_net = [Linear, act, Linear, act, ...], action_net = Linear -> même chose que Actor.head
    sb3_linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)] + [policy.action_net]
    head_linears = [m for m in actor.head if isinstance(m, torch.nn.Linear)]
    for dst, src in zip(head_linears, sb3_linears, strict=True):
        dst.load_state_dict(src.state_dict())

    save_actor(out_path, actor, cfg, "deepsets", encoder_kwargs, head_hidden, meta={"algo": "ppo", **(meta or {})})
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("checkpoints/ppo_sb3.zip"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/ppo.pt"))
    args = parser.parse_args()
    export_ppo(PPO.load(args.model, device="cpu"), EnvConfig(), args.out)
    print(f"Exporté : {args.out}")


if __name__ == "__main__":
    main()
