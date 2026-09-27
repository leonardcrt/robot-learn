"""Registre des politiques comparées : un nom -> une fonction obs -> action."""

from __future__ import annotations

from pathlib import Path

from env.config import EnvConfig
from expert import PotentialFieldExpert, VFHExpert
from models import ActorPolicy

CHECKPOINTS = Path("checkpoints")

LABELS = {
    "expert": "Expert (VFH)",
    "potential_field": "Champ de potentiel",
    "bc": "BC (DeepSets)",
    "bc_mlp": "BC (MLP aplati)",
    "dagger": "DAgger",
    "ppo": "PPO",
}


def available_policies(ckpt_dir: Path = CHECKPOINTS) -> list[str]:
    learned = [n for n in ("bc", "bc_mlp", "dagger", "ppo") if (ckpt_dir / f"{n}.pt").exists()]
    return ["expert", "potential_field", *learned]


def load_policy(name: str, cfg: EnvConfig | None = None, ckpt_dir: Path = CHECKPOINTS):
    cfg = cfg or EnvConfig()
    if name == "expert":
        return VFHExpert(cfg)
    if name == "potential_field":
        return PotentialFieldExpert(cfg)
    return ActorPolicy.load(ckpt_dir / f"{name}.pt", cfg)
