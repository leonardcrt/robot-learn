"""Contrôleurs experts : référence chiffrée et source des démonstrations pour le BC."""

from expert.potential_field import ExpertConfig, PotentialFieldExpert
from expert.vfh import VFHConfig, VFHExpert

EXPERTS = {"vfh": VFHExpert, "potential_field": PotentialFieldExpert}


def make_expert(name: str = "vfh", env_cfg=None):
    return EXPERTS[name](env_cfg)


__all__ = ["ExpertConfig", "PotentialFieldExpert", "VFHConfig", "VFHExpert", "EXPERTS", "make_expert"]
