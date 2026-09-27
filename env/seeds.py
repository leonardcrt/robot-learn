"""Plages de graines disjointes : aucune carte de test n'est vue pendant l'entraînement."""

DEMO_SEEDS = 0  # démonstrations de l'expert pour le BC : 0, 1, 2, ...
DAGGER_SEEDS = 100_000  # rollouts de DAgger
VALIDATION_SEEDS = 200_000  # suivi pendant l'entraînement (BC, DAgger, PPO)
TEST_SEEDS = 1_000_000  # benchmark final uniquement


def seed_range(base: int, n: int) -> list[int]:
    return list(range(base, base + n))
