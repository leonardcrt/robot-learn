"""Fait rouler l'expert VFH dans les deux simulateurs, sur les mêmes cartes de test.

    python -m sim_mujoco.run_expert --n 30

Pour chaque carte : l'issue dans chaque simulateur, la durée, et l'écart maximal (en mètres) entre les deux
trajectoires, comparées pas de temps par pas de temps tant que les deux épisodes durent.
"""

import argparse

import numpy as np

from env import EnvConfig, NavigationEnv
from env.rollout import run_episode
from env.seeds import TEST_SEEDS, seed_range
from expert import VFHExpert
from sim_mujoco.mujoco_env import MuJoCoNavEnv

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=30)
args = parser.parse_args()

cfg = EnvConfig()
expert = VFHExpert(cfg)
kin_env, mj_env = NavigationEnv(cfg), MuJoCoNavEnv(cfg)

print("graine     cinématique        MuJoCo           écart max")
ok_kin = ok_mj = 0
for seed in seed_range(TEST_SEEDS, args.n):
    # 1. Même expert, même carte (même graine) dans les deux simulateurs.
    kin = run_episode(kin_env, expert, seed=seed)
    mj = run_episode(mj_env, expert, seed=seed)
    # 2. Écart entre les trajectoires, tant que les deux épisodes durent.
    n = min(len(kin.trajectory), len(mj.trajectory))
    gap = np.linalg.norm(kin.trajectory[:n, :2] - mj.trajectory[:n, :2], axis=1).max()
    ok_kin += kin.success
    ok_mj += mj.success
    duree_kin, duree_mj = kin.steps * cfg.dt, mj.steps * cfg.dt
    print(f"{seed}  {kin.outcome:>9} {duree_kin:5.1f} s  {mj.outcome:>9} {duree_mj:5.1f} s  {gap:6.2f} m")
print(f"succès de l'expert : cinématique {ok_kin}/{args.n}, MuJoCo {ok_mj}/{args.n}")
