"""GIF de comparaison : la même politique PPO dans les deux simulateurs, sur une même carte de test.

    python -m sim_mujoco.render_compare

Le rendu est celui de env/render.py : une vue du dessus tracée à partir des poses du robot, donc
identique pour les deux simulateurs. Écrit results/mujoco/comparaison_ppo.gif et .png.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from env.config import EnvConfig
from env.render import POLICY_STYLE, animate_trajectories, plot_trajectories
from env.robot_env import NavigationEnv
from env.rollout import run_episode
from env.seeds import TEST_SEEDS, seed_range
from eval.policies import load_policy
from sim_mujoco.mujoco_env import MuJoCoNavEnv
from sim_mujoco.robot_model import RobotParams

MUJOCO_CKPT = Path("checkpoints/mujoco")
ROBOT_LENT = replace(RobotParams(), k_v=6.7, k_w=0.25)  # constantes de temps 0,75 s et 0,4 s

# Styles en mémoire seulement (render.py n'est pas modifié) : une couleur et un trait par courbe.
POLICY_STYLE.update(
    {
        "ppo_cin": ("#eb6834", ":"),
        "ppo_mj": ("#eb6834", "-"),
        "ppo_lent": ("#d03b3b", "--"),
        "ppo_mujoco": ("#2a78d6", "-"),
    }
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/mujoco"))
    parser.add_argument("--candidates", type=int, default=80, help="nombre de cartes de test examinées")
    args = parser.parse_args()

    cfg = EnvConfig()
    ppo = load_policy("ppo", cfg)
    ppo_mujoco = load_policy("ppo_mujoco", cfg, MUJOCO_CKPT) if (MUJOCO_CKPT / "ppo_mujoco.pt").exists() else None

    # 1. Les quatre (ou trois) courbes : même carte, mêmes graines, simulateurs et politiques différents.
    runs = {
        "ppo_cin": ("PPO existant, simulateur cinématique", NavigationEnv(cfg), ppo),
        "ppo_mj": ("PPO existant, MuJoCo", MuJoCoNavEnv(cfg), ppo),
        "ppo_lent": ("PPO existant, MuJoCo avec robot lent", MuJoCoNavEnv(cfg, params=ROBOT_LENT), ppo),
    }
    if ppo_mujoco:
        runs["ppo_mujoco"] = ("PPO réentraîné dans MuJoCo", MuJoCoNavEnv(cfg), ppo_mujoco)

    def roule(seed):
        return {key: run_episode(env, policy, seed=seed) for key, (_, env, policy) in runs.items()}

    # 2. Une carte où la physique se voit : PPO réussit dans le simulateur cinématique mais heurte un
    #    obstacle avec le robot lent. À défaut, la carte où les trajectoires s'écartent le plus.
    choix, meilleur_ecart = None, -1.0
    for seed in seed_range(TEST_SEEDS, args.candidates):
        eps = roule(seed)
        if eps["ppo_cin"].success and eps["ppo_lent"].collision:
            choix = (seed, eps)
            break
        n = min(len(eps["ppo_cin"].trajectory), len(eps["ppo_lent"].trajectory))
        ecart = abs(eps["ppo_cin"].trajectory[:n, :2] - eps["ppo_lent"].trajectory[:n, :2]).max()
        if ecart > meilleur_ecart:
            choix, meilleur_ecart = (seed, eps), ecart
    seed, eps = choix
    print(f"Carte de test {seed} : " + ", ".join(f"{k}={e.outcome}" for k, e in eps.items()))

    # 3. Légendes avec l'issue de chaque courbe, puis GIF et image fixe.
    labels = {
        k: f"{runs[k][0]} : {e.outcome} ({e.steps * cfg.dt:.1f} s)" if e.success else f"{runs[k][0]} : {e.outcome}"
        for k, e in eps.items()
    }
    trajectoires = {k: e.trajectory for k, e in eps.items()}
    scenario = next(iter(eps.values())).scenario
    args.out.mkdir(parents=True, exist_ok=True)
    animate_trajectories(scenario, trajectoires, cfg, args.out / "comparaison_ppo.gif", labels=labels, stride=2)
    plot_trajectories(scenario, trajectoires, cfg, labels=labels)
    plt.savefig(args.out / "comparaison_ppo.png", bbox_inches="tight")
    print(f"Écrit : {args.out / 'comparaison_ppo.gif'} et .png")


if __name__ == "__main__":
    main()
