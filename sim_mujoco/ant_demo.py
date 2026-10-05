"""Exercice de prise en main de MuJoCo : PPO sur le robot à quatre pattes Ant-v5, puis une vidéo.

python -m sim_mujoco.ant_demo                  # 500 000 pas (la durée dépend de la machine)
python -m sim_mujoco.ant_demo --timesteps 5000 # essai rapide
"""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from env.video import VideoWriter

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", type=int, default=500_000)
parser.add_argument("--out", type=Path, default=Path("results/mujoco/ant_demo.mp4"))
args = parser.parse_args()

# 1. Entraînement : politique classique (l'observation de l'Ant n'a pas de structure d'ensemble).
model = PPO("MlpPolicy", gym.make("Ant-v5"), device="cpu", seed=0, verbose=1)
model.learn(total_timesteps=args.timesteps)

# 2. Vidéo d'un épisode : des images "rgb_array" écrites avec le VideoWriter du projet
#    (le RecordVideo de Gymnasium demanderait moviepy, une bibliothèque en plus).
env = gym.make("Ant-v5", render_mode="rgb_array")
obs, _ = env.reset(seed=0)
with VideoWriter(args.out, fps=env.metadata["render_fps"]) as video:
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        video.append(env.render())
        if terminated or truncated:
            break
print(f"Vidéo écrite : {args.out} ({video.frames} images)")
