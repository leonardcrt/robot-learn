"""Paramètres de l'environnement, regroupés en un seul endroit.

Toutes les grandeurs sont en unités SI (mètres, secondes, radians).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardConfig:
    """Pondération des termes de récompense (utilisée uniquement par le RL)."""

    progress: float = 1.0  # par mètre gagné vers la cible (shaping basé sur un potentiel)
    success: float = 10.0
    collision: float = -10.0
    out_of_bounds: float = -10.0
    time: float = -0.01  # par pas de temps : pousse à arriver vite
    proximity: float = -0.1  # max par pas, quand le robot frôle un obstacle
    proximity_margin: float = 0.3  # distance de surface sous laquelle la pénalité s'active


@dataclass(frozen=True)
class EnvConfig:
    # Arène carrée [0, size] x [0, size]
    arena_size: float = 10.0

    # Robot différentiel modélisé comme un unicycle
    robot_radius: float = 0.2
    v_max: float = 1.0  # m/s, marche avant uniquement
    omega_max: float = 2.0  # rad/s
    dt: float = 0.1  # s
    max_steps: int = 300  # 30 s simulées

    # Cible
    goal_radius: float = 0.3
    min_start_goal_dist: float = 5.0

    # Obstacles circulaires
    n_obstacles: int = 10
    n_obstacles_on_path: int = 3  # placés près du segment départ-cible pour forcer l'évitement
    obstacle_radius_range: tuple[float, float] = (0.3, 0.8)
    spawn_clearance: float = 0.6  # distance libre minimale autour du départ et de la cible

    # Capteur : le robot ne "voit" que les K obstacles les plus proches dans un rayon donné
    sensing_range: float = 4.0
    k_nearest: int = 5

    reward: RewardConfig = field(default_factory=RewardConfig)
