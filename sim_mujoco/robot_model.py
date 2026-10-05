"""Le monde MuJoCo d'une carte : sol, obstacles cylindriques statiques et robot à trois articulations.

Une carte (`Scenario`, voir env/scenario.py) devient un texte XML (format MJCF) que MuJoCo compile.
On régénère ce XML à chaque épisode : les rayons et les positions des obstacles changent à chaque
carte, et modifier les tailles dans un modèle déjà compilé est fragile (certaines grandeurs dérivées,
comme le rayon englobant utilisé pour écarter les paires d'objets trop éloignées, sont calculées à la
compilation).
"""

from dataclasses import dataclass

from env.config import EnvConfig
from env.scenario import Scenario


@dataclass(frozen=True)
class RobotParams:
    """Dynamique du robot et gains du contrôleur de vitesse (voir mujoco_env.py)."""

    mass: float = 5.0  # kg
    timestep: float = 0.01  # s : un sous-pas de physique (le pas de contrôle de 0,1 s en contient dix)
    k_v: float = 33.0  # N·s/m : gain de vitesse avant, constante de temps m/k_v ≈ 0,15 s
    f_max: float = 20.0  # N : force maximale le long du cap
    k_w: float = 1.0  # N·m·s/rad : gain de rotation, constante de temps I/k_w = 0,1 s
    t_max: float = 1.0  # N·m : couple maximal
    c_lat: float = 200.0  # N·s/m : amortissement latéral (roulement sans glissement)


def build_xml(sc: Scenario, cfg: EnvConfig, params: RobotParams | None = None) -> str:
    """Texte XML MuJoCo d'une carte."""
    params = params or RobotParams()
    half_height = 0.15  # demi-hauteur des obstacles (m) ; le robot en fait 0,1

    # 1. Un cylindre statique par obstacle : rattaché au monde, sans articulation.
    #    `size` d'un cylindre MuJoCo = (rayon, demi-hauteur) ; son axe est vertical.
    obstacles = "\n".join(
        f'    <geom name="obs{i}" type="cylinder" pos="{x:.6f} {y:.6f} {half_height}" size="{r:.6f} {half_height}"/>'
        for i, (x, y, r) in enumerate(sc.obstacles)
    )
    centre = cfg.arena_size / 2
    sol = centre + cfg.boundary_margin + 1.0  # demi-côté du sol décoratif

    return f"""<mujoco>
  <option timestep="{params.timestep}" gravity="0 0 0"/>
  <worldbody>
    <!-- 2. Sol : simple décor (contype=0 : il ne collisionne avec rien). Gravité nulle et pas
         d'articulation verticale : le robot reste dans le plan sans qu'il y ait besoin de sol. -->
    <geom name="sol" type="plane" pos="{centre} {centre} 0" size="{sol} {sol} 0.1" contype="0" conaffinity="0"/>
{obstacles}
    <!-- 3. Robot : cylindre de rayon robot_radius et de masse 5 kg. Les articulations s'appliquent
         dans l'ordre : deux translations puis le lacet, donc les axes x et y restent fixes dans le monde. -->
    <body name="robot" pos="0 0 0.1">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom name="corps" type="cylinder" size="{cfg.robot_radius} 0.1" mass="{params.mass}"/>
    </body>
  </worldbody>
</mujoco>
"""
