"""Environnement Gymnasium `MuJoCoNavEnv` : la même tâche que env/robot_env.py, avec de la physique.

Ce qui est identique à `NavigationEnv` (on en hérite) : l'observation (`build_observation`), les cartes
et les graines (`reset` appelle le même `sample_scenario`), la récompense, les conditions de fin et
l'interface d'une politique. Les checkpoints existants se branchent donc sans modification.

Ce qui change : le robot n'obéit plus instantanément. Un contrôleur de vitesse à force limitée
transforme la commande (v, omega) en forces appliquées à un corps rigide de 5 kg, et la collision est
détectée par les contacts de MuJoCo plutôt que par un test de distance.
"""

import mujoco
import numpy as np

from env.config import EnvConfig
from env.robot_env import NavigationEnv, action_to_velocity
from sim_mujoco.robot_model import RobotParams, build_xml


class MuJoCoNavEnv(NavigationEnv):
    def __init__(
        self, config: EnvConfig | None = None, render_mode: str | None = None, params: RobotParams | None = None
    ):
        super().__init__(config, render_mode)
        self.params = params or RobotParams()
        self.n_substeps = round(self.cfg.dt / self.params.timestep)  # 10 sous-pas de 0,01 s pour 0,1 s
        self.model = None
        self.data = None

    # --- API Gymnasium -------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # 1. Même tirage de carte que NavigationEnv (même appel à sample_scenario avec la même graine) :
        #    super().reset() choisit la carte et le point de départ, et initialise self.pose.
        super().reset(seed=seed, options=options)
        # 2. Un nouveau monde MuJoCo pour cette carte.
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.scenario, self.cfg, self.params))
        self.data = mujoco.MjData(self.model)
        # 3. Pose de départ (x, y, cap) dans qpos, vitesses nulles.
        self.data.qpos[:] = self.pose
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), self._info()

    def step(self, action):
        cfg, rw = self.cfg, self.cfg.reward
        # 1. Action normalisée -> commande (v, omega), comme dans le simulateur cinématique.
        v_cmd, w_cmd = action_to_velocity(action, cfg)
        # 2. Dix sous-pas de physique ; `collision` vaut True si un contact a eu lieu pendant le pas.
        collision = self._advance(v_cmd, w_cmd)
        # 3. Lecture de la pose (le cap, non borné dans qpos, est ramené dans [-pi, pi[).
        self.pose = np.array([self.data.qpos[0], self.data.qpos[1], (self.data.qpos[2] + np.pi) % (2 * np.pi) - np.pi])
        self.steps += 1
        self.trajectory.append(self.pose.copy())

        # 4. Fin d'épisode : mêmes règles que NavigationEnv, sauf la collision (venue des contacts).
        dist = self._goal_distance()
        clearance = self._min_clearance()
        lo, hi = -cfg.boundary_margin, cfg.arena_size + cfg.boundary_margin
        out_of_bounds = not (lo <= self.pose[0] <= hi and lo <= self.pose[1] <= hi)
        success = dist <= cfg.goal_radius and not collision

        # 5. Récompense : mêmes formules que NavigationEnv.step (recopiées, voir tests/test_mujoco.py).
        reward = rw.progress * (self._prev_dist - dist) + rw.time
        if clearance < rw.proximity_margin:
            reward += rw.proximity * (1.0 - max(clearance, 0.0) / rw.proximity_margin)
        if success:
            reward += rw.success
        if collision:
            reward += rw.collision
        if out_of_bounds:
            reward += rw.out_of_bounds
        self._prev_dist = dist

        terminated = bool(success or collision or out_of_bounds)
        truncated = bool(not terminated and self.steps >= cfg.max_steps)
        info = self._info(collision=collision, out_of_bounds=out_of_bounds, success=success)
        return self._obs(), float(reward), terminated, truncated, info

    # --- dynamique -----------------------------------------------------------------

    def _advance(self, v_cmd: float, w_cmd: float) -> bool:
        """Fait avancer la physique d'un pas de contrôle. Renvoie True si le robot a touché un obstacle."""
        p, d = self.params, self.data
        for _ in range(self.n_substeps):
            # 1. Vitesses du robot dans son propre repère : le long du cap et sur le côté.
            x_dot, y_dot, theta, w = d.qvel[0], d.qvel[1], d.qpos[2], d.qvel[2]
            c, s = np.cos(theta), np.sin(theta)
            v_avant = x_dot * c + y_dot * s
            v_lateral = -x_dot * s + y_dot * c
            # 2. Force le long du cap (proportionnelle à l'écart de vitesse, bornée) et force latérale
            #    qui s'oppose au glissement : c'est elle qui fait du disque libre un robot à roues.
            f_avant = np.clip(p.k_v * (v_cmd - v_avant), -p.f_max, p.f_max)
            f_lateral = -p.c_lat * v_lateral
            # 3. Forces dans le repère du monde (les translations x et y sont fixes dans le monde) et couple.
            d.qfrc_applied[0] = f_avant * c - f_lateral * s
            d.qfrc_applied[1] = f_avant * s + f_lateral * c
            d.qfrc_applied[2] = np.clip(p.k_w * (w_cmd - w), -p.t_max, p.t_max)
            # 4. Un sous-pas de physique, puis test des contacts. Le sol ne collisionne avec rien et les
            #    obstacles sont statiques : le seul contact possible est robot contre obstacle. On teste à
            #    chaque sous-pas pour ne pas manquer un frôlement que le solveur ferait disparaître.
            mujoco.mj_step(self.model, d)
            if d.ncon > 0:
                return True
        return False
