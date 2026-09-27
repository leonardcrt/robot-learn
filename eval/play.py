"""Simulateur interactif : dessine ta carte et regarde les politiques la résoudre.

    python -m eval.play            # carte de test aléatoire
    python -m eval.play --seed 3   # carte précise

Souris (sur la carte, quand la simulation est à l'arrêt) :
    clic gauche          place la cible
    clic droit           ajoute un obstacle (ou supprime celui sous le curseur)
    Maj + clic gauche    place le départ du robot

Clavier :
    espace   lancer / mettre en pause         r   revenir au départ
    n        nouvelle carte aléatoire          c   effacer les obstacles
    ← / →    tourner le robot au départ        s   afficher / masquer le capteur
    v        démarrer / arrêter l'enregistrement vidéo (MP4 dans results/videos/)

Panneau de droite : choix de la politique (dont « Comparer » qui lance l'expert, le BC,
DAgger et PPO en même temps), curseur du stade d'apprentissage de PPO (de « avant
entraînement » à la fin), taille des obstacles et vitesse de simulation.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "TkAgg")  # fenêtre interactive (MPLBACKEND=Agg pour les tests)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.widgets import Button, RadioButtons, Slider  # noqa: E402

from env.config import EnvConfig  # noqa: E402
from env.render import INK, INK_SECONDARY, MUTED, SURFACE, draw_scenario, style_for  # noqa: E402
from env.robot_env import NavigationEnv  # noqa: E402
from env.scenario import Scenario, surface_distances  # noqa: E402
from env.seeds import TEST_SEEDS  # noqa: E402
from env.video import VideoWriter  # noqa: E402
from eval.learning_progress import color_for  # noqa: E402
from eval.policies import CHECKPOINTS, LABELS, load_policy  # noqa: E402
from models import ActorPolicy, load_actor  # noqa: E402

SNAPSHOT_DIR = CHECKPOINTS / "ppo_snapshots"
RECORD_DIR = Path("results/videos")
COMPARE = ["expert", "bc", "dagger", "ppo"]
CHOICES = [
    ("expert", "Expert (VFH)"),
    ("potential_field", "Champ de potentiel"),
    ("bc", "BC (DeepSets)"),
    ("bc_mlp", "BC (MLP aplati)"),
    ("dagger", "DAgger"),
    ("ppo", "PPO final"),
    ("ppo_stage", "PPO au stade du curseur"),
    ("ppo_evolution", "Évolution PPO (tous les stades)"),
    ("compare", "Comparer les 4"),
]


def steps_label(steps: int) -> str:
    if steps == 0:
        return "avant entraînement"
    return f"{steps / 1e6:.2f} M pas" if steps >= 1_000_000 else f"{steps // 1000} k pas"


class Playground:
    def __init__(self, cfg: EnvConfig, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.choices = [(k, v) for k, v in CHOICES if self._available(k)]
        self.mode = "expert"
        self.policy_cache: dict[str, object] = {}
        self.snapshots = self._load_snapshot_meta()
        self.stage = len(self.snapshots) - 1
        self.obstacle_radius = 0.5
        self.steps_per_tick = 1
        self.show_sensor = True
        self.running = False
        self.recorder: VideoWriter | None = None
        self.runs: dict[str, dict] = {}

        self._build_ui()
        self._load_scenario(self._sample(seed))
        self.timer = self.fig.canvas.new_timer(interval=int(1000 * cfg.dt))
        self.timer.add_callback(self._tick)
        self.timer.start()

    # --- politiques ----------------------------------------------------------------------

    @staticmethod
    def _available(key: str) -> bool:
        if key in ("expert", "potential_field", "compare"):
            return True
        if key in ("ppo_stage", "ppo_evolution"):
            return SNAPSHOT_DIR.exists() and any(SNAPSHOT_DIR.glob("ppo_*.pt"))
        return (CHECKPOINTS / f"{key}.pt").exists()

    def _load_snapshot_meta(self) -> list[dict]:
        snaps = []
        for path in sorted(SNAPSHOT_DIR.glob("ppo_*.pt")) if SNAPSHOT_DIR.exists() else []:
            _, ckpt = load_actor(path, self.cfg)
            snaps.append({"path": path, **ckpt["meta"]})
        return sorted(snaps, key=lambda s: s["timesteps"])

    def _policy(self, key: str):
        if key not in self.policy_cache:
            if key.startswith("stage:"):
                self.policy_cache[key] = ActorPolicy.load(self.snapshots[int(key[6:])]["path"], self.cfg)
            else:
                self.policy_cache[key] = load_policy(key, self.cfg)
        return self.policy_cache[key]

    def _active(self) -> list[tuple[str, str]]:
        """(clé de politique, nom de style) des robots à simuler."""
        if self.mode == "compare":
            return [(k, k) for k in COMPARE if self._available(k)]
        if self.mode == "ppo_stage":
            return [(f"stage:{self.stage}", "ppo")]
        if self.mode == "ppo_evolution":
            n = len(self.snapshots)
            return [(f"stage:{i}", color_for(i, n)) for i in range(n)]
        return [(self.mode, self.mode)]

    # --- scénario -----------------------------------------------------------------------

    def _sample(self, seed: int) -> Scenario:
        env = NavigationEnv(self.cfg)
        env.reset(seed=seed)  # exactement la carte que voit le benchmark pour cette graine
        return env.scenario

    def _load_scenario(self, sc: Scenario):
        self.start = np.array(sc.start, dtype=float)
        self.theta = float(sc.start_theta)
        self.goal = np.array(sc.goal, dtype=float)
        self.obstacles = [list(o) for o in sc.obstacles]
        self._reset_runs()

    def _scenario(self) -> Scenario:
        return Scenario(self.start.copy(), self.theta, self.goal.copy(), np.array(self.obstacles).reshape(-1, 3))

    def _reset_runs(self):
        self.running = False
        sc = self._scenario()
        self.runs = {}
        for key, style in self._active():
            env = NavigationEnv(self.cfg)
            obs, info = env.reset(options={"scenario": sc})
            self.runs[key] = {"env": env, "obs": obs, "style": style, "done": False, "outcome": None}
        clearance = surface_distances(self.start, sc.obstacles, self.cfg.robot_radius)
        if len(clearance) and clearance.min() <= 0:
            self.status = "Le départ est dans un obstacle : déplace-le (Maj + clic)."
        else:
            self.status = "Prêt — espace pour lancer."
        self._draw()

    # --- simulation ---------------------------------------------------------------------

    def _tick(self):
        if not self.running:
            return
        for _ in range(self.steps_per_tick):
            for key, run in self.runs.items():
                if run["done"]:
                    continue
                action = self._policy(key)(run["obs"])
                run["obs"], _, terminated, truncated, info = run["env"].step(action)
                if terminated or truncated:
                    run["done"] = True
                    run["outcome"] = (
                        "succès"
                        if info["is_success"]
                        else "collision"
                        if info["collision"]
                        else "sortie"
                        if info["out_of_bounds"]
                        else "timeout"
                    )
        if all(r["done"] for r in self.runs.values()):
            self.running = False
            self.status = "Terminé — r pour rejouer, n pour une nouvelle carte."
        self._update()

    # --- dessin -------------------------------------------------------------------------

    def _label(self, key: str, run: dict) -> str:
        if key.startswith("stage:"):
            name = f"PPO ({steps_label(self.snapshots[int(key[6:])]['timesteps'])})"
        else:
            name = LABELS.get(key, key)
        env = run["env"]
        t = f"{env.steps * self.cfg.dt:.1f} s"
        return f"{name} — {run['outcome']} ({t})" if run["done"] else f"{name} — {t}"

    def _draw(self):
        """Redessin complet (carte modifiée ou politique changée) ; pendant la simulation on passe par _update."""
        ax, cfg = self.ax, self.cfg
        ax.clear()
        sc = self._scenario()
        draw_scenario(ax, sc, cfg)
        tip = self.start + 0.6 * np.array([np.cos(self.theta), np.sin(self.theta)])
        ax.annotate("", tip, self.start, arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.2})

        # Éléments mobiles : marqués « animated », ils sont exclus du rendu complet et redessinés seuls
        # à chaque pas par-dessus un arrière-plan mis en cache (blitting) -> ~10x plus rapide.
        self.dynamic = []
        for key, run in self.runs.items():
            color, ls = (run["style"], "-") if run["style"].startswith("#") else style_for(run["style"])
            (run["line"],) = ax.plot([], [], color=color, ls=ls, lw=2, label=self._label(key, run), zorder=4)
            run["body"] = ax.add_patch(
                Circle((0, 0), cfg.robot_radius, facecolor=color, edgecolor=SURFACE, lw=1.5, zorder=6)
            )
            (run["heading"],) = ax.plot([], [], color=INK, lw=1.5, zorder=7)
            self.dynamic += [run["line"], run["body"], run["heading"]]
        self.sensor = ax.add_patch(
            Circle((0, 0), cfg.sensing_range + cfg.robot_radius, fill=False, ls=":", lw=0.8, edgecolor=MUTED)
        )
        self.sensed = [
            ax.add_patch(Circle((0, 0), 0.1, fill=False, lw=1.8, edgecolor=INK_SECONDARY, zorder=3))
            for _ in range(cfg.k_nearest)
        ]
        self.legend = ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9, labelcolor=INK_SECONDARY)
        self.legend.get_frame().set_edgecolor("#e1e0d9")
        self.title = ax.set_title("", fontsize=10, color=INK, loc="left")
        self.dynamic = self.sensed + [self.sensor, *self.dynamic, self.legend, self.title]
        for artist in self.dynamic:
            artist.set_animated(True)
        self._set_dynamic_state()
        self.fig.canvas.draw()  # rendu complet -> déclenche _on_draw qui met l'arrière-plan en cache

    def screenshot(self, path):
        """savefig ignore les artistes animés : on les rend temporairement statiques."""
        for artist in self.dynamic:
            artist.set_animated(False)
        self.fig.savefig(path, facecolor=SURFACE)
        for artist in self.dynamic:
            artist.set_animated(True)
        self.fig.canvas.draw()

    def _on_draw(self, _event):
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_dynamic()
        self._record_frame()

    # --- enregistrement vidéo ---------------------------------------------------------------

    def _toggle_recording(self):
        if self.recorder is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recorder = VideoWriter(RECORD_DIR / f"simulateur_{stamp}.mp4", fps=int(round(1 / self.cfg.dt)))
            self.status = f"● REC — {self.recorder.path.name} (v pour arrêter)"
        else:
            path, frames = self.recorder.path, self.recorder.frames
            self.recorder.close()
            self.recorder = None
            self.status = f"Vidéo enregistrée : {path} ({frames / (1 / self.cfg.dt):.0f} s)"
            print(self.status)
        self._update()

    def _record_frame(self):
        if self.recorder is not None:
            self.recorder.append(np.asarray(self.fig.canvas.buffer_rgba())[..., :3])

    def _on_close(self, _event):
        if self.recorder is not None:
            self._toggle_recording()

    def _draw_dynamic(self):
        for artist in getattr(self, "dynamic", []):
            self.ax.draw_artist(artist)

    def _update(self):
        self._set_dynamic_state()
        canvas = self.fig.canvas
        if getattr(self, "background", None) is None:
            canvas.draw_idle()
            return
        canvas.restore_region(self.background)
        self._draw_dynamic()
        canvas.blit(self.fig.bbox)
        self._record_frame()

    def _set_dynamic_state(self):
        cfg = self.cfg
        for (key, run), text in zip(self.runs.items(), self.legend.get_texts(), strict=True):
            traj = np.array(run["env"].trajectory)
            x, y, th = traj[-1]
            run["line"].set_data(traj[:, 0], traj[:, 1])
            run["body"].center = (x, y)
            tip = 1.6 * cfg.robot_radius
            run["heading"].set_data([x, x + tip * np.cos(th)], [y, y + tip * np.sin(th)])
            text.set_text(self._label(key, run))
        self.title.set_text(self.status)
        self._update_sensor()

    def _update_sensor(self):
        """Portée du capteur et obstacles effectivement présents dans l'observation (les K plus proches)."""
        for patch in self.sensed:
            patch.set_visible(False)
        self.sensor.set_visible(self.show_sensor and bool(self.runs))
        if not self.sensor.get_visible():
            return
        # en mode évolution, le capteur suit le stade le plus entraîné (le dernier)
        run = list(self.runs.values())[-1 if self.mode == "ppo_evolution" else 0]
        cfg, env = self.cfg, run["env"]
        pose, obstacles = np.array(env.trajectory[-1]), env.scenario.obstacles
        self.sensor.center = tuple(pose[:2])
        if len(obstacles) == 0:
            return
        d = surface_distances(pose[:2], obstacles, cfg.robot_radius)
        visible = np.flatnonzero(d < cfg.sensing_range)
        for patch, i in zip(self.sensed, visible[np.argsort(d[visible])][: cfg.k_nearest], strict=False):
            patch.center, patch.radius = tuple(obstacles[i, :2]), obstacles[i, 2]
            patch.set_visible(True)

    # --- interface ----------------------------------------------------------------------

    def _build_ui(self):
        self.fig = plt.figure("robot-learn — simulateur interactif", figsize=(12, 7.4))
        self.fig.patch.set_facecolor(SURFACE)
        self.ax = self.fig.add_axes([0.03, 0.06, 0.6, 0.88])

        x0, w = 0.68, 0.29
        self.fig.text(x0, 0.95, "Politique", fontsize=10, color=INK, weight="bold")
        labels = [v for _, v in self.choices]
        ax_radio = self.fig.add_axes([x0, 0.58, w, 0.36], facecolor=SURFACE)
        ax_radio.axis("off")
        self.radio = RadioButtons(ax_radio, labels, active=0)
        self.radio.on_clicked(self._on_radio)

        self.stage_text = self.fig.text(x0, 0.535, "", fontsize=8, color=MUTED)
        if self.snapshots:
            ax_stage = self.fig.add_axes([x0 + 0.07, 0.49, w - 0.1, 0.03])
            self.stage_slider = Slider(
                ax_stage, "Stade PPO", 0, len(self.snapshots) - 1, valinit=self.stage, valstep=1, color="#eb6834"
            )
            self.stage_slider.valtext.set_visible(False)
            self.stage_slider.on_changed(self._on_stage)
            self._update_stage_text()

        ax_rad = self.fig.add_axes([x0 + 0.07, 0.43, w - 0.1, 0.03])
        self.radius_slider = Slider(ax_rad, "Rayon obst.", 0.2, 1.2, valinit=self.obstacle_radius, color="#c3c2b7")
        self.radius_slider.on_changed(lambda v: setattr(self, "obstacle_radius", float(v)))
        ax_speed = self.fig.add_axes([x0 + 0.07, 0.38, w - 0.1, 0.03])
        self.speed_slider = Slider(ax_speed, "Vitesse ×", 1, 5, valinit=1, valstep=1, color="#c3c2b7")
        self.speed_slider.on_changed(lambda v: setattr(self, "steps_per_tick", int(v)))

        buttons = [
            ("Lancer / pause  [espace]", self._toggle),
            ("Revenir au départ  [r]", self._reset_runs),
            ("Nouvelle carte  [n]", self._new_map),
            ("Effacer les obstacles  [c]", self._clear),
            ("Enregistrer une vidéo  [v]", self._toggle_recording),
        ]
        self.buttons = []
        for i, (text, cb) in enumerate(buttons):
            b = Button(self.fig.add_axes([x0, 0.31 - i * 0.052, w, 0.043]), text, color="#f0efec", hovercolor="#e1e0d9")
            b.label.set_fontsize(9)
            b.on_clicked(lambda _e, cb=cb: cb())
            self.buttons.append(b)

        help_text = (
            "clic gauche : cible   ·   clic droit : obstacle (+/−)\n"
            "Maj + clic : départ   ·   ← → : cap initial\n"
            "s : capteur   ·   v : vidéo   ·   n : nouvelle carte"
        )
        self.fig.text(x0, 0.005, help_text, fontsize=8, color=MUTED)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update_stage_text(self):
        s = self.snapshots[self.stage]
        success = s.get("val_success")
        extra = f" · succès en validation {success:.0%}" if success is not None else ""
        self.stage_text.set_text(f"Stade PPO : {steps_label(int(s['timesteps']))}{extra}")

    def _on_radio(self, label: str):
        self.mode = next(k for k, v in self.choices if v == label)
        self._reset_runs()

    def _on_stage(self, value):
        self.stage = int(value)
        self._update_stage_text()
        if self.mode != "ppo_stage":
            idx = [k for k, _ in self.choices].index("ppo_stage")
            self.radio.set_active(idx)  # déclenche _on_radio -> reset
        else:
            self._reset_runs()

    def _toggle(self):
        if all(r["done"] for r in self.runs.values()):
            self._reset_runs()
        self.running = not self.running
        self.status = "En cours…" if self.running else "Pause — espace pour reprendre."
        self._update()

    def _new_map(self):
        self._load_scenario(self._sample(int(self.rng.integers(1_000_000))))

    def _clear(self):
        self.obstacles = []
        self._reset_runs()

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None or self.running:
            return
        p = np.array([event.xdata, event.ydata])
        shift = event.key is not None and "shift" in event.key
        if event.button == 1 and shift:
            self.start = p
        elif event.button == 1:
            self.goal = p
        elif event.button == 3:
            hit = [i for i, (x, y, r) in enumerate(self.obstacles) if np.hypot(p[0] - x, p[1] - y) <= r]
            if hit:
                self.obstacles.pop(hit[-1])
            else:
                self.obstacles.append([p[0], p[1], self.obstacle_radius])
        else:
            return
        self._reset_runs()

    def _on_key(self, event):
        actions = {
            " ": self._toggle,
            "r": self._reset_runs,
            "n": self._new_map,
            "c": self._clear,
            "v": self._toggle_recording,
        }
        if event.key in actions:
            actions[event.key]()
        elif event.key in ("left", "right") and not self.running:
            self.theta += np.deg2rad(15 if event.key == "left" else -15)
            self._reset_runs()
        elif event.key == "s":
            self.show_sensor = not self.show_sensor
            self._update()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=TEST_SEEDS)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)  # inférence d'un seul état par pas : le multi-threading ne fait que ralentir
    if not Path("checkpoints").exists():
        print("Lance ce script depuis la racine du dépôt (dossier checkpoints/ introuvable).")
    Playground(EnvConfig(), args.seed)
    plt.show()


if __name__ == "__main__":
    main()
