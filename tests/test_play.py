"""Le simulateur interactif fonctionne sans fenêtre (backend Agg) : clics, lancement, comparaison."""

import os
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

from env.config import EnvConfig  # noqa: E402
from eval.play import Playground  # noqa: E402


def _click(pg, x, y, button=1, key=None):
    pg._on_click(SimpleNamespace(inaxes=pg.ax, xdata=x, ydata=y, button=button, key=key))


def _run_to_end(pg, max_ticks=400):
    pg._toggle()
    update = pg._update
    pg._update = lambda: None  # pas de rendu à chaque pas : seule la logique est testée ici
    for _ in range(max_ticks):
        if not pg.running:
            break
        pg._tick()
    pg._update = update
    pg._update()


def test_edit_map_and_run_expert():
    pg = Playground(EnvConfig(), seed=1_000_000)
    pg._clear()
    _click(pg, 2.0, 2.0, key="shift")  # départ
    _click(pg, 8.0, 8.0)  # cible
    _click(pg, 5.0, 5.0, button=3)  # obstacle sur la diagonale
    assert len(pg.obstacles) == 1
    _run_to_end(pg)
    assert pg.runs["expert"]["outcome"] == "succès"


def test_right_click_removes_obstacle():
    pg = Playground(EnvConfig(), seed=1_000_000)
    n = len(pg.obstacles)
    x, y, _ = pg.obstacles[0]
    _click(pg, x, y, button=3)
    assert len(pg.obstacles) == n - 1


def test_video_recording(tmp_path, monkeypatch):
    import eval.play as play

    monkeypatch.setattr(play, "RECORD_DIR", tmp_path)
    pg = Playground(EnvConfig(), seed=1_000_000)
    pg._toggle_recording()
    pg._toggle()
    for _ in range(10):
        pg._tick()
    pg._toggle_recording()
    videos = list(tmp_path.glob("*.mp4"))
    assert len(videos) == 1 and videos[0].stat().st_size > 1000


def test_compare_mode_runs_several_robots():
    pg = Playground(EnvConfig(), seed=1_000_000)
    pg._on_radio("Comparer les 4")
    assert len(pg.runs) >= 2
    _run_to_end(pg)
    assert all(r["done"] for r in pg.runs.values())
    assert np.isfinite(pg.runs["expert"]["env"].steps)
