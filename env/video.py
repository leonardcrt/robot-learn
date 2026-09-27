"""Écriture de vidéos MP4 (H.264) via imageio-ffmpeg, qui embarque son propre binaire ffmpeg."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VideoWriter:
    """Encode des images RGB (H, W, 3) en MP4 lisible partout (yuv420p, dimensions paires)."""

    def __init__(self, path, fps: int = 20):
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            self.path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=2,
            ffmpeg_log_level="error",
        )
        self.frames = 0

    def append(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        self._writer.append_data(np.ascontiguousarray(frame[: h - h % 2, : w - w % 2, :3]))
        self.frames += 1

    def close(self):
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def figure_frame(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
