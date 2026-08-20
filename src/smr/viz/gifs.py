"""GIF writers (imageio)."""
from __future__ import annotations

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt


def write_gif(frames, path, fps=14):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"  gif -> {path}  ({len(frames)} frames)")


def fig_to_frame(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = buf.reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return frame


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
