"""Depth conversion shared by the VizDoom environment and collectors."""
from __future__ import annotations

import numpy as np

# ── Frame geometry ───────────────────────────────────────────────────
GRID_W = 40
GRID_H = 25

DEPTH_NUM_BINS = 16

DEPTH_LIVE_SCALE = 7.5

DEFAULT_CELL = 8


def depth_grid_to_rgb(grid: np.ndarray, cell: int = DEFAULT_CELL) -> np.ndarray:
    """Render a 16-bin depth grid as a block-upscaled RGB image."""
    grid = np.asarray(grid).astype(np.int16).copy()
    grid[grid >= DEPTH_NUM_BINS] = DEPTH_NUM_BINS - 1
    gray = np.round(grid.astype(np.float32) * (255.0 / (DEPTH_NUM_BINS - 1)))
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    if cell > 1:
        gray = np.kron(gray, np.ones((cell, cell), dtype=np.uint8))
    return np.repeat(gray[:, :, None], 3, axis=2)


# ── Live-engine side: real VizDoom depth → grid ─────────────────────

def depth_buffer_to_grid(depth_buffer: np.ndarray) -> np.ndarray:
    """Downsample and quantize a live depth buffer with a fixed scale."""
    import cv2

    small = cv2.resize(
        depth_buffer.astype(np.float32), (GRID_W, GRID_H),
        interpolation=cv2.INTER_AREA,
    )
    binned = np.floor(small / DEPTH_LIVE_SCALE)
    return np.clip(binned, 0, DEPTH_NUM_BINS - 1).astype(np.int16)

