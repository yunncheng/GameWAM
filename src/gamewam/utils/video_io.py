import os
import tempfile
from itertools import chain
from typing import Iterable

import imageio
import numpy as np
from PIL import Image

from .misc import ensure_dir


def resize_frame_to_size(frame, size: tuple[int, int]) -> Image.Image:
    """Resize an RGB frame to ``(width, height)`` without cropping."""
    image = (
        frame
        if isinstance(frame, Image.Image)
        else Image.fromarray(np.asarray(frame))
    )
    return image.convert("RGB").resize(
        (int(size[0]), int(size[1])),
        resample=Image.Resampling.BILINEAR,
    )


def _to_even_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _to_rgb_array(frame: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL image or an HWC numpy frame to an RGB uint8 array."""
    if isinstance(frame, Image.Image):
        image = frame
    else:
        array = np.asarray(frame)
        try:
            image = Image.fromarray(array)
        except (KeyError, TypeError) as exc:
            raise TypeError(
                "Video frames must be PIL images or numpy arrays supported by "
                f"PIL.Image.fromarray; got shape={array.shape}, dtype={array.dtype}."
            ) from exc
    return np.asarray(image.convert("RGB"))


def save_mp4(
    frames: Iterable[Image.Image | np.ndarray], path: str, fps: int = 8
):
    frame_iter = iter(frames)
    try:
        first_frame = next(frame_iter)
    except StopIteration:
        return None

    ensure_dir(os.path.dirname(path) or ".")
    output_dir = os.path.dirname(os.path.abspath(path))
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp.mp4",
        dir=output_dir,
    )
    os.close(fd)
    os.unlink(temp_path)
    try:
        writer = imageio.get_writer(
            temp_path,
            fps=max(fps, 1),
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            macro_block_size=2,
        )
        try:
            for frame in chain((first_frame,), frame_iter):
                arr = _to_rgb_array(frame)
                writer.append_data(_to_even_frame(arr))
        finally:
            writer.close()
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise
