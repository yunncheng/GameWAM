"""Shared video helpers for game data converters.

``build_video_info`` works around a pyav 13+ API change (canonical codec
name was removed); ``_encode_frames_to_mp4`` is the shared frame-stack
encoder used by converters and the live data collector;
``_extract_video_segment`` trims a frame range out of an existing MP4.
"""
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _extract_video_segment(
    src_mp4: Path, dst_mp4: Path, start_frame: int, num_frames: int
) -> None:
    """Re-encode src_mp4[start_frame : start_frame + num_frames] to dst_mp4.
    Uses ffmpeg's select filter for frame-accurate trimming."""
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if dst_mp4.exists() or dst_mp4.is_symlink():
        dst_mp4.unlink()

    end_frame = start_frame + num_frames
    select_expr = f"between(n\\,{start_frame}\\,{end_frame - 1})"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_mp4),
        "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
        "-vsync", "vfr",
        "-frames:v", str(num_frames),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-an",
        str(dst_mp4),
    ]
    subprocess.run(cmd, check=True)


def _encode_frames_to_mp4(frames: np.ndarray, dst_mp4: Path, fps: int) -> None:
    """Encode a stack of RGB frames ``(N, H, W, 3)`` uint8 to ``dst_mp4`` via
    ffmpeg's rawvideo pipe (frame-accurate, no intermediate PNGs)."""
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if dst_mp4.exists() or dst_mp4.is_symlink():
        dst_mp4.unlink()

    n, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        # One thread per encoder: during massively parallel collection many
        # workers encode simultaneously, and libx264's default auto-threading
        # causes core oversubscription / contention that slows everyone down.
        "-threads", "1",
        str(dst_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.communicate(input=frames.astype(np.uint8).tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed encoding {dst_mp4} (rc={proc.returncode})")


def build_video_info(video_path: Path, fps: int) -> Dict[str, Any]:
    """Pre-compute the ``info["features"][images_key]["info"]`` dict that
    LeRobot's ``update_video_info()`` would otherwise try to fill in via
    ``av.codec.Codec.canonical_name`` (removed in pyav 13+).
    """
    import av
    video_info: Dict[str, Any] = {}
    with av.open(str(video_path), "r") as f:
        vs = f.streams.video[0]
        video_info["video.height"] = vs.height
        video_info["video.width"] = vs.width
        # `name` is the canonical short codec name on all pyav versions;
        # `canonical_name` was removed in pyav 13.
        video_info["video.codec"] = getattr(
            vs.codec, "canonical_name", vs.codec.name
        )
        video_info["video.pix_fmt"] = vs.pix_fmt
        video_info["video.is_depth_map"] = False
        video_info["video.fps"] = int(vs.base_rate) if vs.base_rate else fps
        pix = vs.pix_fmt or ""
        if "gray" in pix or "depth" in pix or "monochrome" in pix:
            video_info["video.channels"] = 1
        elif "rgba" in pix or "yuva" in pix:
            video_info["video.channels"] = 4
        else:
            video_info["video.channels"] = 3
        video_info["has_audio"] = len(f.streams.audio) > 0
    return video_info
