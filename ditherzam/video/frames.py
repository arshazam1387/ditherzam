"""Headless per-frame dithering. Qt-free: imports only numpy, PIL, and the Phase 1/4
render contracts. Loads extracted frames in sorted order, renders each through the
pipeline, and writes `frame%06d.png` to the output directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from ditherzam.imaging import to_gray_f32

_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _sorted_frames(frames_dir) -> list[Path]:
    d = Path(frames_dir)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _EXTS)


def dither_frames(
    in_dir,
    out_dir,
    pipeline,
    settings,
    progress: Callable[[int, int], None] = lambda i, n: None,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> int:
    """Dither every frame in `in_dir` through `pipeline`, writing to `out_dir`.

    Returns the number of frames written. Honors cooperative cancellation: when
    `is_cancelled()` is truthy at the start of an iteration, processing stops and
    the count of frames already written is returned. `progress(done, total)` is
    emitted after each successful frame.
    """
    frames = _sorted_frames(in_dir)
    total = len(frames)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    for idx, src in enumerate(frames):
        if is_cancelled():
            break
        with Image.open(src) as im:
            gray = to_gray_f32(im.convert("L"))
        rgb_u8 = pipeline.render(gray, settings)  # uint8 HxWx3
        dst = out / f"frame{idx:06d}.png"
        Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8)).save(dst)
        written += 1
        progress(written, total)
    return written


def detect_preview_frame(frames_dir, min_mean: float = 5.0) -> str | None:
    """First frame (sorted) whose mean intensity exceeds `min_mean`, else None.

    Used to skip leading mostly-black frames when choosing a preview thumbnail.
    """
    for p in _sorted_frames(frames_dir):
        with Image.open(p) as im:
            if float(np.asarray(im.convert("L"), dtype=np.float32).mean()) > min_mean:
                return str(p)
    return None
