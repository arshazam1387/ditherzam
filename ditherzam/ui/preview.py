"""Interactive preview proxy (Qt-free).

While a control is actively being dragged, render a downscaled proxy for instant
feedback, then a full-resolution pass once the drag settles. The proxy is an
approximation shown on screen only; the committed/exported image always comes
from the full-resolution ``render``/``render_cached`` path.

To keep the proxy visually close to the full render, the base is downscaled by
``proxy_factor`` and the dither block size (``scale``) is reduced by the same
factor, then the result is nearest-upscaled back to the display resolution so the
pixel-art blocks stay crisp and the on-screen image size does not pop.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..imaging import nearest_downscale, nearest_upscale_to


def proxy_factor(h: int, w: int, max_side: int) -> int:
    """Integer downscale factor so the longest side is <= max_side (>=1)."""
    longest = max(int(h), int(w))
    if longest <= max_side:
        return 1
    return int(math.ceil(longest / float(max_side)))


def proxy_scale(scale: int, factor: int) -> int:
    """Dither block size for the proxy, preserving visual block size (>=1)."""
    return max(1, int(round(int(scale) / float(factor))))


def render_preview(pipeline, base_gray, settings, max_side: int) -> np.ndarray:
    """Downscaled proxy render upscaled back to full display size (uint8 HxWx3).

    Falls back to a normal full render when the image already fits within
    ``max_side`` (factor 1), in which case the output is identical to
    ``pipeline.render(base_gray, settings)``.
    """
    h, w = base_gray.shape[:2]
    factor = proxy_factor(h, w, max_side)
    if factor <= 1:
        return pipeline.render(base_gray, settings)
    small = nearest_downscale(base_gray, factor)
    psettings = replace(settings, scale=proxy_scale(settings.scale, factor))
    rgb_small = pipeline.render(small, psettings)
    return nearest_upscale_to(rgb_small, (int(w), int(h))).astype(np.uint8)
