from __future__ import annotations

from numba import njit


@njit(cache=True)
def quantize_to_levels(v, levels):
    """Snap ``v`` (0..255) to the nearest of ``levels`` evenly-spaced values."""
    if levels <= 1:
        return 0.0
    x = v
    if x < 0.0:
        x = 0.0
    elif x > 255.0:
        x = 255.0
    step = 255.0 / (levels - 1)
    q = round(x / step)
    return q * step
