from __future__ import annotations

import colorsys

import numpy as np

from .palette import Palette

RAMP_MODES: tuple[str, ...] = (
    "match", "interpolated", "glitch", "reverse", "hue_cycle", "banded",
)

_LUM_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _clamp_depth(depth: int) -> int:
    return int(min(64, max(1, int(depth))))


def _lum_sorted(colors: np.ndarray) -> np.ndarray:
    lum = colors @ _LUM_W
    order = np.argsort(lum, kind="stable")   # stable: deterministic on ties
    return colors[order]


def _nearest_sample(sorted_colors: np.ndarray, depth: int) -> np.ndarray:
    k = sorted_colors.shape[0]
    if depth == 1:
        return sorted_colors[:1].copy()
    idx = np.round(np.linspace(0, k - 1, depth)).astype(np.int64)
    return sorted_colors[idx].copy()


def _interp_sample(sorted_colors: np.ndarray, depth: int) -> np.ndarray:
    k = sorted_colors.shape[0]
    if depth == 1 or k == 1:
        return np.repeat(sorted_colors[:1], depth, axis=0).astype(np.float32)
    pos = np.linspace(0.0, k - 1, depth)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, k - 1)
    frac = (pos - lo)[:, None].astype(np.float32)
    return (sorted_colors[lo] * (1.0 - frac) + sorted_colors[hi] * frac).astype(np.float32)


def _hue_cycle(depth: int, phase: float) -> np.ndarray:
    out = np.empty((depth, 3), dtype=np.float32)
    for i in range(depth):
        h = (phase + (i / depth if depth else 0.0)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        out[i] = (r * 255.0, g * 255.0, b * 255.0)
    return out


def _apply_phase(ramp: np.ndarray, phase: float) -> np.ndarray:
    depth = ramp.shape[0]
    if depth <= 1 or not phase:
        return ramp
    shift = int(round((phase % 1.0) * depth)) % depth
    if shift == 0:
        return ramp
    return np.roll(ramp, shift, axis=0)


def build_ramp(palette: Palette, depth: int, mapping: str,
               phase: float = 0.0) -> np.ndarray:
    """Build a float32[depth,3] RGB tone ramp (0..255) from ``palette``.

    ``depth`` is clamped to [1,64]. ``phase`` in [0,1] cyclically rotates the
    ramp (reserved for animation). See RAMP_MODES for ``mapping`` values.
    """
    depth = _clamp_depth(depth)
    colors = np.ascontiguousarray(palette.colors, dtype=np.float32)

    if mapping == "hue_cycle":
        return _apply_phase(_hue_cycle(depth, phase), 0.0)
    if mapping == "glitch":
        idx = np.arange(depth) % colors.shape[0]
        return _apply_phase(colors[idx].astype(np.float32), phase)
    if mapping == "banded":
        s = _lum_sorted(colors)
        idx = np.arange(depth) % s.shape[0]
        return _apply_phase(s[idx].astype(np.float32), phase)

    s = _lum_sorted(colors)
    if mapping == "match":
        ramp = _nearest_sample(s, depth)
    elif mapping == "interpolated":
        ramp = _interp_sample(s, depth)
    elif mapping == "reverse":
        ramp = _nearest_sample(s, depth)[::-1].copy()
    else:
        raise ValueError(f"unknown ramp mapping: {mapping!r}")
    return _apply_phase(ramp.astype(np.float32), phase)
