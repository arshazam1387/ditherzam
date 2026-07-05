from __future__ import annotations

import numpy as np

from ..imaging import clamp_u8
from .palette import Palette


def nearest_indices(rgb_f32: np.ndarray, palette_f32: np.ndarray) -> np.ndarray:
    """Index of the nearest palette color (squared RGB distance) per pixel."""
    diff = rgb_f32[:, :, None, :] - palette_f32[None, None, :, :]
    dist = (diff * diff).sum(axis=-1)
    return dist.argmin(axis=-1)


def _bayer_matrix(n: int) -> np.ndarray:
    if n == 1:
        return np.zeros((1, 1), dtype=np.float32)
    smaller = _bayer_matrix(n // 2)
    return np.block([
        [4 * smaller + 0, 4 * smaller + 2],
        [4 * smaller + 3, 4 * smaller + 1],
    ]).astype(np.float32)


# 4x4 Bayer thresholds normalized to the range [-0.5, 0.5)
_BAYER4 = (_bayer_matrix(4) + 0.5) / 16.0 - 0.5


def _to_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    return arr[..., :3].astype(np.float32)


class ColorEngine:
    def __init__(self, palette: Palette, mode: str = "nearest") -> None:
        self.palette = palette
        self.mode = mode

    def map(self, gray_or_rgb_f32: np.ndarray) -> np.ndarray:
        rgb = _to_rgb(gray_or_rgb_f32)
        if self.mode == "off":
            return clamp_u8(rgb)
        pal = self.palette.colors.astype(np.float32)
        if self.mode == "nearest":
            idx = nearest_indices(rgb, pal)
            return clamp_u8(pal[idx])
        if self.mode == "ordered":
            k = pal.shape[0]
            spread = 255.0 / max(1, k - 1)
            h, w = rgb.shape[:2]
            mh, mw = _BAYER4.shape
            offset = _BAYER4[np.arange(h)[:, None] % mh,
                             np.arange(w)[None, :] % mw]
            biased = rgb + offset[:, :, None] * spread
            idx = nearest_indices(biased.astype(np.float32), pal)
            return clamp_u8(pal[idx])
        raise ValueError(f"unknown ColorEngine mode: {self.mode!r}")
