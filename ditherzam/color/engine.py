from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..imaging import clamp_u8
from .palette import Palette


@njit(cache=True, parallel=True)
def _nearest_indices_njit(rgb_f32, pal_f32):
    h, w = rgb_f32.shape[0], rgb_f32.shape[1]
    k = pal_f32.shape[0]
    out = np.empty((h, w), np.int64)
    for y in prange(h):
        for x in range(w):
            r = rgb_f32[y, x, 0]
            g = rgb_f32[y, x, 1]
            b = rgb_f32[y, x, 2]
            best_i = 0
            # squared distance to palette[0], summed in the same left-to-right
            # float32 order as the reference (dr*dr + dg*dg) + db*db
            dr = r - pal_f32[0, 0]
            dg = g - pal_f32[0, 1]
            db = b - pal_f32[0, 2]
            best_d = (dr * dr + dg * dg) + db * db
            for i in range(1, k):
                dr = r - pal_f32[i, 0]
                dg = g - pal_f32[i, 1]
                db = b - pal_f32[i, 2]
                d = (dr * dr + dg * dg) + db * db
                if d < best_d:          # strict: keep the first (lowest) index on ties
                    best_d = d
                    best_i = i
            out[y, x] = best_i
    return out


def nearest_indices(rgb_f32: np.ndarray, palette_f32: np.ndarray) -> np.ndarray:
    """Index of the nearest palette color (squared RGB distance) per pixel.

    Per-pixel loop over the (small) palette instead of a (H, W, K, 3) broadcast:
    same squared-distance argmin, no ~100 MB temporary. Output is bit-identical to
    the broadcast reference (see test_color_engine equivalence tests).
    """
    rgb = np.ascontiguousarray(rgb_f32, dtype=np.float32)
    pal = np.ascontiguousarray(palette_f32, dtype=np.float32)
    return _nearest_indices_njit(rgb, pal)


def _floyd_steinberg_rgb(rgb: np.ndarray, pal: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    work = rgb.astype(np.float32).copy()
    out = np.empty((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            old = work[y, x].copy()
            diff = pal - old
            idx = int((diff * diff).sum(axis=1).argmin())
            new = pal[idx]
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                work[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < h:
                if x - 1 >= 0:
                    work[y + 1, x - 1] += err * (3.0 / 16.0)
                work[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < w:
                    work[y + 1, x + 1] += err * (1.0 / 16.0)
    return out


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
        if self.mode == "diffused":
            return clamp_u8(_floyd_steinberg_rgb(rgb, pal))
        raise ValueError(f"unknown ColorEngine mode: {self.mode!r}")
