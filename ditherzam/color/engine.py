from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..imaging import clamp_u8
from .palette import Palette
from .ramp import build_ramp


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


@njit(cache=True, parallel=True)
def _ordered_rgb_njit(rgb_f32, pal_f32, bayer_f32):
    """Bayer-bias and palette-map RGB directly, without frame intermediates."""
    h, w = rgb_f32.shape[:2]
    k = pal_f32.shape[0]
    mh, mw = bayer_f32.shape
    out = np.empty((h, w, 3), dtype=np.uint8)
    spread = np.float32(255.0 / max(1, k - 1))
    for y in prange(h):
        for x in range(w):
            offset = bayer_f32[y % mh, x % mw] * spread
            r = rgb_f32[y, x, 0] + offset
            g = rgb_f32[y, x, 1] + offset
            b = rgb_f32[y, x, 2] + offset

            dr = r - pal_f32[0, 0]
            dg = g - pal_f32[0, 1]
            db = b - pal_f32[0, 2]
            best_distance = (dr * dr + dg * dg) + db * db
            best_index = 0
            for i in range(1, k):
                dr = r - pal_f32[i, 0]
                dg = g - pal_f32[i, 1]
                db = b - pal_f32[i, 2]
                distance = (dr * dr + dg * dg) + db * db
                if distance < best_distance:
                    best_distance = distance
                    best_index = i

            # Palette colors are constrained to 0..255. Assignment preserves
            # clamp_u8's truncation for fractional extracted-palette colors.
            out[y, x, 0] = pal_f32[best_index, 0]
            out[y, x, 1] = pal_f32[best_index, 1]
            out[y, x, 2] = pal_f32[best_index, 2]
    return out


@njit(cache=True)
def _floyd_steinberg_rgb_njit(rgb: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Sequential scalar RGB Floyd-Steinberg with first-minimum palette ties."""
    h, w = rgb.shape[:2]
    work = rgb.copy()
    out = np.empty((h, w, 3), dtype=np.float32)
    weight_right = np.float32(7.0 / 16.0)
    weight_down_left = np.float32(3.0 / 16.0)
    weight_down = np.float32(5.0 / 16.0)
    weight_down_right = np.float32(1.0 / 16.0)
    for y in range(h):
        for x in range(w):
            old_r = work[y, x, 0]
            old_g = work[y, x, 1]
            old_b = work[y, x, 2]

            dr = pal[0, 0] - old_r
            dg = pal[0, 1] - old_g
            db = pal[0, 2] - old_b
            best_distance = (dr * dr + dg * dg) + db * db
            best_index = 0
            for i in range(1, pal.shape[0]):
                dr = pal[i, 0] - old_r
                dg = pal[i, 1] - old_g
                db = pal[i, 2] - old_b
                distance = (dr * dr + dg * dg) + db * db
                if distance < best_distance:
                    best_distance = distance
                    best_index = i

            new_r = pal[best_index, 0]
            new_g = pal[best_index, 1]
            new_b = pal[best_index, 2]
            out[y, x, 0] = new_r
            out[y, x, 1] = new_g
            out[y, x, 2] = new_b
            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b
            if x + 1 < w:
                work[y, x + 1, 0] += err_r * weight_right
                work[y, x + 1, 1] += err_g * weight_right
                work[y, x + 1, 2] += err_b * weight_right
            if y + 1 < h:
                if x - 1 >= 0:
                    work[y + 1, x - 1, 0] += err_r * weight_down_left
                    work[y + 1, x - 1, 1] += err_g * weight_down_left
                    work[y + 1, x - 1, 2] += err_b * weight_down_left
                work[y + 1, x, 0] += err_r * weight_down
                work[y + 1, x, 1] += err_g * weight_down
                work[y + 1, x, 2] += err_b * weight_down
                if x + 1 < w:
                    work[y + 1, x + 1, 0] += err_r * weight_down_right
                    work[y + 1, x + 1, 1] += err_g * weight_down_right
                    work[y + 1, x + 1, 2] += err_b * weight_down_right
    return out


def _floyd_steinberg_rgb(rgb: np.ndarray, pal: np.ndarray) -> np.ndarray:
    rgb_f32 = np.ascontiguousarray(rgb, dtype=np.float32)
    pal_f32 = np.ascontiguousarray(pal, dtype=np.float32)
    return _floyd_steinberg_rgb_njit(rgb_f32, pal_f32)


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
    def __init__(self, palette: Palette, mode: str = "nearest", *,
                 depth: int = 2, mapping: str = "match", phase: float = 0.0) -> None:
        self.palette = palette
        self.mode = mode
        self.depth = depth
        self.mapping = mapping
        self.phase = phase
        self._ramp = None
        self._ramp_key = None

    def _get_ramp(self) -> np.ndarray:
        key = (self.palette.colors.tobytes(), int(self.depth),
               self.mapping, float(self.phase))
        if self._ramp_key != key:
            self._ramp = build_ramp(self.palette, self.depth, self.mapping, self.phase)
            self._ramp_key = key
        return self._ramp

    def map(self, gray_or_rgb_f32: np.ndarray) -> np.ndarray:
        rgb = _to_rgb(gray_or_rgb_f32)
        if self.mode == "off":
            return clamp_u8(rgb)
        if self.mode == "ramp":
            ramp = self._get_ramp()
            depth = ramp.shape[0]
            gray = rgb[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32) \
                if rgb.ndim == 3 else rgb
            if depth == 1:
                level = np.zeros(gray.shape, dtype=np.int64)
            else:
                level = np.clip(np.round(gray / 255.0 * (depth - 1)), 0, depth - 1)
                level = level.astype(np.int64)
            return clamp_u8(ramp[level])
        pal = self.palette.colors.astype(np.float32)
        if self.mode == "nearest":
            idx = nearest_indices(rgb, pal)
            return clamp_u8(pal[idx])
        if self.mode == "ordered":
            rgb_f32 = np.ascontiguousarray(rgb, dtype=np.float32)
            pal_f32 = np.ascontiguousarray(pal, dtype=np.float32)
            return _ordered_rgb_njit(rgb_f32, pal_f32, _BAYER4)
        if self.mode == "diffused":
            return clamp_u8(_floyd_steinberg_rgb(rgb, pal))
        raise ValueError(f"unknown ColorEngine mode: {self.mode!r}")
