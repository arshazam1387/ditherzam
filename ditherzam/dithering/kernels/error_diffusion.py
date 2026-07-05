from __future__ import annotations
import numpy as np
from numba import njit, prange
from ditherzam.dithering import registry


@njit(cache=True)
def _floyd_steinberg(img, thr):
    h, w = img.shape
    out = img.copy()
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                out[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    out[y + 1, x - 1] += err * 3 / 16
                out[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    out[y + 1, x + 1] += err * 1 / 16
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@registry.register("Floyd-Steinberg", "Error Diffusion", dims=2)
def floyd_steinberg(image_array, parameter, luminance_threshold_value):
    return _floyd_steinberg(image_array.astype(np.float32), luminance_threshold_value)


@njit(cache=True)
def _atkinson(img, thr):
    h, w = img.shape
    out = img.copy()
    offs = ((0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0))
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = (old - new) / 8.0
            for dy, dx in offs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    out[ny, nx] += err
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@registry.register("Atkinson", "Error Diffusion", dims=2)
def atkinson(image_array, parameter, luminance_threshold_value):
    return _atkinson(image_array.astype(np.float32), luminance_threshold_value)


def _bayer_matrix(n: int) -> np.ndarray:
    if n == 1:
        return np.zeros((1, 1), dtype=np.float32)
    smaller = _bayer_matrix(n // 2)
    m = np.block([
        [4 * smaller + 0, 4 * smaller + 2],
        [4 * smaller + 3, 4 * smaller + 1],
    ]).astype(np.float32)
    return m


_BAYER4 = (_bayer_matrix(4) + 0.5) / 16.0 * 255.0  # thresholds 0..255


@njit(cache=True, parallel=True)
def _ordered(img, thresholds):
    h, w = img.shape
    mh, mw = thresholds.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            t = thresholds[y % mh, x % mw]
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@registry.register("Bayer-Matrix 4x4", "Ordered Dither", dims=2)
def bayer_4(image_array, parameter, luminance_threshold_value):
    return _ordered(image_array.astype(np.float32), _BAYER4)


@njit(cache=True)
def _diffuse(img, thr, offsets, weights, divisor):
    h, w = img.shape
    out = img.copy()
    n = offsets.shape[0]
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = old - new
            for k in range(n):
                ny = y + offsets[k, 0]
                nx = x + offsets[k, 1]
                if 0 <= ny < h and 0 <= nx < w:
                    out[ny, nx] += err * weights[k] / divisor
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@njit(cache=True)
def _diffuse_row(img, thr, w_right):
    h, w = img.shape
    out = img.copy()
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                out[y, x + 1] += err * w_right
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out
