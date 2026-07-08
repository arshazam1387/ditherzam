from __future__ import annotations
import numpy as np
from numba import njit, prange
from ditherzam.dithering import registry
from ditherzam.dithering.nlevels import quantize_to_levels


@njit(cache=True)
def _floyd_steinberg(img, thr, levels=2):
    h, w = img.shape
    out = img.copy()
    if levels <= 2:
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
    bias = 127.5 - thr
    for y in range(h):
        for x in range(w):
            old = out[y, x] + bias
            new = quantize_to_levels(old, levels)
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
    return out


@registry.register("Floyd-Steinberg", "Error Diffusion", dims=2)
def floyd_steinberg(image_array, parameter, luminance_threshold_value, levels=2):
    return _floyd_steinberg(image_array.astype(np.float32),
                            luminance_threshold_value, levels)


@njit(cache=True)
def _atkinson(img, thr, levels=2):
    h, w = img.shape
    out = img.copy()
    offs = ((0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0))
    if levels <= 2:
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
    bias = 127.5 - thr
    for y in range(h):
        for x in range(w):
            old = out[y, x] + bias
            new = quantize_to_levels(old, levels)
            out[y, x] = new
            err = (old - new) / 8.0
            for dy, dx in offs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    out[ny, nx] += err
    return out


@registry.register("Atkinson", "Error Diffusion", dims=2)
def atkinson(image_array, parameter, luminance_threshold_value, levels=2):
    return _atkinson(image_array.astype(np.float32), luminance_threshold_value, levels)


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
def _ordered(img, thresholds, levels=2):
    h, w = img.shape
    mh, mw = thresholds.shape
    out = np.empty_like(img)
    if levels <= 2:
        for y in prange(h):
            for x in range(w):
                t = thresholds[y % mh, x % mw]
                out[y, x] = 255.0 if img[y, x] >= t else 0.0
        return out
    step = 255.0 / (levels - 1)
    for y in prange(h):
        for x in range(w):
            off = (thresholds[y % mh, x % mw] / 255.0 - 0.5) * step
            out[y, x] = quantize_to_levels(img[y, x] + off, levels)
    return out


@registry.register("Bayer-Matrix 4x4", "Ordered Dither", dims=2)
def bayer_4(image_array, parameter, luminance_threshold_value, levels=2):
    return _ordered(image_array.astype(np.float32), _BAYER4, levels)


@njit(cache=True)
def _diffuse(img, thr, offsets, weights, divisor, levels=2):
    h, w = img.shape
    out = img.copy()
    n = offsets.shape[0]
    if levels <= 2:
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
    bias = 127.5 - thr
    for y in range(h):
        for x in range(w):
            old = out[y, x] + bias
            new = quantize_to_levels(old, levels)
            out[y, x] = new
            err = old - new
            for k in range(n):
                ny = y + offsets[k, 0]
                nx = x + offsets[k, 1]
                if 0 <= ny < h and 0 <= nx < w:
                    out[ny, nx] += err * weights[k] / divisor
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


# ── Kernel: None · Error Diffusion · dims=2 · no sliders (no-op passthrough) ──
@registry.register("None", "Error Diffusion", dims=2)
def no_dither(image_array, parameter, luminance_threshold_value):
    return image_array.astype(np.float32)


# ── Classic weighted-diffusion offset/weight tables (float32 weights) ──
_JJN_OFF = np.array([[0, 1], [0, 2], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                     [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]], dtype=np.int64)
_JJN_W = np.array([7, 5, 3, 5, 7, 5, 3, 1, 3, 5, 3, 1], dtype=np.float32)
_JJN_DIV = 48.0

_STUCKI_OFF = _JJN_OFF
_STUCKI_W = np.array([8, 4, 2, 4, 8, 4, 2, 1, 2, 4, 2, 1], dtype=np.float32)

_BURKES_OFF = np.array([[0, 1], [0, 2], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2]],
                       dtype=np.int64)
_BURKES_W = np.array([8, 4, 2, 4, 8, 4, 2], dtype=np.float32)

_SIERRA_OFF = np.array([[0, 1], [0, 2], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                        [2, -1], [2, 0], [2, 1]], dtype=np.int64)
_SIERRA_W = np.array([5, 3, 2, 4, 5, 4, 2, 2, 3, 2], dtype=np.float32)

_SIERRA_LITE_OFF = np.array([[0, 1], [1, -1], [1, 0]], dtype=np.int64)
_SIERRA_LITE_W = np.array([2, 1, 1], dtype=np.float32)

_TWO_ROW_OFF = np.array([[0, 1], [0, 2], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2]],
                        dtype=np.int64)
_TWO_ROW_W = np.array([4, 3, 1, 2, 3, 2, 1], dtype=np.float32)

_STEVENSON_OFF = np.array([[0, 2],
                           [1, -3], [1, -1], [1, 1], [1, 3],
                           [2, -2], [2, 0], [2, 2],
                           [3, -3], [3, -1], [3, 1], [3, 3]], dtype=np.int64)
_STEVENSON_W = np.array([32, 12, 26, 30, 16, 12, 26, 12, 5, 12, 12, 5],
                        dtype=np.float32)

_FAN_OFF = np.array([[0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int64)
_FAN_W = np.array([7, 1, 3, 5], dtype=np.float32)

_SHIAU_OFF = np.array([[0, 1], [1, -2], [1, -1], [1, 0], [1, 1]], dtype=np.int64)
_SHIAU_W = np.array([8, 1, 1, 2, 4], dtype=np.float32)

_FALSE_FS_OFF = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int64)
_FALSE_FS_W = np.array([3, 3, 2], dtype=np.float32)

_ATK_LIGHT_OFF = np.array([[0, 1], [0, 2], [1, 0], [1, 1]], dtype=np.int64)
_ATK_LIGHT_W = np.array([1, 1, 1, 1], dtype=np.float32)  # /8 (Atkinson-style bleed)


@registry.register("Jarvis-Judice-Ninke", "Error Diffusion", dims=2)
def jjn(image_array, parameter, luminance_threshold_value, levels=2):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _JJN_OFF, _JJN_W, _JJN_DIV, levels)


@registry.register("Stucki", "Error Diffusion", dims=2)
def stucki(image_array, parameter, luminance_threshold_value, levels=2):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _STUCKI_OFF, _STUCKI_W, 42.0, levels)


@registry.register("Burkes", "Error Diffusion", dims=2)
def burkes(image_array, parameter, luminance_threshold_value, levels=2):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _BURKES_OFF, _BURKES_W, 32.0, levels)


@registry.register("Sierra", "Error Diffusion", dims=2)
def sierra(image_array, parameter, luminance_threshold_value, levels=2):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _SIERRA_OFF, _SIERRA_W, 32.0, levels)


@registry.register("Sierra-Lite", "Error Diffusion", dims=2)
def sierra_lite(image_array, parameter, luminance_threshold_value, levels=2):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _SIERRA_LITE_OFF, _SIERRA_LITE_W, 4.0, levels)


@registry.register("Two-Row-Sierra", "Error Diffusion", dims=2)
def two_row_sierra(image_array, parameter, luminance_threshold_value, levels=2):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _TWO_ROW_OFF, _TWO_ROW_W, 16.0, levels)


@registry.register("Stevenson-Arce", "Error Diffusion", dims=2)
def stevenson_arce(image_array, parameter, luminance_threshold_value, levels=2):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _STEVENSON_OFF, _STEVENSON_W, 200.0, levels)


@registry.register("Fan", "Error Diffusion", dims=2)
def fan(image_array, parameter, luminance_threshold_value, levels=2):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _FAN_OFF, _FAN_W, 16.0, levels)


@registry.register("Shiau-Fan", "Error Diffusion", dims=2)
def shiau_fan(image_array, parameter, luminance_threshold_value, levels=2):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _SHIAU_OFF, _SHIAU_W, 16.0, levels)


@registry.register("False Floyd-Steinberg", "Error Diffusion", dims=2)
def false_floyd_steinberg(image_array, parameter, luminance_threshold_value, levels=2):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _FALSE_FS_OFF, _FALSE_FS_W, 8.0, levels)


@registry.register("Atkinson-Light", "Error Diffusion", dims=2)
def atkinson_light(image_array, parameter, luminance_threshold_value, levels=2):
    # Atkinson-style: only 4/8 of the error propagates (softer than full Atkinson).
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _ATK_LIGHT_OFF, _ATK_LIGHT_W, 8.0, levels)


# ── Kernel: Ostromukhov · Error Diffusion · dims=2 · simplified variable coeffs ──
@njit(cache=True)
def _ostromukhov(img, thr):
    h, w = img.shape
    out = img.copy()
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = old - new
            v = old / 255.0
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            # value-dependent coefficients (right, down-left, down)
            d1 = 13.0 + v * 8.0
            d2 = (1.0 - abs(2.0 * v - 1.0)) * 7.0
            d3 = 5.0 + (1.0 - v) * 8.0
            s = d1 + d2 + d3
            if x + 1 < w:
                out[y, x + 1] += err * d1 / s
            if y + 1 < h:
                if x - 1 >= 0:
                    out[y + 1, x - 1] += err * d2 / s
                out[y + 1, x] += err * d3 / s
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@registry.register("Ostromukhov", "Error Diffusion", dims=2)
def ostromukhov(image_array, parameter, luminance_threshold_value):
    return _ostromukhov(image_array.astype(np.float32), luminance_threshold_value)


# ── Kernel: Gaussian · Error Diffusion · dims=2 · slider Distribution Spread 1-20-1 ──
@njit(cache=True)
def _gaussian_dither(img, spread, thr):
    np.random.seed(0)
    h, w = img.shape
    out = np.empty_like(img)
    sigma = spread * 3.0
    for y in range(h):
        for x in range(w):
            n = np.random.standard_normal() * sigma
            out[y, x] = 255.0 if (img[y, x] + n) >= thr else 0.0
    return out


@registry.register("Gaussian", "Error Diffusion", dims=2,
                   param_sliders=("dither_parameter_slider",))
def gaussian(image_array, parameter, luminance_threshold_value):
    spread = float(parameter) if parameter else 1.0
    return _gaussian_dither(image_array.astype(np.float32), spread,
                            luminance_threshold_value)
