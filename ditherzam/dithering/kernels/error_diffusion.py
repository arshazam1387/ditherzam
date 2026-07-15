from __future__ import annotations
import numpy as np
from numba import njit, prange
from ditherzam.dithering import registry
from ditherzam.dithering.nlevels import quantize_to_levels


_DIFFUSION_SLIDERS = (
    "diffusion_strength_slider", "lateral_spread_slider",
    "downward_spread_slider", "direction_bias_slider",
    "error_response_slider",
)


def _diffusion_params(parameter):
    """Return normalized native diffusion controls (legacy-safe)."""
    defaults = (100.0, 100.0, 100.0, 50.0, 100.0)
    if not isinstance(parameter, (tuple, list)):
        return defaults
    vals = list(parameter[:5]) + list(defaults[len(parameter):])
    return tuple(float(v) for v in vals[:5])


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


@registry.register("Floyd-Steinberg", "Error Diffusion", dims=2, supports_levels=True,
                   param_sliders=_DIFFUSION_SLIDERS)
def floyd_steinberg(image_array, parameter, luminance_threshold_value, levels=2):
    controls = _diffusion_params(parameter)
    if controls != (100.0, 100.0, 100.0, 50.0, 100.0):
        return _diffuse_controlled(image_array.astype(np.float32),
                                   luminance_threshold_value,
                                   np.array([[0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int64),
                                   np.array([7, 3, 5, 1], dtype=np.float32), 16.0,
                                   levels, *controls)
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


@registry.register("Atkinson", "Error Diffusion", dims=2, supports_levels=True,
                   param_sliders=_DIFFUSION_SLIDERS)
def atkinson(image_array, parameter, luminance_threshold_value, levels=2):
    controls = _diffusion_params(parameter)
    if controls != (100.0, 100.0, 100.0, 50.0, 100.0):
        offsets = np.array([[0, 1], [0, 2], [1, -1], [1, 0], [1, 1], [2, 0]], dtype=np.int64)
        return _diffuse_controlled(image_array.astype(np.float32), luminance_threshold_value,
                                   offsets, np.ones(6, dtype=np.float32), 8.0,
                                   levels, *controls)
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


@njit(cache=True, parallel=True)
def _bayer4_controlled(img, levels, cell_size, matrix_turn, contrast, offset_x, offset_y):
    h, w = img.shape
    out = np.empty_like(img)
    step = 255.0 / (levels - 1) if levels > 2 else 255.0
    for y in prange(h):
        for x in range(w):
            row = (y // cell_size + offset_y) % 4
            col = (x // cell_size + offset_x) % 4
            for _ in range(matrix_turn % 4):
                row, col = col, 3 - row
            threshold = (_BAYER4[row, col] - np.float32(127.5)) * contrast + np.float32(127.5)
            if levels <= 2:
                out[y, x] = np.float32(255.0) if img[y, x] >= threshold else np.float32(0.0)
            else:
                off = (threshold / np.float32(255.0) - np.float32(0.5)) * step
                out[y, x] = quantize_to_levels(img[y, x] + off, levels)
    return out


@registry.register("Bayer-Matrix 4x4", "Ordered Dither", dims=2, supports_levels=True,
                   param_sliders=("bayer_cell_size_slider", "matrix_rotation_slider",
                                  "threshold_contrast_slider", "matrix_offset_x_slider",
                                  "matrix_offset_y_slider"))
def bayer_4(image_array, parameter, luminance_threshold_value, levels=2):
    defaults = (1, 0, 100, 0, 0)
    if isinstance(parameter, (tuple, list)):
        values = list(parameter[:5]) + list(defaults[len(parameter):])
        cell, turn, contrast, ox, oy = (int(v) for v in values[:5])
    else:
        cell, turn, contrast, ox, oy = defaults
    if (cell, turn, contrast, ox, oy) != defaults:
        return _bayer4_controlled(image_array.astype(np.float32), levels,
                                  max(1, cell), turn, np.float32(contrast / 100.0),
                                  ox, oy)
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
def _diffuse_controlled(img, thr, offsets, weights, divisor, levels,
                        strength, lateral, downward, direction, response):
    """Weighted diffusion with five orthogonal, kernel-native controls."""
    h, w = img.shape
    out = img.copy()
    n = offsets.shape[0]
    bias = 127.5 - thr
    exponent = 100.0 / max(response, 1.0)
    for y in range(h):
        for x in range(w):
            old = out[y, x] if levels <= 2 else out[y, x] + bias
            new = (255.0 if old >= thr else 0.0) if levels <= 2 else quantize_to_levels(old, levels)
            out[y, x] = new
            err = old - new
            mag = min(abs(err) / 255.0, 4.0)
            if mag > 0.0:
                err = (1.0 if err >= 0.0 else -1.0) * (mag ** exponent) * 255.0
            for k in range(n):
                ny, nx = y + offsets[k, 0], x + offsets[k, 1]
                if 0 <= ny < h and 0 <= nx < w:
                    factor = strength / 100.0
                    if offsets[k, 0] == 0:
                        factor *= lateral / 100.0
                    else:
                        factor *= downward / 100.0
                    if offsets[k, 1] < 0:
                        factor *= (100.0 - direction) / 50.0
                    elif offsets[k, 1] > 0 and offsets[k, 0] > 0:
                        factor *= direction / 50.0
                    out[ny, nx] += err * weights[k] / divisor * factor
    if levels <= 2:
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


def _run_diffusion(image_array, parameter, threshold, offsets, weights, divisor, levels):
    controls = _diffusion_params(parameter)
    image = image_array.astype(np.float32)
    if controls == (100.0, 100.0, 100.0, 50.0, 100.0):
        return _diffuse(image, threshold, offsets, weights, divisor, levels)
    return _diffuse_controlled(image, threshold, offsets, weights, divisor, levels, *controls)


@registry.register("Jarvis-Judice-Ninke", "Error Diffusion", dims=2, supports_levels=True,
                   param_sliders=_DIFFUSION_SLIDERS)
def jjn(image_array, parameter, luminance_threshold_value, levels=2):
    return _run_diffusion(image_array, parameter, luminance_threshold_value,
                          _JJN_OFF, _JJN_W, _JJN_DIV, levels)


@registry.register("Stucki", "Error Diffusion", dims=2, supports_levels=True, param_sliders=_DIFFUSION_SLIDERS)
def stucki(image_array, parameter, luminance_threshold_value, levels=2):
    return _run_diffusion(image_array, parameter, luminance_threshold_value, _STUCKI_OFF, _STUCKI_W, 42.0, levels)


@registry.register("Burkes", "Error Diffusion", dims=2, supports_levels=True, param_sliders=_DIFFUSION_SLIDERS)
def burkes(image_array, parameter, luminance_threshold_value, levels=2):
    return _run_diffusion(image_array, parameter, luminance_threshold_value, _BURKES_OFF, _BURKES_W, 32.0, levels)


@registry.register("Sierra", "Error Diffusion", dims=2, supports_levels=True, param_sliders=_DIFFUSION_SLIDERS)
def sierra(image_array, parameter, luminance_threshold_value, levels=2):
    return _run_diffusion(image_array, parameter, luminance_threshold_value, _SIERRA_OFF, _SIERRA_W, 32.0, levels)


@registry.register("Sierra-Lite", "Error Diffusion", dims=2, supports_levels=True, param_sliders=_DIFFUSION_SLIDERS)
def sierra_lite(image_array, parameter, luminance_threshold_value, levels=2):
    return _run_diffusion(image_array, parameter, luminance_threshold_value, _SIERRA_LITE_OFF, _SIERRA_LITE_W, 4.0, levels)


@registry.register("Two-Row-Sierra", "Error Diffusion", dims=2, supports_levels=True, param_sliders=_DIFFUSION_SLIDERS)
def two_row_sierra(image_array, parameter, luminance_threshold_value, levels=2):
    return _run_diffusion(image_array, parameter, luminance_threshold_value, _TWO_ROW_OFF, _TWO_ROW_W, 16.0, levels)


@registry.register("Stevenson-Arce", "Error Diffusion", dims=2, supports_levels=True, param_sliders=_DIFFUSION_SLIDERS)
def stevenson_arce(image_array, parameter, luminance_threshold_value, levels=2):
    return _run_diffusion(image_array, parameter, luminance_threshold_value, _STEVENSON_OFF, _STEVENSON_W, 200.0, levels)


@registry.register("Fan", "Error Diffusion", dims=2, supports_levels=True, param_sliders=_DIFFUSION_SLIDERS)
def fan(image_array, parameter, luminance_threshold_value, levels=2):
    return _run_diffusion(image_array, parameter, luminance_threshold_value, _FAN_OFF, _FAN_W, 16.0, levels)


@registry.register("Shiau-Fan", "Error Diffusion", dims=2, supports_levels=True, param_sliders=_DIFFUSION_SLIDERS)
def shiau_fan(image_array, parameter, luminance_threshold_value, levels=2):
    return _run_diffusion(image_array, parameter, luminance_threshold_value, _SHIAU_OFF, _SHIAU_W, 16.0, levels)


@registry.register("False Floyd-Steinberg", "Error Diffusion", dims=2, supports_levels=True, param_sliders=_DIFFUSION_SLIDERS)
def false_floyd_steinberg(image_array, parameter, luminance_threshold_value, levels=2):
    return _run_diffusion(image_array, parameter, luminance_threshold_value, _FALSE_FS_OFF, _FALSE_FS_W, 8.0, levels)


@registry.register("Atkinson-Light", "Error Diffusion", dims=2, supports_levels=True, param_sliders=_DIFFUSION_SLIDERS)
def atkinson_light(image_array, parameter, luminance_threshold_value, levels=2):
    # Atkinson-style: only 4/8 of the error propagates (softer than full Atkinson).
    return _run_diffusion(image_array, parameter, luminance_threshold_value, _ATK_LIGHT_OFF, _ATK_LIGHT_W, 8.0, levels)


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


@njit(cache=True)
def _ostromukhov_controlled(img, thr, strength, lateral, downward, direction, response):
    h, w = img.shape
    out = img.copy()
    exponent = 100.0 / max(response, 1.0)
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = old - new
            mag = min(abs(err) / 255.0, 4.0)
            if mag > 0.0:
                err = (1.0 if err >= 0.0 else -1.0) * mag ** exponent * 255.0
            v = min(1.0, max(0.0, old / 255.0))
            d1 = (13.0 + v * 8.0) * lateral / 100.0
            d2 = (1.0 - abs(2.0 * v - 1.0)) * 7.0 * downward / 100.0 * (100.0 - direction) / 50.0
            d3 = (5.0 + (1.0 - v) * 8.0) * downward / 100.0
            s = max(d1 + d2 + d3, 1e-6)
            scale = strength / 100.0
            if x + 1 < w:
                out[y, x + 1] += err * d1 / s * scale
            if y + 1 < h:
                if x - 1 >= 0:
                    out[y + 1, x - 1] += err * d2 / s * scale
                out[y + 1, x] += err * d3 / s * scale
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@registry.register("Ostromukhov", "Error Diffusion", dims=2, param_sliders=_DIFFUSION_SLIDERS)
def ostromukhov(image_array, parameter, luminance_threshold_value):
    controls = _diffusion_params(parameter)
    if controls != (100.0, 100.0, 100.0, 50.0, 100.0):
        return _ostromukhov_controlled(image_array.astype(np.float32),
                                       luminance_threshold_value, *controls)
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


@njit(cache=True)
def _gaussian_controlled(img, spread, thr, tail, horizontal, vertical, seed):
    np.random.seed(max(0, int(seed)))
    h, w = img.shape
    noise = np.empty_like(img)
    sigma = spread * 3.0
    exponent = 50.0 / max(tail, 1.0)
    for y in range(h):
        for x in range(w):
            z = np.random.standard_normal()
            z = (1.0 if z >= 0 else -1.0) * abs(z) ** exponent
            noise[y, x] = z * sigma
    # Directional grain: 50 is independent legacy noise; larger values blend
    # neighbouring noise along that axis, smaller values subtract it.
    hx = (horizontal - 50.0) / 100.0
    vy = (vertical - 50.0) / 100.0
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            n = noise[y, x]
            if x > 0:
                n += noise[y, x - 1] * hx
            if y > 0:
                n += noise[y - 1, x] * vy
            out[y, x] = 255.0 if img[y, x] + n >= thr else 0.0
    return out


@registry.register("Gaussian", "Error Diffusion", dims=2,
                   param_sliders=("dither_parameter_slider", "noise_tail_slider",
                                  "horizontal_grain_slider", "vertical_grain_slider",
                                  "noise_seed_slider"))
def gaussian(image_array, parameter, luminance_threshold_value):
    if isinstance(parameter, (tuple, list)):
        vals = list(parameter) + [50, 50, 50, 0]
        spread, tail, horizontal, vertical, seed = (float(v) for v in vals[:5])
    else:
        spread = float(parameter) if parameter else 1.0
        tail = horizontal = vertical = 50.0
        seed = 0.0
    if (tail, horizontal, vertical, seed) != (50.0, 50.0, 50.0, 0.0):
        return _gaussian_controlled(image_array.astype(np.float32), spread,
                                    luminance_threshold_value, tail, horizontal,
                                    vertical, seed)
    return _gaussian_dither(image_array.astype(np.float32), spread,
                            luminance_threshold_value)
