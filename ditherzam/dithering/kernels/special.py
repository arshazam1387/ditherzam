from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from ditherzam.dithering import registry


@njit(cache=True, parallel=True)
def _radial_burst(img, thr):
    h, w = img.shape
    cx = w / 2.0
    cy = h / 2.0
    rays = 24.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            ang = math.atan2(y - cy, x - cx)
            t = (math.sin(ang * rays) * 0.5 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _wave(img, thr):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            t = ((math.sin(x * 0.15) + math.sin(y * 0.15)) * 0.25 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True)
def _noise(img, thr):
    np.random.seed(0)
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            n = (np.random.random() - 0.5) * 255.0
            out[y, x] = 255.0 if (img[y, x] + n) >= thr else 0.0
    return out


@njit(cache=True, parallel=True)
def _topography(img, warp):
    h, w = img.shape
    bands = 8.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            sx = int(x + math.sin(y * 0.1) * warp)
            if sx < 0:
                sx = 0
            elif sx >= w:
                sx = w - 1
            b0 = int(img[y, sx] / 256.0 * bands)
            xr = sx + 1 if sx + 1 < w else sx
            yd = y + 1 if y + 1 < h else y
            br = int(img[y, xr] / 256.0 * bands)
            bd = int(img[int(yd), sx] / 256.0 * bands)
            out[y, x] = 0.0 if (b0 != br or b0 != bd) else 255.0
    return out


@njit(cache=True, parallel=True)
def _thresholder(img, freq):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            t = 128.0 + math.sin(x / freq) * math.cos(y / freq) * 64.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _diagonal(img, sensitivity):
    h, w = img.shape
    out = np.empty_like(img)
    thr_edge = 200.0 / sensitivity
    for y in prange(h):
        for x in range(w):
            xl = x - 1 if x > 0 else x
            xr = x + 1 if x < w - 1 else x
            yu = y - 1 if y > 0 else y
            yd = y + 1 if y < h - 1 else y
            gx = img[y, xr] - img[y, xl]
            gy = img[int(yd), x] - img[int(yu), x]
            mag = math.sqrt(gx * gx + gy * gy)
            out[y, x] = 0.0 if mag > thr_edge else 255.0
    return out


@njit(cache=True, parallel=True)
def _displace_contour(img, contour_thr, line_mode, smoothing, line_space):
    h, w = img.shape
    bands = line_space if line_space >= 1 else 1
    thick = line_mode if line_mode >= 1 else 1
    step = 256.0 / (bands * 4.0)
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            val = img[y, x]
            b0 = int(val / step)
            is_line = False
            for t in range(thick):
                xr = x + 1 + t if x + 1 + t < w else w - 1
                yd = y + 1 + t if y + 1 + t < h else h - 1
                if int(img[y, xr] / step) != b0 or int(img[yd, x] / step) != b0:
                    is_line = True
            gate = val < (contour_thr / 100.0 * 255.0)
            out[y, x] = 0.0 if (is_line and gate) else 255.0
    return out


@njit(cache=True, parallel=True)
def _sine_wave_modulation(img, freq, wave_thr):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            line = math.sin(x * freq * 0.05 + y * 0.1) * 0.5 + 0.5
            gate = darkness * (wave_thr / 15.0)
            out[y, x] = 0.0 if line < gate else 255.0
    return out


@njit(cache=True, parallel=True)
def _vortex(img, thr):
    h, w = img.shape
    cx = w / 2.0
    cy = h / 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            ang = math.atan2(dy, dx)
            r = math.sqrt(dx * dx + dy * dy)
            t = (math.sin(ang * 6.0 + r * 0.15) * 0.5 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _concentric(img, thr):
    h, w = img.shape
    cx = w / 2.0
    cy = h / 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            r = math.sqrt(dx * dx + dy * dy)
            t = (math.sin(r * 0.3) * 0.5 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _wireframe_alt(img, sensitivity):
    h, w = img.shape
    out = np.empty_like(img)
    thr_edge = 160.0 / sensitivity
    for y in prange(h):
        for x in range(w):
            xl = x - 1 if x > 0 else x
            xr = x + 1 if x < w - 1 else x
            yu = y - 1 if y > 0 else y
            yd = y + 1 if y < h - 1 else y
            gx = img[y, xr] - img[y, xl]
            gy = img[int(yd), x] - img[int(yu), x]
            gd = img[int(yd), xr] - img[int(yu), xl]
            mag = math.sqrt(gx * gx + gy * gy + gd * gd)
            out[y, x] = 0.0 if mag > thr_edge else 255.0
    return out


@njit(cache=True, parallel=True)
def _crosshatch_alt(img, s):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            hit = False
            if darkness > 0.15 and (x % s == 0):
                hit = True
            if darkness > 0.40 and (y % s == 0):
                hit = True
            if darkness > 0.65 and ((x + y) % s == 0):
                hit = True
            if darkness > 0.85 and ((x - y) % s == 0):
                hit = True
            out[y, x] = 0.0 if hit else 255.0
    return out


# ── Kernel: Radial Burst · Special Effects · dims=2 · no sliders ──
@registry.register("Radial Burst", "Special Effects", dims=2)
def radial_burst(image_array, parameter, luminance_threshold_value):
    return _radial_burst(image_array.astype(np.float32), luminance_threshold_value)


# ── Kernel: Wave · Special Effects · dims=2 · no sliders ──
@registry.register("Wave", "Special Effects", dims=2)
def wave(image_array, parameter, luminance_threshold_value):
    return _wave(image_array.astype(np.float32), luminance_threshold_value)


# ── Kernel: Noise · Special Effects · dims=2 · no sliders ──
@registry.register("Noise", "Special Effects", dims=2)
def noise(image_array, parameter, luminance_threshold_value):
    return _noise(image_array.astype(np.float32), luminance_threshold_value)


# ── Kernel: Topography · Special Effects · dims=2 · Warp Intensity 1-20-1 ──
@registry.register("Topography", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def topography(image_array, parameter, luminance_threshold_value):
    warp = float(parameter) if parameter else 1.0
    return _topography(image_array.astype(np.float32), warp)


# ── Kernel: Thresholder · Special Effects · dims=2 · Modulation Frequency 1-20-1 ──
@registry.register("Thresholder", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def thresholder(image_array, parameter, luminance_threshold_value):
    freq = float(parameter) if parameter else 1.0
    if freq < 1.0:
        freq = 1.0
    return _thresholder(image_array.astype(np.float32), freq)


# ── Kernel: Diagonal · Special Effects · dims=2 · Edge Sensitivity 1-20-1 ──
@registry.register("Diagonal", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def diagonal(image_array, parameter, luminance_threshold_value):
    s = float(parameter) if parameter else 1.0
    if s < 1.0:
        s = 1.0
    return _diagonal(image_array.astype(np.float32), s)


# ── Kernel: Displace Contour · Special Effects · dims=2 ──
#    sliders (Contour Threshold 0-100-50, Line Mode 1-3-1, Smoothing 0-5-0, Line Spacing 1-5-1)
@registry.register("Displace Contour", "Special Effects", dims=2,
                   param_sliders=("contour_thresh_slider", "line_mode_slider",
                                  "smoothing_slider", "line_space_slider"))
def displace_contour(image_array, parameter, luminance_threshold_value):
    ct, lm, sm, ls = _unpack4(parameter, 50, 1, 0, 1)
    return _displace_contour(image_array.astype(np.float32),
                             float(ct), int(lm), int(sm), int(ls))


# ── Kernel: Sine Wave Modulation · Special Effects · dims=2 ──
#    sliders (Wave Frequency 1-20-5, Wave Threshold 1-30-10)
@registry.register("Sine Wave Modulation", "Special Effects", dims=2,
                   param_sliders=("wave_frequency_slider", "wave_threshold_slider"))
def sine_wave_modulation(image_array, parameter, luminance_threshold_value):
    freq, wthr = _unpack2s(parameter, 5, 10)
    return _sine_wave_modulation(image_array.astype(np.float32),
                                 float(freq), float(wthr))


# ── Kernel: Vortex · Special Effects · dims=2 · no sliders (extra) ──
@registry.register("Vortex", "Special Effects", dims=2)
def vortex(image_array, parameter, luminance_threshold_value):
    return _vortex(image_array.astype(np.float32), luminance_threshold_value)


# ── Kernel: Concentric Rings · Special Effects · dims=2 · no sliders (extra) ──
@registry.register("Concentric Rings", "Special Effects", dims=2)
def concentric_rings(image_array, parameter, luminance_threshold_value):
    return _concentric(image_array.astype(np.float32), luminance_threshold_value)


# ── Kernel: Wireframe Alt · Special Effects · dims=2 · Edge Sensitivity 1-20-1 (extra) ──
@registry.register("Wireframe Alt", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def wireframe_alt(image_array, parameter, luminance_threshold_value):
    s = float(parameter) if parameter else 1.0
    if s < 1.0:
        s = 1.0
    return _wireframe_alt(image_array.astype(np.float32), s)


# ── Kernel: Crosshatch Alt · Special Effects · dims=2 · Line Spacing 1-20-1 (extra) ──
@registry.register("Crosshatch Alt", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def crosshatch_alt(image_array, parameter, luminance_threshold_value):
    s = int(parameter) if parameter else 4
    if s < 1:
        s = 1
    return _crosshatch_alt(image_array.astype(np.float32), s)


# ── Tuple-unpack helpers (plain Python) ──
def _unpack4(parameter, d0, d1, d2, d3):
    if isinstance(parameter, (tuple, list)):
        vals = list(parameter) + [d0, d1, d2, d3]
        return vals[0], vals[1], vals[2], vals[3]
    return parameter, d1, d2, d3


def _unpack2s(parameter, d0, d1):
    if isinstance(parameter, (tuple, list)):
        a = parameter[0] if len(parameter) > 0 else d0
        b = parameter[1] if len(parameter) > 1 else d1
        return a, b
    return parameter, d1
