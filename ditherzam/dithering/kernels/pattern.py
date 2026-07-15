from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from ditherzam.dithering import registry


@njit(cache=True, parallel=True)
def _checkers(img, s, thr):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            cell = ((x // s) + (y // s)) % 2
            # Alternate around the chosen tonal threshold. The old
            # ``thr``/``255-thr`` pair collapses to the same 127.5 value at the
            # default, making the checkerboard disappear entirely.
            base = thr - 64.0 if cell == 0 else thr + 64.0
            base = min(255.0, max(0.0, base))
            out[y, x] = 255.0 if img[y, x] >= base else 0.0
    return out


@njit(cache=True, parallel=True)
def _diamond(img, s):
    h, w = img.shape
    half = s / 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            dx = abs((x % s) - half)
            dy = abs((y % s) - half)
            t = (dx + dy) / s * 255.0
            # A zero-threshold centre must not leak into pure black, while a
            # 255-threshold corner must still clear for pure white.
            out[y, x] = 255.0 if img[y, x] > 0.0 and img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _gridlock(img, g, thr):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            on_line = (x % g == 0) or (y % g == 0)
            t = thr * 0.5 if on_line else thr
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _print_pattern(img, cell):
    # CMYK-style rotated dot screen (single-channel halftone simulation).
    h, w = img.shape
    ca = math.cos(0.261799)  # 15 degrees
    sa = math.sin(0.261799)
    half = cell / 2.0
    r2max = half * half * 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            xr = x * ca - y * sa
            yr = x * sa + y * ca
            dx = (xr % cell) - half
            dy = (yr % cell) - half
            t = (dx * dx + dy * dy) / r2max * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _block_tone(img, dot):
    # Classic round-dot halftone; dot radius grows with local darkness.
    h, w = img.shape
    c = dot if dot >= 2 else 2
    half = c / 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            r = darkness * half
            dx = (x % c) - half + 0.5
            dy = (y % c) - half + 0.5
            out[y, x] = 0.0 if (dx * dx + dy * dy) <= r * r else 255.0
    return out


@njit(cache=True)
def _stippling(img, density):
    np.random.seed(0)
    h, w = img.shape
    scale = density if density >= 1 else 1
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            p = darkness / (1.0 + (scale - 1) * 0.15)
            out[y, x] = 0.0 if np.random.random() < p else 255.0
    return out


@njit(cache=True, parallel=True)
def _crosshatch(img, s):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            # At the black endpoint ink must fill the cell; otherwise the four
            # hatch directions can cover only a sparse subset of pixels.
            hit = darkness >= 0.999
            if darkness > 0.20 and ((x + y) % s == 0):
                hit = True
            if darkness > 0.45 and ((x - y) % s == 0):
                hit = True
            if darkness > 0.70 and (x % s == 0):
                hit = True
            if darkness > 0.88 and (y % s == 0):
                hit = True
            out[y, x] = 0.0 if hit else 255.0
    return out


@njit(cache=True, parallel=True)
def _dot_screen_p(img, cell):
    h, w = img.shape
    c = cell if cell >= 2 else 2
    half = c / 2.0
    r2max = half * half * 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            dx = (x % c) - half + 0.5
            dy = (y % c) - half + 0.5
            t = (dx * dx + dy * dy) / r2max * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _line_screen(img, period):
    h, w = img.shape
    p = period if period >= 2 else 2
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            thick = darkness * p
            out[y, x] = 0.0 if (y % p) < thick else 255.0
    return out


def _unpack5(parameter, defaults):
    if isinstance(parameter, (tuple, list, np.ndarray)):
        return tuple(parameter[i] if i < len(parameter) else defaults[i] for i in range(5))
    if parameter is None:
        return defaults
    return (parameter,) + defaults[1:]


_PATTERN_SLIDERS = ("pattern_scale_slider", "pattern_contrast_slider",
                    "pattern_bias_slider", "pattern_skew_slider",
                    "pattern_phase_y_slider")


def _pattern_input(image_array, contrast, bias, phase_x, phase_y):
    img = image_array.astype(np.float32)
    if float(contrast) != 100.0 or float(bias) != 0.0:
        img = np.clip(128.0 + (img - 128.0) * float(contrast) / 100.0 +
                      float(bias), 0.0, 255.0).astype(np.float32)
    if int(phase_x) or int(phase_y):
        img = np.roll(img, (int(phase_y), int(phase_x)), axis=(0, 1))
    return img


def _pattern_result(image_array, parameter, default_scale, kernel, threshold=None):
    scale, contrast, bias, skew, py = _unpack5(
        parameter, (default_scale, 100, 0, 0, 0))
    img = _pattern_input(image_array, contrast, bias, 0, py)
    if threshold is None:
        out = kernel(img, max(1, int(scale)))
    else:
        out = kernel(img, max(1, int(scale)), threshold)
    if int(py):
        out = np.roll(out, -int(py), axis=0)
    if int(skew):
        sheared = np.empty_like(out)
        for y in range(out.shape[0]):
            sheared[y] = np.roll(out[y], y * int(skew))
        out = sheared
    return out


# ── Kernel: Checkers - Small · Patterned · dims=2 · no sliders (board 2) ──
@registry.register("Checkers - Small", "Patterned", dims=2, param_sliders=_PATTERN_SLIDERS)
def checkers_small(image_array, parameter, luminance_threshold_value):
    return _pattern_result(image_array, parameter, 2, _checkers, luminance_threshold_value)


# ── Kernel: Checkers - Medium · Patterned · dims=2 · no sliders (board 4) ──
@registry.register("Checkers - Medium", "Patterned", dims=2, param_sliders=_PATTERN_SLIDERS)
def checkers_medium(image_array, parameter, luminance_threshold_value):
    return _pattern_result(image_array, parameter, 4, _checkers, luminance_threshold_value)


# ── Kernel: Checkers - Large · Patterned · dims=2 · no sliders (board 8) ──
@registry.register("Checkers - Large", "Patterned", dims=2, param_sliders=_PATTERN_SLIDERS)
def checkers_large(image_array, parameter, luminance_threshold_value):
    return _pattern_result(image_array, parameter, 8, _checkers, luminance_threshold_value)


# ── Kernel: Diamond · Patterned · dims=2 · no sliders ──
@registry.register("Diamond", "Patterned", dims=2, param_sliders=_PATTERN_SLIDERS)
def diamond(image_array, parameter, luminance_threshold_value):
    return _pattern_result(image_array, parameter, 8, _diamond)


# ── Kernel: Gridlock/Traffic · Patterned · dims=2 · no sliders ──
@registry.register("Gridlock/Traffic", "Patterned", dims=2, param_sliders=_PATTERN_SLIDERS)
def gridlock_traffic(image_array, parameter, luminance_threshold_value):
    return _pattern_result(image_array, parameter, 6, _gridlock, luminance_threshold_value)


# ── Kernel: Print Pattern · Patterned · dims=2 · no sliders (CMYK halftone) ──
@registry.register("Print Pattern", "Patterned", dims=2, param_sliders=_PATTERN_SLIDERS)
def print_pattern(image_array, parameter, luminance_threshold_value):
    return _pattern_result(image_array, parameter, 6, _print_pattern)


# ── Kernel: Block Tone · Patterned · dims=2 · Dot Size 4-30-4 ──
@registry.register("Block Tone", "Patterned", dims=2,
                   param_sliders=_PATTERN_SLIDERS)
def block_tone(image_array, parameter, luminance_threshold_value):
    return _pattern_result(image_array, parameter, 4, _block_tone)


# ── Kernel: Stippling · Patterned · dims=2 · Dot Density 1-20-1 ──
@registry.register("Stippling", "Patterned", dims=2,
                   param_sliders=_PATTERN_SLIDERS)
def stippling(image_array, parameter, luminance_threshold_value):
    return _pattern_result(image_array, parameter, 1, _stippling)


# ── Kernel: Crosshatch · Patterned · dims=2 · Line Spacing 1-20-1 ──
@registry.register("Crosshatch", "Patterned", dims=2,
                   param_sliders=_PATTERN_SLIDERS)
def crosshatch(image_array, parameter, luminance_threshold_value):
    return _pattern_result(image_array, parameter, 4, _crosshatch)


# ── Kernel: Dot Screen · Patterned · dims=2 · Cell Size 2-20-6 (extra) ──
@registry.register("Dot Screen", "Patterned", dims=2,
                   param_sliders=_PATTERN_SLIDERS)
def dot_screen(image_array, parameter, luminance_threshold_value):
    return _pattern_result(image_array, parameter, 6, _dot_screen_p)


# ── Kernel: Line Screen · Patterned · dims=2 · Line Period 2-20-6 (extra) ──
@registry.register("Line Screen", "Patterned", dims=2,
                   param_sliders=_PATTERN_SLIDERS)
def line_screen(image_array, parameter, luminance_threshold_value):
    return _pattern_result(image_array, parameter, 6, _line_screen)
