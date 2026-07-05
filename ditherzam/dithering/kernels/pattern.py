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
            base = thr if cell == 0 else (255.0 - thr)
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
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
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
            hit = False
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


# ── Kernel: Checkers - Small · Patterned · dims=2 · no sliders (board 2) ──
@registry.register("Checkers - Small", "Patterned", dims=2)
def checkers_small(image_array, parameter, luminance_threshold_value):
    return _checkers(image_array.astype(np.float32), 2, luminance_threshold_value)


# ── Kernel: Checkers - Medium · Patterned · dims=2 · no sliders (board 4) ──
@registry.register("Checkers - Medium", "Patterned", dims=2)
def checkers_medium(image_array, parameter, luminance_threshold_value):
    return _checkers(image_array.astype(np.float32), 4, luminance_threshold_value)


# ── Kernel: Checkers - Large · Patterned · dims=2 · no sliders (board 8) ──
@registry.register("Checkers - Large", "Patterned", dims=2)
def checkers_large(image_array, parameter, luminance_threshold_value):
    return _checkers(image_array.astype(np.float32), 8, luminance_threshold_value)


# ── Kernel: Diamond · Patterned · dims=2 · no sliders ──
@registry.register("Diamond", "Patterned", dims=2)
def diamond(image_array, parameter, luminance_threshold_value):
    return _diamond(image_array.astype(np.float32), 8)


# ── Kernel: Gridlock/Traffic · Patterned · dims=2 · no sliders ──
@registry.register("Gridlock/Traffic", "Patterned", dims=2)
def gridlock_traffic(image_array, parameter, luminance_threshold_value):
    return _gridlock(image_array.astype(np.float32), 6, luminance_threshold_value)


# ── Kernel: Print Pattern · Patterned · dims=2 · no sliders (CMYK halftone) ──
@registry.register("Print Pattern", "Patterned", dims=2)
def print_pattern(image_array, parameter, luminance_threshold_value):
    return _print_pattern(image_array.astype(np.float32), 6)


# ── Kernel: Block Tone · Patterned · dims=2 · Dot Size 4-30-4 ──
@registry.register("Block Tone", "Patterned", dims=2,
                   param_sliders=("dither_parameter_slider",))
def block_tone(image_array, parameter, luminance_threshold_value):
    dot = int(parameter) if parameter else 4
    return _block_tone(image_array.astype(np.float32), dot)


# ── Kernel: Stippling · Patterned · dims=2 · Dot Density 1-20-1 ──
@registry.register("Stippling", "Patterned", dims=2,
                   param_sliders=("dither_parameter_slider",))
def stippling(image_array, parameter, luminance_threshold_value):
    density = int(parameter) if parameter else 1
    return _stippling(image_array.astype(np.float32), density)


# ── Kernel: Crosshatch · Patterned · dims=2 · Line Spacing 1-20-1 ──
@registry.register("Crosshatch", "Patterned", dims=2,
                   param_sliders=("dither_parameter_slider",))
def crosshatch(image_array, parameter, luminance_threshold_value):
    s = int(parameter) if parameter else 4
    if s < 1:
        s = 1
    return _crosshatch(image_array.astype(np.float32), s)


# ── Kernel: Dot Screen · Patterned · dims=2 · Cell Size 2-20-6 (extra) ──
@registry.register("Dot Screen", "Patterned", dims=2,
                   param_sliders=("dither_parameter_slider",))
def dot_screen(image_array, parameter, luminance_threshold_value):
    cell = int(parameter) if parameter else 6
    return _dot_screen_p(image_array.astype(np.float32), cell)


# ── Kernel: Line Screen · Patterned · dims=2 · Line Period 2-20-6 (extra) ──
@registry.register("Line Screen", "Patterned", dims=2,
                   param_sliders=("dither_parameter_slider",))
def line_screen(image_array, parameter, luminance_threshold_value):
    period = int(parameter) if parameter else 6
    return _line_screen(image_array.astype(np.float32), period)
