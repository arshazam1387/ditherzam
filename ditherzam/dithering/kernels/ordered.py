from __future__ import annotations
import numpy as np
from numba import njit, prange
from ditherzam.dithering import registry
from ditherzam.dithering.nlevels import quantize_to_levels


def _bayer_matrix(n: int) -> np.ndarray:
    """Recursive Bayer index matrix of order n (n a power of two)."""
    if n == 1:
        return np.zeros((1, 1), dtype=np.float32)
    s = _bayer_matrix(n // 2)
    return np.block([
        [4 * s + 0, 4 * s + 2],
        [4 * s + 3, 4 * s + 1],
    ]).astype(np.float32)


def _bayer_thresholds(n: int) -> np.ndarray:
    """Bayer matrix normalized to 0..255 threshold values."""
    return ((_bayer_matrix(n) + 0.5) / float(n * n) * 255.0).astype(np.float32)


_BAYER2 = _bayer_thresholds(2)
_BAYER4 = _bayer_thresholds(4)
_BAYER8 = _bayer_thresholds(8)
_BAYER16 = _bayer_thresholds(16)

# Classic 4x4 clustered-dot (spiral) screen, normalized to 0..255 thresholds.
_CLUSTER4_IDX = np.array([[12, 5, 6, 13],
                          [4, 0, 1, 7],
                          [11, 3, 2, 8],
                          [15, 10, 9, 14]], dtype=np.float32)
_CLUSTER4 = ((_CLUSTER4_IDX + 0.5) / 16.0 * 255.0).astype(np.float32)


@njit(cache=True, parallel=True)
def _ordered(img, thresholds, levels=2):
    h, w = img.shape
    mh, mw = thresholds.shape
    out = np.empty_like(img)
    if levels <= 2:
        for y in prange(h):
            for x in range(w):
                out[y, x] = 255.0 if img[y, x] >= thresholds[y % mh, x % mw] else 0.0
        return out
    step = 255.0 / (levels - 1)
    for y in prange(h):
        for x in range(w):
            # thresholds are 0..255; recenter to [-0.5,0.5]*step as a sub-step offset
            off = (thresholds[y % mh, x % mw] / 255.0 - 0.5) * step
            out[y, x] = quantize_to_levels(img[y, x] + off, levels)
    return out


@njit(cache=True)
def _random_ordered(img):
    np.random.seed(0)
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            t = np.random.random() * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _bit_tone(img, dot, base):
    h, w = img.shape
    mh, mw = base.shape
    d = dot if dot >= 1 else 1
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            t = base[(y // d) % mh, (x // d) % mw]
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _mosaic(img, block, thr):
    h, w = img.shape
    b = block if block >= 1 else 1
    nby = (h + b - 1) // b
    nbx = (w + b - 1) // b
    out = np.empty_like(img)
    for by in prange(nby):
        for bx in range(nbx):
            y0 = by * b
            x0 = bx * b
            s = 0.0
            c = 0
            for yy in range(y0, min(y0 + b, h)):
                for xx in range(x0, min(x0 + b, w)):
                    s += img[yy, xx]
                    c += 1
            v = 255.0 if (s / c) >= thr else 0.0
            for yy in range(y0, min(y0 + b, h)):
                for xx in range(x0, min(x0 + b, w)):
                    out[yy, xx] = v
    return out


@njit(cache=True)
def _bayer_void(img, warp, thr, base):
    h, w = img.shape
    mh, mw = base.shape
    strength = warp / 50.0
    out = np.empty_like(img)
    for y in range(h):
        fy = y / (h - 1) if h > 1 else 0.0
        shift = int((fy * fy) * warp)
        for x in range(w):
            sx = (x + shift) % w
            b = base[y % mh, sx % mw]
            final = thr + (b - 128.0) * strength
            out[y, x] = 255.0 if img[y, sx] >= final else 0.0
    return out


@njit(cache=True, parallel=True)
def _modulated_bayer(img, thresholds):
    h, w = img.shape
    mh, mw = thresholds.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            local = img[y, x] / 255.0
            t = thresholds[y % mh, x % mw] * (0.5 + local)
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _dot_screen(img, cell):
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


# ── Kernel: Bayer-Matrix 2x2 · Ordered Dither · dims=2 · no sliders ──
@registry.register("Bayer-Matrix 2x2", "Ordered Dither", dims=2)
def bayer_2(image_array, parameter, luminance_threshold_value, levels=2):
    return _ordered(image_array.astype(np.float32), _BAYER2, levels)


# ── Kernel: Bayer-Matrix 8x8 · Ordered Dither · dims=2 · no sliders ──
@registry.register("Bayer-Matrix 8x8", "Ordered Dither", dims=2)
def bayer_8(image_array, parameter, luminance_threshold_value, levels=2):
    return _ordered(image_array.astype(np.float32), _BAYER8, levels)


# ── Kernel: Bayer-Matrix 16x16 · Ordered Dither · dims=2 · no sliders ──
@registry.register("Bayer-Matrix 16x16", "Ordered Dither", dims=2)
def bayer_16(image_array, parameter, luminance_threshold_value, levels=2):
    return _ordered(image_array.astype(np.float32), _BAYER16, levels)


# ── Kernel: Bayer-Ordered · Ordered Dither · dims=2 · alias of 4x4 ──
@registry.register("Bayer-Ordered", "Ordered Dither", dims=2)
def bayer_ordered(image_array, parameter, luminance_threshold_value, levels=2):
    return _ordered(image_array.astype(np.float32), _BAYER4, levels)


# ── Kernel: Bayer-Void · Ordered Dither · dims=2 · Warp Intensity 1-50-10 ──
@registry.register("Bayer-Void", "Ordered Dither", dims=2,
                   param_sliders=("dither_parameter_slider",))
def bayer_void(image_array, parameter, luminance_threshold_value):
    warp = float(parameter) if parameter else 10.0
    return _bayer_void(image_array.astype(np.float32), warp,
                       luminance_threshold_value, _BAYER4)


# ── Kernel: Random Ordered · Ordered Dither · dims=2 · no sliders ──
@registry.register("Random Ordered", "Ordered Dither", dims=2)
def random_ordered(image_array, parameter, luminance_threshold_value):
    return _random_ordered(image_array.astype(np.float32))


# ── Kernel: Bit Tone · Ordered Dither · dims=2 · Dot Size 1-20-1 ──
@registry.register("Bit Tone", "Ordered Dither", dims=2,
                   param_sliders=("dither_parameter_slider",))
def bit_tone(image_array, parameter, luminance_threshold_value):
    dot = int(parameter) if parameter else 1
    return _bit_tone(image_array.astype(np.float32), dot, _BAYER4)


# ── Kernel: Mosaic · Ordered Dither · dims=2 · Block Size 1-50-10 ──
@registry.register("Mosaic", "Ordered Dither", dims=2,
                   param_sliders=("dither_parameter_slider",))
def mosaic(image_array, parameter, luminance_threshold_value):
    block = int(parameter) if parameter else 10
    return _mosaic(image_array.astype(np.float32), block,
                   luminance_threshold_value)


# ── Kernel: Modulated Bayer Dither · Ordered Dither · dims=2 · Matrix Size 2-3-2 ──
@registry.register("Modulated Bayer Dither", "Ordered Dither", dims=2,
                   param_sliders=("matrix_size_slider",))
def modulated_bayer(image_array, parameter, luminance_threshold_value):
    size = int(parameter) if parameter else 2
    thr = _BAYER8 if size >= 3 else _BAYER4
    return _modulated_bayer(image_array.astype(np.float32), thr)


# ── Kernel: Cluster-Dot · Ordered Dither · dims=2 · no sliders (extra) ──
@registry.register("Cluster-Dot", "Ordered Dither", dims=2)
def cluster_dot(image_array, parameter, luminance_threshold_value, levels=2):
    return _ordered(image_array.astype(np.float32), _CLUSTER4, levels)


# ── Kernel: Halftone-Ordered · Ordered Dither · dims=2 · Cell Size 2-20-6 (extra) ──
@registry.register("Halftone-Ordered", "Ordered Dither", dims=2,
                   param_sliders=("dither_parameter_slider",))
def halftone_ordered(image_array, parameter, luminance_threshold_value):
    cell = int(parameter) if parameter else 6
    return _dot_screen(image_array.astype(np.float32), cell)
