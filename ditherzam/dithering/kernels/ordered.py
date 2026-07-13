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
def _ordered(img, thresholds, levels=2, contrast=100.0, bias=0.0,
             rotation=0, offset_x=0, offset_y=0):
    h, w = img.shape
    mh, mw = thresholds.shape
    out = np.empty_like(img)
    if levels <= 2:
        for y in prange(h):
            for x in range(w):
                yy = (y + offset_y) % mh
                xx = (x + offset_x) % mw
                r = rotation % 4
                if r == 1:
                    yy, xx = xx % mh, (mw - 1 - yy) % mw
                elif r == 2:
                    yy, xx = (mh - 1 - yy) % mh, (mw - 1 - xx) % mw
                elif r == 3:
                    yy, xx = (mh - 1 - xx) % mh, yy % mw
                t = 128.0 + (thresholds[yy, xx] - 128.0) * contrast / 100.0 + bias
                out[y, x] = 255.0 if img[y, x] >= t else 0.0
        return out
    step = 255.0 / (levels - 1)
    for y in prange(h):
        for x in range(w):
            # thresholds are 0..255; recenter to [-0.5,0.5]*step as a sub-step offset
            yy = (y + offset_y) % mh
            xx = (x + offset_x) % mw
            r = rotation % 4
            if r == 1:
                yy, xx = xx % mh, (mw - 1 - yy) % mw
            elif r == 2:
                yy, xx = (mh - 1 - yy) % mh, (mw - 1 - xx) % mw
            elif r == 3:
                yy, xx = (mh - 1 - xx) % mh, yy % mw
            t = 128.0 + (thresholds[yy, xx] - 128.0) * contrast / 100.0 + bias
            off = (t / 255.0 - 0.5) * step
            out[y, x] = quantize_to_levels(img[y, x] + off, levels)
    return out


@njit(cache=True)
def _random_ordered(img, seed=0, contrast=100.0, bias=0.0, grain_x=1, grain_y=1):
    np.random.seed(seed)
    h, w = img.shape
    out = np.empty_like(img)
    if grain_x == 1 and grain_y == 1:
        for y in range(h):
            for x in range(w):
                t = np.random.random() * 255.0
                t = 128.0 + (t - 128.0) * contrast / 100.0 + bias
                out[y, x] = 255.0 if img[y, x] >= t else 0.0
    else:
        for y in range(h):
            for x in range(w):
                gy = y // grain_y
                gx = x // grain_x
                # Coordinate hash makes each rectangular grain deterministic.
                z = (gx * 374761393 + gy * 668265263 + seed * 69069) & 0x7fffffff
                t = (z % 104729) / 104728.0 * 255.0
                t = 128.0 + (t - 128.0) * contrast / 100.0 + bias
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
            # Keep the locally modulated threshold in the valid luminance
            # domain.  Without the upper clamp, bright Bayer cells can exceed
            # 255, leaving black holes even in a pure-white image.
            t = thresholds[y % mh, x % mw] * (0.5 + local)
            t = min(255.0, max(0.0, t))
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


@njit(cache=True, parallel=True)
def _dot_screen_45(img, cell):
    """A 45-degree round-dot screen, distinct from the axis-aligned screen."""
    h, w = img.shape
    c = cell if cell >= 2 else 2
    half = c / 2.0
    inv_sqrt2 = 0.7071067811865476
    r2max = half * half * 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            xr = (x - y) * inv_sqrt2
            yr = (x + y) * inv_sqrt2
            dx = (xr % c) - half
            dy = (yr % c) - half
            t = (dx * dx + dy * dy) / r2max * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


def _unpack(parameter, defaults):
    if isinstance(parameter, (tuple, list, np.ndarray)):
        return tuple(parameter[i] if i < len(parameter) else defaults[i]
                     for i in range(len(defaults)))
    if parameter is None:
        return defaults
    return (parameter,) + defaults[1:]


_ORDERED_SLIDERS = ("threshold_contrast_slider", "threshold_bias_slider",
                    "matrix_rotation_slider", "matrix_offset_x_slider",
                    "matrix_offset_y_slider")


def _ordered_entry(image_array, parameter, base, levels):
    contrast, bias, rotation, ox, oy = _unpack(parameter, (100, 0, 0, 0, 0))
    return _ordered(image_array.astype(np.float32), base, levels, float(contrast),
                    float(bias), int(rotation), int(ox), int(oy))


_PRIMARY_ORDERED_SLIDERS = ("dither_parameter_slider", "threshold_contrast_slider",
                            "threshold_bias_slider", "matrix_offset_x_slider",
                            "matrix_offset_y_slider")
_MATRIX_ORDERED_SLIDERS = ("matrix_size_slider", "threshold_contrast_slider",
                           "threshold_bias_slider", "matrix_offset_x_slider",
                           "matrix_offset_y_slider")
_MOSAIC_SLIDERS = ("dither_parameter_slider", "tile_threshold_scale_slider",
                   "threshold_bias_slider", "matrix_offset_x_slider",
                   "matrix_offset_y_slider")


def _input_controls(image_array, parameter, default_primary):
    primary, contrast, bias, ox, oy = _unpack(
        parameter, (default_primary, 100, 0, 0, 0))
    img = image_array.astype(np.float32)
    if float(contrast) != 100.0 or float(bias) != 0.0:
        img = np.clip(128.0 + (img - 128.0) * float(contrast) / 100.0 +
                      float(bias), 0.0, 255.0).astype(np.float32)
    if int(ox) or int(oy):
        img = np.roll(img, (int(oy), int(ox)), axis=(0, 1))
    return img, primary, int(ox), int(oy)


def _unphase(out, ox, oy):
    return np.roll(out, (-oy, -ox), axis=(0, 1)) if ox or oy else out


# ── Kernel: Bayer-Matrix 2x2 · Ordered Dither · dims=2 · no sliders ──
@registry.register("Bayer-Matrix 2x2", "Ordered Dither", dims=2, supports_levels=True,
                   param_sliders=_ORDERED_SLIDERS)
def bayer_2(image_array, parameter, luminance_threshold_value, levels=2):
    return _ordered_entry(image_array, parameter, _BAYER2, levels)


# ── Kernel: Bayer-Matrix 8x8 · Ordered Dither · dims=2 · no sliders ──
@registry.register("Bayer-Matrix 8x8", "Ordered Dither", dims=2, supports_levels=True,
                   param_sliders=_ORDERED_SLIDERS)
def bayer_8(image_array, parameter, luminance_threshold_value, levels=2):
    return _ordered_entry(image_array, parameter, _BAYER8, levels)


# ── Kernel: Bayer-Matrix 16x16 · Ordered Dither · dims=2 · no sliders ──
@registry.register("Bayer-Matrix 16x16", "Ordered Dither", dims=2, supports_levels=True,
                   param_sliders=_ORDERED_SLIDERS)
def bayer_16(image_array, parameter, luminance_threshold_value, levels=2):
    return _ordered_entry(image_array, parameter, _BAYER16, levels)


# ── Kernel: Bayer-Ordered · Ordered Dither · dims=2 · alias of 4x4 ──
@registry.register("Bayer-Ordered", "Ordered Dither", dims=2, supports_levels=True,
                   param_sliders=_ORDERED_SLIDERS)
def bayer_ordered(image_array, parameter, luminance_threshold_value, levels=2):
    return _ordered_entry(image_array, parameter, _BAYER4, levels)


# ── Kernel: Bayer-Void · Ordered Dither · dims=2 · Warp Intensity 1-50-10 ──
@registry.register("Bayer-Void", "Ordered Dither", dims=2,
                   param_sliders=_PRIMARY_ORDERED_SLIDERS)
def bayer_void(image_array, parameter, luminance_threshold_value):
    img, warp, ox, oy = _input_controls(image_array, parameter, 10)
    return _unphase(_bayer_void(img, float(warp), luminance_threshold_value, _BAYER4), ox, oy)


# ── Kernel: Random Ordered · Ordered Dither · dims=2 · no sliders ──
@registry.register("Random Ordered", "Ordered Dither", dims=2,
                   param_sliders=("random_seed_slider", "threshold_contrast_slider",
                                  "threshold_bias_slider", "grain_width_slider",
                                  "grain_height_slider"))
def random_ordered(image_array, parameter, luminance_threshold_value):
    seed, contrast, bias, gx, gy = _unpack(parameter, (0, 100, 0, 1, 1))
    return _random_ordered(image_array.astype(np.float32), int(seed), float(contrast),
                           float(bias), max(1, int(gx)), max(1, int(gy)))


# ── Kernel: Bit Tone · Ordered Dither · dims=2 · Dot Size 1-20-1 ──
@registry.register("Bit Tone", "Ordered Dither", dims=2,
                   param_sliders=_PRIMARY_ORDERED_SLIDERS)
def bit_tone(image_array, parameter, luminance_threshold_value):
    img, dot, ox, oy = _input_controls(image_array, parameter, 1)
    return _unphase(_bit_tone(img, int(dot), _BAYER4), ox, oy)


# ── Kernel: Mosaic · Ordered Dither · dims=2 · Block Size 1-50-10 ──
@registry.register("Mosaic", "Ordered Dither", dims=2,
                   param_sliders=_MOSAIC_SLIDERS)
def mosaic(image_array, parameter, luminance_threshold_value):
    block, threshold_scale, bias, ox, oy = _unpack(parameter, (10, 100, 0, 0, 0))
    img = image_array.astype(np.float32)
    if int(ox) or int(oy):
        img = np.roll(img, (int(oy), int(ox)), axis=(0, 1))
    threshold = float(luminance_threshold_value) * float(threshold_scale) / 100.0 + float(bias)
    return _unphase(_mosaic(img, int(block), threshold), int(ox), int(oy))


# ── Kernel: Modulated Bayer Dither · Ordered Dither · dims=2 · Matrix Size 2-3-2 ──
@registry.register("Modulated Bayer Dither", "Ordered Dither", dims=2,
                   param_sliders=_MATRIX_ORDERED_SLIDERS)
def modulated_bayer(image_array, parameter, luminance_threshold_value):
    img, size, ox, oy = _input_controls(image_array, parameter, 2)
    thr = _BAYER8 if size >= 3 else _BAYER4
    return _unphase(_modulated_bayer(img, thr), ox, oy)


# ── Kernel: Cluster-Dot · Ordered Dither · dims=2 · no sliders (extra) ──
@registry.register("Cluster-Dot", "Ordered Dither", dims=2, supports_levels=True,
                   param_sliders=_ORDERED_SLIDERS)
def cluster_dot(image_array, parameter, luminance_threshold_value, levels=2):
    return _ordered_entry(image_array, parameter, _CLUSTER4, levels)


# ── Kernel: Halftone-Ordered · Ordered Dither · dims=2 · Cell Size 2-20-6 (extra) ──
@registry.register("Halftone-Ordered", "Ordered Dither", dims=2,
                   param_sliders=_PRIMARY_ORDERED_SLIDERS)
def halftone_ordered(image_array, parameter, luminance_threshold_value):
    img, cell, ox, oy = _input_controls(image_array, parameter, 6)
    return _unphase(_dot_screen_45(img, int(cell)), ox, oy)
