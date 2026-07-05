from __future__ import annotations
import math
import numpy as np
from numba import njit
from ditherzam.dithering import registry
from ditherzam.dithering.kernels.error_diffusion import _diffuse_row
from ditherzam.dithering.kernels.ordered import _BAYER4


@njit(cache=True)
def _line_diffuse(img, thr, line_scale, horizontal):
    """1-D error diffusion producing banded line patterns; density ~ brightness."""
    h, w = img.shape
    out = img.copy()
    s = line_scale if line_scale >= 1 else 1
    if horizontal:
        for y in range(h):
            carry = 0.0
            for x in range(w):
                old = out[y, x] + carry
                new = 255.0 if old >= thr else 0.0
                out[y, x] = new
                carry = (old - new) / s
    else:
        for x in range(w):
            carry = 0.0
            for y in range(h):
                old = out[y, x] + carry
                new = 255.0 if old >= thr else 0.0
                out[y, x] = new
                carry = (old - new) / s
    return out


@njit(cache=True)
def _uniform_modulation(img, thr, line_scale, smoothing, bleed, horizontal):
    """Row/column diffusion with EMA-smoothed vertical bleed."""
    h, w = img.shape
    out = img.copy()
    s = line_scale if line_scale >= 1 else 1
    if horizontal:
        prev_err = np.zeros(w, dtype=np.float32)
        for y in range(h):
            carry = 0.0
            for x in range(w):
                base = out[y, x] + carry + prev_err[x] * bleed
                new = 255.0 if base >= thr else 0.0
                out[y, x] = new
                err = (base - new)
                carry = err / s
                prev_err[x] = prev_err[x] * smoothing + err * (1.0 - smoothing)
    else:
        prev_err = np.zeros(h, dtype=np.float32)
        for x in range(w):
            carry = 0.0
            for y in range(h):
                base = out[y, x] + carry + prev_err[y] * bleed
                new = 255.0 if base >= thr else 0.0
                out[y, x] = new
                err = (base - new)
                carry = err / s
                prev_err[y] = prev_err[y] * smoothing + err * (1.0 - smoothing)
    return out


@njit(cache=True)
def _atkinson_vhs(img, thr, line_count):
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if img[y, x] >= thr else 0.0
    lc = line_count if line_count >= 1 else 1
    for k in range(lc):
        ry = int((k + 0.5) / lc * h)
        if ry >= h:
            ry = h - 1
        for x in range(w):
            out[ry, x] = 255.0  # bright horizontal tracking line
    return out


@njit(cache=True)
def _glitch(img, thr, intensity):
    np.random.seed(0)
    h, w = img.shape
    out = np.empty_like(img)
    amp = intensity if intensity >= 1 else 1
    for y in range(h):
        shift = int((np.random.random() - 0.5) * 2.0 * amp)
        for x in range(w):
            sx = (x + shift) % w
            out[y, x] = 255.0 if img[y, sx] >= thr else 0.0
    return out


@njit(cache=True)
def _waveform(img, thr, density):
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            freq = 0.05 + (1.0 - img[y, x] / 255.0) * density * 0.05
            v = (math.sin(x * freq + y * 0.3) * 0.5 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= v else 0.0
    return out


@njit(cache=True)
def _waveform_alt(img, thr, blend):
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            gx = 0.0
            if 0 < x < w - 1:
                gx = (img[y, x + 1] - img[y, x - 1]) / 255.0
            phase = x * (0.05 + (1.0 - img[y, x] / 255.0) * 0.1) + gx * blend
            v = (math.sin(phase) * 0.5 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= v else 0.0
    return out


@njit(cache=True)
def _ordered_modulation(img, thr, param, base):
    h, w = img.shape
    mh, mw = base.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            wob = math.sin((x + y) * 0.1 * param) * 40.0
            t = base[y % mh, x % mw] + wob
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True)
def _smooth_diffuse(img, thr, line_scale, smoothness):
    h, w = img.shape
    out = img.copy()
    s = line_scale if line_scale >= 1 else 1
    a = 1.0 / smoothness if smoothness >= 1 else 1.0
    for y in range(h):
        carry = 0.0
        for x in range(w):
            old = out[y, x] + carry
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            carry = ((old - new) / s) * a + carry * (1.0 - a)
    return out


@njit(cache=True)
def _stucki_diffusion_lines(img, thr, emphasis):
    """Stucki diffusion biased to horizontal carry to form line patterns."""
    h, w = img.shape
    out = img.copy()
    e = emphasis / 5.0
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                out[y, x + 1] += err * (0.5 + 0.4 * e)
            if x + 2 < w:
                out[y, x + 2] += err * (0.2 * e)
            if y + 1 < h:
                out[y + 1, x] += err * (0.3 - 0.15 * e)
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@njit(cache=True)
def _atkinson_line_modulation(img, thr, strength, hbias):
    h, w = img.shape
    out = img.copy()
    hb = hbias / 5.0
    st = strength / 5.0
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = (old - new) / 8.0 * st
            if x + 1 < w:
                out[y, x + 1] += err * (1.0 + hb)
            if x + 2 < w:
                out[y, x + 2] += err * hb
            if y + 1 < h:
                out[y + 1, x] += err * (1.0 - 0.5 * hb)
                if x + 1 < w:
                    out[y + 1, x + 1] += err
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@njit(cache=True)
def _contrast_aware(img, thr, line_scale, horizontal):
    """1-D diffusion whose threshold warps with local contrast."""
    h, w = img.shape
    out = img.copy()
    s = line_scale if line_scale >= 1 else 1
    if horizontal:
        for y in range(h):
            carry = 0.0
            for x in range(w):
                lo = img[y, x - 1] if x > 0 else img[y, x]
                hi = img[y, x + 1] if x < w - 1 else img[y, x]
                local = abs(hi - lo)
                t = thr + (local - 64.0) * 0.25
                old = out[y, x] + carry
                new = 255.0 if old >= t else 0.0
                out[y, x] = new
                carry = (old - new) / s
    else:
        for x in range(w):
            carry = 0.0
            for y in range(h):
                lo = img[y - 1, x] if y > 0 else img[y, x]
                hi = img[y + 1, x] if y < h - 1 else img[y, x]
                local = abs(hi - lo)
                t = thr + (local - 64.0) * 0.25
                old = out[y, x] + carry
                new = 255.0 if old >= t else 0.0
                out[y, x] = new
                carry = (old - new) / s
    return out


# ── Kernel: Artifact Modulation · Glitch · dims=2 · Dither Param 1-20-1 ──
@registry.register("Artifact Modulation", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def artifact_modulation(image_array, parameter, luminance_threshold_value):
    p = int(parameter) if parameter else 1
    return _waveform_alt(image_array.astype(np.float32),
                         luminance_threshold_value, float(p))


# ── Kernel: Atkinson-VHS · Glitch · dims=2 · Line Count 1-20-1 ──
@registry.register("Atkinson-VHS", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def atkinson_vhs(image_array, parameter, luminance_threshold_value):
    lc = int(parameter) if parameter else 1
    return _atkinson_vhs(image_array.astype(np.float32),
                         luminance_threshold_value, lc)


# ── Kernel: Glitch · Glitch · dims=2 · Glitch Intensity 1-20-1 ──
@registry.register("Glitch", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def glitch(image_array, parameter, luminance_threshold_value):
    intensity = int(parameter) if parameter else 1
    return _glitch(image_array.astype(np.float32),
                   luminance_threshold_value, intensity)


# ── Kernel: Modulated Diffuse Y · Glitch · dims=2 · Line Scale 1-20-1 ──
@registry.register("Modulated Diffuse Y", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def modulated_diffuse_y(image_array, parameter, luminance_threshold_value):
    ls = int(parameter) if parameter else 1
    return _line_diffuse(image_array.astype(np.float32),
                         luminance_threshold_value, ls, True)


# ── Kernel: Modulated Diffuse X · Glitch · dims=2 · Line Scale 1-20-1 ──
@registry.register("Modulated Diffuse X", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def modulated_diffuse_x(image_array, parameter, luminance_threshold_value):
    ls = int(parameter) if parameter else 1
    return _line_diffuse(image_array.astype(np.float32),
                         luminance_threshold_value, ls, False)


# ── Kernel: Uniform Modulation Y · Glitch · dims=2 ──
#    sliders (Line Scale 1-20-1, Smoothing Factor 0-1-0, Bleed Fraction 0-100-0)
@registry.register("Uniform Modulation Y", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",
                                  "smoothing_factor_slider",
                                  "bleed_fraction_slider"))
def uniform_modulation_y(image_array, parameter, luminance_threshold_value):
    ls, smooth, bleed = _unpack3(parameter)
    return _uniform_modulation(image_array.astype(np.float32),
                               luminance_threshold_value,
                               int(ls), float(smooth), float(bleed) / 100.0, True)


# ── Kernel: Uniform Modulation X · Glitch · dims=2 (same three sliders) ──
@registry.register("Uniform Modulation X", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",
                                  "smoothing_factor_slider",
                                  "bleed_fraction_slider"))
def uniform_modulation_x(image_array, parameter, luminance_threshold_value):
    ls, smooth, bleed = _unpack3(parameter)
    return _uniform_modulation(image_array.astype(np.float32),
                               luminance_threshold_value,
                               int(ls), float(smooth), float(bleed) / 100.0, False)


# ── Kernel: Waveform · Glitch · dims=2 · Wave Density 1-20-1 ──
@registry.register("Waveform", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def waveform(image_array, parameter, luminance_threshold_value):
    d = float(parameter) if parameter else 1.0
    return _waveform(image_array.astype(np.float32),
                     luminance_threshold_value, d)


# ── Kernel: Waveform Alt · Glitch · dims=2 · Modulation Blend 1-20-1 ──
@registry.register("Waveform Alt", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def waveform_alt(image_array, parameter, luminance_threshold_value):
    b = float(parameter) if parameter else 1.0
    return _waveform_alt(image_array.astype(np.float32),
                         luminance_threshold_value, b)


# ── Kernel: Ordered Modulation · Glitch · dims=2 · Dither Param 1-20-1 ──
@registry.register("Ordered Modulation", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def ordered_modulation(image_array, parameter, luminance_threshold_value):
    p = float(parameter) if parameter else 1.0
    return _ordered_modulation(image_array.astype(np.float32),
                               luminance_threshold_value, p, _BAYER4)


# ── Kernel: Smooth Diffuse · Glitch · dims=2 (Line Scale 1-20-1, Smoothness 1-10-5) ──
@registry.register("Smooth Diffuse", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "smoothness_slider"))
def smooth_diffuse(image_array, parameter, luminance_threshold_value):
    ls, smoothness = _unpack2(parameter, 1, 5)
    return _smooth_diffuse(image_array.astype(np.float32),
                           luminance_threshold_value, int(ls), int(smoothness))


# ── Kernel: Stucki Diffusion Lines · Glitch · dims=2 · Line Emphasis 1-10-5 ──
@registry.register("Stucki Diffusion Lines", "Glitch Effects", dims=2,
                   param_sliders=("line_emphasis_slider",))
def stucki_diffusion_lines(image_array, parameter, luminance_threshold_value):
    e = float(parameter) if parameter else 5.0
    return _stucki_diffusion_lines(image_array.astype(np.float32),
                                   luminance_threshold_value, e)


# ── Kernel: Atkinson Line Modulation · Glitch · dims=2 ──
#    sliders (Modulation Strength 1-10-5, Horizontal Bias 1-10-5)
@registry.register("Atkinson Line Modulation", "Glitch Effects", dims=2,
                   param_sliders=("modulation_strength_slider",
                                  "horizontal_bias_slider"))
def atkinson_line_modulation(image_array, parameter, luminance_threshold_value):
    strength, hbias = _unpack2(parameter, 5, 5)
    return _atkinson_line_modulation(image_array.astype(np.float32),
                                     luminance_threshold_value,
                                     float(strength), float(hbias))


# ── Kernel: Contrast Aware Y · Glitch · dims=2 · Line Scale 1-20-1 ──
@registry.register("Contrast Aware Y", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def contrast_aware_y(image_array, parameter, luminance_threshold_value):
    ls = int(parameter) if parameter else 1
    return _contrast_aware(image_array.astype(np.float32),
                           luminance_threshold_value, ls, True)


# ── Kernel: Contrast Aware X · Glitch · dims=2 · Line Scale 1-20-1 ──
@registry.register("Contrast Aware X", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def contrast_aware_x(image_array, parameter, luminance_threshold_value):
    ls = int(parameter) if parameter else 1
    return _contrast_aware(image_array.astype(np.float32),
                           luminance_threshold_value, ls, False)


# ── Tuple-unpack helpers (plain Python; run outside njit) ──
def _unpack3(parameter):
    if isinstance(parameter, (tuple, list)):
        a = parameter[0] if len(parameter) > 0 else 1
        b = parameter[1] if len(parameter) > 1 else 0.0
        c = parameter[2] if len(parameter) > 2 else 0.0
        return a, b, c
    return parameter, 0.0, 0.0


def _unpack2(parameter, d0, d1):
    if isinstance(parameter, (tuple, list)):
        a = parameter[0] if len(parameter) > 0 else d0
        b = parameter[1] if len(parameter) > 1 else d1
        return a, b
    return parameter, d1
