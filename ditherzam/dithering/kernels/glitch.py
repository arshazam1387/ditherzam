from __future__ import annotations
import math
import numpy as np
from numba import njit
from ditherzam.dithering import registry
from ditherzam.dithering.kernels.error_diffusion import _diffuse_row
from ditherzam.dithering.kernels.ordered import _BAYER4


@njit(cache=True)
def _line_diffuse(img, thr, line_scale, horizontal, error_gain, decay, row_phase, clamp_error, line_spacing):
    """1-D error diffusion producing banded line patterns; density ~ brightness."""
    h, w = img.shape
    out = img.copy()
    s = line_scale if line_scale >= 1 else 1
    if horizontal:
        for y in range(h):
            carry = row_phase if y % 2 else 0.0
            for x in range(w):
                old = out[y, x] + carry
                new = 255.0 if old >= thr else 0.0
                out[y, x] = new
                carry = (old - new) / s * error_gain + carry * decay
                if clamp_error > 0:
                    carry = max(-clamp_error, min(clamp_error, carry))
    else:
        for x in range(w):
            carry = row_phase if x % 2 else 0.0
            for y in range(h):
                old = out[y, x] + carry
                new = 255.0 if old >= thr else 0.0
                out[y, x] = new
                carry = (old - new) / s * error_gain + carry * decay
                if clamp_error > 0:
                    carry = max(-clamp_error, min(clamp_error, carry))
    spacing = max(1, int(line_spacing))
    if spacing > 1:
        if horizontal:
            for y in range(h):
                if y % spacing != 0:
                    out[y, :] = 255.0
        else:
            for x in range(w):
                if x % spacing != 0:
                    out[:, x] = 255.0
    return out


@njit(cache=True)
def _uniform_modulation(img, thr, line_scale, smoothing, bleed, horizontal, error_gain, decay):
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
                carry = err / s * error_gain + carry * decay
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
                carry = err / s * error_gain + carry * decay
                prev_err[y] = prev_err[y] * smoothing + err * (1.0 - smoothing)
    return out


@njit(cache=True)
def _atkinson_vhs(img, thr, line_count, line_width, brightness, spacing_curve, offset):
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if img[y, x] >= thr else 0.0
    lc = line_count if line_count >= 1 else 1
    for k in range(lc):
        pos = (k + 0.5) / lc
        ry = int((pos ** spacing_curve) * h + offset)
        if ry < 0:
            ry = 0
        if ry >= h:
            ry = h - 1
        for dy in range(line_width):
            ly = ry + dy
            if ly < h:
                for x in range(w):
                    out[ly, x] = brightness
    return out


@njit(cache=True)
def _glitch(img, thr, intensity, seed, row_hold, direction_bias, wrap):
    np.random.seed(seed)
    h, w = img.shape
    out = np.empty_like(img)
    amp = intensity if intensity >= 1 else 1
    shift = 0
    for y in range(h):
        if y % row_hold == 0:
            shift = int((np.random.random() - direction_bias) * 2.0 * amp)
        for x in range(w):
            sx = x + shift
            if wrap:
                sx %= w
            elif sx < 0: sx = 0
            elif sx >= w: sx = w - 1
            out[y, x] = 255.0 if img[y, sx] >= thr else 0.0
    return out


@njit(cache=True)
def _waveform(img, thr, density, base_freq, y_phase, amplitude, phase, spacing):
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            freq = (base_freq / 100.0 + (1.0 - img[y, x] / 255.0) * density * 0.05) / (spacing / 100.0)
            v = 127.5 + math.sin(x * freq + y * y_phase / 100.0 + phase * math.pi / 180.0) * amplitude
            out[y, x] = 255.0 if img[y, x] >= v else 0.0
    return out


@njit(cache=True)
def _waveform_alt(img, thr, blend, base_freq, tone_freq, amplitude, phase, spacing):
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            gx = 0.0
            if 0 < x < w - 1:
                gx = (img[y, x + 1] - img[y, x - 1]) / 255.0
            p = x * (base_freq / 100.0 + (1.0 - img[y, x] / 255.0) * tone_freq / 100.0) / (spacing / 100.0) + gx * blend + phase * math.pi / 180.0
            v = 127.5 + math.sin(p) * amplitude
            out[y, x] = 255.0 if img[y, x] >= v else 0.0
    return out


@njit(cache=True)
def _ordered_modulation(img, thr, param, base, frequency, amplitude, xweight, phase, spacing):
    h, w = img.shape
    mh, mw = base.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            coord = x * xweight / 100.0 + y * (1.0 - xweight / 100.0)
            wob = math.sin(coord * frequency / 100.0 * param / (spacing / 100.0) + phase * math.pi / 180.0) * amplitude
            t = base[y % mh, x % mw] + wob
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True)
def _smooth_diffuse(img, thr, line_scale, smoothness, error_gain, memory, row_bias):
    h, w = img.shape
    out = img.copy()
    s = line_scale if line_scale >= 1 else 1
    a = 1.0 / smoothness if smoothness >= 1 else 1.0
    for y in range(h):
        carry = row_bias if y % 2 else 0.0
        for x in range(w):
            old = out[y, x] + carry
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            carry = ((old - new) / s) * a * error_gain + carry * (1.0 - a) * memory
    return out


@njit(cache=True)
def _stucki_diffusion_lines(img, thr, emphasis, near_weight, far_weight, vertical_weight, final_threshold):
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
                out[y, x + 1] += err * near_weight * (1.0 + 0.8 * e)
            if x + 2 < w:
                out[y, x + 2] += err * far_weight * e
            if y + 1 < h:
                out[y + 1, x] += err * vertical_weight * (1.0 - 0.5 * e)
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= final_threshold else 0.0
    return out


@njit(cache=True)
def _atkinson_line_modulation(img, thr, strength, hbias, divisor, far_weight, vertical_weight):
    h, w = img.shape
    out = img.copy()
    hb = hbias / 5.0
    st = strength / 5.0
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = (old - new) / divisor * st
            if x + 1 < w:
                out[y, x + 1] += err * (1.0 + hb)
            if x + 2 < w:
                out[y, x + 2] += err * hb * far_weight
            if y + 1 < h:
                out[y + 1, x] += err * (1.0 - 0.5 * hb) * vertical_weight
                if x + 1 < w:
                    out[y + 1, x + 1] += err
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@njit(cache=True)
def _contrast_aware(img, thr, line_scale, horizontal, contrast_gain, contrast_center, error_gain, radius, line_spacing):
    """1-D diffusion whose threshold warps with local contrast."""
    h, w = img.shape
    out = img.copy()
    s = line_scale if line_scale >= 1 else 1
    if horizontal:
        for y in range(h):
            carry = 0.0
            for x in range(w):
                lo = img[y, x - radius] if x >= radius else img[y, x]
                hi = img[y, x + radius] if x + radius < w else img[y, x]
                local = abs(hi - lo)
                t = thr + (local - contrast_center) * contrast_gain
                old = out[y, x] + carry
                new = 255.0 if old >= t else 0.0
                out[y, x] = new
                carry = (old - new) / s * error_gain
    else:
        for x in range(w):
            carry = 0.0
            for y in range(h):
                lo = img[y - radius, x] if y >= radius else img[y, x]
                hi = img[y + radius, x] if y + radius < h else img[y, x]
                local = abs(hi - lo)
                t = thr + (local - contrast_center) * contrast_gain
                old = out[y, x] + carry
                new = 255.0 if old >= t else 0.0
                out[y, x] = new
                carry = (old - new) / s * error_gain
    spacing = max(1, int(line_spacing))
    if spacing > 1:
        if horizontal:
            for y in range(h):
                if y % spacing != 0:
                    out[y, :] = 255.0
        else:
            for x in range(w):
                if x % spacing != 0:
                    out[:, x] = 255.0
    return out


# ── Kernel: Artifact Modulation · Glitch · dims=2 · Dither Param 1-20-1 ──
@registry.register("Artifact Modulation", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "artifact_base_frequency_slider", "artifact_tone_frequency_slider", "artifact_amplitude_slider", "artifact_phase_slider", "wave_line_spacing_slider"))
def artifact_modulation(image_array, parameter, luminance_threshold_value):
    p, base, tone, amp, phase, spacing = _unpack6(parameter, 1, 5, 10, 128, 0, 100)
    return _waveform_alt(image_array.astype(np.float32),
                         luminance_threshold_value, float(p), float(base), float(tone), _half_span(amp), float(phase), float(spacing))


# ── Kernel: Atkinson-VHS · Glitch · dims=2 · Line Count 1-20-1 ──
@registry.register("Atkinson-VHS", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "vhs_line_width_slider", "vhs_line_brightness_slider", "vhs_spacing_curve_slider", "vhs_line_offset_slider"))
def atkinson_vhs(image_array, parameter, luminance_threshold_value):
    lc, width, bright, curve, offset = _unpack5(parameter, 1, 1, 255, 100, 0)
    return _atkinson_vhs(image_array.astype(np.float32),
                         luminance_threshold_value, int(lc), max(1, int(width)), float(bright), float(curve) / 100.0, int(offset))


# ── Kernel: Glitch · Glitch · dims=2 · Glitch Intensity 1-20-1 ──
@registry.register("Glitch", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "glitch_seed_slider", "glitch_row_hold_slider", "glitch_direction_bias_slider", "glitch_wrap_slider"))
def glitch(image_array, parameter, luminance_threshold_value):
    intensity, seed, hold, bias, wrap = _unpack5(parameter, 1, 0, 1, 50, 1)
    return _glitch(image_array.astype(np.float32),
                   luminance_threshold_value, int(intensity), int(seed), max(1, int(hold)), float(bias) / 100.0, bool(wrap))


# ── Kernel: Modulated Diffuse Y · Glitch · dims=2 · Line Scale 1-20-1 ──
@registry.register("Modulated Diffuse Y", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "diffusion_error_gain_slider", "diffusion_decay_slider", "diffusion_row_phase_slider", "diffusion_error_clamp_slider", "diffusion_line_spacing_slider"))
def modulated_diffuse_y(image_array, parameter, luminance_threshold_value):
    ls, gain, decay, phase, clamp, spacing = _unpack6(parameter, 1, 100, 0, 0, 0, 1)
    return _line_diffuse(image_array.astype(np.float32),
                         luminance_threshold_value, int(ls), True, float(gain) / 100.0, float(decay) / 100.0, float(phase), float(clamp), int(spacing))


# ── Kernel: Modulated Diffuse X · Glitch · dims=2 · Line Scale 1-20-1 ──
@registry.register("Modulated Diffuse X", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "diffusion_error_gain_slider", "diffusion_decay_slider", "diffusion_row_phase_slider", "diffusion_error_clamp_slider", "diffusion_line_spacing_slider"))
def modulated_diffuse_x(image_array, parameter, luminance_threshold_value):
    ls, gain, decay, phase, clamp, spacing = _unpack6(parameter, 1, 100, 0, 0, 0, 1)
    return _line_diffuse(image_array.astype(np.float32),
                         luminance_threshold_value, int(ls), False, float(gain) / 100.0, float(decay) / 100.0, float(phase), float(clamp), int(spacing))


# ── Kernel: Uniform Modulation Y · Glitch · dims=2 ──
#    sliders (Line Scale 1-20-1, Smoothing Factor 0-1-0, Bleed Fraction 0-100-0)
@registry.register("Uniform Modulation Y", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",
                                  "smoothing_factor_slider",
                                  "bleed_fraction_slider", "diffusion_error_gain_slider", "diffusion_decay_slider"))
def uniform_modulation_y(image_array, parameter, luminance_threshold_value):
    ls, smooth, bleed, gain, decay = _unpack5(parameter, 2, 50, 25, 100, 10)
    return _uniform_modulation(image_array.astype(np.float32),
                               luminance_threshold_value,
                               int(ls), float(smooth) / 100.0, float(bleed) / 100.0, True, float(gain) / 100.0, float(decay) / 100.0)


# ── Kernel: Uniform Modulation X · Glitch · dims=2 (same three sliders) ──
@registry.register("Uniform Modulation X", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",
                                  "smoothing_factor_slider",
                                  "bleed_fraction_slider", "diffusion_error_gain_slider", "diffusion_decay_slider"))
def uniform_modulation_x(image_array, parameter, luminance_threshold_value):
    ls, smooth, bleed, gain, decay = _unpack5(parameter, 1, 0, 0, 100, 0)
    return _uniform_modulation(image_array.astype(np.float32),
                               luminance_threshold_value,
                               int(ls), float(smooth) / 100.0, float(bleed) / 100.0, False, float(gain) / 100.0, float(decay) / 100.0)


# ── Kernel: Waveform · Glitch · dims=2 · Wave Density 1-20-1 ──
@registry.register("Waveform", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "waveform_base_frequency_slider", "waveform_y_phase_slider", "waveform_amplitude_slider", "waveform_phase_slider", "wave_line_spacing_slider"))
def waveform(image_array, parameter, luminance_threshold_value):
    d, base, yp, amp, phase, spacing = _unpack6(parameter, 1, 5, 30, 128, 0, 100)
    return _waveform(image_array.astype(np.float32),
                     luminance_threshold_value, float(d), float(base), float(yp), _half_span(amp), float(phase), float(spacing))


# ── Kernel: Waveform Alt · Glitch · dims=2 · Modulation Blend 1-20-1 ──
@registry.register("Waveform Alt", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "artifact_base_frequency_slider", "artifact_tone_frequency_slider", "artifact_amplitude_slider", "artifact_phase_slider", "wave_line_spacing_slider"))
def waveform_alt(image_array, parameter, luminance_threshold_value):
    b, base, tone, amp, phase, spacing = _unpack6(parameter, 1, 5, 10, 128, 0, 100)
    return _waveform_alt(image_array.astype(np.float32),
                         luminance_threshold_value, float(b), float(base), float(tone), _half_span(amp), float(phase), float(spacing))


# ── Kernel: Ordered Modulation · Glitch · dims=2 · Dither Param 1-20-1 ──
@registry.register("Ordered Modulation", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "ordered_wobble_frequency_slider", "ordered_wobble_amplitude_slider", "ordered_axis_weight_slider", "ordered_phase_slider", "wave_line_spacing_slider"))
def ordered_modulation(image_array, parameter, luminance_threshold_value):
    p, freq, amp, axis, phase, spacing = _unpack6(parameter, 1, 20, 40, 50, 0, 100)
    return _ordered_modulation(image_array.astype(np.float32),
                               luminance_threshold_value, float(p), _BAYER4, float(freq), float(amp), float(axis), float(phase), float(spacing))


# ── Kernel: Smooth Diffuse · Glitch · dims=2 (Line Scale 1-20-1, Smoothness 1-10-5) ──
@registry.register("Smooth Diffuse", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "smoothness_slider", "diffusion_error_gain_slider", "smooth_memory_slider", "diffusion_row_phase_slider"))
def smooth_diffuse(image_array, parameter, luminance_threshold_value):
    ls, smoothness, gain, memory, bias = _unpack5(parameter, 1, 5, 100, 100, 0)
    return _smooth_diffuse(image_array.astype(np.float32),
                           luminance_threshold_value, int(ls), int(smoothness), float(gain) / 100.0, float(memory) / 100.0, float(bias))


# ── Kernel: Stucki Diffusion Lines · Glitch · dims=2 · Line Emphasis 1-10-5 ──
@registry.register("Stucki Diffusion Lines", "Glitch Effects", dims=2,
                   param_sliders=("line_emphasis_slider", "stucki_near_weight_slider", "stucki_far_weight_slider", "stucki_vertical_weight_slider", "stucki_final_threshold_slider"))
def stucki_diffusion_lines(image_array, parameter, luminance_threshold_value):
    e, near, far, vertical, final_thr = _unpack5(parameter, 5, 50, 20, 30, 128)
    return _stucki_diffusion_lines(image_array.astype(np.float32),
                                   luminance_threshold_value, float(e), float(near) / 100.0, float(far) / 100.0, float(vertical) / 100.0, float(final_thr))


# ── Kernel: Atkinson Line Modulation · Glitch · dims=2 ──
#    sliders (Modulation Strength 1-10-5, Horizontal Bias 1-10-5)
@registry.register("Atkinson Line Modulation", "Glitch Effects", dims=2,
                   param_sliders=("modulation_strength_slider",
                                  "horizontal_bias_slider", "atkinson_divisor_slider", "atkinson_far_weight_slider", "atkinson_vertical_weight_slider"))
def atkinson_line_modulation(image_array, parameter, luminance_threshold_value):
    strength, hbias, divisor, far, vertical = _unpack5(parameter, 5, 5, 8, 100, 100)
    return _atkinson_line_modulation(image_array.astype(np.float32),
                                     luminance_threshold_value,
                                     float(strength), float(hbias), max(1.0, float(divisor)), float(far) / 100.0, float(vertical) / 100.0)


# ── Kernel: Contrast Aware Y · Glitch · dims=2 · Line Scale 1-20-1 ──
@registry.register("Contrast Aware Y", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "contrast_gain_slider", "contrast_center_slider", "diffusion_error_gain_slider", "contrast_radius_slider", "diffusion_line_spacing_slider"))
def contrast_aware_y(image_array, parameter, luminance_threshold_value):
    ls, gain, center, error_gain, radius, spacing = _unpack6(parameter, 1, 25, 64, 100, 1, 1)
    return _contrast_aware(image_array.astype(np.float32),
                           luminance_threshold_value, int(ls), True, float(gain) / 100.0, float(center), float(error_gain) / 100.0, max(1, int(radius)), int(spacing))


# ── Kernel: Contrast Aware X · Glitch · dims=2 · Line Scale 1-20-1 ──
@registry.register("Contrast Aware X", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "contrast_gain_slider", "contrast_center_slider", "diffusion_error_gain_slider", "contrast_radius_slider", "diffusion_line_spacing_slider"))
def contrast_aware_x(image_array, parameter, luminance_threshold_value):
    ls, gain, center, error_gain, radius, spacing = _unpack6(parameter, 1, 25, 64, 100, 1, 1)
    return _contrast_aware(image_array.astype(np.float32),
                           luminance_threshold_value, int(ls), False, float(gain) / 100.0, float(center), float(error_gain) / 100.0, max(1, int(radius)), int(spacing))


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


def _unpack5(parameter, d0, d1, d2, d3, d4):
    if isinstance(parameter, (tuple, list)):
        defaults = (d0, d1, d2, d3, d4)
        return tuple(parameter[i] if i < len(parameter) else defaults[i] for i in range(5))
    return (parameter if parameter not in (None, 0) else d0), d1, d2, d3, d4


def _unpack6(parameter, d0, d1, d2, d3, d4, d5):
    if isinstance(parameter, (tuple, list)):
        defaults = (d0, d1, d2, d3, d4, d5)
        return tuple(parameter[i] if i < len(parameter) else defaults[i] for i in range(6))
    return (parameter if parameter not in (None, 0) else d0), d1, d2, d3, d4, d5


def _half_span(value):
    return max(0.0, float(value) - 0.5)
