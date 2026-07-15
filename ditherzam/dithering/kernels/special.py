from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from ditherzam.dithering import registry


@njit(cache=True, parallel=True)
def _radial_burst(img, thr, rays, phase, center_x, center_y, threshold_span):
    h, w = img.shape
    cx = w / 2.0 + center_x
    cy = h / 2.0 + center_y
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            ang = math.atan2(y - cy, x - cx)
            t = 127.5 + math.sin(ang * rays + phase * math.pi / 180.0) * threshold_span
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _wave(img, thr, xfreq, yfreq, phase, xweight, amplitude, line_spacing):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            wx = xweight / 100.0
            p = phase * math.pi / 180.0
            spacing = line_spacing / 100.0
            t = 127.5 + (math.sin(x * xfreq / (100.0 * spacing) + p) * wx + math.sin(y * yfreq / (100.0 * spacing) + p) * (1.0 - wx)) * amplitude
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True)
def _noise(img, thr, amplitude, seed, bias, grain, image_mix):
    np.random.seed(seed)
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            if grain == 1:
                n = (np.random.random() - 0.5) * amplitude + bias
            else:
                gy = (y // grain) * grain
                gx = (x // grain) * grain
                np.random.seed(seed + gy * 65537 + gx)
                n = (np.random.random() - 0.5) * amplitude + bias
            value = img[y, x] * image_mix / 100.0
            out[y, x] = 255.0 if (value + n) >= thr else 0.0
    return out


@njit(cache=True, parallel=True)
def _topography(img, warp, bands, warp_freq, sample_step, phase):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            sx = int(x + math.sin(y * warp_freq / 100.0 + phase * math.pi / 180.0) * warp)
            if sx < 0:
                sx = 0
            elif sx >= w:
                sx = w - 1
            b0 = int(img[y, sx] / 256.0 * bands)
            xr = sx + sample_step if sx + sample_step < w else sx
            yd = y + sample_step if y + sample_step < h else y
            br = int(img[y, xr] / 256.0 * bands)
            bd = int(img[int(yd), sx] / 256.0 * bands)
            out[y, x] = 0.0 if (b0 != br or b0 != bd) else 255.0
    return out


@njit(cache=True, parallel=True)
def _thresholder(img, freq, xscale, yscale, amplitude, phase):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            p = phase * math.pi / 180.0
            t = 128.0 + math.sin(x * xscale / freq + p) * math.cos(y * yscale / freq + p) * amplitude
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _diagonal(img, sensitivity, xweight, yweight, radius, edge_bias):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            xl = x - radius if x >= radius else x
            xr = x + radius if x + radius < w else x
            yu = y - radius if y >= radius else y
            yd = y + radius if y + radius < h else y
            gx = (img[y, xr] - img[y, xl]) * xweight / 100.0
            gy = (img[int(yd), x] - img[int(yu), x]) * yweight / 100.0
            mag = math.sqrt(gx * gx + gy * gy)
            thr_edge = 200.0 / sensitivity
            out[y, x] = 0.0 if mag > thr_edge + edge_bias else 255.0
    return out


@njit(cache=True, parallel=True)
def _displace_contour(img, contour_thr, line_mode, smoothing, line_space, displacement):
    h, w = img.shape
    bands = line_space if line_space >= 1 else 1
    thick = line_mode if line_mode >= 1 else 1
    step = 256.0 / (bands * 4.0)
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            sx = int(x + math.sin(y * 0.1) * displacement)
            if sx < 0: sx = 0
            elif sx >= w: sx = w - 1
            val = img[y, sx]
            b0 = int(val / step)
            is_line = False
            for t in range(thick):
                reach = 1 + t + smoothing
                xr = sx + reach if sx + reach < w else w - 1
                yd = y + 1 + t if y + 1 + t < h else h - 1
                if int(img[y, xr] / step) != b0 or int(img[yd, x] / step) != b0:
                    is_line = True
            gate = val < (contour_thr / 100.0 * 255.0)
            out[y, x] = 0.0 if (is_line and gate) else 255.0
    return out


@njit(cache=True, parallel=True)
def _sine_wave_modulation(img, freq, wave_thr, yfreq, phase, contrast, line_spacing):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            spacing = line_spacing / 100.0
            line = math.sin((x * freq * 0.05 + y * yfreq / 100.0) / spacing + phase * math.pi / 180.0) * 0.5 + 0.5
            gate = darkness * (wave_thr / 15.0) * contrast / 100.0
            out[y, x] = 0.0 if line < gate else 255.0
    return out


@njit(cache=True, parallel=True)
def _vortex(img, thr, arms, radial_freq, phase, center_x, center_y):
    h, w = img.shape
    cx = w / 2.0 + center_x
    cy = h / 2.0 + center_y
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            ang = math.atan2(dy, dx)
            r = math.sqrt(dx * dx + dy * dy)
            t = (math.sin(ang * arms + r * radial_freq / 100.0 + phase * math.pi / 180.0) * 0.5 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _concentric(img, thr, frequency, phase, center_x, center_y, ellipticity):
    h, w = img.shape
    cx = w / 2.0 + center_x
    cy = h / 2.0 + center_y
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            r = math.sqrt(dx * dx + dy * dy * ellipticity * ellipticity / 10000.0)
            t = (math.sin(r * frequency / 100.0 + phase * math.pi / 180.0) * 0.5 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _wireframe_alt(img, sensitivity, xweight, yweight, diagweight, radius):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            xl = x - radius if x >= radius else x
            xr = x + radius if x + radius < w else x
            yu = y - radius if y >= radius else y
            yd = y + radius if y + radius < h else y
            gx = (img[y, xr] - img[y, xl]) * xweight / 100.0
            gy = (img[int(yd), x] - img[int(yu), x]) * yweight / 100.0
            gd = (img[int(yd), xr] - img[int(yu), xl]) * diagweight / 100.0
            mag = math.sqrt(gx * gx + gy * gy + gd * gd)
            thr_edge = 160.0 / sensitivity
            out[y, x] = 0.0 if mag > thr_edge else 255.0
    return out


@njit(cache=True, parallel=True)
def _crosshatch_alt(img, s, vertical_gate, horizontal_gate, diagonal_gate, steep_gate):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            hit = False
            if darkness > vertical_gate / 100.0 and (x % s == 0):
                hit = True
            if darkness > horizontal_gate / 100.0 and (y % s == 0):
                hit = True
            if darkness > diagonal_gate / 100.0 and ((x + y) % s == 0):
                hit = True
            if darkness > steep_gate / 100.0 and ((x - y) % s == 0):
                hit = True
            out[y, x] = 0.0 if hit else 255.0
    return out


@njit(cache=True)
def _hash01(x, y, salt):
    # Same integer-hash family as pipeline jitter; pure function of (x, y, salt),
    # identical under JIT on/off because it is plain masked int arithmetic.
    h = (x * 374761393 + y * 668265263 + salt * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h >> 8) & 0xFFFF) / 65535.0


@njit(cache=True)
def _vnoise(x, y, k, salt):
    # Smooth 2D value noise on an integer hash lattice; iteration index k is
    # folded into the lattice so every feedback step gets its own field.
    xi = int(math.floor(x)) + k * 8191
    yi = int(math.floor(y))
    fx = x - math.floor(x)
    fy = y - math.floor(y)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    n00 = _hash01(xi, yi, salt)
    n10 = _hash01(xi + 1, yi, salt)
    n01 = _hash01(xi, yi + 1, salt)
    n11 = _hash01(xi + 1, yi + 1, salt)
    return (n00 * (1.0 - ux) + n10 * ux) * (1.0 - uy) + (n01 * (1.0 - ux) + n11 * ux) * uy


@njit(cache=True, parallel=True)
def _echo_smear(img, thr, count, spacing, wave, phase, streak, dissolve, breath, wave_freq):
    h, w = img.shape
    b = breath / 100.0
    dissolve_gate = b * dissolve / 100.0 * 2.0
    phase_rad = phase * math.pi / 180.0
    freq = wave_freq / 100.0

    # Streak pre-pass: selected columns drip straight down from the column's
    # lowest subject pixel; -1 marks columns with no subject or not selected.
    streak_prob = streak / 100.0 * 0.12
    drip_from = np.full(w, -1, dtype=np.int64)
    for x in prange(w):
        if _hash01(x, 0, 303) < streak_prob:
            for y in range(h - 1, -1, -1):
                if img[y, x] < thr:
                    drip_from[x] = y
                    break

    # Wavy drip mask: each selected column draws a fading, swaying trail
    # downward from the subject's lowest edge. Parallel writes all store the
    # same value (1), so write races are benign.
    drip_mask = np.zeros((h, w), dtype=np.uint8)
    for x0 in prange(w):
        y0 = drip_from[x0]
        if y0 < 0:
            continue
        drop = h - 1 - y0
        if drop <= 0:
            continue
        jitter = _hash01(x0, 0, 606) * 6.2831853
        for y in range(y0 + 1, h):
            prog = (y - y0) / drop
            density = 1.0 - 0.85 * prog
            if _hash01(x0, y, 707) >= density:
                continue
            sway = math.sin(y * freq + phase_rad + jitter) * wave * 0.75
            xc = int(x0 + sway)
            half = 1 if prog < 0.3 else 0
            for dx in range(-half, half + 1):
                xi = xc + dx
                if 0 <= xi < w:
                    drip_mask[y, xi] = 1

    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            ink = False
            if img[y, x] < thr:
                xl = x - 3 if x >= 3 else 0
                xr2 = x + 3 if x + 3 < w else w - 1
                near_edge = img[y, xl] >= thr or img[y, xr2] >= thr
                local = dissolve_gate * (1.5 if near_edge else 0.75)
                if _hash01(x, y, 101) >= local:
                    ink = True
            if not ink and drip_mask[y, x] == 1:
                ink = True
            if not ink and count > 0 and b > 0.0:
                visible = b * count
                t = (phase % 360.0) / 360.0     # travel fraction along the smear axis
                for n in range(1, count + 2):   # +1 line so the cycle wraps seamlessly
                    e_idx = n - t               # continuous echo index (line identity)
                    d = e_idx * spacing
                    if d < 1.0:
                        continue                # this line has arrived at the subject
                    off = math.sin(y * freq + phase_rad + e_idx * 0.7) * wave
                    sx = int(x - d - off)
                    if sx < 0 or sx + 1 >= w:
                        continue
                    # trailing (right) edge only: subject at sx, background at sx+1
                    if img[y, sx] < thr and img[y, sx + 1] >= thr:
                        if e_idx <= visible:
                            ink = True
                            break
                        if _hash01(x, y, 202 + n) < (visible + 1.0 - e_idx):
                            ink = True
                            break
            if not ink and img[y, x] >= thr:
                dust_gate = b * dissolve / 100.0 * 0.5
                if dust_gate > 0.0:
                    reach = spacing * (count if count > 0 else 4)
                    xs = x + 1 + int(_hash01(x, y, 404) * reach)
                    if xs < w and img[y, xs] < thr:
                        if _hash01(x, y, 505) < dust_gate:
                            ink = True
            out[y, x] = 0.0 if ink else 255.0
    return out


@njit(cache=True, parallel=True)
def _feedback_smear(img, thr, k_iters, drift, namount, nscale, decay, time_v,
                    erode, density, lines, lspacing):
    h, w = img.shape
    s = nscale / 1000.0
    tshift = time_v / 360.0 * 37.0
    t_frac = (time_v % 360.0) / 360.0
    dgain = density / 100.0
    surv_rate = decay / 100.0
    lgain = lines / 100.0
    falloff = lspacing * 2.0

    # Silhouette profile: rightmost subject pixel per row, box-smoothed
    # vertically with a presence weight so the profile feathers out above
    # and below the subject instead of snapping between bent and straight.
    edge = np.full(h, -1.0, dtype=np.float32)
    for y in prange(h):
        for x in range(w - 1, -1, -1):
            if img[y, x] < thr:
                edge[y] = float(x)
                break
    edge_s = np.empty(h, dtype=np.float32)
    pres = np.empty(h, dtype=np.float32)
    for y in prange(h):
        acc = 0.0
        cnt = 0.0
        tot = 0.0
        lo = y - 8 if y >= 8 else 0
        hi = y + 8 if y + 8 < h else h - 1
        for yy in range(lo, hi + 1):
            tot += 1.0
            if edge[yy] >= 0.0:
                acc += edge[yy]
                cnt += 1.0
        edge_s[y] = acc / cnt if cnt > 0.0 else -1.0
        pres[y] = cnt / tot

    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            ink = False
            subject = img[y, x] < thr
            if subject:
                # Coherent eaten patches: the smooth field itself is the gate,
                # so erosion carves holes instead of per-pixel static. Bites
                # concentrate near the silhouette edges; the core stays solid.
                xl = x - 4 if x >= 4 else 0
                xr4 = x + 4 if x + 4 < w else w - 1
                near_edge = img[y, xl] >= thr or img[y, xr4] >= thr
                cut = erode / 100.0 * (1.3 if near_edge else 0.55)
                field = _vnoise(x * s * 3.0 + tshift, y * s * 3.0, 0, 909)
                if field >= cut:
                    ink = True
            if not ink and not subject and lgain > 0.0:
                # Feedback lines: a lattice of long continuous vertical lines
                # right of the silhouette. The warped coordinate q blends from
                # screen-x (straight verticals far away) to distance-from-
                # silhouette (contour-hugging waves up close); the warp
                # gradient piles lines into nested waves at the edge, and
                # Time marches the whole lattice INTO the subject, one
                # spacing per full sweep.
                e = edge_s[y]
                if e < 0.0 or float(x) > e:
                    wgt = 0.0
                    dqdx = 1.0
                    if e >= 0.0:
                        wgt = pres[y] * math.exp(-(float(x) - e) / falloff)
                        dqdx = 1.0 + wgt * e / falloff
                        if dqdx > 6.0:
                            dqdx = 6.0
                    q = float(x) - wgt * e
                    wob = (_vnoise(x * s * 0.5 + tshift, y * s * 1.6, 0, 808)
                           - 0.5) * 2.0 * namount * (0.3 + 1.7 * wgt)
                    v = (q + wob) / lspacing + t_frac
                    fv = v - math.floor(v)
                    # width x dqdx keeps lines ~1.6px on screen where the
                    # warp compresses the lattice near the silhouette
                    if fv * lspacing < 1.6 * dqdx:
                        idx = int(math.floor(v))
                        if lgain >= 1.0 or _hash01(idx, 0, 111) < lgain:
                            ink = True
            if not ink and dgain > 0.0:
                # Fractional feedback age: Time gives the walk a partial
                # first step, so every trail mark marches away from the
                # subject as it ages and wraps seamlessly each full sweep.
                surv = surv_rate ** t_frac
                px = float(x) - drift * t_frac
                py = float(y)
                for k in range(1, k_iters + 1):
                    surv *= surv_rate
                    if surv * dgain < 0.02:
                        break
                    # Sample the field at the WALKED position with a slow
                    # per-iteration slide: paths bend through the spatial field
                    # like real feedback history — coherent onion-skin lines,
                    # not per-pixel random walks or straight sprayed rays.
                    kk = (k + t_frac) * 0.08
                    nx = _vnoise(px * s + tshift + kk, py * s, 0, 606)
                    ny = _vnoise(px * s + tshift + kk, py * s, 0, 707)
                    px -= drift + (nx - 0.5) * 2.0 * namount * 0.4
                    py -= (ny - 0.5) * 2.0 * namount * 0.24
                    xi = int(px)
                    yi = int(py)
                    if xi < 0 or xi >= w or yi < 0 or yi >= h:
                        break
                    # The k-th displaced copy contributes only its trailing
                    # edge: thin bent lines with black gaps (spacing = drift),
                    # solid near the subject, dissolving into dots with decay.
                    xr = xi + 2 if xi + 2 < w else w - 1
                    if img[yi, xi] < thr and img[yi, xr] >= thr:
                        if _hash01(x, y, 202 + k) < surv * dgain * 1.5:
                            ink = True
                            break
            out[y, x] = 0.0 if ink else 255.0
    return out


# ── Kernel: Radial Burst · Special Effects · dims=2 · no sliders ──
@registry.register("Radial Burst", "Special Effects", dims=2,
                   param_sliders=("ray_count_slider", "ray_phase_slider", "center_x_slider", "center_y_slider", "threshold_span_slider"))
def radial_burst(image_array, parameter, luminance_threshold_value):
    rays, phase, cx, cy, span = _unpack5(parameter, 24, 0, 0, 0, 128)
    return _radial_burst(image_array.astype(np.float32), luminance_threshold_value, float(rays), float(phase), float(cx), float(cy), _half_span(span))


# ── Kernel: Wave · Special Effects · dims=2 · no sliders ──
@registry.register("Wave", "Special Effects", dims=2,
                   param_sliders=("wave_x_frequency_slider", "wave_y_frequency_slider", "special_wave_phase_slider", "wave_x_weight_slider", "wave_amplitude_slider", "wave_line_spacing_slider"))
def wave(image_array, parameter, luminance_threshold_value):
    xf, yf, phase, xw, amp, spacing = _unpack6(parameter, 15, 15, 0, 50, 128, 100)
    return _wave(image_array.astype(np.float32), luminance_threshold_value, float(xf), float(yf), float(phase), float(xw), _half_span(amp), float(spacing))


# ── Kernel: Noise · Special Effects · dims=2 · no sliders ──
@registry.register("Noise", "Special Effects", dims=2,
                   param_sliders=("noise_amplitude_slider", "noise_seed_slider", "noise_bias_slider", "noise_grain_slider", "noise_image_mix_slider"))
def noise(image_array, parameter, luminance_threshold_value):
    amp, seed, bias, grain, mix = _unpack5(parameter, 255, 0, 0, 1, 100)
    return _noise(image_array.astype(np.float32), luminance_threshold_value, float(amp), int(seed), float(bias), max(1, int(grain)), float(mix))


# ── Kernel: Topography · Special Effects · dims=2 · Warp Intensity 1-20-1 ──
@registry.register("Topography", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "topo_band_count_slider", "topo_warp_frequency_slider", "topo_sample_step_slider", "topo_phase_slider"))
def topography(image_array, parameter, luminance_threshold_value):
    warp, bands, freq, step, phase = _unpack5(parameter, 1, 8, 10, 1, 0)
    return _topography(image_array.astype(np.float32), float(warp), float(bands), float(freq), max(1, int(step)), float(phase))


# ── Kernel: Topography Alt · Special Effects · dims=2 · denser bands + wider sampling ──
@registry.register("Topography Alt", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "topo_band_count_slider", "topo_warp_frequency_slider", "topo_sample_step_slider", "topo_phase_slider"))
def topography_alt(image_array, parameter, luminance_threshold_value):
    warp, bands, freq, step, phase = _unpack5(parameter, 3, 12, 10, 2, 0)
    return _topography(image_array.astype(np.float32), float(warp), float(bands), float(freq), max(1, int(step)), float(phase))


# ── Kernel: Thresholder · Special Effects · dims=2 · Modulation Frequency 1-20-1 ──
@registry.register("Thresholder", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "threshold_x_scale_slider", "threshold_y_scale_slider", "threshold_amplitude_slider", "threshold_phase_slider"))
def thresholder(image_array, parameter, luminance_threshold_value):
    freq, xs, ys, amp, phase = _unpack5(parameter, 1, 1, 1, 64, 0)
    freq = float(freq)
    if freq < 1.0:
        freq = 1.0
    return _thresholder(image_array.astype(np.float32), freq, float(xs), float(ys), float(amp), float(phase))


# ── Kernel: Diagonal · Special Effects · dims=2 · Edge Sensitivity 1-20-1 ──
@registry.register("Diagonal", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "edge_x_weight_slider", "edge_y_weight_slider", "edge_radius_slider", "edge_bias_slider"))
def diagonal(image_array, parameter, luminance_threshold_value):
    s, xw, yw, radius, bias = _unpack5(parameter, 1, 100, 100, 1, 0)
    s = float(s)
    if s < 1.0:
        s = 1.0
    return _diagonal(image_array.astype(np.float32), s, float(xw), float(yw), max(1, int(radius)), float(bias))


# ── Kernel: Displace Contour · Special Effects · dims=2 ──
#    sliders (Contour Threshold 0-100-50, Line Mode 1-3-1, Smoothing 0-5-0, Line Spacing 1-5-1)
@registry.register("Displace Contour", "Special Effects", dims=2,
                   param_sliders=("contour_thresh_slider", "line_mode_slider",
                                  "smoothing_slider", "line_space_slider", "contour_displacement_slider"))
def displace_contour(image_array, parameter, luminance_threshold_value):
    ct, lm, sm, ls, displacement = _unpack5(parameter, 50, 1, 0, 1, 0)
    return _displace_contour(image_array.astype(np.float32),
                             float(ct), int(lm), int(sm), int(ls), float(displacement))


# ── Kernel: Sine Wave Modulation · Special Effects · dims=2 ──
#    sliders (Wave Frequency 1-20-5, Wave Threshold 1-30-10)
@registry.register("Sine Wave Modulation", "Special Effects", dims=2,
                   param_sliders=("wave_frequency_slider", "wave_threshold_slider", "sine_y_frequency_slider", "sine_phase_slider", "sine_contrast_slider", "wave_line_spacing_slider"))
def sine_wave_modulation(image_array, parameter, luminance_threshold_value):
    freq, wthr, yfreq, phase, contrast, spacing = _unpack6(parameter, 5, 10, 10, 0, 100, 100)
    return _sine_wave_modulation(image_array.astype(np.float32),
                                 float(freq), float(wthr), float(yfreq), float(phase), float(contrast), float(spacing))


# ── Kernel: Vortex · Special Effects · dims=2 · no sliders (extra) ──
@registry.register("Vortex", "Special Effects", dims=2,
                   param_sliders=("vortex_arms_slider", "vortex_radial_frequency_slider", "vortex_phase_slider", "center_x_slider", "center_y_slider"))
def vortex(image_array, parameter, luminance_threshold_value):
    arms, rf, phase, cx, cy = _unpack5(parameter, 6, 15, 0, 0, 0)
    return _vortex(image_array.astype(np.float32), luminance_threshold_value, float(arms), float(rf), float(phase), float(cx), float(cy))


# ── Kernel: Concentric Rings · Special Effects · dims=2 · no sliders (extra) ──
@registry.register("Concentric Rings", "Special Effects", dims=2,
                   param_sliders=("ring_frequency_slider", "ring_phase_slider", "center_x_slider", "center_y_slider", "ring_ellipticity_slider"))
def concentric_rings(image_array, parameter, luminance_threshold_value):
    freq, phase, cx, cy, ellipse = _unpack5(parameter, 30, 0, 0, 0, 100)
    return _concentric(image_array.astype(np.float32), luminance_threshold_value, float(freq), float(phase), float(cx), float(cy), float(ellipse))


# ── Kernel: Wireframe Alt · Special Effects · dims=2 · Edge Sensitivity 1-20-1 (extra) ──
@registry.register("Wireframe Alt", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "wire_x_weight_slider", "wire_y_weight_slider", "wire_diagonal_weight_slider", "wire_radius_slider"))
def wireframe_alt(image_array, parameter, luminance_threshold_value):
    s, xw, yw, dw, radius = _unpack5(parameter, 1, 100, 100, 100, 1)
    s = float(s)
    if s < 1.0:
        s = 1.0
    return _wireframe_alt(image_array.astype(np.float32), s, float(xw), float(yw), float(dw), max(1, int(radius)))


# ── Kernel: Crosshatch Alt · Special Effects · dims=2 · Line Spacing 1-20-1 (extra) ──
@registry.register("Crosshatch Alt", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "hatch_vertical_gate_slider", "hatch_horizontal_gate_slider", "hatch_diagonal_gate_slider", "hatch_steep_gate_slider"))
def crosshatch_alt(image_array, parameter, luminance_threshold_value):
    s, vg, hg, dg, sg = _unpack5(parameter, 4, 15, 40, 65, 85)
    s = int(s)
    if s < 1:
        s = 1
    return _crosshatch_alt(image_array.astype(np.float32), s, float(vg), float(hg), float(dg), float(sg))


# ── Kernel: Echo Smear · Special Effects · dims=2 ──
#    sliders (Echo Count 0-16-6, Echo Spacing 2-40-10, Wave Amount 0-32-8,
#             Wave Phase 0-360-0, Streak 0-100-20, Dissolve 0-100-30, Breath 0-100-50,
#             Wave Frequency 1-100-10)
@registry.register("Echo Smear", "Special Effects", dims=2,
                   param_sliders=("echo_count_slider", "echo_spacing_slider",
                                  "echo_wave_amount_slider", "echo_wave_phase_slider",
                                  "echo_streak_slider", "echo_dissolve_slider",
                                  "echo_breath_slider", "echo_wave_frequency_slider"))
def echo_smear(image_array, parameter, luminance_threshold_value):
    count, spacing, wave, phase, streak, dissolve, breath, wfreq = _unpack8(
        parameter, 6, 10, 8, 0, 20, 30, 50, 10)
    return _echo_smear(image_array.astype(np.float32),
                       float(luminance_threshold_value), max(0, int(count)),
                       float(spacing), float(wave), float(phase),
                       float(streak), float(dissolve), float(breath), float(wfreq))


# ── Kernel: Feedback Smear · Special Effects · dims=2 ──
#    sliders (Trail Length 4-64-32, Drift 1-8-2, Noise Amount 0-24-6, Noise Scale 1-100-20,
#             Decay 50-100-88, Time 0-360-0, Erode 0-100-30, Density 0-200-100,
#             Lines 0-100-60, Line Spacing 8-160-48)
@registry.register("Feedback Smear", "Special Effects", dims=2,
                   param_sliders=("fs_length_slider", "fs_drift_slider",
                                  "fs_noise_amount_slider", "fs_noise_scale_slider",
                                  "fs_decay_slider", "fs_time_slider",
                                  "fs_erode_slider", "fs_density_slider",
                                  "fs_lines_slider", "fs_line_spacing_slider"))
def feedback_smear(image_array, parameter, luminance_threshold_value):
    k_iters, drift, namount, nscale, decay, time_v, erode, density, lines, lspacing = _unpack10(
        parameter, 32, 2, 6, 20, 88, 0, 30, 100, 60, 48)
    return _feedback_smear(image_array.astype(np.float32),
                           float(luminance_threshold_value),
                           max(1, int(k_iters)), float(drift), float(namount),
                           max(1.0, float(nscale)), float(decay), float(time_v),
                           float(erode), float(density), float(lines),
                           max(4.0, float(lspacing)))


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


def _unpack7(parameter, d0, d1, d2, d3, d4, d5, d6):
    if isinstance(parameter, (tuple, list)):
        defaults = (d0, d1, d2, d3, d4, d5, d6)
        return tuple(parameter[i] if i < len(parameter) else defaults[i] for i in range(7))
    return (parameter if parameter not in (None, 0) else d0), d1, d2, d3, d4, d5, d6


def _unpack8(parameter, d0, d1, d2, d3, d4, d5, d6, d7):
    if isinstance(parameter, (tuple, list)):
        defaults = (d0, d1, d2, d3, d4, d5, d6, d7)
        return tuple(parameter[i] if i < len(parameter) else defaults[i] for i in range(8))
    return (parameter if parameter not in (None, 0) else d0), d1, d2, d3, d4, d5, d6, d7


def _unpack10(parameter, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9):
    if isinstance(parameter, (tuple, list)):
        defaults = (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)
        return tuple(parameter[i] if i < len(parameter) else defaults[i] for i in range(10))
    return (parameter if parameter not in (None, 0) else d0), d1, d2, d3, d4, d5, d6, d7, d8, d9


def _half_span(value):
    return max(0.0, float(value) - 0.5)
