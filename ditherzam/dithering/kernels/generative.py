from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ditherzam.dithering import registry


@njit(cache=True)
def _hilbert_xy(index, side):
    x = 0
    y = 0
    scale = 1
    value = index
    while scale < side:
        rx = (value // 2) & 1
        ry = (value ^ rx) & 1
        if ry == 0:
            if rx == 1:
                x = scale - 1 - x
                y = scale - 1 - y
            x, y = y, x
        x += scale * rx
        y += scale * ry
        value //= 4
        scale *= 2
    return x, y


@njit(cache=True)
def _hilbert_riemersma(img, threshold, queue_length):
    h, w = img.shape
    side = 1
    while side < max(h, w):
        side *= 2
    errors = np.zeros(queue_length, dtype=np.float32)
    out = np.empty_like(img)
    cursor = 0
    for index in range(side * side):
        x, y = _hilbert_xy(index, side)
        if x >= w or y >= h:
            continue
        carried = np.float32(0.0)
        weight = np.float32(0.5)
        for age in range(queue_length):
            slot = (cursor - 1 - age) % queue_length
            carried += errors[slot] * weight
            weight *= np.float32(0.5)
        value = img[y, x] + carried
        quantized = np.float32(255.0) if value >= threshold else np.float32(0.0)
        out[y, x] = quantized
        errors[cursor] = value - quantized
        cursor = (cursor + 1) % queue_length
    return out


@njit(cache=True)
def _spiral_path(img, threshold, retention):
    h, w = img.shape
    out = np.empty_like(img)
    x = (w - 1) // 2
    y = (h - 1) // 2
    dx = 1
    dy = 0
    segment_length = 1
    segment_used = 0
    turns = 0
    visited = 0
    error = np.float32(0.0)
    while visited < h * w:
        if 0 <= x < w and 0 <= y < h:
            value = img[y, x] + error
            quantized = np.float32(255.0) if value >= threshold else np.float32(0.0)
            out[y, x] = quantized
            error = (value - quantized) * retention
            visited += 1
        x += dx
        y += dy
        segment_used += 1
        if segment_used == segment_length:
            segment_used = 0
            dx, dy = -dy, dx
            turns += 1
            if turns % 2 == 0:
                segment_length += 1
    return out


@njit(cache=True, parallel=True)
def _flow_hatch(img, threshold, spacing):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            xl = x - 1 if x > 0 else x
            xr = x + 1 if x + 1 < w else x
            yu = y - 1 if y > 0 else y
            yd = y + 1 if y + 1 < h else y
            gx = (img[int(yu), int(xr)] + np.float32(2.0) * img[y, int(xr)] +
                  img[int(yd), int(xr)] - img[int(yu), int(xl)] -
                  np.float32(2.0) * img[y, int(xl)] - img[int(yd), int(xl)])
            gy = (img[int(yd), int(xl)] + np.float32(2.0) * img[int(yd), x] +
                  img[int(yd), int(xr)] - img[int(yu), int(xl)] -
                  np.float32(2.0) * img[int(yu), x] - img[int(yu), int(xr)])
            magnitude = math.sqrt(gx * gx + gy * gy)
            if magnitude < np.float32(0.001):
                projection = np.float32(x + y) * np.float32(0.70710677)
            else:
                projection = (np.float32(x) * gx + np.float32(y) * gy) / magnitude
            phase = projection / spacing
            phase -= math.floor(phase)
            distance = phase if phase < np.float32(0.5) else np.float32(1.0) - phase
            darkness = np.float32(1.0) - img[y, x] / np.float32(255.0)
            width = darkness * (np.float32(0.08) + threshold / np.float32(1020.0))
            out[y, x] = np.float32(0.0) if distance < width else np.float32(255.0)
    return out


@njit(cache=True, parallel=True)
def _hex_bayer(img, cell_size):
    h, w = img.shape
    out = np.empty_like(img)
    matrix = np.array(((0, 8, 2, 10), (12, 4, 14, 6),
                       (3, 11, 1, 9), (15, 7, 13, 5)), dtype=np.float32)
    row_height = cell_size * np.float32(0.75)
    for y in prange(h):
        for x in range(w):
            row = int(np.float32(y) / row_height)
            shifted_x = np.float32(x) - (cell_size * np.float32(0.5) if row & 1 else np.float32(0.0))
            col = math.floor(shifted_x / cell_size)
            local_x = (shifted_x / cell_size) - col
            local_y = (np.float32(y) / row_height) - row
            hex_bias = abs(local_x - np.float32(0.5)) + abs(local_y - np.float32(0.5)) * np.float32(0.35)
            level = (matrix[int(row % 4), int(col % 4)] + np.float32(0.5)) / np.float32(16.0)
            level = min(np.float32(1.0), max(np.float32(0.0), level + (hex_bias - np.float32(0.35)) * np.float32(0.18)))
            out[y, x] = np.float32(255.0) if img[y, x] / np.float32(255.0) >= level else np.float32(0.0)
    return out


@njit(cache=True, parallel=True)
def _triangular(img, cell_size):
    h, w = img.shape
    out = np.empty_like(img)
    matrix = np.array(((0, 8, 2, 10), (12, 4, 14, 6),
                       (3, 11, 1, 9), (15, 7, 13, 5)), dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            col = int(np.float32(x) / cell_size)
            row = int(np.float32(y) / cell_size)
            fx = np.float32(x) / cell_size - col
            fy = np.float32(y) / cell_size - row
            upper = fx + fy < np.float32(1.0)
            base = matrix[int(row % 4), int(col % 4)]
            offset = np.float32(-2.0) if upper else np.float32(2.0)
            level = min(np.float32(15.5), max(np.float32(0.5), base + np.float32(0.5) + offset)) / np.float32(16.0)
            out[y, x] = np.float32(255.0) if img[y, x] / np.float32(255.0) >= level else np.float32(0.0)
    return out


@njit(cache=True, parallel=True)
def _spiral_engrave(img, threshold, pitch):
    h, w = img.shape
    out = np.empty_like(img)
    cx = np.float32(w - 1) * np.float32(0.5)
    cy = np.float32(h - 1) * np.float32(0.5)
    two_pi = np.float32(6.283185307179586)
    for y in prange(h):
        for x in range(w):
            dx = np.float32(x) - cx
            dy = np.float32(y) - cy
            radius = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx)
            phase = radius / pitch - angle / two_pi
            phase -= math.floor(phase)
            distance = abs(phase - np.float32(0.5))
            darkness = np.float32(1.0) - img[y, x] / np.float32(255.0)
            thickness = darkness * (np.float32(0.25) + threshold / np.float32(1020.0))
            out[y, x] = np.float32(0.0) if distance < thickness else np.float32(255.0)
    return out


@njit(cache=True)
def _reaction_diffusion(img, threshold, iterations):
    h, w = img.shape
    u = np.ones((h, w), dtype=np.float32)
    v = np.zeros((h, w), dtype=np.float32)
    next_u = np.empty_like(u)
    next_v = np.empty_like(v)
    for y in range(h):
        for x in range(w):
            hashed = (x * 37 + y * 61 + x * y * 17 + 13) % 97
            if hashed < 9:
                v[y, x] = np.float32(0.75)
                u[y, x] = np.float32(0.25)
    for _ in range(iterations):
        for y in range(h):
            yu = y - 1 if y > 0 else h - 1
            yd = y + 1 if y + 1 < h else 0
            for x in range(w):
                xl = x - 1 if x > 0 else w - 1
                xr = x + 1 if x + 1 < w else 0
                uv = u[y, x]
                vv = v[y, x]
                lap_u = (u[int(yu), x] + u[int(yd), x] + u[y, int(xl)] + u[y, int(xr)] - np.float32(4.0) * uv)
                lap_v = (v[int(yu), x] + v[int(yd), x] + v[y, int(xl)] + v[y, int(xr)] - np.float32(4.0) * vv)
                lum = img[y, x] / np.float32(255.0)
                feed = np.float32(0.028) + lum * np.float32(0.018)
                kill = np.float32(0.055) + (np.float32(1.0) - lum) * np.float32(0.012)
                reaction = uv * vv * vv
                nu = uv + np.float32(0.16) * lap_u - reaction + feed * (np.float32(1.0) - uv)
                nv = vv + np.float32(0.08) * lap_v + reaction - (feed + kill) * vv
                next_u[y, x] = min(np.float32(1.0), max(np.float32(0.0), nu))
                next_v[y, x] = min(np.float32(1.0), max(np.float32(0.0), nv))
        u, next_u = next_u, u
        v, next_v = next_v, v
    cutoff = np.float32(0.04) + threshold / np.float32(255.0) * np.float32(0.16)
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            out[y, x] = np.float32(0.0) if v[y, x] >= cutoff else np.float32(255.0)
    return out


@njit(cache=True, parallel=True)
def _quasicrystal(img, threshold, waves):
    h, w = img.shape
    out = np.empty_like(img)
    golden_angle = np.float32(2.399963229728653)
    frequency = np.float32(0.32)
    phase_offset = threshold / np.float32(255.0) * np.float32(6.283185307179586)
    for y in prange(h):
        for x in range(w):
            total = np.float32(0.0)
            for wave in range(waves):
                angle = np.float32(wave) * golden_angle
                projection = (np.float32(x) * math.cos(angle) + np.float32(y) * math.sin(angle)) * frequency
                total += math.cos(projection + np.float32(wave) * phase_offset)
            normalized = total / np.float32(waves) * np.float32(0.5) + np.float32(0.5)
            luminance = img[y, x] / np.float32(255.0)
            out[y, x] = np.float32(255.0) if normalized >= np.float32(1.0) - luminance else np.float32(0.0)
    return out


def _scalar_parameter(parameter, default):
    if isinstance(parameter, (tuple, list)):
        return parameter[0] if parameter else default
    return parameter if parameter else default


@registry.register("Hilbert (Riemersma)", "Error Diffusion", dims=2,
                   param_sliders=("dither_parameter_slider",))
def hilbert_riemersma(image_array, parameter, luminance_threshold_value):
    length = min(32, max(8, int(_scalar_parameter(parameter, 8)) + 8))
    return _hilbert_riemersma(image_array.astype(np.float32),
                              np.float32(luminance_threshold_value), length)


@registry.register("Spiral Path", "Error Diffusion", dims=2,
                   param_sliders=("dither_parameter_slider",))
def spiral_path(image_array, parameter, luminance_threshold_value):
    retention = min(95, max(20, int(_scalar_parameter(parameter, 70)))) / 100.0
    return _spiral_path(image_array.astype(np.float32),
                        np.float32(luminance_threshold_value), np.float32(retention))


@registry.register("Flow Hatch", "Patterned", dims=2,
                   param_sliders=("dither_parameter_slider",))
def flow_hatch(image_array, parameter, luminance_threshold_value):
    spacing = min(24, max(2, int(_scalar_parameter(parameter, 6))))
    return _flow_hatch(image_array.astype(np.float32),
                       np.float32(luminance_threshold_value), np.float32(spacing))


@registry.register("Hex Bayer", "Ordered Dither", dims=2,
                   param_sliders=("dither_parameter_slider",))
def hex_bayer(image_array, parameter, luminance_threshold_value):
    cell = min(16, max(1, int(_scalar_parameter(parameter, 3))))
    return _hex_bayer(image_array.astype(np.float32), np.float32(cell))


@registry.register("Triangular", "Ordered Dither", dims=2,
                   param_sliders=("dither_parameter_slider",))
def triangular(image_array, parameter, luminance_threshold_value):
    cell = min(16, max(1, int(_scalar_parameter(parameter, 3))))
    return _triangular(image_array.astype(np.float32), np.float32(cell))


@registry.register("Spiral Engrave", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def spiral_engrave(image_array, parameter, luminance_threshold_value):
    pitch = min(24, max(2, int(_scalar_parameter(parameter, 6))))
    return _spiral_engrave(image_array.astype(np.float32),
                           np.float32(luminance_threshold_value), np.float32(pitch))


@registry.register("Reaction-Diffusion", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def reaction_diffusion(image_array, parameter, luminance_threshold_value):
    iterations = min(60, max(10, int(_scalar_parameter(parameter, 20)) * 3))
    return _reaction_diffusion(image_array.astype(np.float32),
                               np.float32(luminance_threshold_value), iterations)


@registry.register("Quasicrystal", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def quasicrystal(image_array, parameter, luminance_threshold_value):
    waves = min(7, max(3, int(_scalar_parameter(parameter, 5))))
    return _quasicrystal(image_array.astype(np.float32),
                         np.float32(luminance_threshold_value), waves)
