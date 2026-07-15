from __future__ import annotations
import numpy as np
from ..imaging import nearest_downscale, nearest_upscale_to
from .parameters import parameter_specs


def _luminance_to_255(luminance_threshold: float) -> float:
    return float(luminance_threshold / 100.0 * 255.0)


def _build_param(entry, params: dict):
    # Spec §8.2 step 3: param_func takes precedence over param_sliders extraction.
    if entry.param_func is not None:
        return entry.param_func(params)
    present = any(name in params for name in entry.param_sliders)
    if not present:
        return 0
    if all(name in params for name in entry.param_sliders):
        vals = [params[name] for name in entry.param_sliders]
        return vals[0] if len(vals) == 1 else tuple(vals)
    defaults = {spec.key: spec.default for spec in parameter_specs(entry)}
    vals = [params.get(name, defaults[name]) for name in entry.param_sliders]
    if len(vals) <= 1:
        return vals[0] if vals else 0
    return tuple(vals)


def _creative_int(params: dict, key: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(round(float(params.get(key, default))))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _creative_input(small: np.ndarray, params: dict) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Apply deterministic art-direction transforms before the style kernel.

    The all-default path returns the original object, preserving historical
    pixels and avoiding allocations. Orientation and offsets transform the
    pattern coordinate system; jitter perturbs threshold input deterministically.
    """
    turns = _creative_int(params, "creative_orientation", 0, 0, 3)
    ox = _creative_int(params, "creative_offset_x", 0, -64, 64)
    oy = _creative_int(params, "creative_offset_y", 0, -64, 64)
    jitter = _creative_int(params, "creative_jitter", 0, 0, 100)
    seed = _creative_int(params, "creative_seed", 0, 0, 999)
    work = small
    if turns:
        work = np.rot90(work, turns)
    if ox or oy:
        work = np.roll(work, shift=(oy, ox), axis=(0, 1))
    if jitter:
        h, w = work.shape[:2]
        yy, xx = np.indices((h, w), dtype=np.uint32)
        hashed = (xx * np.uint32(374761393) + yy * np.uint32(668265263) +
                  np.uint32(seed * 2246822519 & 0xFFFFFFFF))
        hashed = (hashed ^ (hashed >> np.uint32(13))) * np.uint32(1274126177)
        noise = ((hashed & np.uint32(0xFFFF)).astype(np.float32) /
                 np.float32(65535.0) - np.float32(0.5))
        amplitude = np.float32(jitter) * np.float32(1.275)
        work = (work + noise * amplitude).astype(np.float32)
    return work, (turns, ox, oy)


def _creative_output(out: np.ndarray, original: np.ndarray, params: dict,
                     transform: tuple[int, int, int]) -> np.ndarray:
    turns, ox, oy = transform
    if ox or oy:
        out = np.roll(out, shift=(-oy, -ox), axis=(0, 1))
    if turns:
        out = np.rot90(out, -turns)
    mix = _creative_int(params, "creative_mix", 100, 0, 100)
    if mix == 100:
        return out
    if mix == 0:
        return original
    alpha = np.float32(mix / 100.0)
    return (np.asarray(original, np.float32) * (np.float32(1.0) - alpha) +
            np.asarray(out, np.float32) * alpha).astype(np.float32)


def _resize_field_nearest(field: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Float-safe nearest-neighbour resize that preserves negative values."""
    th, tw = int(target_hw[0]), int(target_hw[1])
    fh, fw = field.shape[:2]
    if (fh, fw) == (th, tw):
        return field
    ys = np.minimum((np.arange(th) * fh) // max(th, 1), fh - 1)
    xs = np.minimum((np.arange(tw) * fw) // max(tw, 1), fw - 1)
    return field[ys][:, xs]


def _binary_to_levels(func, small, param, tval, levels) -> np.ndarray:
    """Promote a binary (0/255) threshold kernel to ``levels`` tones.

    Most pattern/screen/threshold kernels only ever emit black or white, so a
    colour palette applied afterward collapses to two colours no matter the
    depth. Here the tonal range is split into ``levels-1`` bands and the kernel
    dithers the in-band fraction: ``out = (floor(v/step) + kernel_bit(frac)) *
    step``. Each kernel keeps its own texture while building up a smooth
    multi-tone gradient the palette can span. Bit-identical kernels are treated
    as black boxes, so this needs no per-kernel change.
    """
    step = 255.0 / (levels - 1)
    v = np.clip(np.asarray(small, dtype=np.float32), 0.0, 255.0)
    q = v / step
    lower = np.floor(q)
    frac = (q - lower).astype(np.float32)             # 0..1 position within band
    bit_img = np.asarray(func((frac * 255.0).astype(np.float32), param, tval))
    bit = (bit_img >= 128.0).astype(np.float32)       # kernel emits 0/255
    out = (lower + bit) * step
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def apply_dither(gray_f32, *, style, scale, luminance_threshold,
                 params, registry, preview_disabled=False,
                 threshold_field=None, levels=2) -> np.ndarray:
    entry = registry.get_entry(style)
    if style == "None" or entry is None or preview_disabled:
        return gray_f32

    tval = _luminance_to_255(luminance_threshold)
    factor = max(1, int(scale))
    h, w = gray_f32.shape[:2]

    small = nearest_downscale(gray_f32, factor)
    original_small = small

    if threshold_field is not None:
        fld = _resize_field_nearest(
            np.asarray(threshold_field, dtype=np.float32), small.shape[:2])
        # per-pixel threshold tval + fld  <=>  compare (small - fld) against tval
        small = (small - fld).astype(np.float32)
        original_small = small

    small, creative_transform = _creative_input(small, params)

    param = _build_param(entry, params)
    lv = int(levels)
    if entry.supports_levels:
        out = entry.func(small, param, tval, lv)
    elif lv > 2:
        # Kernel only does binary thresholding; promote it to multi-tone so a
        # colour palette can span its full range instead of showing 1-2 colours.
        out = _binary_to_levels(entry.func, small, param, tval, lv)
    else:
        out = entry.func(small, param, tval)

    out = _creative_output(out, original_small, params, creative_transform)
    return nearest_upscale_to(out, (w, h))
