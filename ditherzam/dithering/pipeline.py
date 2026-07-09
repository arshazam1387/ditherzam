from __future__ import annotations
import numpy as np
from ..imaging import nearest_downscale, nearest_upscale_to


def _luminance_to_255(luminance_threshold: float) -> float:
    return float(luminance_threshold / 100.0 * 255.0)


def _build_param(entry, params: dict):
    # Spec §8.2 step 3: param_func takes precedence over param_sliders extraction.
    if entry.param_func is not None:
        return entry.param_func(params)
    vals = [params[name] for name in entry.param_sliders if name in params]
    if len(vals) <= 1:
        return vals[0] if vals else 0
    return tuple(vals)


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

    if threshold_field is not None:
        fld = _resize_field_nearest(
            np.asarray(threshold_field, dtype=np.float32), small.shape[:2])
        # per-pixel threshold tval + fld  <=>  compare (small - fld) against tval
        small = (small - fld).astype(np.float32)

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

    return nearest_upscale_to(out, (w, h))
