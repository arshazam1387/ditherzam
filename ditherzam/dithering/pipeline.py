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


def apply_dither(gray_f32, *, style, scale, luminance_threshold,
                 params, registry, preview_disabled=False,
                 threshold_field=None) -> np.ndarray:
    entry = registry.get_entry(style)
    if style == "None" or entry is None or preview_disabled:
        return gray_f32

    tval = _luminance_to_255(luminance_threshold)
    factor = max(1, int(scale))
    h, w = gray_f32.shape[:2]

    small = nearest_downscale(gray_f32, factor)
    param = _build_param(entry, params)
    out = entry.func(small, param, tval)

    # threshold_field is accepted for the frozen contract; Phase 8 (temporal) wires it.
    return nearest_upscale_to(out, (w, h))
