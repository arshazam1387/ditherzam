"""Qt-free interactive preview proxy."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .imaging import nearest_upscale_to
from .masking.render import _mask_identity, render_with_mask


PREVIEW_RESOLUTIONS = ("Auto", "480", "720", "1080", "1440", "2160", "Full")
_NUMERIC_RESOLUTION_LABELS = frozenset(PREVIEW_RESOLUTIONS[1:-1])
_NUMERIC_RESOLUTIONS = (480, 720, 1080, 1440, 2160)
_AUTO_BUCKETS = (720, 1080, 1440)


def normalize_preview_resolution(value) -> str:
    if isinstance(value, bool):
        return "Auto"
    if isinstance(value, int):
        candidate = str(value)
    elif isinstance(value, float):
        if not value.is_integer():
            return "Auto"
        candidate = str(int(value))
    elif isinstance(value, str):
        candidate = value.strip()
    else:
        return "Auto"
    lowered = candidate.casefold()
    if lowered == "auto":
        return "Auto"
    if lowered == "full":
        return "Full"
    return candidate if candidate in _NUMERIC_RESOLUTION_LABELS else "Auto"


def preview_cap(resolution, source_longest: int, auto_cap: int = 1440) -> int:
    source_longest = max(1, int(source_longest))
    normalized = normalize_preview_resolution(resolution)
    if normalized == "Full":
        return source_longest
    cap = int(auto_cap) if normalized == "Auto" else int(normalized)
    return min(source_longest, max(1, cap))


def preview_target_size(h: int, w: int, max_side: int) -> tuple[int, int]:
    h, w = max(1, int(h)), max(1, int(w))
    longest = max(h, w)
    cap = max(1, int(max_side))
    if longest <= cap:
        return h, w
    ratio = cap / float(longest)
    return max(1, int(round(h * ratio))), max(1, int(round(w * ratio)))


def resize_preview_bucket(required_pixels: float) -> int:
    required = max(0.0, float(required_pixels))
    for bucket in _AUTO_BUCKETS:
        if required <= bucket:
            return bucket
    return _AUTO_BUCKETS[-1]


def auto_preview_resolution(source_hw: tuple[int, int], viewport_wh: tuple[int, int],
                            device_pixel_ratio: float = 1.0) -> int:
    h, w = (max(1, int(v)) for v in source_hw)
    viewport_w, viewport_h = (max(1, int(v)) for v in viewport_wh)
    fit = min(viewport_w / float(w), viewport_h / float(h))
    fitted_longest = max(h * fit, w * fit) * max(0.01, float(device_pixel_ratio))
    return min(max(h, w), resize_preview_bucket(fitted_longest))


def zoom_preview_bucket(current: int, required_pixels: float, ceiling: int,
                        source_longest: int) -> int:
    current = max(1, int(current))
    limit = min(max(1, int(ceiling)), max(1, int(source_longest)))
    if required_pixels <= current or current >= limit:
        return min(current, limit)
    candidates = (*_NUMERIC_RESOLUTIONS, limit)
    target = next((value for value in candidates
                   if value > current and value >= required_pixels), limit)
    return min(target, limit)


def proxy_factor(h: int, w: int, max_side: int) -> int:
    longest = max(int(h), int(w))
    if longest <= max_side:
        return 1
    return int(math.ceil(longest / float(max_side)))


def proxy_scale(scale: int, factor: int) -> int:
    return max(1, int(round(int(scale) / float(factor))))


def render_preview(pipeline, base_gray, settings, max_side: int,
                   is_cancelled=None, temporal_field=None,
                   mask_context=None, mask_caches=None,
                   rendered_identity=None) -> np.ndarray:
    h, w = base_gray.shape[:2]
    factor = proxy_factor(h, w, max_side)
    target_shape = (h, w) if factor <= 1 else preview_target_size(h, w, max_side)

    def baked_cache_key():
        return ("mask-proxy-baked", rendered_identity, target_shape,
                _mask_identity(mask_context), mask_context.settings.outside)

    def render_complete_branch(bake=None) -> np.ndarray:
        if factor <= 1:
            base = base_gray if bake is None else bake(base_gray)
            if mask_context is not None and temporal_field is None:
                key = baked_cache_key() if bake is not None else (
                    "mask-proxy", rendered_identity, target_shape)
                return pipeline.render_cached(
                    base, settings, is_cancelled=is_cancelled, cache_key=key)
            return pipeline.render(base, settings, temporal_field=temporal_field,
                                   is_cancelled=is_cancelled)
        target_h, target_w = target_shape
        small = nearest_upscale_to(base_gray, (target_w, target_h))
        psettings = replace(settings, scale=proxy_scale(settings.scale, factor))
        if bake is not None and temporal_field is None:
            rgb_small = pipeline.render_cached(
                bake(small), psettings, is_cancelled=is_cancelled,
                cache_key=baked_cache_key())
        elif bake is not None:
            rgb_small = pipeline.render(
                nearest_upscale_to(bake(base_gray), (target_w, target_h)),
                psettings, temporal_field=temporal_field, is_cancelled=is_cancelled)
        elif mask_context is not None and temporal_field is None:
            rgb_small = pipeline.render_cached(
                small, psettings, is_cancelled=is_cancelled,
                cache_key=("mask-proxy", rendered_identity, target_shape))
        else:
            rgb_small = pipeline.render(small, psettings, temporal_field=temporal_field,
                                        is_cancelled=is_cancelled)
        return np.asarray(rgb_small, dtype=np.uint8)

    return render_with_mask(
        render_complete_branch, mask_context, caches=mask_caches,
        rendered_identity=rendered_identity, is_cancelled=is_cancelled,
        target_shape=target_shape)
