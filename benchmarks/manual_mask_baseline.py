"""Repeatable Phase-0 alpha/composite/thumbnail baseline."""

import os
import json
import platform
import statistics
import time
import tracemalloc

import numba
import numpy as np
import PIL
from PIL import Image

from ditherzam.layers.blend import blend_layer
from ditherzam.layers.model import RasterLayerMask
from ditherzam.layers.render import _apply_raster_mask
from ditherzam.layers.render_cache import LayerLookRenderCache
from ditherzam.layers.mask_contracts import (
    EXACT_MASK_COMPOSITE_1080P_TARGET_P95_MS,
    EXACT_MASK_COMPOSITE_4K_TARGET_P95_MS,
    masked_alpha,
    LookInputsKey,
    RenderedLookKey,
    source_mask_bytes,
)


def _measure(callable_, repeats, *, warmups=1):
    for _ in range(warmups):
        callable_()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": statistics.quantiles(
            samples, n=100, method="inclusive")[94],
    }


def _case(width, height):
    alpha = np.arange(width * height, dtype=np.uint8).reshape(height, width)
    mask = np.flipud(alpha)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3], rgba[..., 3] = 127, alpha
    canvas = np.zeros_like(rgba)

    def alpha_composite():
        effective = 255 - (
            100 * (255 - mask.astype(np.uint16)) + 50) // 100
        out = rgba.copy()
        out[..., 3] = (
            alpha.astype(np.uint16) * effective + 127) // 255
        return out

    masked = alpha_composite()
    raster_mask = RasterLayerMask(mask, density=73)
    production_call = lambda: _apply_raster_mask(
        rgba, raster_mask, (height, width))
    production_timing = _measure(production_call, 31, warmups=1)
    tracemalloc.start()
    production_call()
    _current, production_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert masked_alpha(127, 127, 100) == 63
    return {
        "mask_bytes": source_mask_bytes(width, height),
        "alpha": _measure(alpha_composite, 31),
        "mt03_mask_composite": production_timing,
        "mt03_tracemalloc_peak_bytes": production_peak,
        "mt03_density": raster_mask.density,
        "composite": _measure(
            lambda: blend_layer(canvas, masked, "normal", 100), 5),
        "thumbnail": _measure(
            lambda: Image.fromarray(masked, "RGBA").resize(
                (56, 56), Image.Resampling.NEAREST), 31),
        "working_mask_bytes": mask.nbytes,
        "incremental_stroke_floor_bytes": mask.nbytes,
    }


def _cache_paths(width, height):
    rgba = np.full((height, width, 4), 127, np.uint8)
    masks = {
        "empty": RasterLayerMask(np.zeros((height, width), np.uint8)),
        "full": RasterLayerMask(np.full((height, width), 255, np.uint8)),
        "general": RasterLayerMask(
            np.arange(width * height, dtype=np.uint8).reshape(height, width),
            density=73),
    }
    key = RenderedLookKey(
        ("benchmark-source", width, height),
        "benchmark-look",
        (height, width),
        LookInputsKey("smart-off", "benchmark-v1"),
    )
    results = {}
    for name, mask in masks.items():
        renders = 0

        def produce():
            nonlocal renders
            renders += 1
            return rgba

        cold_samples = []
        for _ in range(31):
            cache = LayerLookRenderCache(rgba.nbytes + 1024)
            start = time.perf_counter()
            completed = cache.get_or_render(key, produce)
            _apply_raster_mask(completed, mask, (height, width))
            cold_samples.append((time.perf_counter() - start) * 1000)
        cold_renders = renders
        warm_cache = LayerLookRenderCache(rgba.nbytes + 1024)
        completed = warm_cache.get_or_render(key, produce)
        renders_before_warm = renders
        warm = _measure(
            lambda: _apply_raster_mask(
                warm_cache.get_or_render(key, produce), mask, (height, width)),
            31,
            warmups=0,
        )
        results[name] = {
            "cold_median_ms": statistics.median(cold_samples),
            "cold_p95_ms": statistics.quantiles(
                cold_samples, n=100, method="inclusive")[94],
            "warm_median_ms": warm["median_ms"],
            "warm_p95_ms": warm["p95_ms"],
            "cold_look_renders": cold_renders,
            "mask_only_warm_look_renders": renders - renders_before_warm,
        }

    thumb_rgba = rgba[:56, :56]
    thumb_mask = RasterLayerMask(
        masks["general"].pixels[:56, :56].copy(), density=73)
    thumb_key = RenderedLookKey(
        ("benchmark-thumb", width, height), "benchmark-look", (56, 56),
        LookInputsKey("smart-off", "benchmark-v1"))
    thumb_cache = LayerLookRenderCache(thumb_rgba.nbytes + 1024)
    thumb_renders = 0

    def thumb_produce():
        nonlocal thumb_renders
        thumb_renders += 1
        return thumb_rgba

    thumb_cold_samples = []
    for _ in range(31):
        cold_cache = LayerLookRenderCache(thumb_rgba.nbytes + 1024)
        start = time.perf_counter()
        thumb_value = _apply_raster_mask(
            cold_cache.get_or_render(thumb_key, thumb_produce),
            thumb_mask,
            (56, 56),
        )
        Image.fromarray(thumb_value, "RGBA").copy()
        thumb_cold_samples.append((time.perf_counter() - start) * 1000)
    thumb_cold_renders = thumb_renders
    thumb_cache.get_or_render(thumb_key, thumb_produce)
    before = thumb_renders
    thumb = _measure(
        lambda: Image.fromarray(
            _apply_raster_mask(
                thumb_cache.get_or_render(thumb_key, thumb_produce),
                thumb_mask,
                (56, 56)),
            "RGBA",
        ).copy(),
        31,
        warmups=0,
    )
    results["thumbnail_warm"] = {
        **thumb,
        "mask_only_look_renders": thumb_renders - before,
        "cold_median_ms": statistics.median(thumb_cold_samples),
        "cold_p95_ms": statistics.quantiles(
            thumb_cold_samples, n=100, method="inclusive")[94],
        "cold_look_renders": thumb_cold_renders,
    }
    return results


def main():
    report = {"metadata": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "numba": numba.__version__,
        "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
        "samples": 31,
        "warmups": 1,
        "tracemalloc": "separate untimed production call",
        "clock": "time.perf_counter",
    }, "cases": {}}
    all_pass = True
    for label, shape in (("1080p", (1920, 1080)), ("4K", (3840, 2160))):
        result = _case(*shape)
        result["mt04_cache_paths"] = _cache_paths(*shape)
        target = (
            EXACT_MASK_COMPOSITE_1080P_TARGET_P95_MS
            if label == "1080p"
            else EXACT_MASK_COMPOSITE_4K_TARGET_P95_MS
        )
        result["mt03_target_ms"] = target
        result["mt03_meets_target"] = (
            result["mt03_mask_composite"]["p95_ms"] <= target)
        all_pass = all_pass and result["mt03_meets_target"]
        report["cases"][label] = result
    report["overall_pass"] = all_pass
    print(json.dumps(report, sort_keys=True))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
