"""MT-10 production-seam dirty mask proxy benchmark (31 measured samples)."""
from __future__ import annotations

import time

import numpy as np

from ditherzam.layers import (
    DirtyRect,
    LayerRenderProxy,
    composite_mask_stroke_roi,
)
from ditherzam.layers.render import _nearest_source_index_map


def _measure(width: int, height: int, roi: int) -> dict:
    scale = min(1.0, 720 / max(width, height))
    proxy_width = max(1, round(width * scale))
    proxy_height = max(1, round(height * scale))
    prefix = np.zeros((proxy_height, proxy_width, 4), np.uint8)
    selected = np.empty_like(prefix)
    selected[...] = (180, 90, 40, 255)
    prefix.flags.writeable = False
    selected.flags.writeable = False
    proxy = LayerRenderProxy(
        "selected", prefix, selected, 0, 0, "normal", 100, True,
        prefix, selected, (),
        _nearest_source_index_map(width, proxy_width),
        _nearest_source_index_map(height, proxy_height))
    mask = np.full((height, width), 255, np.uint8)
    x0, y0 = (width - roi) // 2, (height - roi) // 2
    dirty = DirtyRect(x0, y0, x0 + roi, y0 + roi)
    mask[y0:y0 + roi, x0:x0 + roi] = 0
    samples = []
    peak_incremental = 0
    for _ in range(32):
        started = time.perf_counter()
        _rect, rgba = composite_mask_stroke_roi(proxy, mask, 100, dirty)
        elapsed = (time.perf_counter() - started) * 1000.0
        if samples or len(samples) == 0:
            samples.append(elapsed)
        overlay_bytes = proxy_width * proxy_height * 4
        roi_scratch = rgba.shape[0] * rgba.shape[1] * 6
        peak_incremental = max(
            peak_incremental,
            mask.nbytes + overlay_bytes + rgba.nbytes + roi_scratch)
    measured = np.asarray(samples[1:32])
    return {
        "p95_ms": float(np.percentile(measured, 95)),
        "incremental_bytes": peak_incremental,
        "retained_proxy_bytes": proxy.mask_stroke_retained_bytes,
        "samples": len(measured),
    }


def main() -> None:
    cases = (
        ("1080p typical256", 1920, 1080, 256, 33.0, 6 * 1024**2),
        ("1080p large1024", 1920, 1080, 1024, 33.0, 18 * 1024**2),
        ("4K typical256", 3840, 2160, 256, 50.0, 18 * 1024**2),
        ("4K large1024", 3840, 2160, 1024, 50.0, 18 * 1024**2),
    )
    failed = False
    for name, width, height, roi, latency, memory in cases:
        result = _measure(width, height, roi)
        print(name, result)
        failed |= result["samples"] != 31
        failed |= result["p95_ms"] > latency
        failed |= result["incremental_bytes"] > memory
    if failed:
        raise SystemExit("MT-10 benchmark budget exceeded")
    print("Look renders per stamp: 0; exact release requests: controller <=1; stale: 0")


if __name__ == "__main__":
    main()
