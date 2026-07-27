"""Reproducible OpenMP scaling/digest and concurrent-render benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ditherzam.composition.transitions import _blend_straight_rgba_native
from ditherzam.layers.blend import _blend_layer_native
from ditherzam.threading_policy import set_native_threads


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(np.ceil(percentile * len(ordered))) - 1)]


def _measure(call, samples: int) -> tuple[float, float, str]:
    output = call()
    digest = _digest(output)
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        result = call()
        times.append((time.perf_counter() - start) * 1000)
        assert _digest(result) == digest
    return statistics.median(times), _percentile(times, 0.95), digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--width", type=int, default=3840)
    args = parser.parse_args()
    rng = np.random.default_rng(20260727)
    shape = (args.height, args.width, 4)
    first = rng.integers(0, 256, shape, dtype=np.uint8)
    second = rng.integers(0, 256, shape, dtype=np.uint8)
    operations = {
        "compositor_overlay": lambda: _blend_layer_native(first, second, "overlay", 73),
        "transition_scalar": lambda: _blend_straight_rgba_native(first, second, 0.37),
    }
    rows = []
    baselines = {}
    for threads in (1, 2, 4, 8):
        set_native_threads(threads)
        for name, call in operations.items():
            median, p95, digest = _measure(call, args.samples)
            if threads == 1:
                baselines[name] = median
            rows.append({"threads": threads, "operation": name,
                         "median_ms": round(median, 3),
                         "p95_ms": round(p95, 3),
                         "speedup_vs_1t": round(baselines[name] / median, 3),
                         "sha256": digest})

    concurrency = []
    concurrent_baseline = None
    for threads in (1, 2, 4, 8):
        set_native_threads(threads)
        call = operations["compositor_overlay"]
        sequential_times, concurrent_times, pair_digests = [], [], None
        for _ in range(args.samples):
            start = time.perf_counter()
            sequential_results = (call(), call())
            sequential_times.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as pool:
                concurrent_results = tuple(pool.map(lambda _: call(), range(2)))
            concurrent_times.append((time.perf_counter() - start) * 1000)
            digests = tuple(_digest(result) for result in concurrent_results)
            assert digests == tuple(_digest(result) for result in sequential_results)
            pair_digests = digests
        concurrent_median = statistics.median(concurrent_times)
        if concurrent_baseline is None:
            concurrent_baseline = concurrent_median
        concurrency.append({
            "native_threads_each": threads,
            "sequential_pair_median_ms": round(statistics.median(sequential_times), 3),
            "sequential_pair_p95_ms": round(_percentile(sequential_times, .95), 3),
            "concurrent_pair_median_ms": round(concurrent_median, 3),
            "concurrent_pair_p95_ms": round(_percentile(concurrent_times, .95), 3),
            "concurrent_speedup_vs_1t": round(concurrent_baseline / concurrent_median, 3),
            "output_sha256": pair_digests,
        })
    set_native_threads(2)
    print(json.dumps({
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "cpu_count": __import__("os").cpu_count(), "shape": shape,
                        "samples": args.samples},
        "scaling": rows, "concurrency": concurrency,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
