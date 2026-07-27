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
from ditherzam.threading_policy import native_threads


def _measure(call, samples: int) -> tuple[float, str]:
    output = call()
    digest = hashlib.sha256(output.tobytes()).hexdigest()
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        result = call()
        times.append((time.perf_counter() - start) * 1000)
        assert hashlib.sha256(result.tobytes()).hexdigest() == digest
    return statistics.median(times), digest


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
    for threads in (1, 2, 4, 8):
        with native_threads(threads):
            for name, call in operations.items():
                median, digest = _measure(call, args.samples)
                rows.append({"threads": threads, "operation": name,
                             "median_ms": round(median, 3), "sha256": digest})

    concurrency = []
    for threads in (1, 2, 4, 8):
        with native_threads(threads):
            call = operations["compositor_overlay"]
            def timed_ms(fn):
                values = []
                for _ in range(args.samples):
                    start = time.perf_counter()
                    fn()
                    values.append((time.perf_counter() - start) * 1000)
                return statistics.median(values)
            sequential = timed_ms(lambda: (call(), call()))
            def pair():
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(lambda _: call(), range(2)))
            concurrent = timed_ms(pair)
            concurrency.append({"native_threads_each": threads,
                                "sequential_pair_median_ms": round(sequential, 3),
                                "concurrent_pair_median_ms": round(concurrent, 3)})
    print(json.dumps({
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "cpu_count": __import__("os").cpu_count(), "shape": shape,
                        "samples": args.samples},
        "scaling": rows, "concurrency": concurrency,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
