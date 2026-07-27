# Native rewrite baseline — 2026-07-27

This baseline was captured after repairing `heavy_effects()` to pass Epsilon
Glow's current `intensity` keyword. The benchmark input and production effect
implementation were otherwise unchanged.

## Environment

- Base commit: `ff2d09d0fef39bf16e6fe3c5abd842d3ef7e242b`
- OS: Windows 11 `10.0.26200`
- CPU: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
- Python: CPython 3.12.13 x64 (MSC v.1944)
- NumPy: 2.4.6
- Numba: 0.66.0
- JIT: enabled (no `NUMBA_DISABLE_JIT`)

Commands:

```powershell
C:\Users\arsha\Desktop\custom dither\.venv\Scripts\python.exe -m benchmarks.bench
C:\Users\arsha\Desktop\custom dither\.venv\Scripts\python.exe -m benchmarks.bench_cache
```

Both commands exited successfully. Each reported warm value is a median of five
samples. `benchmarks.bench` performs one untimed warmup before each median.
`benchmarks.bench_cache` primes JIT and cache state before its five-sample
medians. Its per-control measurements re-prime the baseline cache before every
timed changed render.

## Entry-point results

Heavy path (`Floyd-Steinberg` + Game Boy palette + Chromatic Aberration +
Epsilon Glow):

| Size | Warm median |
|---|---:|
| 512×512 | 82.2 ms |
| 1080p | 690.8 ms |

The 1080p heavy-path stage breakdown was: effects 574.55 ms, color 79.23 ms,
dither 11.26 ms, midtones 5.77 ms, saturation 3.91 ms, contrast 3.08 ms,
highlights 0.71 ms, and blur 0.01 ms.

Cached-render entry point:

| Scenario | Median |
|---|---:|
| 1080p full uncached | 698.6 ms |
| 1080p saturation change | 602.6 ms |
| 1080p luminance-threshold change | 677.4 ms |
| 1080p contrast change | 679.0 ms |
| 1080p invert change | 22.4 ms |
| 1080p effects-parameter change | 659.3 ms |
| 1080p preview proxy (`max_side=640`) | 80.9 ms |
| 4K preview proxy (`max_side=640`) | 115.7 ms |

For comparison, the full renders paired with the preview proxy were 709.9 ms at
1080p and 2899.8 ms at 4K.

## Supplemental distribution sample

The entry points intentionally print medians only. To freeze tail latency for
the native-rewrite baseline, the same callables were also measured with one
untimed warmup and ten timed samples using `time.perf_counter()`. p95 is
`numpy.percentile(samples, 95)`.

| Scenario | Samples | Warmup | Median | p95 |
|---|---:|---:|---:|---:|
| Heavy render, 512×512 | 10 | 1 | 99.1 ms | 103.6 ms |
| Heavy render, 1080p | 10 | 1 | 773.9 ms | 805.4 ms |
| Full render, 1080p cache baseline | 10 | 1 | 723.1 ms | 839.6 ms |
| Effects-only cached tick, 1080p | 10 | 1 | 641.3 ms | 711.6 ms |

The first-call “cold” columns in `benchmarks.bench` are not clean-cache compiler
measurements because Numba's on-disk cache persists across processes; the
benchmark itself prints this limitation. Warm medians and the supplemental
distribution are the reproducible comparison baseline.
