# High-resolution baseline — 2026-07-09

This planning baseline supplements, rather than replaces, the historical results
in `RESULTS.md`. Measurements used deterministic synthetic inputs with JIT enabled
on Windows 11 and CPython 3.12.

## Observed baseline

| case | resolution | observed wall time |
|---|---:|---:|
| color off | 4K | ~615 ms |
| ramp | 4K | ~859 ms |
| nearest | 4K | ~869 ms |
| ordered | 4K | ~1,079 ms |
| RGB diffused | 480p | ~5.56 s |
| unchanged cached render | 4K | effectively free |
| saturation-only cached mutation | 4K | ~412 ms |
| upstream cached mutation | 4K | ~773 ms |
| first drag feedback | current capped proxy | ~67 ms |

The 4K staged cache retained about 182 MiB, excluding source and display copies.
A capped proxy was still nearest-upscaled into a source-sized 4K RGB result,
adding about 24 MiB per displayed result. These figures are investigation
measurements, not acceptance thresholds; rerun the harness before and after each
implementation change on the same machine.

## Reproduction

```powershell
.venv/Scripts/python.exe -m benchmarks.high_res
.venv/Scripts/python.exe -m benchmarks.high_res --tier quick --repeats 1 --section cache
.venv/Scripts/python.exe -m benchmarks.high_res --tier full
.venv/Scripts/python.exe -m benchmarks.high_res --tier full --large-diffused
.venv/Scripts/python.exe -m benchmarks.high_res --tier quick --repeats 1 --section effects
.venv/Scripts/python.exe -m benchmarks.high_res --tier full --section effects
```

## Effects profiling

The `effects` section isolates post-processing from dithering and color mapping.
It runs deterministic RGB `uint8` input through Blur, Sharpen, Chromatic
Aberration, JPEG Glitch, Epsilon Glow, and a stack containing all five effects.
The safe `quick` and `default` tiers cover 1080p. The intentionally expensive
1080p plus 4K matrix is available only with `--tier full`.

Each row reports the first call, every warm timing sample (rather than hiding
variance behind only a median), peak Python-tracked allocation, optional process
working-set delta, and the first 16 hexadecimal digits of the output SHA-256.
Checksums make before/after exactness comparisons reproducible; memory figures
retain the limitations described below. The benchmark also asserts that each
effect preserves the input shape, returns RGB `uint8` output, and does not
mutate its input.

Quick smoke measurements on the baseline machine (`--tier quick --repeats 1`):

| effect | 1080p first | 1080p warm | Python peak | SHA-256/16 |
|---|---:|---:|---:|---|
| Blur | 105.9 ms | 104.9 ms | 17.9 MiB | `7486973db285adf6` |
| Sharpen | 227.9 ms | 200.0 ms | 100.9 MiB | `334479bdcf9195de` |
| Chromatic Aberration | 9.0 ms | 13.8 ms | 13.9 MiB | `25bec0ce738ad021` |
| JPEG Glitch | 96.5 ms | 44.8 ms | 18.8 MiB | `0ff59b0bb62c74c0` |
| Epsilon Glow | 1,461.7 ms | 994.8 ms | 193.8 MiB | `863f8c372dbbb876` |
| all five, stacked | 1,626.9 ms | 1,622.8 ms | 199.7 MiB | `1a2ef0e9ef8fb127` |

These smoke figures identify Epsilon Glow as the dominant individual effect and
Sharpen as the next-largest Python-allocation case. They are evidence for where
to investigate, not proof that a candidate optimization preserves exact output.

`first ms` is the first invocation in the current process; persistent Numba
cache files may mean it is not a true compiler-cold result. `warm ms` is the
median of repeated invocations. `py peak MB` comes from `tracemalloc`; `rss dMB`
is the process working-set delta when optional `psutil` is installed. Neither is
an exact retained-cache measurement, so focused memory investigations must also
enumerate retained arrays by `nbytes`.
