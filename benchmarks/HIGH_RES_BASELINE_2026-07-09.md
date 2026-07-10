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
```

`first ms` is the first invocation in the current process; persistent Numba
cache files may mean it is not a true compiler-cold result. `warm ms` is the
median of repeated invocations. `py peak MB` comes from `tracemalloc`; `rss dMB`
is the process working-set delta when optional `psutil` is installed. Neither is
an exact retained-cache measurement, so focused memory investigations must also
enumerate retained arrays by `nbytes`.
