# Native integration report — 2026-07-27

Environment: Windows 11 `10.0.26200`, CPython 3.12.13 x64, NumPy 2.4.6,
Cython 3.1.6, PyInstaller 6.21.0, 8 logical CPUs. Inputs were deterministic
3840×2160 RGBA uint8 arrays (seed `20260727`), one warm call and five measured
samples. Command: `python -m benchmarks.native_thread_scaling --samples 5`.

| Threads | Overlay median / p95 / speedup | Transition median / p95 / speedup |
|---:|---:|---:|
| 1 | 190.322 / 223.315 ms / 1.000× | 147.996 / 164.637 ms / 1.000× |
| 2 | 115.103 / 153.761 ms / 1.653× | 90.120 / 105.114 ms / 1.642× |
| 4 | 67.550 / 73.184 ms / 2.817× | 63.746 / 77.836 ms / 2.322× |
| 8 | 67.114 / 67.525 ms / 2.836× | 62.534 / 69.722 ms / 2.367× |

Every overlay result had SHA-256
`4ffb53e49e240e52568702157933ae4337b100c9d6a82afe4253cdcc48fff111`;
every transition result had
`d71fad5cb48034f80e8d30bc504372b7934da457ca44bc2b3ffed7fdf52fd8d9`.

Two simultaneous overlay calls had concurrent pair median/p95 results of
242.453/258.470, 138.840/166.377, 113.725/169.520, and 107.183/123.573 ms at
1, 2, 4, and 8 threads per call, or 1.000×/1.746×/2.132×/2.262× versus the
one-thread concurrent median. Both outputs in every pair matched the overlay
digest above. The default remains two: it preserves the independently accepted
performance baseline while bounding nested render/export workers, and higher
counts show diminishing or variable tail gains. Applications may set 1/2/4/8
as process-wide configuration; each operation snapshots the value before its
nogil loop. No save/restore context is exposed. This policy does not alter
Numba. Selection and brush remain single-threaded.

Focused enabled-native integration: 290 passed. A separate forced-fallback
public-wrapper run passed 62 tests with `DITHERZAM_DISABLE_NATIVE=1`. The clean
wheel smoke imported and exercised every module without a compiler.

The standard frozen artifact built successfully and
`ditherzam.exe --native-smoke` exited zero, executing all four accepted native
areas. Smart Mask remains correctly disabled without approved locked model
assets; its licensed release build was not attempted.

Artifacts from the verification workspace:

- wheel SHA-256:
  `109840384c385929207a553b3a8540666c5d89f2b02d28bc170a78b08396acd3`
- standard ZIP SHA-256:
  `26f0fa6e2620b4e8d18717a12ce9637190f60c7ab34c2937925439a31e725646`
- frozen executable SHA-256:
  `9be969140d79938903bfe0d327c92b05de06540a058eb3d9a048238b227a4606`
