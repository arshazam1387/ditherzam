# Native integration report — 2026-07-27

Environment: Windows 11 `10.0.26200`, CPython 3.12.13 x64, NumPy 2.4.6,
Cython 3.1.6, PyInstaller 6.21.0, 8 logical CPUs. Inputs were deterministic
3840×2160 RGBA uint8 arrays (seed `20260727`), one warm call and five measured
samples. Command: `python -m benchmarks.native_thread_scaling --samples 5`.

| Threads | Overlay median | Scalar transition median |
|---:|---:|---:|
| 1 | 201.714 ms | 153.645 ms |
| 2 | 100.886 ms | 87.957 ms |
| 4 | 74.286 ms | 69.899 ms |
| 8 | 52.920 ms | 45.139 ms |

Every overlay result had SHA-256
`4ffb53e49e240e52568702157933ae4337b100c9d6a82afe4253cdcc48fff111`;
every transition result had
`d71fad5cb48034f80e8d30bc504372b7934da457ca44bc2b3ffed7fdf52fd8d9`.

Two simultaneous overlay calls had concurrent pair medians of 239.776, 142.614,
113.484, and 124.561 ms at 1, 2, 4, and 8 threads per call respectively. Eight
threads regressed when nested (16 requested workers on 8 logical CPUs). The
default remains two: it preserves the independently accepted performance
baseline, leaves headroom for concurrent render/export work, and is nearly twice
as fast as one thread. Applications may temporarily select 1/2/4/8 through the
explicit native policy; this does not alter the Numba policy. Selection and
brush remain single-threaded.

Focused enabled-native integration: 289 passed. A separate forced-fallback
public-wrapper run passed 62 tests with `DITHERZAM_DISABLE_NATIVE=1`. The clean
wheel smoke imported and exercised every module without a compiler.

The standard frozen artifact built successfully and
`ditherzam.exe --native-smoke` exited zero, executing all four accepted native
areas. Smart Mask remains correctly disabled without approved locked model
assets; its licensed release build was not attempted.

Artifacts from the verification workspace:

- wheel SHA-256:
  `832dd19a897501c992ae4665e0f6618b0e009e7c62a9f38912b7fb6750eda701`
- standard ZIP SHA-256:
  `42d2911b5d0e978aaa5881e022814bf5706d7e5e21f6edc1f6be0f172eaee386`
- frozen executable SHA-256:
  `4494dce49eec05b620d5cb7329c4163ddab38fd01306ec031e4657acda77be0b`
