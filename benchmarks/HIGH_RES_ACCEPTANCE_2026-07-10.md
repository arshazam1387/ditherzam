# High-resolution performance — acceptance results (2026-07-10)

Final validation for the High-Resolution Preview & Render Performance program
(Waves 1–4). Measured on an 8-CPU host, JIT enabled unless noted.

Reproduce:

```
.venv/Scripts/python.exe -m benchmarks.high_res --tier default --repeats 2
.venv/Scripts/python.exe -m benchmarks.thread_scaling --tier quick --repeats 3
```

## Acceptance criteria — verdicts with evidence

| Criterion | Target | Measured | Verdict |
|-----------|--------|----------|---------|
| Compiled RGB diffusion speedup | ≥5× vs 5.56 s @480p | 480p warm **18.4 ms** → ~**302×** | ✅ |
| Retained staged cache | ≤192 MiB | **142.4 MiB** after 60 distinct 4K complete-groups; bound never exceeded | ✅ |
| Capped preview allocation | not source-sized | cap480@4K peak **39.9 MB** vs full-4K render **300–459 MB** (scales with cap, not source) | ✅ |
| Interactive preview latency | responsive at cap | 4K warm: cap480 **32 ms**, cap720 **54 ms**, cap1080 **77 ms** | ✅ |
| Cache-hit mutation | near-instant | "unchanged" @4K warm **1.2 ms** | ✅ |
| Exact Full / exports under JIT | byte-exact | exactness suites **172 passed** JIT-on (color-engine, render-cache, tonal fusion, scratch reuse) | ✅ |
| Export isolated from preview state | exact, cap-independent | video/anim/batch/still snapshot dedicated contexts; preview cap never reaches export (Tasks 4.1/4.2 tests) | ✅ |
| No stale paints / worker wedge | one terminal outcome | latest-wins + cooperative cancellation, atomic no-partial-publish (Tasks 3.6/4.1) | ✅ |
| Full regression suite | green | **915 passed / 0 failed** JIT-off | ✅ |

### Known-red (pre-existing, not from this program)
Seven `special.py` float-array-index kernel tests fail under **JIT-on**; they
predate Wave 1 and are unrelated to any high-res change. JIT-off is fully green.

## Exact render — first vs warm (ms) / tracemalloc peak (MB)

| case | size | first | warm | peak MB |
|------|------|------:|-----:|--------:|
| off/k4 | 1080p | 583.4 | 50.7 | 89.6 |
| ramp/k4 | 1080p | 31.9 | 24.8 | 33.7 |
| nearest/k16 | 1080p | 91.6 | 120.7 | 114.7 |
| ordered/k16 | 1080p | 66.6 | 60.4 | 51.4 |
| diffused/k4 | 480p | 27.8 | 18.4 | 19.6 |
| off/k4 | 4K | 232.8 | 268.3 | 300.6 |
| ramp/k4 | 4K | 96.4 | 96.3 | 134.5 |
| nearest/k16 | 4K | 458.6 | 457.8 | 458.8 |
| ordered/k16 | 4K | 241.3 | 323.6 | 205.7 |

## Capped preview — warm ms / peak MB (source-independent)

| cap | @1080p | @4K |
|-----|-------:|----:|
| 480 | 14.6 / 10.3 | 31.7 / 39.9 |
| 720 | 20.2 / 17.2 | 54.1 / 40.4 |
| 1080 | 50.4 / 38.8 | 77.4 / 41.4 |
| 1440 | 70.5 / 69.0 | 105.2 / 69.0 |
| 2160 | 123.7 / 114.7 | 202.0 / 155.2 |

Peak tracks the cap, not the 4K source — a capped preview never allocates a
source-sized buffer.

## Cached mutations @4K (re-primed per sample, warm ms)

| mutation | warm ms |
|----------|--------:|
| unchanged (cache hit) | 1.2 |
| saturation | 22.4 |
| invert | 74.6 |
| palette/mode | 575.0 |
| dither | 864.7 |

## Thread scaling (see THREAD_SCALING_2026-07-10.md)

Ordered scales best (2.57× @8t), nearest/ramp/preview plateau ~1.4× by 2–4t,
diffusion stays flat (sequential by design), effects regress past 2t (GIL-bound).
Policy (Task 4.4): interactive Numba capped at `min(4, cpu)`; async export drops
to `cpu − interactive` reserve; diffusion never parallelized.

## Manual 4K+ QA checklist

Automated coverage above exercises the render/export/cache/threading core
headlessly. The following interactive items require a human at the GUI and are
**pending sign-off** — they cannot be driven from this harness:

- [ ] Load a real ≥4K photo; drag adjustment sliders — preview stays fluid at the
      selected cap, no UI freeze.
- [ ] `Ctrl+Enter` Full Quality Preview renders exact full-res, then a later edit
      returns to the cap.
- [ ] Switch preview resolution (Auto/480…2160/Full) via the View menu — geometry,
      pan, zoom, and 100% semantics hold in source coordinates.
- [ ] Export PNG/JPG, batch folder, SVG, animation MP4, and video MP4 while editing
      mid-export — exported output is unaffected by the concurrent edit.
- [ ] Animation scrub/playback and video display stay responsive; exported frames
      are exact.
- [ ] Observe no stale frame painted after a newer one; app does not wedge after
      heavy use.
