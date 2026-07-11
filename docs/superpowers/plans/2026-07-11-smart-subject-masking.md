# Smart Subject/Background Masking — Implementation Plan

**Date:** 2026-07-11  
**Status:** Proposed — awaiting approval  
**Design:** `docs/superpowers/specs/2026-07-11-smart-subject-masking-design.md`  
**Method:** dependency-ordered TDD; red → green → refactor; one reviewed commit per task  
**Scope:** planning only until this plan and its one-to-one execution ledger are approved

## Global constraints

- Clean-room: no Studio AAA/Dither Boy code, strings, binaries, endpoints, models,
  or copied implementation details.
- No network, telemetry, license endpoint, remote inference, or runtime model download.
- Python 3.12 via `.venv/Scripts/python.exe`.
- Core remains Qt-free. PySide6 imports are limited to `ditherzam/ui/`,
  `ditherzam/app.py`, and `ditherzam/video/workers.py`.
- Preserve the frozen pipeline order: contrast → midtones → highlights → blur →
  dither → color → saturation → effects → invert.
- Smart Mask disabled must be byte-identical and do no mask hashing, resizing,
  compositing, inference, or mask allocation.
- Complete-pipeline masking is an outer compositor; do not crop or mask individual
  stages.
- Preserve Source Colors, Colored Dither, all styles/effects, exact export contexts,
  immutable one-read snapshots, stage-boundary cancellation, worker terminal
  outcomes, preview caps, and the 192 MiB total retained-cache ceiling.
- Preview masks never become export authority; exact export uses source resolution.
- Do not commit weights, generated fixtures, or binaries before license,
  provenance, checksum, size, and `.gitignore` approval.
- Preserve all unrelated dirty-tree work. Branch from the correct eventual base in
  place, never with a worktree, only after the user approves execution.

## Verification conventions

Focused JIT-off command prefix:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:NUMBA_DISABLE_JIT='1'
.venv/Scripts/python.exe -m pytest -q
```

Real-JIT gates remove `NUMBA_DISABLE_JIT`:

```powershell
Remove-Item Env:NUMBA_DISABLE_JIT -ErrorAction SilentlyContinue
.venv/Scripts/python.exe -m pytest -q <targets>
```

Full wave gate:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:NUMBA_DISABLE_JIT='1'
.venv/Scripts/python.exe -m pytest -q
```

Existing unrelated failures, if any, must be demonstrated from the execution
branch base. No new failure is accepted.

## Wave 0 — Legal, provenance, and measurable selection gates

### SM-01 — Offline asset manifest and fail-closed resolver

**Goal / prerequisite:** Establish the release-asset contract before any model is
downloaded or converted. No prerequisite.

**Files and interfaces**

- Add `ditherzam/masking/__init__.py`.
- Add `ditherzam/masking/model_assets.py`:
  `ModelManifest`, `ModelAssetError`, `load_manifest(path)`,
  `verify_model_asset(asset_root, manifest)`.
- Add `assets/models/smart_mask/README.md` and a manifest schema/example only;
  no weights.
- Add developer-only `tools/stage_smart_mask_model.py`.
- Modify `.gitignore` and `THIRD_PARTY_NOTICES.md` only as the policy requires.
- Add `tests/test_mask_model_assets.py` and `tests/test_offline_security.py`.

**Red first:** Tests import the missing module and assert rejection of absent files,
wrong bytes/hash, incomplete metadata, absolute/traversing paths, unapproved model
IDs, tensor-contract mismatch, and arbitrary preset paths. An approved temporary
local asset must verify. Static/import tests prohibit downloader/network modules.
Expected failure: `ModuleNotFoundError: ditherzam.masking`.

**Implementation:** Use a frozen manifest value object, fixed application-owned
asset root, streamed SHA-256, exact byte count, and explicit fields for logical
model/version, upstream repository+commit+URL, source hash, license/attribution,
conversion revision/opset/tool versions, ONNX hash, tensors, preprocessing, output
semantics, and algorithm version. Developer staging may fetch only when explicitly
run; application modules never fetch.

**Verify:** `... pytest -q tests/test_mask_model_assets.py tests/test_offline_security.py`
(JIT-off only). Manually inspect license/NOTICE layout and missing/corrupt messages.

**Acceptance:** Fail-closed local verification works; no weights are committed; no
runtime networking exists; U-2-Net upstream state is pinned to
`ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29`.

**Commit:** `feat(mask): add offline model provenance gate`

### SM-02 — Quality metrics, fixture provenance, and winner policy

**Goal / prerequisite:** Freeze the decision math before observing model results.
Depends on SM-01 manifest terminology, not on real assets.

**Files and interfaces**

- Add `ditherzam/masking/quality.py`:
  `dice_score`, `iou_score`, `boundary_f_score`, `aggregate_quality`,
  `select_model_candidate`.
- Add `tests/fixtures/smart_mask/README.md` and provenance-manifest format only.
- Add `tests/test_mask_quality_metrics.py`.
- Add `benchmarks/smart_mask.py` CLI skeleton and report schema.

**Red first:** Synthetic masks test exact empty/full/partial metrics, invalid shapes,
boundary tolerance, category floors, and deterministic selection. Full U2NET wins
only if eligible, within budgets, manually approved, and at least `0.03` absolute
better than U2NETP in aggregate IoU or boundary F; otherwise eligible U2NETP wins;
neither returns unavailable. Expected failure: missing quality module.

**Implementation:** Pure NumPy metric oracle and explicit immutable result records.
Record Dice ≥0.90, IoU ≥0.82, each category Dice ≥0.82, boundary F ≥0.80, warm
median ≤500 ms/p95 ≤800 ms, cold ≤2.0 s. Do not add fixtures lacking documented
redistribution rights.

**Verify:** `... pytest -q tests/test_mask_quality_metrics.py`; JIT-off only.

**Acceptance:** Thresholds and quality-first selection are fixed before the bakeoff;
fixture metadata requires source URL, author, license, checksum, transformations,
and ground-truth provenance.

**Commit:** `test(mask): freeze quality and model selection gates`

## Wave 1 — Qt-free data, geometry, alpha, and caching

### SM-03 — Immutable source, model, mask, and settings contracts

**Goal / prerequisite:** Define the one canonical data model shared by core, UI,
workers, caches, and export. Depends on SM-01 terminology.

**Files and interfaces**

- Add `ditherzam/masking/contracts.py`:
  `SourceIdentity`, `ModelIdentity`, `InferenceIdentity`, `MaskIdentity`,
  `ProbabilityMap`, validation helpers, `source_identity(rgba_u8)`.
- Add `ditherzam/masking/settings.py`:
  `MaskTarget`, `OutsideMode`, frozen `SmartMaskSettings`.
- Add `tests/test_mask_contracts.py` and `tests/test_mask_settings.py`.

**Red first:** Missing types; tests require content rather than filename identity,
RGBA/dimensions/alpha participation, stable equality/hash, exact defaults, enum and
range validation, immutable arrays, and rejection of wrong dtype/shape/range,
NaN/Inf, or non-C-contiguous confidence. Expected import failure.

**Implementation:** Canonical straight `uint8 RGBA`; immutable C-contiguous
`float32[H,W]` confidence/mask in `[0,1]`; frozen enums/dataclasses. Defaults:
disabled, Subject, sensitivity 50, feather 8, expansion 0, invert false, Original.
Identity includes source content, exact model hash, preprocessing/candidate and all
derived-mask settings/algorithm versions.

**Verify:** `... pytest -q tests/test_mask_contracts.py tests/test_mask_settings.py`
(JIT-off only).

**Acceptance:** No Qt imports; arrays publish read-only; shape mismatch is never
silently resized at the master boundary; no path-based identity.

**Commit:** `feat(mask): define immutable masking contracts`

### SM-04 — Deterministic mask derivation and preview resize

**Goal / prerequisite:** Derive all mask edits without rerunning inference. Depends
on SM-03.

**Files and interfaces**

- Add `ditherzam/masking/geometry.py`:
  `sensitivity_threshold`, `derive_master_mask`, `expand_contract`, `feather`,
  `resize_mask_area` and algorithm-version constants.
- Add `tests/test_mask_geometry.py`.

**Red first:** Tests require monotonic sensitivity, Subject/Background/Whole Image,
target then invert then signed geometry then feather order, ± expansion, preserved
holes/thin edges, hard zero-feather boundary, symmetric feather, deterministic area
resize, source-pixel semantics, invalid-input rejection, and immutable output.
Expected missing module/functions.

**Implementation:** Prefer focused NumPy/Pillow operations already available; do
not introduce SciPy or Numba without a measured need. Freeze mapping and rounding.
Whole Image returns the semantic all-ones result without inference dependency.

**Verify:** `... pytest -q tests/test_mask_geometry.py`; JIT-off. Inspect synthetic
boundary images manually.

**Acceptance:** Source-resolution master contract holds; higher sensitivity includes
more uncertain foreground; preview resize is deterministic and never promoted back.

**Commit:** `feat(mask): add deterministic mask geometry`

### SM-05 — Straight-alpha complete-branch compositor

**Goal / prerequisite:** Implement the sole outer compositing boundary. Depends on
SM-03–SM-04.

**Files and interfaces**

- Add `ditherzam/masking/composite.py`:
  frozen `CompositeContext`, `composite_masked(rendered_rgb, source_rgba, mask,
  outside_mode)`, `flatten_rgba_white`.
- Add `tests/test_mask_composite.py`.

**Red first:** Exact masks 0/1/0.5, Original/Transparent/White/Black, RGB/RGBA/
grayscale-derived sources, source alpha, dimension errors, deterministic rounding,
straight-alpha edge colors, and no halo after white/dark flatten. Expected import
failure.

**Implementation:** Composite the completed RGB render against decoded source or
solid outside in the byte domain with one documented rounding rule; alpha separately
under straight-alpha semantics. Never mutate inputs. Preserve deterministic hidden
RGB when alpha is zero.

**Verify:** `... pytest -q tests/test_mask_composite.py`; JIT-off initially. If
profiling later justifies Numba, add byte-parity real-JIT tests. Manually inspect a
feathered transparent edge over light/dark backgrounds.

**Acceptance:** Complete branch is the only processed input; output is RGB when
opaque and RGBA when required; stage order remains untouched.

**Commit:** `feat(mask): composite complete renders with straight alpha`

### SM-06 — Bounded inference, derived-mask, and composite caches

**Goal / prerequisite:** Reuse expensive work without coupling mask edits to the
staged render cache. Depends on SM-03–SM-05.

**Files and interfaces**

- Add `ditherzam/masking/cache.py`: `MaskCaches`, metrics, `clear_source`.
- Modify `ditherzam/render_cache.py` only if extracting its existing unique-backing
  byte-accounting helper is measurably simpler than duplication.
- Add `tests/test_mask_cache.py`; retain `tests/test_render_cache.py` and
  `tests/test_render_cache_budget.py` as regressions.

**Red first:** Partition by source/model/preprocessing; derived key includes every
mask setting/version; composite key includes rendered identity, mask, outside,
source and alpha version; atomic LRU eviction; unique backing counted once;
oversized results not retained; source clear; 50-source bounded soak. Expected
missing cache.

**Implementation:** Three explicit bounded caches, not a generic framework. Existing
render cache remains mask-unaware so its completed branch is reusable. Establish a
measured mask/inference sub-budget while total editor retained data stays ≤192 MiB.

**Verify:** `... pytest -q tests/test_mask_cache.py tests/test_render_cache.py tests/test_render_cache_budget.py`
(JIT-off).

**Acceptance:** No stale collisions or cross-call mutable buffers; cache edits do
not change render output; retained memory is bounded.

**Commit:** `feat(mask): add bounded inference and composite caches`

### SM-07 — Canonical decoded source RGBA retention

**Goal / prerequisite:** Preserve the original pixels and alpha required by
Original/Transparent. Depends on SM-03.

**Files and interfaces**

- Modify `ditherzam/ui/main_window.py::_DecodeSignals`, `_DecodeWorker.run`,
  `ImageEditor.__init__`, `load_array`, `_on_image_decoded`.
- Add `tests/test_source_rgba.py`; extend `tests/test_preview_lifecycle.py`.

**Red first:** Transparent PNG decode currently loses alpha. Tests require exact
straight RGBA survival, opaque alpha synthesis for RGB/grayscale programmatic load,
shape validation, immutable owned storage, and atomic source replacement.

**Implementation:** Decode once to straight RGBA, derive current RGB/gray from RGB
channels, retain `_base_rgba`, and preserve `_base_rgb` for current Source Colors.
Do not premultiply. Avoid redundant copies after ownership is established.

**Verify:** `... pytest -q tests/test_source_rgba.py tests/test_preview_lifecycle.py`
(JIT-off). Manually open a transparent PNG.

**Acceptance:** Exact source RGBA is retained without changing current grayscale/RGB
render inputs or source-color behavior.

**Commit:** `feat(mask): retain canonical source RGBA`

## Wave 2 — Offline adapter and resilient inference lifecycle

### SM-08 — Lazy ONNX Runtime adapter and session contract

**Goal / prerequisite:** Provide offline CPU inference behind a testable Qt-free
boundary. Depends on SM-01, SM-03; enables real SM-15 bakeoff.

**Files and interfaces**

- Add `ditherzam/masking/adapter.py`:
  `SegmentationAdapter` protocol, `InferenceResult`, `InferenceCancelled`,
  `NoClearSubject`.
- Add `ditherzam/masking/ort_adapter.py`:
  `OrtSegmentationAdapter`, `preprocess_u2net`, `postprocess_probability`.
- Add `ditherzam/masking/session.py`: synchronized lazy session holder.
- Add optional/release pin `onnxruntime==1.22.1` only in the appropriate dependency
  group; do not require it for fake-session unit tests.
- Add `tests/test_mask_adapter_contract.py`, `tests/test_mask_session.py` and a
  generated/license-safe tiny test model or injected fake session.

**Red first:** Tests require exact manifest tensor validation, documented 320-square
RGB normalization, seven-output selection semantics, deterministic source resize,
stable `primary`, source-resolution immutable confidence, lazy single session reuse,
CPU provider, safe-boundary cancellation, and explicit no-subject/missing/runtime
errors. Expected import failure.

**Implementation:** Dependency-inject session factory; import ORT lazily; validate
local asset before session creation; configure measured intra-op threads; never
probe a network or silently install/use GPU. Freeze exact converted tensor names.

**Verify:** `... pytest -q tests/test_mask_adapter_contract.py tests/test_mask_session.py tests/test_offline_security.py`
(JIT-off); optional approved-asset integration marker.

**Acceptance:** One reusable validated local session; no weights needed for default
suite; errors are honest; output satisfies SM-03.

**Commit:** `feat(mask): add offline ONNX segmentation adapter`

### SM-09 — Immutable inference requests and latest-wins scheduler

**Goal / prerequisite:** Separate inference state from render scheduling. Depends on
SM-03 and SM-08 error vocabulary.

**Files and interfaces**

- Add `ditherzam/masking/inference_request.py`: thread-safe `CancellationToken`,
  frozen `InferenceRequest`, terminal outcome enum/value.
- Add `ditherzam/masking/inference_scheduler.py`: `request`, `is_current`,
  `should_cancel`, `invalidate_source`, `on_terminal`.
- Add `tests/test_inference_request.py`, `tests/test_inference_scheduler.py`.

**Red first:** First launch, one newest trailing request, advisory cancellation,
source invalidation, stale source/model/generation rejection, idempotent exactly-one
terminal handling for success/no-subject/cancel/failure, duplicate terminal safety,
and recovery after exception. Expected missing modules.

**Implementation:** One inference-specific scheduler; do not generalize
`RenderScheduler`. Match publication on generation + source identity + exact model
hash. Duplicate terminal notifications cannot launch twice.

**Verify:** `... pytest -q tests/test_inference_request.py tests/test_inference_scheduler.py tests/test_render_scheduler.py`
(JIT-off).

**Acceptance:** At most one active plus one complete trailing request; stale work
never publishes; render scheduling is unchanged.

**Commit:** `feat(mask): schedule latest-wins inference requests`

### SM-10 — Resilient Qt inference worker

**Goal / prerequisite:** Run inference asynchronously with terminal guarantees.
Depends on SM-08–SM-09.

**Files and interfaces**

- Add `ditherzam/ui/mask_workers.py`: `InferenceSignals`, `InferenceWorker(QRunnable)`.
- Add `tests/test_mask_worker.py`; retain `tests/test_render_resilience.py`.

**Red first:** Fake adapter success emits succeeded only; degenerate result emits
no-subject only; cancellation at every safe boundary emits cancelled only;
missing/corrupt/incompatible emits model-unavailable/failure only; exception logs
and emits failed only; every path releases scheduler. Expected missing worker.

**Implementation:** Mirror `_RenderWorker.run` recovery shape. Consume only frozen
request and adapter/session dependencies; never read `ImageEditor`, widgets, or live
settings. Do not forcibly terminate ONNX.

**Verify:** `... pytest -q tests/test_mask_worker.py tests/test_mask_session.py tests/test_render_resilience.py`
(JIT-off); optional tiny-model smoke.

**Acceptance:** Exactly one terminal signal on all paths; no wedge; missing model
does not alter rendering.

**Commit:** `feat(mask): add resilient inference worker lifecycle`

## Wave 3 — UI, immutable rendering, presets, and v1 scope guards

### SM-11 — Smart Mask panel and exact enablement matrix

**Goal / prerequisite:** Add the approved compact controls without synchronous
inference. Depends on SM-03 settings.

**Files and interfaces**

- Add `ditherzam/ui/smart_mask_panel.py`: `SmartMaskPanel`, focused settings,
  re-detect, cancel and overlay signals.
- Modify `ditherzam/ui/controls.py::ControlPanel.__init__` to host the separate group.
- Add `tests/test_smart_mask_panel.py`; extend `tests/test_controls.py`.

**Red first:** Exact defaults/ranges/labels; Primary (1 of 1) disabled; Whole Image
disablement; Subject/Background restore; cancel detecting-only; re-detect requires
source/model; all lifecycle statuses; overlay separate/reset. Expected import failure.

**Implementation:** One state-to-enablement method. Keep mask state outside style
`params` and `ControlPanel.state`. Expansion −64…64; sensitivity 0…100; source-pixel
labels. No fake variations or download affordance.

**Verify:** `... pytest -q tests/test_smart_mask_panel.py tests/test_controls.py tests/test_widgets.py`
(JIT-off/offscreen). Manual narrow layout, tab order and Whole Image round-trip.

**Acceptance:** Matches specification; enable never blocks GUI; overlay does not
emit creative render-setting changes.

**Commit:** `feat(ui): add Smart Mask controls`

### SM-12 — Editor inference coordination and source invalidation

**Goal / prerequisite:** Connect panel, scheduler, worker, probability/derived cache,
and lifecycle. Depends on SM-04, SM-06–SM-11.

**Files and interfaces**

- Modify `ditherzam/ui/main_window.py::ImageEditor.__init__`, `load_array`,
  `_build_request`, and add `_request_mask_detection`, `_cancel_mask_detection`,
  terminal slots, `_current_mask_context`.
- Extend `ditherzam/ui/render_request.py::RenderRequest` with one optional frozen
  `mask_context=None` field, preserving existing callers.
- Add `tests/test_mask_editor_lifecycle.py`, `tests/test_mask_render_request.py`;
  extend lifecycle/thread-safety/resilience tests.

**Red first:** Disabled load and Whole Image infer nothing; enabling starts once;
edits reuse confidence; re-detect/cancel/failure retain last valid mask; no prior
mask stays unmasked; source replacement invalidates all source-specific state;
stale progress/results ignored; terminal launches at most one trailing request;
render request freezes one context; missing/no-subject uses `None`.

**Implementation:** Dedicated inference scheduler and, if benchmark-supported,
single-thread pool. Invalidate before publishing new source. Derive off GUI thread
only if 4K heartbeat measurement requires it. Snapshot rather than reread live UI.

**Verify:** JIT-off focused editor/lifecycle/thread/resilience/scheduler tests; then
real-JIT `tests/test_mask_editor_lifecycle.py tests/test_render_thread_safety.py tests/test_render_cancellation.py`.
Manual rapid re-detect, source swap, cancel, render edits, injected failure.

**Acceptance:** Responsive latest-wins lifecycle; no stale publication/cache; every
terminal recovers; disabled and failures remain unmasked.

**Commit:** `feat(ui): coordinate Smart Mask inference lifecycle`

### SM-13 — Preview/full/exact render integration and overlay

**Goal / prerequisite:** Composite immutable masks across display and exact still
render without changing `RenderPipeline`. Depends on SM-04–SM-07 and SM-12.

**Files and interfaces**

- Add `ditherzam/masking/render.py`: thin `render_with_mask` outer wrapper with
  explicit disabled bypass.
- Add `ditherzam/ui/mask_overlay.py`: pure `apply_mask_overlay`.
- Modify `ditherzam/ui/preview.py::render_preview` only at its outer result boundary.
- Modify `ditherzam/ui/main_window.py::_build_request`, `_RenderWorker.run`,
  `_on_rendered`, `render_now`, `_rendered_rgb` (rename to result-neutral helper if
  needed), and `_export_pipeline` context wiring.
- Modify `ditherzam/ui/convert.py` only for safe RGBA QImage conversion.
- Add `tests/test_mask_render_integration.py`, `tests/test_mask_preview.py`,
  `tests/test_mask_overlay.py`; extend render order/cache/thread/zoom tests.

**Red first:** Disabled spy proves no mask operations and exact historical bytes;
enabled renders complete pipeline once then composites; capped requests derive exact
target mask/source; Full/export use master; one-read source/engine/effect/mask/
outside snapshots survive concurrent reassignment; Source Colors remain spatial;
overlay changes display only and never cache/export; stale request cannot paint.

**Implementation:** Keep `RenderPipeline.STAGE_ORDER` and staged keys untouched.
Apply overlay after composite without mutating cached result. Preserve logical source
geometry. Reuse complete rendered branch across mask edits.

**Verify:** JIT-off mask preview/integration/overlay plus render/order/cache/thread/
zoom suites; real-JIT mask integration + color exactness/source/style targets.
Manual 480/1080/Full edges, glow/blur boundary, overlay toggle, exact export compare.

**Acceptance:** Zero-work disabled path; deterministic aligned preview; exact Full/
export; no stage or source-color regression; overlay excluded everywhere durable.

**Commit:** `feat(mask): integrate immutable preview and exact compositing`

### SM-14 — Presets and explicit unsupported-media guards

**Goal / prerequisite:** Persist reusable settings only and prevent silent masked
exports outside v1. Depends on SM-11–SM-13.

**Files and interfaces**

- Modify `ditherzam/presets.py::settings_to_preset`, `preset_to_settings`; introduce
  a named `PresetContents` result if needed to avoid an opaque four-tuple.
- Modify `ImageEditor` preset save/load/apply handlers.
- Add `ditherzam/masking/scope.py`: pure `mask_allows_media(kind, settings)`.
- Guard `ImageEditor._on_export_svg`, `_on_batch_folder`, `_export_animation`, video
  export/action wiring; do not add mask args to those APIs.
- Add `tests/test_mask_preset_wiring.py`, `tests/test_mask_scope_gating.py`; extend
  preset, animation, video and UI export tests.

**Red first:** Seven reusable settings round-trip; old/malformed presets default/
clamp safely; arrays/source/candidate/overlay/progress/error/model paths excluded;
restore triggers at most one detection after complete apply. Subject/Background
blocks SVG/batch/video/animation before work with clear message; disabled/Whole
preserves existing behavior.

**Implementation:** Top-level `smart_mask` mapping. Coalesce apply signals. One pure
scope policy plus handler defense. Never silently export unmasked.

**Verify:** JIT-off preset/scope/UI export/animation/video tests. Manual old YAML,
enabled YAML with/without model/source, and menu/shortcut guards.

**Acceptance:** Backward compatible; only approved settings persist; unsupported
masked media cannot start; all unmasked paths remain unchanged.

**Commit:** `feat(mask): persist settings and guard deferred media`

## Wave 4 — Raster formats, model selection, packaging, and acceptance

### SM-15 — Exact PNG RGBA and JPEG white flattening

**Goal / prerequisite:** Complete approved still format behavior. Depends on SM-05,
SM-07, SM-13.

**Files and interfaces**

- Modify `ditherzam/export/raster.py::save_raster`.
- Modify `ImageEditor._on_export_raster` and result-neutral exact render helper.
- Extend `tests/test_export_raster.py` and mask integration tests.

**Red first:** PNG round-trips four exact channels; JPEG explicitly flattens straight
RGBA onto white with documented deterministic math; unsupported channel counts and
extensions fail; historical RGB/grayscale behavior remains.

**Implementation:** Accept H×W, H×W×3, H×W×4 uint8-like arrays. PNG preserves RGBA;
JPEG calls the shared white flattener before quality 95 encoding. Warn/label JPEG
flatten behavior once in UI.

**Verify:** `... pytest -q tests/test_export_raster.py tests/test_mask_render_integration.py`
(JIT-off). Manual transparent hair/thin-edge PNG and JPEG inspection.

**Acceptance:** No dark halos; PNG alpha exact; JPEG corners white within codec
tolerance; unmasked exports unchanged.

**Commit:** `feat(export): preserve mask alpha across raster formats`

### SM-16 — Licensed U2NETP/full-U2NET bakeoff and release selection

**Goal / prerequisite:** Select the smallest model that does not compromise approved
quality. Depends on SM-01–SM-02 and SM-08; requires written redistribution evidence
and approved licensed fixtures before real assets are staged.

**Files and interfaces**

- Add developer-only `tools/convert_u2net_onnx.py`.
- Complete `benchmarks/smart_mask.py` and write dated
  `benchmarks/SMART_MASK_ACCEPTANCE_2026-07-11.md`.
- Add only approved fixture manifest/assets and opt-in integration tests.
- Finalize manifest source/conversion hashes and tensor contract; do not commit
  weights unless separately approved.

**Red first:** Unit metric/runner tests pass but no candidate has a measured eligible
report, so the release-selection gate is red/unavailable.

**Implementation:** Convert both variants reproducibly from the pinned upstream
state. Run identical fixtures/preprocessing/hardware. Measure quality, manual edges,
cold/warm latency, preprocessing/postprocessing, 1080p/4K geometry/composite, peak/
retained RSS, heartbeat, cancellation, and 50-cycle growth. Full wins only by SM-02
policy. Never lower thresholds after results.

**Verify:** unit benchmark tests, then opt-in `.venv/Scripts/python.exe -m benchmarks.smart_mask ...`.
Manual portrait/product/animal/full-body/multiple people/busy/low contrast/transparent/
thin/no-subject matrix.

**Acceptance:** Exact hardware/software/asset hashes recorded; all quality and
performance gates pass; model rights are explicit; user approves any model asset
before repository inclusion.

**Commit:** `bench(mask): select licensed offline segmentation model` (report and
approved metadata only; weights follow asset policy)

### SM-17 — Frozen Windows packaging, offline security, and end-to-end certification

**Goal / prerequisite:** Prove release readiness and close every acceptance criterion.
Depends on SM-01–SM-16 and approved selected asset.

**Files and interfaces**

- Finalize release dependency/package configuration for `onnxruntime==1.22.1`,
  model asset, ORT DLLs, manifest, license/NOTICE/provenance.
- Add minimal frozen Windows build/smoke configuration only if one does not already
  exist; avoid creating a general packaging framework.
- Add `tests/test_mask_error_states.py`, `tests/test_mask_ui_smoke.py`,
  `tests/test_mask_e2e.py`; extend app/resilience/offline tests.
- Add dated end-to-end acceptance report.

**Red first:** Frozen build initially lacks verified runtime/asset. E2E tests cover
RGB/RGBA/grayscale × outside modes, target/invert, Source Colors/Colored Dither,
effects/invert, preview/full/export, missing/corrupt/no-subject/OOM/cancel/source
replacement, progress staleness, and unsupported media.

**Implementation:** Bundle content-addressed local assets and notices. Verify before
session load. With sockets denied, start, detect, render and export; missing/corrupt
asset remains usable/unmasked. Centralize lifecycle presentation; ignore stale
progress. Fix defects in separate focused commits, not by weakening tests.

**Verify:** full JIT-off suite; focused real-JIT mask/render/color/style/thread tests;
quality benchmark; frozen Windows offline smoke; 4K GUI rapid-edit/re-detect/source
swap; overlay/export; 50-cycle RSS; PNG/JPEG inspection.

**Acceptance:** All eleven specification criteria pass; total retained caches
≤192 MiB; latency/quality/heartbeat/cancellation budgets pass; every worker terminal
recovers; disabled hashes match baseline; no outbound socket/network/download; no
unapproved asset or provenance gap.

**Commit:** `test(mask): certify offline Smart Mask end to end`

## Dependency map

```text
SM-01 ──▶ SM-02
  │
  └──▶ SM-03 ──▶ SM-04 ──▶ SM-05 ──▶ SM-06
           │                         │
           ├──▶ SM-07               │
           └──▶ SM-08 ──▶ SM-09 ──▶ SM-10
                    │                 │
                    └─────────────────┴──▶ SM-11 ──▶ SM-12 ──▶ SM-13 ──▶ SM-14
                                                │          │
                                                └──────────┴──▶ SM-15
SM-01 + SM-02 + SM-08 ──▶ SM-16
SM-14 + SM-15 + SM-16 ──▶ SM-17
```

Parallelism is limited to independent work: after SM-03, SM-04/SM-07/SM-08 may be
owned separately; SM-11 UI can begin after contracts while scheduler/worker work
continues; SM-14 and SM-15 may split after SM-13. Integration tasks remain serial.

## Plan self-review

- Every approved control/default/state maps to SM-03/04/11/12.
- Model licensing, provenance, conversion, metrics, bakeoff and packaging map to
  SM-01/02/08/16/17.
- Mask/alpha contracts and complete-branch composition map to SM-03–SM-07/13/15.
- Immutable requests, latest-wins, cancellation and terminal recovery map to
  SM-09/10/12/17.
- Cache identity/bounds and exact preview/export ownership map to SM-06/13/17.
- Presets and deferred SVG/batch/video/animation behavior map to SM-14.
- Security/privacy/offline behavior maps to SM-01/08/17.
- Performance and quality budgets map to SM-02/16/17.
- No placeholder task, fake model, runtime downloader, internal-stage masking,
  source-specific preset mask, or speculative layer/tracking system is included.
- Task IDs are stable and will map one-to-one to the execution ledger.

## Approval gate

After approval, derive `docs/tasklists/09-smart-subject-masking-tasks.md` with one
ledger section per SM-01…SM-17. Approving this plan does not approve model assets,
licenses, fixture inclusion, commits, branch changes, or execution; those remain
explicit ledger/release gates.
