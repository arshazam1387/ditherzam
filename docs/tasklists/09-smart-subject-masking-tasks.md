# Smart Subject/Background Masking — Execution Ledger

Source: `docs/superpowers/plans/2026-07-11-smart-subject-masking.md`

Branch: `feat/smart-subject-masking` · checkpoint HEAD: `ad1f418` (2026-07-12)

This ledger maps one-to-one to SM-01…SM-17. SM-01…15 implementation and review
are complete. SM-16/17 have reviewed, fail-closed skeletons only: unchecked items
are release gates, not evidence already obtained. The branch inherits **71 known
pre-existing JIT-off `test_kernels_all` golden failures**; do not count them as
Smart Mask regressions or merge until they are separately resolved/regenerated.

## Commands and operating rules

JIT-off prefix (PowerShell):

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:NUMBA_DISABLE_JIT='1'
.venv/Scripts/python.exe -m pytest -q <targets>
```

Real-JIT prefix:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
Remove-Item Env:NUMBA_DISABLE_JIT -ErrorAction SilentlyContinue
.venv/Scripts/python.exe -m pytest -q <targets>
```

- Never fetch, stage, or commit weights/binaries without explicit user approval.
- Do not lower frozen quality/performance thresholds after seeing results.
- Keep runtime fully offline and core Qt-free; masking remains an outer compositor.
- Disabled Smart Mask must remain byte-identical with zero mask work.

## Wave 0 — provenance and selection policy

### SM-01 — Offline asset manifest and fail-closed resolver (no dependency)

- [x] Red: missing/corrupt/unapproved/traversing assets and runtime networking fail.
- [x] Green: frozen manifest, streamed verification, local resolver and staging policy.
- [x] Refactor/review complete (`3fad7ed`).
- [x] Verify: `pytest -q tests/test_mask_model_assets.py tests/test_offline_security.py`.
- [ ] Approve real license, redistribution evidence, hashes and binary inclusion.

### SM-02 — Quality metrics and winner policy (depends SM-01 terminology)

- [x] Red: synthetic metric, threshold and deterministic-selection cases.
- [x] Green: Dice/IoU/boundary-F aggregation and frozen winner policy.
- [x] Refactor/review complete (`9eca9f6`).
- [x] Verify: `pytest -q tests/test_mask_quality_metrics.py`.
- [ ] Supply approved, redistributable fixture corpus and provenance manifest.

## Wave 1 — Qt-free contracts, geometry, alpha and caches

### SM-03 — Immutable contracts and settings (depends SM-01)

- [x] Red / [x] Green / [x] Refactor/review complete (`c9f8922`).
- [x] Verify: `pytest -q tests/test_mask_contracts.py tests/test_mask_settings.py`.

### SM-04 — Deterministic derivation and resize (depends SM-03)

- [x] Red / [x] Green / [x] Refactor/review complete (`ab6fea9`).
- [x] Verify: `pytest -q tests/test_mask_geometry.py`.

### SM-05 — Straight-alpha outer compositor (depends SM-04)

- [x] Red / [x] Green / [x] Refactor/review complete (`d699223`).
- [x] Verify JIT-off and real-JIT: `pytest -q tests/test_mask_composite.py`.

### SM-06 — Bounded mask caches (depends SM-05)

- [x] Red / [x] Green / [x] Refactor/review complete (`5ccd0c4`).
- [x] Verify: `pytest -q tests/test_mask_cache.py tests/test_render_cache.py tests/test_render_cache_budget.py`.

### SM-07 — Canonical source RGBA retention (depends SM-03)

- [x] Red / [x] Green / [x] Refactor/review complete (`57836b5`).
- [x] Verify: `pytest -q tests/test_source_rgba.py tests/test_preview_lifecycle.py`.

## Wave 2 — offline inference lifecycle

### SM-08 — Lazy ONNX Runtime adapter (depends SM-03)

- [x] Red / [x] Green / [x] Refactor/review complete (`9150e9e`).
- [x] Verify: `pytest -q tests/test_mask_adapter_contract.py tests/test_mask_session.py tests/test_offline_security.py`.
- [ ] Freeze and validate the converted model's exact seven output names/order.

### SM-09 — Immutable requests and latest-wins scheduler (depends SM-08)

- [x] Red / [x] Green / [x] Refactor/review complete (`c6d8c35`).
- [x] Verify: `pytest -q tests/test_inference_request.py tests/test_inference_scheduler.py tests/test_render_scheduler.py`.

### SM-10 — Resilient Qt inference worker (depends SM-09)

- [x] Red / [x] Green / [x] Refactor/review complete (`b36879e`).
- [x] Verify: `pytest -q tests/test_mask_worker.py tests/test_mask_session.py tests/test_render_resilience.py`.

## Wave 3 — UI and render integration

### SM-11 — Smart Mask panel (depends SM-03 and lifecycle interfaces)

- [x] Red / [x] Green / [x] Refactor/review complete (`e3e1935`).
- [x] Verify: `pytest -q tests/test_smart_mask_panel.py tests/test_controls.py tests/test_widgets.py`.

### SM-12 — Editor inference coordination (depends SM-09–11)

- [x] Red / [x] Green / [x] Refactor/review complete (`eb5a115`).
- [x] Verify: `pytest -q tests/test_mask_editor_lifecycle.py tests/test_mask_render_request.py tests/test_mask_cache.py tests/test_render_resilience.py`.
- [x] Prove actual editor allocation disabled=192/0 MiB, enabled=128/64 MiB.

### SM-13 — Preview/full/exact render and overlay (depends SM-05/06/12)

- [x] Red / [x] Green / [x] Refactor/review complete (`32b3b1e`).
- [x] Verify JIT-off and real-JIT: `pytest -q tests/test_mask_render_integration.py tests/test_mask_overlay.py tests/test_render_cache.py`.

### SM-14 — Presets and unsupported-media guards (depends SM-13)

- [x] Red / [x] Green / [x] Refactor/review complete (`68fb782`).
- [x] Verify: `pytest -q tests/test_mask_preset_wiring.py tests/test_mask_scope_gating.py tests/test_presets.py tests/test_ui_export_actions.py`.

## Wave 4 — raster, real assets and release certification

### SM-15 — Exact PNG RGBA and JPEG white flatten (depends SM-13)

- [x] Red / [x] Green / [x] Refactor/review complete (`059b3dd`).
- [x] Verify JIT-off and real-JIT: `pytest -q tests/test_export_raster.py tests/test_mask_render_integration.py`.

### SM-16 — Licensed U2NETP/full-U2NET bakeoff (depends SM-01/02/08)

- [x] Red-pending-asset gate and local-only converter/benchmark skeleton.
- [x] Green for fail-closed skeleton behavior only.
- [x] Refactor/review complete (`7150b8a`; 83 green, 1 expected asset skip).
- [ ] User approves licensed U2NETP and full-U2NET source weights and redistribution.
- [ ] Reproducibly convert both candidates; record source/ONNX hashes and toolchain.
- [ ] Run exact opt-in command emitted by `python -m benchmarks.smart_mask --help`
  against the approved local candidates and fixture manifest on recorded Windows hardware.
- [ ] Pass frozen Dice ≥0.90, IoU ≥0.82, each-category Dice ≥0.82, boundary-F ≥0.80.
- [ ] Pass warm median ≤500 ms, p95 ≤800 ms, cold ≤2.0 s, geometry/composite,
  memory, heartbeat, cancellation and 50-cycle growth gates.
- [ ] Complete manual portrait/product/animal/full-body/multiple-people/busy/
  low-contrast/transparent/thin/no-subject matrix.
- [ ] Apply frozen winner policy and obtain human winner/binary-inclusion approval.
- [ ] Publish dated acceptance report without inventing or lowering evidence.

### SM-17 — Frozen Windows offline certification (depends SM-14/15/16)

- [x] Red-pending-asset packaging/E2E gate and fail-closed skeleton.
- [x] Green for skeleton behavior only.
- [x] Refactor/review complete (`cada4e7`; focused tests green, asset skips expected).
- [ ] Include only approved content-addressed winner, manifest, ORT payload and notices.
- [ ] Build the frozen Windows artifact with `onnxruntime==1.22.1`.
- [ ] Run full JIT-off suite: `pytest -q` and account separately for the 71 inherited goldens.
- [ ] Run focused real-JIT mask/render/color/style/thread suites.
- [ ] Run frozen build offline with sockets denied: start, detect, render and export.
- [ ] Prove missing/corrupt/no-subject/OOM/cancel/source-replacement recovery and
  byte-identical unmasked fallback.
- [ ] Complete 4K GUI rapid-edit/re-detect/source-swap, overlay/export, 50-cycle RSS,
  PNG hair/thin-edge and JPEG white-corner manual QA.
- [ ] Record frozen hardware/software/asset hashes and all acceptance evidence.
- [ ] Obtain final human release sign-off.

## Documentation and merge readiness

- [x] SM-01…15 implementation and independent reviews complete at `ad1f418`.
- [x] SM-16/17 fail-closed skeleton implementation and reviews complete.
- [x] Shared status recorded in `docs/memory/048-smart-mask-sdd-execution.md`.
- [ ] Update memory after licensed bakeoff, winner decision and certification.
- [ ] Resolve or explicitly rebaseline the 71 inherited golden failures before merge.
- [ ] Confirm clean status excludes `.codex/`, `purple harrow.png`, and other user/temp files.
- [ ] Confirm no unapproved fixtures, weights, generated binaries or network paths are staged.
- [ ] Final whole-branch review, release-signoff evidence, and merge to main.
