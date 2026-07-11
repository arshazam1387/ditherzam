# Smart Subject/Background Masking — Design Specification

**Date:** 2026-07-11  
**Status:** Approved  
**Scope:** Automatic offline masking for still-image preview and raster export.  
**Workflow gate:** Product code, implementation plan, and execution task list must
not begin until this specification is approved.

## Goal

Add one reusable Smart Mask system around the completed render pipeline so every
dither style, color mode, and effect can process the whole image, the detected
subject, or its background. Detection and rendering remain fully local. Disabled
Smart Mask output is byte-identical to the existing renderer.

The implementation should be small and measured: one mask contract, one inference
adapter boundary, one outer compositor, and separate bounded inference/render
caches. Abstractions without a demonstrated second use are out of scope.

## User stories

- As an artist, I can isolate a subject or background without changing styles.
- I can preserve the original outside the processed area, replace it with a solid
  color, or make it transparent.
- I can tune sensitivity and mask geometry while seeing an optional overlay.
- I can cancel or replace slow detection without freezing the editor.
- I can export the exact full-resolution result even while using capped previews.
- I can use Source Colors and Colored Dither without moving source hues to the
  wrong spatial regions.
- I can work with no network connection and verify which local model is in use.

## v1 scope

### Included

- Still-image editor preview and exact PNG/JPEG export.
- Automatic, class-agnostic primary-foreground detection.
- Enable, target, candidate display, sensitivity, feather, expand/contract,
  invert, outside-region selection, overlay, re-detect, progress, and cancel.
- Original, Transparent, White, and Black outside modes.
- Source-resolution master masks and deterministic preview derivatives.
- Missing/corrupt/incompatible model handling without fallback heuristics.
- CPU inference; optional accelerators only after measured packaging validation.

### Deferred to v2

- Brush add/subtract, polygon/lasso tools, mask layers, undo history for strokes.
- Mask import/export and source-specific sidecars.
- Multiple-instance segmentation and cycling genuinely distinct subject masks.
- Portrait-specialist model selection or automatic model ensembles.
- Batch, video, and animation masking, including tracking and per-frame inference.
- SVG masked output.

These are deliberate scope boundaries, not placeholders in the v1 UI.

## Non-goals

- A Smart Styles category or changes to individual dither kernels.
- Cloud inference, telemetry, runtime downloads, license checks, or network access.
- A Photoshop-style layer system.
- Semantic object labeling, face recognition, identity tracking, or prompting.
- Pretending thresholding or another heuristic is AI detection.
- Changing the frozen internal stage order.

## UX and state model

### Placement

Add a collapsible **Smart Mask** group in the main control panel, separate from
style-native parameters, Color, and Effects. It wraps their combined result.

```text
Smart Mask
  [ ] Enabled
  Target                 [ Subject v]
  Detected subject       [ Primary (1 of 1) ]   (disabled when only one)
  Sensitivity            [-------50-------]
  Edge feather           [---- 8 px ----]
  Expand / contract      [----- 0 px -----]
  [ ] Invert mask
  Outside region         [ Original v]
  [ ] Show mask overlay
  [ Re-detect ] [ Cancel ]          Detecting 62%
  status: Ready / Detecting / No clear subject / Model unavailable / Error
```

Pixel-valued controls use source-image pixels. Preview display scales their
effect to the preview dimensions; exact export applies them at source resolution.

### Defaults and exact behavior

| control | default | behavior |
|---|---|---|
| Enabled | Off | Off bypasses all mask work and preserves historical output exactly. Turning On starts detection only if no valid master mask exists. |
| Target | Subject | Subject uses mask `M`; Background uses `1-M`; Whole Image uses all ones and needs no detection. |
| Detected subject | Primary (1 of 1) | Read-only/disabled for the v1 single salient-foreground adapter. The control exists to state what was selected, not imply fake variations. It becomes a selector only when a future adapter returns stable ranked candidate IDs. |
| Sensitivity | 50 | Maps monotonically to the adapter's documented foreground-confidence calibration. Higher values include more uncertain foreground. Changing it derives a mask from cached probability output and does not rerun inference. |
| Edge feather | 8 source px | Applies a documented symmetric soft-edge operation after sensitivity and expand/contract. Zero is hard edged. |
| Expand/contract | 0 source px | Signed range, provisionally −64…+64 px. Negative contracts; positive expands. Applied before feathering. |
| Invert mask | Off | Complements the selected target mask after target selection and before geometry operations. It is independent of Subject/Background so its effect is explicit, though Background+Invert can equal Subject. |
| Outside region | Original | Original uses decoded source RGBA/RGB at the same spatial position. Transparent sets outside alpha to zero while retaining straight-RGBA edge colors. White/Black are opaque. |
| Show mask overlay | Off | Preview-only translucent color overlay of the final effective mask boundary/fill. It never enters render caches, presets, or exports. Automatically turns off on source replacement; remains user-toggleable during edits. |
| Re-detect | enabled with source/model | Starts a new generation, cancels/obsoletes older work, and retains the last valid mask until a new valid result publishes. |
| Cancel | detecting only | Requests cooperative cancellation. Exactly one terminal outcome is emitted. The last valid mask remains; if none exists, masking reports Not detected and does not alter the image. |

Target=Whole Image disables candidate, sensitivity, feather, expand/contract,
invert, overlay, re-detect, and cancel because there is no inferred boundary.
Outside region has no visible effect and is disabled. Smart Mask remains enabled
so returning to Subject/Background restores the prior valid settings.

### State transitions

`Disabled → Needs detection → Detecting → Ready` is the normal path. Detection may
end in `Ready`, `No clear subject`, `Cancelled`, `Model unavailable`, or `Error`.
Every worker generation has exactly one terminal outcome. Only the latest matching
source/model generation may publish. A source replacement invalidates the master
mask, probability map, candidate ID, inference cache entry, and overlay.

Loading a source never runs inference while Smart Mask is disabled. Enabling it,
pressing Re-detect, or restoring an enabled preset without a valid source-specific
mask may trigger inference. No background-idle surprise detection is performed.

## Offline model/runtime decision

### Comparison

All candidates require local Windows/Python 3.12 measurements before approval.
Sizes below are upstream claims or approximate checkpoint sizes, not packaged
ONNX sizes; the asset manifest records measured bytes.

| candidate | strengths | material limits | license/provenance | v1 disposition |
|---|---|---|---|---|
| U-2-Net **U2NETP / full U2NET**, 320×320 salient-object detection | Class-agnostic; aligns with people, products, animals; upstream reports ~4.7 MB / 176.3 MB checkpoints | Single salient foreground; U2NETP may lose fine boundaries; no stable model release/checksum | Repository Apache-2.0, but pretrained-weight redistribution scope requires written confirmation; Google Drive assets lack upstream checksums | **Provisional family**, selected by bakeoff and conditional hard gates |
| MODNet photographic portrait matting | Strong continuous hair/portrait alpha; repository explicitly says code and models are Apache-2.0 | People/portraits only; cannot meet general-subject goal alone; canonical ONNX/checksum absent | Clearer model grant, but public checkpoint must be pinned, hashed, and reproducibly converted | v2 optional portrait specialist |
| MediaPipe Selfie Segmenter / DeepLab-V3 | Small established runtime; published task contracts; fast mobile evidence | Selfie is person-only; DeepLab has fixed semantic classes rather than arbitrary salient subjects; desktop quality/latency unknown | Runtime Apache-2.0; downloadable model redistribution/checksum terms need confirmation | Rejected as general v1 default |
| IS-Net general-use, commonly ~176 MB at 1024² | High-resolution general foreground quality challenger | Much larger/slower; old PyTorch reference; ONNX conversion and Python 3.12 path unproven | Upstream explicitly licenses code/metric but not clearly weights; no checksum | Benchmark challenger only; never redistribute without permission |

Primary sources: [U-2-Net repository](https://github.com/xuebinqin/U-2-Net),
[U-2-Net paper](https://arxiv.org/abs/2005.09007),
[MODNet repository](https://github.com/ZHKKKe/MODNet),
[MODNet paper](https://arxiv.org/abs/2011.11961),
[MediaPipe Image Segmenter](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter),
[MediaPipe selfie model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf),
[IS-Net/DIS repository](https://github.com/xuebinqin/DIS), and
[ONNX Runtime installation documentation](https://onnxruntime.ai/docs/install/).

### Recommendation and hard gates

Use a reproducibly converted U-2-Net-family ONNX model through pinned
`onnxruntime==1.22.1` CPU, only if all gates pass. Bake off U2NETP and full U2NET
at upstream commit `ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29`; do not select
the smaller model before measuring quality:

1. Written confirmation or equivalent authoritative evidence establishes that
   the exact pretrained weights may be redistributed under compatible terms.
2. A manifest pins upstream repository commit, source URL, original SHA-256,
   conversion script revision/opset, ONNX SHA-256 and bytes, tensor contract,
   preprocessing, license, attribution, and modification notice.
3. Windows x86-64 Python 3.12, frozen-app packaging, numerical parity, latency,
   retained memory, and quality acceptance pass.
4. At least one U-2-Net variant beats the minimum general-subject quality budgets.
   Full U2NET is selected when it improves aggregate IoU or boundary F-score by
   at least 0.03 absolute over U2NETP, wins manual edge QA, and stays inside the
   approved latency/memory budgets; otherwise U2NETP is selected. IS-Net may be
   used only as a local benchmark challenger unless its weights receive clear rights.

Model size is evidence-driven: select the smallest eligible asset meeting quality
budgets. Assets up to roughly 200 MB may be considered only when measured quality
materially improves and packaging is explicitly approved. No quality compromise
is made merely to hit a smaller arbitrary size.

If both U-2-Net variants fail licensing or quality, do not ship their weights and
do not silently substitute a heuristic. Keep the adapter/UX functional in Model
unavailable state while an eligible model is selected or trained.

### Runtime and packaging policy

- Application code contains no downloader, URL fetch, telemetry, remote fallback,
  update check, or license endpoint.
- A developer-only packaging script may fetch an approved source asset, verify its
  pinned SHA-256, convert it reproducibly, verify output SHA-256, and stage it for
  release. It is never called by the application.
- The model ships inside an installer/release asset, not initially in Git. Model
  weights, generated fixtures, and binaries remain uncommitted until license,
  provenance, size, and `.gitignore` policy are approved.
- ONNX Runtime CPU is the dependable default. GPU/DirectML is deferred unless one
  provider can be optional without conflicting runtime packages or destabilizing
  packaging. Provider probing must always fall back locally to CPU.
- Model metadata and license/NOTICE ship beside the asset. Startup does not load
  the session until detection is requested; one warm session is reused afterward.
- A local manifest hash mismatch, missing asset, incompatible opset, or runtime
  load failure produces Model unavailable with remediation text and no render
  alteration.

## Data contracts

### Source identity

A source identity is a stable content digest plus decoded dimensions and alpha
contract, not a path or filename. Reopening changed bytes creates a new identity.

### Inference output and master mask

- Adapter output: immutable C-contiguous `float32[H,W]`, finite confidence in
  `[0,1]`, at source resolution, where 1 means foreground confidence.
- Canonical master mask: immutable C-contiguous `float32[H,W]` in `[0,1]`, at
  source resolution, after sensitivity, target, inversion, geometry, and feather.
- No NaN/Inf is accepted. Shape mismatch is an error, never silently resized at
  the master boundary.
- Mask identity includes source content identity, model logical ID/version/hash,
  inference preprocessing version, candidate ID, sensitivity, target, invert,
  expand/contract, feather algorithm/version, and values.
- Preview masks are deterministic area-resampled derivatives of the master mask
  at exact preview geometry. Full preview/export uses the source-resolution master.
- Arrays published across workers are read-only. UI state owns references, not
  mutable buffers.

The probability map is cached separately from derived geometry so sensitivity,
target, invert, expand, and feather edits do not rerun inference.

### Image and alpha contract

- Decode and retain source as straight (unpremultiplied) `uint8 RGBA`; an opaque
  RGB source receives alpha 255. Grayscale input expands to neutral RGB while
  retaining any source alpha.
- Existing render pipeline still produces opaque `uint8 RGB`; Source Colors gets
  source RGB resized exactly as today.
- The outer compositor produces straight `uint8 RGBA` only when Transparent or
  source alpha requires it; otherwise it may retain `uint8 RGB` at the existing
  boundary.
- Composite color uses linear mask weights over current byte-domain pipeline and
  outside pixels, with one frozen rounding rule documented by differential tests.
  Alpha is composited separately under straight-alpha rules; hidden RGB at alpha
  zero is deterministic to avoid edge halos on later flattening.
- Original means decoded source color/alpha, never the grayscale render input and
  never an earlier partial pipeline stage. There is no ambiguous “normal unfiltered
  pipeline input/output” option in v1.

## Render and compositing architecture

The frozen internal order remains:

`contrast → midtones → highlights → blur → dither → color → saturation → effects → invert`

Smart Mask renders that complete branch normally, then composites it against the
selected outside source/color. This makes “apply to subject/background” mean all
pipeline stages—including blur, diffusion, glow, and chromatic aberration—run on
the complete image before the final soft mask. Neighborhood operations may draw
context from both sides of a boundary; the mask gates their final visibility, not
their input samples. This avoids seams and does not mutate stage semantics.

The disabled path does not allocate, resize, hash, or composite a mask. Its output
and existing cache behavior remain byte-identical.

### Cache keys and ownership

- Inference cache is separate, bounded, and keyed by source identity + exact model
  hash + preprocessing version. It stores the immutable probability map.
- Derived-mask cache is bounded and keyed by inference identity + all mask settings.
- Existing staged render caches remain unaware of masks because the complete
  rendered branch is reusable across mask edits.
- A small outer-composite cache may key rendered-output identity + mask identity +
  outside mode + outside source identity + alpha/algorithm version.
- Byte accounting counts unique NumPy backing storage once and follows the existing
  192 MiB per-editor retained-cache policy; a benchmark may assign a smaller
  inference/mask sub-budget. Oversized entries render but are not retained.

Frozen render requests receive a single mask/composite context snapshot containing
immutable array references and IDs. Workers never reread live mask UI state. Exact
exports create a dedicated pipeline and mask context at launch. Engine, effect,
source, mask, and outside snapshots are each read once.

## Inference lifecycle and concurrency

- A dedicated inference scheduler is independent from render scheduling/caches.
- At most one inference runs per editor plus one complete latest trailing request.
- Request fields include generation, source identity, exact local model identity,
  preprocessing version, and cancellation token.
- Session creation is lazy and synchronized; one validated session is reused.
- Preprocess at the model's documented resolution. Postprocess the confidence map
  to source resolution exactly once before publication.
- Cancellation is cooperative at safe boundaries: before session creation, after
  preprocessing, after inference, and during source-resolution postprocessing
  where chunking is safe. ONNX calls are not forcibly terminated.
- Success, no-subject, cancelled, and failure are distinct terminal outcomes.
  Exceptions are logged locally and always release scheduler busy state.
- A result publishes only when request generation, source identity, and model hash
  still match. Stale success is discarded and cannot replace a newer mask.
- Inference and interactive Numba rendering must not intentionally oversubscribe
  CPU. Benchmark one ORT intra-op thread setting at a time against the existing
  interactive budget; choose the lowest-latency responsive configuration.

The v1 single-output adapter uses stable candidate ID `primary`. “No clear subject”
is decided from documented confidence/area/degeneracy rules, not visual guesswork.
It preserves the last valid mask after a failed re-detect; without one, Smart Mask
does not alter the image and clearly reports the state.

## Preview, export, and format behavior

| surface | v1 behavior |
|---|---|
| Capped preview | Complete pipeline and derived mask render at the same capped geometry. Overlay is added only for display. |
| Full Quality Preview | Uses source-resolution master mask; never a preview-derived mask promoted upward. |
| PNG | Exact source dimensions. Saves straight RGBA when output/source has alpha; otherwise RGB is allowed. |
| JPEG | Exact source dimensions and opaque RGB. Any transparency is flattened against white using the same tested compositor; UI warns once/labels this behavior. |
| SVG | Smart Mask export unavailable in v1 with a clear message; existing unmasked SVG remains unchanged. |
| Batch | Smart Mask unsupported in v1; existing unmasked batch behavior unchanged. |
| Video/animation preview/export | Smart Mask unsupported in v1; existing behavior unchanged. Tracking or per-frame inference is not approximated with a stale still mask. |

Preview pixels and preview masks never feed export. Export cancellation/progress
uses existing exact export contexts and does not share mutable editor state.

## Presets and source changes

Creative presets store reusable Smart Mask settings: enabled, target, sensitivity,
feather pixels, expansion pixels, invert, and outside mode. They exclude probability
maps, master-mask pixels, source identity, candidate ID, overlay state, progress,
errors, and local asset paths.

Loading an enabled preset against a source with no valid matching inference result
requests detection after the preset is applied. Loading it with no model reports
Model unavailable and leaves output unmasked. Old presets default Smart Mask Off.
Source byte, dimensions, decode-alpha contract, model hash, or preprocessing-version
changes invalidate source-specific inference and masks.

## Errors and recovery

- **Missing/corrupt/incompatible model:** detection controls explain the local
  problem; rendering stays unmasked; no download action exists.
- **No clear subject:** keep source and settings; show status and allow sensitivity
  adjustment or Re-detect. Never publish an all-zero/all-one mask as success.
- **Inference exception:** one failure terminal outcome, scheduler recovers, last
  valid mask remains, and technical details go to local logs.
- **Out of memory:** discard partial results, clear only safe mask/inference cache
  entries, report failure, and keep normal rendering usable.
- **Source replaced or request superseded:** cancel/obsolete old work; stale results
  never paint or cache under the new source.
- **Unsupported export:** fail before rendering with a clear format/scope message.

## Performance and quality program

### Fixtures

Use redistributable or synthetic fixtures with documented origin/license for:
portrait/hair, product, animal, full body, multiple people, busy background,
low contrast, transparent source, thin structures, and no clear subject. Keep a
small licensed ground-truth set; do not commit generated/third-party assets until
provenance policy is approved.

### Measurements

Record cold runtime/session load, warm inference, preprocessing/postprocessing,
preview-mask latency, 1080p/4K geometry and compositing, exact export latency,
peak/retained RSS, cancellation-to-terminal time, stale-result rejection, and UI
heartbeat. Compare U2NETP against eligible quality challengers at documented input
sizes and report CPU separately from any optional accelerator.

### Acceptance budgets

- Warm CPU inference median ≤500 ms and p95 ≤800 ms on the documented reference
  Windows machine; cold first detection ≤2.0 s excluding application startup.
- Sensitivity/target/invert updates from cached confidence ≤50 ms at 1080p and
  ≤150 ms at 4K; feather/expand/composite combined ≤100 ms at 1080p and ≤300 ms
  at 4K, measured separately from the unchanged render branch.
- Obsolete/cancel request reaches terminal scheduler state within 100 ms of the
  next safe boundary; UI heartbeat p95 gap stays ≤100 ms during CPU inference.
- Peak and retained memory are measured before finalizing the cache sub-budget;
  total retained editor caches remain ≤192 MiB. No unbounded session/result growth
  across 50 detections/source replacements.
- Licensed ground-truth aggregate foreground Dice ≥0.90 and IoU ≥0.82, with no
  category (portrait/product/animal/full-body) Dice below 0.82. Boundary F-score
  at a documented tolerance ≥0.80. Thresholds may be raised after baseline; they
  may not be lowered merely to approve the provisional model without user review.
- Manual QA must pass hair/fur, thin objects, holes, multiple people treated as one
  salient foreground, low contrast, busy background, transparency, and no-subject
  behavior. Metrics do not replace visual edge review.
- Disabled Smart Mask is byte-identical across the existing full JIT-off suite and
  focused real-JIT render/color/style tests. Full-resolution exports remain exact.

If the provisional model misses quality budgets, optimize preprocessing only when
it preserves the documented model contract, or choose a better eligible model.
Do not hide quality loss through excessive feathering.

## Testing strategy

- TDD every task: red → green → refactor, then a safe commit boundary.
- Pure Qt-free unit tests for mask validation, sensitivity mapping, geometry,
  deterministic resize, compositing/alpha, identities, cache bounds, manifests,
  and adapter protocol.
- Adapter contract tests use a tiny synthetic ONNX fixture or fake session; real
  approved weights are an explicit opt-in integration gate, not required silently.
- Worker tests cover success/no-subject/cancel/failure, exactly one terminal signal,
  latest-wins publication, source replacement, and exception recovery.
- Differential tests cover disabled equality, RGB/RGBA/grayscale, Source Colors,
  Colored Dither, all outside modes, stage-neighborhood edges, preview/full/export,
  and one-read engine/effect/mask snapshots.
- Offscreen Qt tests cover exact control enablement, overlay exclusion, progress,
  missing model, presets, export warnings, and rapid re-detect/source replacement.
- Focused JIT-off tests run after every task. Real JIT is required when render,
  Numba thread interaction, source-color paths, or performance are touched.
- Frozen Windows packaging smoke test proves runtime/model discovery with networking
  disabled and verifies missing/corrupt model behavior.

## Security, privacy, and clean-room guarantees

- Image pixels, masks, filenames, model status, and metrics never leave the process.
- No networking dependency or dormant network code is introduced.
- Model and runtime are content-addressed and verified locally before loading.
- Parse only the fixed approved ONNX model; do not accept arbitrary preset-supplied
  model paths. Presets cannot select executables/providers or escape asset roots.
- Ship complete license, attribution, provenance, and modification records.
- Do not use Studio AAA/Dither Boy code, strings, binaries, endpoints, models, or
  implementation details. Public papers and official open-source model/runtime
  documentation are the only external implementation references.

## Alternatives rejected

- **Masking individual internal stages:** changes frozen semantics and creates edge
  inconsistencies. One complete rendered branch plus outer compositor is simpler.
- **Running the pipeline only on a cropped subject:** neighborhood operations and
  dither coordinates change at crop edges, breaking style continuity.
- **Runtime model download:** violates the offline/privacy requirement and makes
  first use nondeterministic.
- **PyTorch as default runtime:** excessive packaging/runtime footprint for a single
  inference model when ONNX Runtime can satisfy the tested contract.
- **MODNet as sole default:** portrait-only and fails product/animal scope.
- **MediaPipe selfie/semantic model as sole default:** person-only or fixed classes,
  not arbitrary salient foreground.
- **IS-Net weights immediately bundled:** weight redistribution is not explicit.
- **Fake subject variations from sensitivity thresholds:** misleading; sensitivity
  is already a separate, explainable control.
- **Storing masks in presets:** masks are source-specific and can silently apply to
  the wrong content.
- **Using one still mask for video/animation:** motion requires tracking or per-frame
  inference and would visibly drift.
- **Automatic detection on every load:** wastes resources and surprises users who
  have not enabled masking.

## Acceptance criteria

1. Every model/license/provenance hard gate is documented and passed before an
   asset enters a release; runtime operation is completely offline.
2. Smart Mask Off remains byte-identical and adds no mask-path allocations/work.
3. Subject, Background, Whole Image, Invert, sensitivity, geometry, and all four
   outside modes follow the state table exactly.
4. Every current style, Source Colors, Colored Dither, effects stack, and invert
   works inside the selected region without changing internal stage order.
5. PNG preserves straight RGBA; JPEG flattens to white; masked SVG/batch/video/
   animation are explicitly unavailable in v1 without changing unmasked behavior.
6. Preview uses deterministic derived masks; Full and export use exact source masks.
7. Immutable source/engine/effect/mask/outside snapshots and separate bounded caches
   prevent TOCTOU races, stale publication, corruption, and unbounded memory.
8. Detection always emits one terminal outcome and never wedges rendering after
   success, no-subject, cancellation, source replacement, or exception.
9. Presets store reusable settings only; old presets remain compatible; source
   changes invalidate source-specific masks.
10. Performance, quality, memory, packaging, JIT-off/JIT-on, and manual QA budgets
    pass on documented fixtures and hardware.
11. The implementation adds no network, telemetry, runtime-download, proprietary,
    or unapproved binary/model behavior.

## Approval gate

Approval freezes v1 scope, terminology, defaults, format behavior, model hard
gates, and acceptance budgets for the detailed implementation plan. Any material
change returns to this specification for revision before planning continues.
