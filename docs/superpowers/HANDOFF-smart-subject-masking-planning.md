# Handoff — Smart Subject/Background Masking

Paste this into a fresh Codex or Claude Code session opened at
`C:\Users\arsha\Desktop\custom dither`.

---

We need to design and plan a new **Smart Mask** feature for ditherzam using the
project's full **spec → implementation plan → task-list** workflow. This should
be a reusable masking system around all styles, color modes, and effects—not a
single dither style or a “Smart Styles” category.

Do not begin product implementation in the planning session. Investigate the
tree, resolve design choices with the user, then produce the three approval-gated
planning artifacts described below.

## Start here

1. Invoke `zam-memory` in recall mode.
2. Read `docs/memory/INDEX.md`, every constraint, the latest progress, and all
   entries relevant to render order, worker safety, previews, exports, color,
   source-faithful colors, and style controls—especially 002, 003, 022, 026,
   037, 042–046.
3. Inspect `git status` and preserve all unrelated/user-owned changes. The tree
   may contain a large uncommitted style/color-control pass.
4. Verify memory claims against the current files before relying on them.
5. Read representative existing artifacts in `docs/superpowers/specs/`,
   `docs/superpowers/plans/`, and `docs/tasklists/` so the new documents match
   repository conventions and detail level.

## Product goal

Add offline subject detection and reusable masking so a user can apply the
current render pipeline to:

- the whole image;
- the detected subject only; or
- the detected background only.

The user must also be able to choose what happens outside the processed region:

- retain the original image;
- retain the normal unfiltered pipeline input/output, if that is meaningfully
  distinct and clearly specified;
- transparent;
- white;
- black.

This masking must work independently of the selected dither style. It should be
usable with wave/modulation styles, color modes (including Source Colors and
Colored Dither), effects, still export, and—after explicitly scoped decisions—
batch/video/animation.

## Required UX to design

At minimum, design these controls and their exact state behavior:

- **Smart Mask enabled**
- **Target:** Subject / Background / Whole Image
- **Detected subject or variation:** define whether this selects among multiple
  instances, changes model prompts/thresholds, or cycles ranked masks; do not use
  a vague slider whose behavior cannot be explained
- **Confidence / sensitivity**
- **Edge feather**
- **Expand/contract mask**
- **Invert mask**
- **Outside region:** Original / Transparent / White / Black
- **Mask preview/overlay** with clear on/off behavior
- **Re-detect** and cancellation/progress behavior

Determine whether manual refinement is in v1. Consider a practical staged scope:
automatic mask first, then optional brush add/subtract, subject cycling, and mask
import/export. Do not silently expand v1 into a full Photoshop-style layer system.

## Offline model constraint

The user approved bundling a local segmentation model for better detection.
There must be **no runtime model download, network request, telemetry, licensing
check, or remote inference**. The feature must work fully offline.

Before selecting a model/runtime:

1. Research current authoritative model cards and runtime documentation.
2. Compare at least three viable offline subject/foreground segmentation options
   on license, redistribution rights, model size, CPU latency, memory, output
   quality, Python 3.12/Windows support, GPU optionality, and packaging impact.
3. Prefer a permissively licensed, redistributable model with documented
   provenance. Record the exact model version, source URL, checksum, license,
   attribution requirements, preprocessing, output semantics, and input size.
4. Decide whether the model is committed, downloaded only by a developer
   packaging script, or shipped as a release asset. Runtime downloading is not
   allowed.
5. Keep model/runtime code outside the Qt-free core boundary where appropriate,
   but keep mask math, compositing, geometry, settings mapping, and inference
   adapters independently testable and Qt-free.
6. Define graceful behavior when the optional model asset/runtime is absent or
   incompatible. Never pretend a heuristic is AI detection.

Potential technologies are candidates, not assumptions: ONNX Runtime with a
small segmentation model, a locally bundled PyTorch model, or another fully
offline runtime. Measure before choosing.

## Architecture questions the spec must settle

### Mask representation and ownership

- Canonical mask dtype/range/shape and whether it is immutable.
- Source-resolution master mask versus preview-resolution derived masks.
- Mask identity/version in render-cache keys.
- Ownership across GUI, workers, exact export contexts, video frames, and preset
  state.
- Whether masks are stored in presets, saved as sidecar assets, regenerated, or
  intentionally excluded because they are source-specific.
- Behavior when image dimensions or source content change.

### Pipeline and compositing

- Preserve the frozen internal render stage order:
  contrast → midtones → highlights → blur → dither → color → saturation →
  effects → invert.
- Prefer masking/compositing around a complete rendered branch unless evidence
  shows a stage-specific mask is required. Specify exactly what “apply filter to
  subject/background” means for blur, diffusion, glow, chromatic aberration, and
  other neighborhood operations at mask edges.
- Define original/outside pixels for grayscale-only input and RGB source input.
- Specify alpha handling through preview conversion, PNG export, JPEG flattening,
  SVG, batch, video, and animation. Transparent output likely requires an RGBA
  contract instead of merely painting white.
- Specify premultiplied versus straight alpha and feathered-edge compositing.
- Ensure Source Colors uses original RGB spatial hues correctly inside masked
  regions and Colored Dither colors the marks themselves.

### Subject inference lifecycle

- Detection trigger: on load, on enable, explicit button, or background idle.
- Asynchronous worker with progress, cancellation, latest-wins behavior, stale
  result rejection, and exactly one terminal outcome.
- Separate inference caching from render caching.
- Preview mask generation versus exact full-resolution mask refinement.
- Multiple-subject ranking and stable variation/instance IDs.
- CPU thread budgets and interaction with Numba render threads.
- Warmup/model-session reuse and bounded memory.

Preserve the worker/coalescer guarantees from memory entry 022: exceptions and
cancellation must never wedge the UI, engine/effect/mask snapshots must be read
once, and stale results must not publish.

## Performance and quality investigation

Benchmark representative portrait, product, animal, full-body, multiple-person,
busy-background, low-contrast, transparent-source, and no-clear-subject images.
Use redistributable test fixtures or synthetic fixtures with documented origins.

Measure:

- cold model load;
- warm inference at chosen input sizes;
- CPU and optional GPU latency;
- peak and retained memory;
- preview mask latency;
- exact export latency;
- feather/expand/composite cost at 1080p and 4K;
- cancellation/stale-result responsiveness.

Define quantitative mask-quality acceptance using a small licensed ground-truth
fixture set where practical (IoU/Dice plus boundary quality), and a manual QA
matrix for cases where metrics are insufficient.

## Compatibility and constraints

- Clean-room: no Studio AAA/Dither Boy source, strings, binaries, endpoints, or
  copied implementation details.
- No network, telemetry, licensing, or runtime model downloads.
- Python 3.12 via `.venv/Scripts/python.exe`.
- Core remains Qt-free; only approved UI/app/worker locations import PySide6.
- TDD: red → green → refactor, with focused JIT-off and real-JIT validation where
  Numba paths are involved.
- Preserve exact unmasked output when Smart Mask is disabled.
- Preserve full-resolution export; preview caps must not silently cap export
  masks.
- Preserve immutable render requests, cancellation semantics, exact export
  contexts, cache bounds, source-color behavior, presets, and current style
  controls.
- Do not commit model weights, generated fixtures, or binaries until their
  license, provenance, size policy, and `.gitignore` handling are approved.

## Required planning deliverables

Produce these in order, with explicit approval gates:

### 1. Design specification

Write:

`docs/superpowers/specs/YYYY-MM-DD-smart-subject-masking-design.md`

It must include:

- user stories and non-goals;
- UX/state model and wireframe-level control placement;
- model/runtime comparison with a recommended option;
- asset provenance/licensing/packaging policy;
- mask and alpha data contracts;
- inference lifecycle and concurrency;
- render/composite architecture and cache keys;
- preview/export/batch/video/animation behavior;
- preset/source-specific persistence decisions;
- error, missing-model, no-subject, and multi-subject behavior;
- performance and quality budgets;
- security/privacy/offline guarantees;
- alternatives rejected and why;
- staged v1/v2 scope;
- explicit acceptance criteria.

Present the specification to the user and obtain approval before writing the
implementation plan. Revise the spec when decisions change.

### 2. Detailed implementation plan

After spec approval, write:

`docs/superpowers/plans/YYYY-MM-DD-smart-subject-masking.md`

The plan must be dependency ordered and split into small TDD tasks. Every task
must state:

- goal and prerequisite;
- exact files/symbols to add or modify;
- failing test to write first and expected failure;
- implementation steps;
- focused test command;
- JIT-off/JIT-on requirement;
- manual QA where needed;
- completion/acceptance condition;
- safe commit boundary and proposed commit message.

Include explicit tasks for model licensing/provenance, inference adapter, mask
math, compositing/alpha, immutable request snapshots, cancellation, cache keys,
UI, persistence, exact export, format behavior, preview caps, batch/video scope,
benchmarks, packaging, and end-to-end QA.

### 3. Execution task list

After plan approval, derive a concise checkbox execution ledger:

`docs/tasklists/09-smart-subject-masking-tasks.md`

The task list must map one-to-one to plan task IDs and include:

- phase/wave grouping;
- dependencies;
- red/green/refactor checkboxes;
- exact verification commands;
- model/license asset gates;
- manual QA gates;
- documentation/memory update gates;
- merge/readiness checklist.

Do not replace the detailed plan with the task list. The task list is the live
execution surface; the spec remains the product contract and the plan remains
the engineering recipe.

## Final handoff behavior

At the end of the planning session:

1. Ensure spec, plan, and task list agree exactly on scope, terminology, defaults,
   formats, and acceptance criteria.
2. Record approved decisions and the next executable task through `zam-memory`,
   deduplicating `docs/memory/INDEX.md`.
3. Provide the next implementation session with the exact starting task, tests,
   branch recommendation, and any required model/license prerequisites.
4. Do not implement product code until the three artifacts are approved.

Begin with memory recall, tree inspection, and focused user questions about v1
scope, transparent output formats, multiple subjects, manual refinement, and
acceptable bundled model size. Then research and write the specification first.
