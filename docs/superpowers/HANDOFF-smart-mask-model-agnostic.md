# Handoff — Model-agnostic Smart Mask (user-installable models)

**Branch:** `feat/smart-subject-masking` (HEAD `c9ca4f7` at handoff)
**Depends on:** the SM-01..SM-17 checkpoint (see
`.superpowers/sdd/progress.md`, `docs/tasklists/09-smart-subject-masking-tasks.md`,
plan `docs/superpowers/plans/2026-07-11-smart-subject-masking.md`,
spec `docs/superpowers/specs/2026-07-11-smart-subject-masking-design.md`).
**Execute via SDD** (`superpowers:subagent-driven-development`): every task TDD +
independent review. Masking work is security/clean-room sensitive — do not skip review.

## Goal (owner-approved decisions)

Let users run Smart Mask with **their own model**, not just one hardcoded export.

1. **Bundle lite by default** — `u2netp` (~4.6 MB) ships/staged so Smart Mask works
   out of the box. (For the OWNER's machine this is already staged locally; see
   "Current state".)
2. **Bring-your-own (option B, approved):** accept a user-supplied model, marked
   **"unverified — use at your own risk."** No provenance/licensing gate for BYO —
   it's the user's own file. The strict *verified* provenance path stays intact for
   any future approved release model.
3. **Easy, offline UX:** NOT an in-app downloader (the spec + `test_offline_security.py`
   forbid any in-app network fetch — do NOT add one). Instead: a **"Get a model ↗"
   link** (opens the model's page in the user's browser), plus **drag-and-drop /
   file-picker** install. "Drag and play."

## Current state (already done, verified — do not redo)

- Owner's local copy WORKS: `u2netp.onnx` (sha256 `309c8469…4ddd8`, 4,574,861 bytes)
  staged at `assets/models/smart_mask/u2netp.onnx` with a local dev
  `assets/models/smart_mask/manifest.yaml`. Verified end-to-end through
  `OrtSegmentationAdapter` (subject 0.996 / background 0.002). `onnxruntime==1.22.1`
  installed in `.venv`.
- Both the model and the local manifest are **gitignored** (see `.gitignore`
  `assets/models/smart_mask/` rules) — never pushed. Nothing committed.
- This exact `u2netp.onnx` (rembg release build) **is** the graph the current strict
  contract was designed around: input `input.1` (1,3,320,320); 7 outputs
  `1959..1965` (each 1,1,320,320); primary `1959`. So the *lite* model already runs
  on the strict path with zero code changes.

## The core problem to solve

The adapter is hardwired to ONE export:
- `ditherzam/masking/ort_adapter.py:20-21` — `INPUT_NAME`/`OUTPUT_NAME` are module
  constants from `EXPECTED_INPUT_TENSOR`/`EXPECTED_OUTPUT_TENSOR` (`input.1`/`1959`),
  used verbatim in `session.run([OUTPUT_NAME], {INPUT_NAME: tensor})` (line 142) and
  in the constructor contract check (line 112).
- `ditherzam/masking/model_assets.py:51-52,107-108,215` — `load_manifest` and
  `ModelManifest` validation REJECT any manifest whose input/output tensor isn't
  exactly `input.1`/`1959`, plus `APPROVED_MODEL_IDS={u2net,u2netp}`, a pinned
  `upstream_commit`, and exactly-7 `output_names`.

A different/bigger export (e.g. full `u2net.onnx`) names its outputs differently, so
it's rejected today. Generalization = read IO identity from the manifest, and add an
**unverified BYO manifest path** alongside the strict verified path.

## Recommended scope: two tiers

**Tier A — U-2-Net family (do this first; contained, high value).**
Accept any export with the U-2-Net contract shape (input 1×3×320×320 float32; a
1×1×320×320 float32 primary output; N side outputs), reading the primary output NAME
from the manifest instead of the `1959` constant. Same preprocessing/postprocessing
(`preprocess_u2net`/`postprocess_probability` already generic). Covers lite + full
u2net + variants — the realistic "bigger model" set.

**Tier B — truly arbitrary models (optional, later; the "full project").**
Pluggable model families, each declaring input/output shapes, preprocessing
(mean/std or none), output selection, and postprocessing. Only pursue if a
non-U-2-Net family is actually wanted; it carries real silent-wrong-mask risk and a
bigger validation/UI surface.

## Task breakdown (SDD; TDD + review each)

1. **Manifest: verified vs unverified.** In `model_assets.py`, add an
   `unverified`/`byo` manifest kind (or `verified: bool`). For BYO: relax the exact
   `input.1`/`1959` equality and the pinned-commit / approved-id checks to STRUCTURAL
   checks (shapes/dtypes, primary present among outputs). Keep the strict verified
   path byte-identical for release assets. Preserve `_validate_relative_path` and the
   fail-closed hash/byte verification unchanged. Do NOT weaken `test_offline_security`
   or the clean-room invariants.
2. **Adapter: per-manifest IO.** In `ort_adapter.py`, derive input/output names from
   the manifest (`self._input_name`/`self._output_name`) instead of the module
   constants; relax the constructor contract check to structural + "primary present".
   Keep cancellation, lazy session, and single-primary-output semantics.
3. **Install/inspect helper (Qt-free core).** A function that, given a user ONNX path,
   loads it via onnxruntime, reads input/output names+shapes, validates the U-2-Net
   contract structurally, computes sha256/bytes, and writes a local **unverified**
   manifest. Reuse for both drag-drop and file-picker. No network.
4. **UI: Smart Mask install panel** (`ditherzam/ui/smart_mask_panel.py`). Add:
   a "Get a model ↗" link (open browser to the model page), a file-picker + drag-drop
   target, an "unverified model — use at your own risk" confirmation, and clear
   present/absent state. Wire to the inspect helper; fail closed with a message on a
   non-compatible file. PySide6 stays only under `ditherzam/ui/`.
5. **Settings: active model selection.** Persist which staged model is active
   (bundled lite vs an installed one). Must not enter creative presets; must not
   change exact-export authority; disabled Smart Mask stays byte-identical / zero work.
6. **Packaging (extends SM-17):** release bundles lite `u2netp` in the installer asset
   (still gitignored in-repo). Installed BYO models live in the user-data/asset dir.

## Invariants to preserve (copy into every review)

- Clean-room; NO in-app network/downloader/telemetry (`test_offline_security.py`).
- Smart Mask disabled == byte-identical, zero mask work (no hash/resize/infer/alloc).
- Frozen pipeline order; masking is an OUTER compositor only.
- Fail-closed on missing/mismatched/incompatible model; exact export uses source
  resolution and never preview/mask-preview authority.
- 192 MiB total retained-cache ceiling; immutable one-read snapshots; stage-boundary
  cancellation; worker terminal outcomes.
- The strict VERIFIED provenance path (approved id + pinned commit + hash + exact
  tensor contract) must remain available and unchanged for a future release model.

## Known caveats

- **71 pre-existing `test_kernels_all` golden failures** on the branch base
  (creative-dither), JIT-off — NOT masking-caused; flagged in the ledger. Must be
  resolved/waived before ANY merge to main. Don't let them mask real regressions.
- Do not commit model binaries or the local dev `manifest.yaml` (both gitignored).
- Full `u2net.onnx` (~168 MB) is the natural "bigger" model; confirm its actual output
  names by inspecting the file (do not guess) when testing Tier A.
