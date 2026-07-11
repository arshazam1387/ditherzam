# Handoff — Continue High-Resolution Performance Program (Wave 3 → Wave 4)

Open a fresh Claude Code session at `C:\Users\arsha\Desktop\custom dither` and use
the prompt below.

---

Continue the approved high-resolution performance program for ditherzam. Invoke
`zam-memory` first: read `docs/memory/INDEX.md`, all `constraint` entries, and
entries 020, 022, 023, 026, 030, 031, 032, 033, 034. Read the approved design and
plan completely:

- `docs/superpowers/specs/2026-07-09-high-resolution-preview-performance-design.md`
- `docs/superpowers/plans/2026-07-09-high-resolution-preview-performance.md`

## Repository state

- Branch: `feat/high-res-perf`
- Product HEAD: `89af350 refactor(render): dispatch tonal out= by cached signature, not try/except`
- Do NOT modify or commit `.codex/`.
- Uncommitted (intentionally, matching prior sessions): `docs/memory/031-034`,
  `docs/memory/INDEX.md`, this handoff file, and `.superpowers/sdd/` (git-ignored
  SDD ledger). Leave them uncommitted unless the user asks to commit; if they do,
  `git add docs/memory docs/superpowers` (not `.superpowers/`, not `.codex/`).
- No partial production edits after `89af350`.
- Preserve inherited Epsilon Glow work from `feat/epsilon-glow-tab`.

Recent commits (newest first): 89af350, 0a51a96, 29e2914, 3574bf3, d9284f7,
bd0cdc1, a9a80c9, 2dba236, 510418f, 5bf0525, bf962a2, 5eb0ee6, 44745d8, ad810c9,
6bf15e0, 306681c, 06cb1ee, 1bcb853.

## Workflow in use (continue it)

Subagent-Driven Development (`superpowers:subagent-driven-development`), sequential:
one fresh implementer subagent per task (model: sonnet unless a task needs opus),
then a task-reviewer subagent (or controller review for small/UI diffs), fix loop
for Critical/Important findings, then next task. Helper scripts in the SDD skill
dir: `scripts/task-brief PLAN N`, `scripts/review-package BASE HEAD`. Track every
task in the ledger `.superpowers/sdd/progress.md` (already reflects everything
done). Record durable decisions/gotchas via `zam-memory`. **Do NOT spawn parallel
implementers in the shared tree** (conflict); parallelism only via worktrees if
ever needed.

## Completed (Wave 1 + Wave 2 + Wave 3-to-date)

- Persisted preview resolution (Auto/480/720/1080/1440/2160/Full), capped drag +
  settled previews, async initial decode, `Ctrl+Enter` Full, optional bucketed
  zoom refinement (Auto-only), source-logical viewport (refit only on new source).
- Immutable prioritized `RenderRequest`/`RenderScheduler` (one in flight + priority
  trailing; `invalidate()` clears pending; terminal-failure recovery; no wedge).
- Exact compiled RGB Floyd–Steinberg; fused exact ordered mapping; fused ramp
  luminance (Option A scalar default, Option B legacy exact-BLAS behind env
  `DITHERZAM_RAMP_EXACT_BLAS` — see memory 034); content-keyed ColorContextCache.
- Exact saturation/RGB fusion; Epsilon Glow allocation reduction; 192 MiB atomic
  complete-group render-cache LRU; QImage zero-copy measured + rejected (033).
- **3.2** exact three-pass in-place tonal fusion (`29e2914`): one reused `out=buf`
  through `apply_contrast/apply_midtones/apply_highlights` (they gained an optional
  `out=` param; `out is None` = unchanged legacy path). `_accepts_out(fn)` (cached
  `inspect.signature`) selects the fused vs allocating branch. Byte-identical
  (frozen hashes + 100k adversarial sweep + e2e). One-pass forbidden (032).
- **3.6** cooperative stage-boundary cancellation (`0a51a96`+`89af350`):
  `render.RenderCancelled`; `render()`/`render_cached()` take `is_cancelled=None`,
  checked ONLY between complete stages; `RenderScheduler.should_cancel(req) =
  is_current(req) and _pending is not None`; `render_preview(..., is_cancelled=)`
  forwards; `_RenderWorker` takes `is_cancelled=`, catches `RenderCancelled` before
  generic `Exception`, emits a DISTINCT `cancelled` signal; `_on_render_cancelled`
  releases scheduler + launches trailing, never paints; cancelled `render_cached`
  publishes NO partial cache (commit is the last line). Export/exact callers omit
  the predicate and are never cancellable.

## Test state

- Full suite JIT-off: **864 passed / 0 failed** (the six once-red cancellation
  tests are green). Run with a controlled temp dir to dodge the Windows teardown
  symlink `PermissionError [WinError 5]` (a teardown artifact, not a failure):

```powershell
$env:QT_QPA_PLATFORM='offscreen'; $env:NUMBA_DISABLE_JIT='1'
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp='C:\tmp\dz-pytest'
```
Bash equivalent: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp="$TEMP/dz"`.
For JIT-on gates drop `NUMBA_DISABLE_JIT`; keep memory 020's seven historical
`special.py` float-index JIT-on failures separate from regressions.

## Exact next task — Plan 3.4 Workspace/buffer reuse

Treat 3.2's tonal `out=buf` private buffer as the first proven workspace win. Add
MORE reuse only with: exclusive per-call/context ownership (never a module-global
mutable workspace), no aliasing of any returned or cached array, retained buffers
accounted under the same 192 MiB cache budget, and a MEASURED benefit. Add tests
for consecutive-output stability and cached/plain concurrency. Commit boundary:
`perf(render): reuse exact scratch buffers safely`. If no further safe/measured win
exists beyond the tonal buffer, record that finding and move on — do not force it.

## Then continue in this order (each: TDD, focused + full gate, small commit)

1. **4.1 capped animation/video screen previews** — animation playback async /
   latest-wins at the selected cap; video display frames capped before QImage
   conversion; exported frames stay EXACT. Files: `ui/timeline_panel.py`,
   `ui/video_controller.py`. Commit `perf(media-ui): cap animation and video previews`.
2. **4.2 dedicated exact export contexts** — still/batch/video/animation snapshot
   source/settings/palette/effects at launch; never share the mutable interactive
   pipeline; preview preferences never enter export APIs; SVG keeps its exact
   contract. Commit `fix(export): isolate exact rendering from preview state`.
3. **4.3 thread scaling benchmark** → **4.4 bounded threading policy** — benchmark
   1/2/4/8 threads first; then a conservative Qt-free `threading_policy.py`; one
   interactive render in flight; diffusion stays sequential; prevent Qt×Numba×export
   oversubscription; reserve UI capacity during exports.
4. **4.5 JIT warmup refinement** — warm only selected optimized paths (nearest/
   ordered/ramp/saturation/diffusion/tonal) on the existing daemon thread, never
   the GUI thread; measure cold separately.
5. **4.6 acceptance matrix + real-photo 4K QA** — all caps/modes/K, cached
   mutations, effects, 1/2/4/8 threads, cold/warm, peak/RSS/retained bytes, output
   hashes; verify preview latency, ≥5× diffusion, ≤192 MiB retained cache, no
   source-sized capped alloc, exact Full/exports, no stale paints/wedges. Record
   before/after in `benchmarks/` and update `zam-memory`. Commit
   `docs: record high-resolution performance results`.

## Non-negotiable invariants

- Clean-room; no Studio AAA code/strings/binaries/network/licensing behavior.
- Core Qt-free; PySide6 only under `ui/`, `app.py`, `video/workers.py`.
- Preserve EXACT pixels/exports, first-minimum palette ties, two-level
  compatibility, depth promotion, threshold tone bias, frozen `STAGE_ORDER`.
- Memory 022: engine/effect one-read snapshots; exactly one terminal worker signal
  (finished XOR failed XOR cancelled); never reintroduce the coalescer wedge.
- Full preview/export stays exact and independent of preview quality. Ramp A/B
  toggle is GLOBAL (preview and export must never diverge).
- TDD; small commits; do not hide the JIT-on `special.py` failures or weaken tests.
- Inspect `git status` before every task; preserve unrelated/user changes.

When a task or durable decision finishes, update `docs/memory/` via `zam-memory`
and the SDD ledger. Do not commit `.codex/`.

---
