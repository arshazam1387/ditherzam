# HANDOFF — Smart Subject/Background Masking, SDD orchestration (SM-04 → SM-17)

**Date:** 2026-07-11
**Branch:** `feat/smart-subject-masking` (created off `4b99c4d`)
**HEAD at handoff:** `c9f8922`
**Method:** superpowers **subagent-driven-development** (fresh implementer subagent per
task → task review (spec + quality) → fix loop → mark complete; broad whole-branch
review at the very end, then finishing-a-development-branch).

You are taking over orchestration from SM-04 onward. SM-01/02/03 are done and
reviewer-approved. Do NOT re-dispatch completed tasks — the ledger + `git log` are the
recovery map.

---

## 0. Read these first (in order)

1. **Ledger (source of truth):** `.superpowers/sdd/progress.md` — task checklist,
   per-task log, global-constraints block (reviewer attention lens), Minor-findings
   roll-up for the final review, and the cross-cutting 71-golden-failures flag.
2. **Plan:** `docs/superpowers/plans/2026-07-11-smart-subject-masking.md` — the 17 tasks,
   dependency map, and per-task files/red-tests/acceptance/commit-message.
3. **Spec:** `docs/superpowers/specs/2026-07-11-smart-subject-masking-design.md`.
4. **Per-task briefs:** `.superpowers/sdd/task-NN-brief.md` (NN = 04..17). Each already
   contains the plan's Global Constraints + Verification conventions + that task's full
   section. Hand the brief PATH to each implementer as "read this first — your
   requirements with exact values verbatim." Do NOT paste task text or prior-task
   history into dispatch prompts.

---

## 1. User-approved execution scope (BINDING)

- **SM-04 .. SM-15:** build fully, green, reviewer-approved. Autonomous.
- **SM-16 (bakeoff) and SM-17 (packaging/E2E):** build **CODE SKELETONS ONLY** — the
  developer converter tool (`tools/convert_u2net_onnx.py`), benchmark harness
  (`benchmarks/smart_mask.py` completion), packaging config, and tests that are
  **red-pending-asset**. Then **STOP** and hand back to the user. Do **NOT**:
  - fetch/download model weights,
  - make a licensing/redistribution judgment,
  - commit any model binary or unlicensed fixture,
  - lower any SM-02 quality threshold to make a gate pass.
  These are explicit human gates per the plan's Approval Gate. The user must supply and
  approve licensed U2NET/U2NETP weights before SM-16/17 can truly complete.
- `onnxruntime` is **not installed**; unit tasks use fake/injected sessions. Only
  `pip install onnxruntime==1.22.1` into `.venv` if an optional integration path
  genuinely needs it (SM-08 has an optional approved-asset marker; SM-16/17 real runs).

---

## 2. Environment & conventions

- Python: `.venv/Scripts/python.exe` (3.12). Working dir: `C:\Users\arsha\Desktop\custom dither`.
- **Verify (JIT-off, per task):**
  `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 .venv/Scripts/python.exe -m pytest -q <targets>`
  then the full suite once before commit.
- **Real-JIT gates** (where the plan asks): remove `NUMBA_DISABLE_JIT`, run only the
  named targets.
- **Known-red baseline:** branch base has **71 pre-existing `test_kernels_all`
  golden-fixture failures** (from creative-dither commit `0335773`), unrelated to
  masking. Every task confirms it adds **ZERO** new failures beyond those 71. Do not
  touch kernel golden tests. **Flag these 71 to the user before any merge to main.**
- If pytest teardown throws a machine-specific temp `PermissionError`, work around with
  `--basetemp=C:/Users/arsha/AppData/Local/Temp/claude/<slug>` (prior tasks did this).
- **Never stage** `.codex/` or `purple harrow.png` — unrelated dirty-tree items that
  must stay uncommitted. Tell every implementer this explicitly.

## 2a. Recurring subagent failure mode (IMPORTANT)

The general-purpose implementers here **repeatedly detach a background pytest and end
their turn without committing or writing their report** (happened on SM-01 and SM-03).
Mitigations:
- In every implementer dispatch, state: "Run all pytest FOREGROUND/blocking. NEVER use
  a background/detached run. Do not end your turn until you have committed and written
  the report file."
- After each implementer returns, **verify from git yourself** before trusting status:
  `git log --oneline <BASE>..HEAD` (expect the task's commit) and `git status --porcelain`.
  If files exist but no commit/report, **resume the same agent** via SendMessage (keeps
  its context) telling it to run tests foreground, stage only its files, commit with the
  exact message, and write the report — rather than cold-restarting.

---

## 3. Per-task loop (mechanics)

Helper scripts live in the skill dir:
`C:/Users/arsha/.claude/plugins/cache/claude-plugins-official/superpowers/6.1.1/skills/subagent-driven-development/scripts/`

For each task N (04..17):

1. **Record BASE** = current HEAD (`git rev-parse HEAD`). Never use `HEAD~1` for the
   review package — multi-commit tasks (e.g. impl + fix) would be truncated.
2. **Dispatch implementer** (general-purpose). Prompt = (a) one line on where the task
   fits; (b) the brief path `.superpowers/sdd/task-NN-brief.md` as "read first,
   requirements verbatim"; (c) interfaces/decisions from earlier tasks the brief can't
   know (see §4); (d) resolution of any ambiguity you spotted; (e) report path
   `.superpowers/sdd/task-NN-report.md` + the <15-line status contract; (f) the
   foreground-pytest + don't-stage-junk + 71-known-red rules from §2/§2a.
   **Model:** Sonnet for most; escalate to Opus for the heavy integration tasks
   **SM-12 and SM-13** (multi-file editor/render coordination, concurrency, one-read
   snapshots). SM-10/SM-14 are standard (Sonnet ok). SM-16/SM-17 skeletons: Sonnet.
3. **Verify from git** (§2a). Handle status: DONE / DONE_WITH_CONCERNS (read concerns) /
   NEEDS_CONTEXT (provide + re-dispatch) / BLOCKED (assess: context? more capable model?
   split? escalate to user if plan is wrong).
4. **Review:** `scripts/review-package BASE HEAD` → prints a diff-file path. Dispatch a
   task reviewer (general-purpose, Sonnet; Opus for SM-12/13) using
   `task-reviewer-prompt.md`. Give it: brief path, the global-constraints block (copy
   from ledger verbatim — it's the attention lens), report path, BASE/HEAD, the diff
   path, and **task-specific named risks** to check (don't add open-ended "check
   everything"; don't pre-judge or tell it what not to flag).
5. **Fix loop:** dispatch fix for Critical/Important (resuming the implementer agent is
   efficient — it holds file context). Fix must re-run covering tests and report the
   command+output. Then re-review (resume the reviewer agent — it holds the findings).
   Repeat until spec ✅ AND quality Approved. Log **Minor** findings in the ledger for
   final-review triage; don't fix now.
   - A finding that contradicts what the plan mandates = the **user's** decision:
     present finding + plan text, ask which governs. Don't silently override the plan.
   - Resolve any reviewer "⚠️ cannot verify from diff" item yourself (you hold
     cross-task context) before marking complete.
6. **Mark complete:** flip the `[ ]`→`[x]` in the ledger Tasks list AND append a one-line
   `Log` entry naming the commit range + review outcome, in the same turn.

**Serial only:** one implementer at a time (parallel implementers conflict on the tree).
The dependency map in the plan allows some parallelism in principle, but this single
working tree is not worktree-isolated — keep it serial.

---

## 4. Cross-task interfaces already established (feed to implementers as needed)

- `ditherzam/masking/` package exists, **Qt-free**, imports only stdlib + numpy + yaml.
  `tests/test_offline_security.py` enforces this via a substring scan + AST import
  allowlist; `FORBIDDEN_TOKENS` = urllib/requests/socket/http/download. If a new
  masking module legitimately needs a new stdlib/compute import, the implementer must
  add it to that allowlist (SM-02 added numpy/types, SM-03 added enum) — but NEVER a
  networking module.
- **SM-01** `ditherzam/masking/model_assets.py`: `ModelManifest`, `ModelAssetError`,
  `load_manifest(path)`, `verify_model_asset(asset_root, manifest)`, `default_asset_root()`,
  streamed SHA-256 + byte count, U-2-Net upstream pinned to
  `ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29`. Asset root under `assets/models/smart_mask/`
  (weights gitignored). `tools/stage_smart_mask_model.py` is the ONLY fetch site.
- **SM-02** `ditherzam/masking/quality.py`: `dice_score`, `iou_score`, `boundary_f_score`,
  `aggregate_quality`, `select_model_candidate`. Threshold constants frozen (Dice≥0.90,
  IoU≥0.82, per-category Dice≥0.82, boundaryF≥0.80, warm median≤500ms/p95≤800ms,
  cold≤2.0s). Winner policy: full-U2NET wins only if eligible AND within budget AND
  manually approved AND ≥0.03 aggregate IoU-or-boundaryF margin; else eligible U2NETP;
  else unavailable. `benchmarks/smart_mask.py` is a SKELETON (complete it in SM-16).
  `DEFAULT_BOUNDARY_TOLERANCE_PX=2` is provisional — confirm against real fixture
  resolution in SM-16.
- **SM-03** `ditherzam/masking/contracts.py`: `SourceIdentity`, `ModelIdentity`,
  `InferenceIdentity`, `MaskIdentity`, `ProbabilityMap`, `source_identity(rgba_u8)`,
  `validate_confidence_array`. `ditherzam/masking/settings.py`: `MaskTarget`,
  `OutsideMode`, frozen `SmartMaskSettings`.
  - **Identity is CONTENT-based** (hash of `array.tobytes(order="C")`), never path.
  - Canonical straight `uint8 RGBA`; confidence/mask are immutable **owned** C-contiguous
    `float32[H,W]` in `[0,1]` (ProbabilityMap takes a defensive copy, then sets
    `writeable=False`). ProbabilityMap uses `eq=False` + identity-keyed `__eq__`/`__hash__`
    (raw ndarray field would crash generated eq/hash — don't reintroduce that pattern).
  - **Exact defaults:** `enabled=False, target=SUBJECT, sensitivity=50, feather_px=8,
    expansion_px=0, invert=False, outside=ORIGINAL`.
  - **MaskIdentity deliberately EXCLUDES `outside`/OutsideMode** (masking is an outer
    compositor; outside-mode belongs to the composite step, not mask identity). Keep this.
  - Expansion range `±64` is spec-provisional; if SM-11/SM-04 need a different bound,
    that's a plan-consistency check, not a free change.

**Open Minor findings** (in ledger, for final whole-branch review triage): helper-fn
duplication between contracts.py:53-69 and settings.py:55-71; `boundary_f_score`
`tolerance_px` isinstance(int) accepts `bool`; a couple of doc/default nits. Point the
final reviewer at the ledger's Minor list.

---

## 5. Next task = SM-04

Deterministic mask geometry + preview resize. Depends on SM-03. BASE = `c9f8922`.
Files: `ditherzam/masking/geometry.py` (`sensitivity_threshold`, `derive_master_mask`,
`expand_contract`, `feather`, `resize_mask_area`, algorithm-version constants),
`tests/test_mask_geometry.py`. Prefer focused NumPy/Pillow already in the project; do
NOT add SciPy/Numba without a measured need. Order: target → invert → signed geometry →
feather. Whole Image returns semantic all-ones WITHOUT inference. Commit:
`feat(mask): add deterministic mask geometry`. Brief: `.superpowers/sdd/task-04-brief.md`.

Then SM-05, SM-06, SM-07 (SM-07 modifies `main_window.py` decode path — first UI-adjacent
task; retain source RGBA, don't premultiply). Continue per plan/dependency map.

---

## 6. Finish (after SM-15, and after SM-16/17 skeletons)

1. Broad whole-branch review: `scripts/review-package $(git merge-base main HEAD) HEAD`
   → dispatch the final code-reviewer (superpowers:requesting-code-review template) on
   the **most capable** model. Feed it the ledger's Minor-findings list for triage.
2. One fix subagent for the whole final-findings list (not one per finding).
3. `superpowers:finishing-a-development-branch`.
4. **STOP for the user** on SM-16/17 real assets. Surface: (a) the 71 pre-existing kernel
   golden failures, (b) that SM-16/17 need licensed weights + sign-off, (c) that
   `onnxruntime==1.22.1` install is deferred.
5. Record a zam-memory `progress` entry updating `048-smart-mask-sdd-execution.md`.

Do not merge to main without the user; the 71 golden failures must be resolved or
explicitly accepted first.
