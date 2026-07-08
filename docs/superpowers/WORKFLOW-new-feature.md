# Workflow — Building a New Feature (ditherzam)

This is the exact pipeline used to ship color Sub-projects A, B, and C. It is for
**new features only** — a bug fix uses `superpowers:systematic-debugging` instead.

The shape: **decompose → spec → plan → orchestrated subagent build → merge → remember.**
Each stage has an owning skill and a durable artifact. The orchestrator (the main
session) never writes feature code itself — it curates context and dispatches
subagents, one task at a time.

```
recall ──▶ brainstorm ──▶ spec.md ──▶ plan.md ──▶ per-task briefs ──▶ subagents ──▶ merge ──▶ memory
(zam-    (superpowers:   (specs/)    (plans/)    (task-brief          (SDD +        (finish   (zam-
 memory)  brainstorming)                          script)             reviewers)     branch)   memory)
```

---

## 0. Recall (start of every session)

Invoke **`zam-memory`** (recall). Read `docs/memory/INDEX.md`, every `constraint`
entry, and the latest `progress` entry. This tells you the current phase, the next
task, and the live invariants before you touch anything.

## 1. Decompose + brainstorm → `spec.md`

Invoke **`superpowers:brainstorming`**.

- If the request is large, **break it into sub-projects** first. Each sub-project
  gets its own spec → plan → build cycle. (Color work was split A = engine,
  B = editing UX, C = library. Don't try to spec three subsystems at once.)
- Ask **one clarifying question at a time** before proposing a design. Prefer
  multiple-choice. Nail down the *one* real design decision early (for C it was
  "where does category metadata live").
- Propose 2–3 approaches with a recommendation, then present the design in sections.
- On approval, write the spec to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
  and **commit** it. Spec must list: Goal, In scope, Out of scope (YAGNI), Invariants,
  Architecture (per component), Testing, Risks.

**Invariants block is mandatory** — copy the live ones verbatim so every later stage
inherits them:
- Clean-room (our own code; Dither Boy is behaviour inspiration only).
- Qt-free core (`ditherzam/color/**`, `dithering/**` never import PySide6; only
  `ui/`, `app.py`, `video/workers.py` do).
- Frozen `RenderPipeline.render()` stage order + `RenderSettings` fields untouched;
  UI-session prefs live in `panel.state` only, never `RenderSettings`.
- Python 3.12; TDD per task. Suite is green in **JIT-off** mode
  (`NUMBA_DISABLE_JIT=1`, set by `tests/conftest.py`). 7 kernel tests fail JIT-**on**
  (`special.py`) — pre-existing, not your feature's problem.

## 2. Plan → `plan.md`

Invoke **`superpowers:writing-plans`**.

- Write to `docs/superpowers/plans/YYYY-MM-DD-<feature>.md` and **commit**.
- Start with the header block + a **Global Constraints** section (the spec's
  invariants, verbatim). Every task implicitly inherits it.
- Break into **bite-sized TDD tasks**. Each task: exact file paths, an Interfaces
  block (what it consumes from earlier tasks / produces for later ones — exact
  signatures), then steps: write failing test → run it, see it fail → minimal
  implementation → run, see it pass → commit. **Full code in every step. No
  placeholders.** A fresh subagent reads only its own task, so names/types must be
  consistent across tasks.
- **Verify every class/name against the actual tree** while writing. (B's plan
  mislabeled `ImageEditor` as `MainWindow` and cost a mid-build escalation.)
- Self-review: spec coverage (every requirement maps to a task), placeholder scan,
  type consistency.

## 3. Orchestrated build (plan.md → per-task briefs → subagents)

Invoke **`superpowers:subagent-driven-development`**. This is where `plan.md` gets
turned into per-task work handed to subagents. The user's "task.md" == the per-task
**brief** the `task-brief` script extracts from the plan.

**One-time setup:**
- **Branch off `main` in-place — NOT a git worktree.** The repo-root `.venv` does
  not exist inside a worktree, so the test runner breaks there. (This overrides the
  SDD skill's default worktree suggestion for *this* repo.)
  ```bash
  git checkout -b feat/<feature>
  ```
- Create the ledger `.superpowers/sdd/progress.md` (git-ignored scratch): list every
  task as `- [ ]`, record the branch merge-base. **The ledger is the recovery map** —
  if context is lost/compacted, trust it + `git log`, never re-dispatch a task it
  marks done. (Mid-build the API session limit was hit once; the ledger + git log
  made resume trivial.)

**Scripts** (in the SDD skill dir
`~/.claude/plugins/cache/claude-plugins-official/superpowers/<ver>/skills/subagent-driven-development/scripts/`):
- `task-brief <plan.md> <N>` → writes `.superpowers/sdd/task-N-brief.md`, prints the path.
- `review-package <BASE> <HEAD>` → writes a single diff file (commits + stat + full
  diff), prints the path. **Always use the BASE recorded before the task**, never
  `HEAD~1` (drops multi-commit tasks).

**Per task, in order:**
1. `task-brief plan.md N`. Record the current HEAD as this task's BASE.
2. **Dispatch one implementer subagent** (see the skill's `implementer-prompt.md`).
   Hand it: the brief path (its requirements), where the task fits, interfaces from
   earlier tasks it can't know, and a report-file path
   (`.superpowers/sdd/task-N-report.md`). Everything heavy moves as **files**, not
   pasted into the prompt.
   - **Model:** cheapest that fits. Transcription (full code in the brief) → **haiku**.
     Integration / multi-file / judgment → **sonnet**. Always set the model explicitly.
3. Handle status: DONE → review. DONE_WITH_CONCERNS → read concerns first.
   NEEDS_CONTEXT → give context, re-dispatch. BLOCKED → give context / bump model /
   split the task / escalate to the human.
4. `review-package BASE HEAD`. **Dispatch a task reviewer subagent** (see
   `task-reviewer-prompt.md`) with the brief, the report, the diff-file path, and the
   binding constraints verbatim. Never tell a reviewer what not to flag or pre-rate a
   finding.
5. Fix loop: dispatch **one** fix subagent for Critical/Important findings (re-runs
   the covering tests, appends to the report). Log Minor findings in the ledger for
   final triage. Re-review after fixes.
6. Mark the task `[x]` and append `Task N: complete (commits <base7>..<head7>, review
   clean)` to the ledger.

**After all tasks: final whole-branch review** on **opus** (most capable). Package it
with `review-package <merge-base> HEAD`. Point it at the ledger's logged Minor
findings to triage what must land before merge. If it returns findings, dispatch
**one** fix subagent with the whole list.

## 4. Finish → merge

Invoke **`superpowers:finishing-a-development-branch`**.

- Verify the full suite yourself: `./.venv/Scripts/python.exe -m pytest -q` (expect
  the JIT-off count; the one `QMouseEvent` deprecation warning is pre-existing).
- Present the 4 options; **let the human choose** (merge is theirs to make). The
  established pattern is **merge to `main` locally, `--no-ff`**, re-verify, delete the
  branch. Nothing is pushed unless the human asks (`main` runs ahead of `origin`).

## 5. Remember

Invoke **`zam-memory`** (record). Add a `progress` entry — feature shipped + merge
commit + green count + what's still deferred — and refresh its one-line `INDEX.md`
pointer. Commit as `chore(memory): …`. Delete any stale in-progress memory a subagent
dropped mid-build.

---

## The rules that keep it clean

- **Orchestrator curates, subagents build.** The main session writes no feature code.
- **Files over pasted text.** Briefs, reports, and diffs are files; prompts stay small.
- **Fresh subagent per task.** Never paste prior-task history into a later dispatch —
  a subagent needs its task, its interfaces, and the constraints. Nothing else.
- **TDD every task**, commit per green, keep the suite green (JIT-off).
- **Two gates per task** (spec compliance + code quality), one broad opus gate at the
  end. Don't skip re-review after a fix.
- **The ledger is durable truth** across compaction/limits.
- **Runner:** `./.venv/Scripts/python.exe -m pytest`. **Launch the app:**
  `./.venv/Scripts/python.exe -c "from ditherzam.app import main; main()"`.
