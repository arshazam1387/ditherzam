---
name: zam-memory
description: ditherzam project memory. Invoke to recall project state (current phase, next task, decisions, gotchas, invariants) or to record a new memory. Use at the START of any ditherzam work session, and whenever a phase/task completes, a design decision is made, a gotcha is hit, or the user says "remember …". Keeps context across sessions and hand-off agents.
---

# zam-memory — ditherzam project memory

A tiny, file-based, version-controlled memory that lives in **this repo** so every
session and hand-off agent shares the same project state. It is the fastest way to
answer "where are we and what's next" without re-reading all the plans.

**Store location:** `docs/memory/`
- `docs/memory/INDEX.md` — the always-load index (one line per entry). Read this first.
- `docs/memory/NNN-slug.md` — one atomic fact per file.

## When to invoke

- **Start of a work session** → run the **recall** flow (default, no args).
- After **finishing a phase/task**, making a **decision**, hitting a **gotcha**, or
  when the user says **"remember …"** → run the **record** flow.

## Recall flow (default / `recall` / `status`)

1. Read `docs/memory/INDEX.md`.
2. Read the entries that match the user's current task (always read every
   `constraint` entry and the latest `progress` entry).
3. Report, concisely:
   - **Current phase & next task** (from the newest `progress` entry).
   - **Active invariants** (all `constraint` entries) — clean-room, Qt-free core,
     Python 3.12, TDD-per-task.
   - Any **gotchas** relevant to the task about to be done.
4. Do NOT dump raw files — synthesize.

## Record flow (`remember <text>` / `done <phase/task>` / `decide <text>` / `gotcha <text>`)

1. Pick the **type**: `progress` | `decision` | `gotcha` | `constraint` | `reference`.
2. **Dedup:** scan INDEX; if an entry already covers this, UPDATE that file instead
   of creating a new one. Delete entries that have become wrong.
3. Create/So update `docs/memory/NNN-slug.md` (next zero-padded number) with:

```markdown
---
type: progress | decision | gotcha | constraint | reference
phase: <0-8 or "-">
status: <done | in-progress | blocked | n/a>
date: <YYYY-MM-DD, convert relative dates to absolute>
---

<one atomic fact. For decision/gotcha add **Why:** and **Fix/Apply:** lines.
Link related entries with [[NNN-slug]].>
```

4. Add/refresh a one-line pointer in `docs/memory/INDEX.md`:
   `- [type|phase N|status] Title — one-line hook (NNN-slug.md)`
5. Keep it atomic — one fact per file. No conversation-only trivia. Don't record
   what the plans/spec/git history already state; record what a fresh agent would
   otherwise re-derive (progress, non-obvious decisions, traps).
6. If the user asks, `git add docs/memory && git commit -m "chore(memory): …"`.

## Rules

- **Atomic:** one fact per file; short.
- **Absolute dates** only (today is whatever the session's current date is).
- **Invariants live forever** — never delete a `constraint` entry unless the
  constraint itself changed; reflect what was true when written and re-verify file
  names/functions against the tree before acting on an old entry.
- The index is the contract: every entry has exactly one INDEX line; never put
  entry bodies in INDEX.
- Prefer updating over duplicating.

## Quick reference — entry types for ditherzam

| type | use for |
|---|---|
| `progress` | which phase/task is done / in-progress / next (drives "where are we") |
| `decision` | design choices + **why** (e.g. "color mapping runs after dither, on the tone bands") |
| `gotcha` | traps + fix (e.g. "Numba njit needs `cache=True`; tests set `NUMBA_DISABLE_JIT=1`") |
| `constraint` | project invariants (clean-room, Qt-free core, Python 3.12, TDD) |
| `reference` | pointers (spec sections, plan files, repo URL, gh account) |
