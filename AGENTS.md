# ditherzam agent entry

ditherzam is a Python 3.12 desktop image editor for dithering, color, effects,
layers, masks, animation, video, and exact export.

## Route by task

| Task | Open first |
|---|---|
| Status, constraints, decisions, gotchas | `docs/memory/INDEX.md` |
| Subsystem ownership or change impact | `docs/map/AGENTS.md` |
| Product requirements | `DITHER_BOY_FULL_SPEC.md` |
| Historical phase contracts | `docs/plans/00-ROADMAP.md` |
| Build and test | `pyproject.toml`, then the relevant map section |

## Rules

- Memory owns project state; code and tests own as-built behavior.
- Preserve clean-room and Qt-free-core constraints (memory 001 and 002).
- The approved native rewrite is on `main`; preserve exact fallbacks, thread
  budgeting, and the Round-only brush boundary (memory 093, 101, and 102).
- Treat force-pushes, releases, and remote branch deletion as separately
  authorized operations even when local changes are already approved.
- Inspect `git status` and preserve unrelated user edits.
