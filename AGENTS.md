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
- Do not promote the native rewrite without explicit approval (memory 093).
- Inspect `git status` and preserve unrelated user edits.
