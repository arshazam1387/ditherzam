---
type: reference
phase: "-"
status: n/a
date: 2026-07-04
---

Fallback when stuck: if a subsystem hits a roadblock, we can't figure out a fix, or
an implementation doesn't come out right, we may consult the **decompiled Dither
Boy 3.0.2** bytecode/disassembly for inspiration on how they solved it — then
implement our own version.

- Source material: extracted from the installed `Dither Boy.exe` (PyInstaller,
  Python 3.12). Full bytecode + disassembly of `main.pyc`, `dither_registry.pyc`,
  `db_tools.pyc` (dumped as `dis_main.txt` ~3.2MB, `dis_dither_registry.txt` ~1MB),
  plus complete string/const/docstring tables. Behavior is already distilled in
  `DITHER_BOY_FULL_SPEC.md`.
- **Where it lives:** a temporary session scratchpad (`.../scratchpad/dbwork/`),
  which is ephemeral and may be gone in a new session. If long-term reference is
  wanted, keep it OUTSIDE the repo (never commit it — `.gitignore` blocks `*.exe`
  and `*.pyc`).
- **Caveat (see [[001-clean-room]]):** consulting their decompiled code for
  "inspiration" weakens the clean-room claim. Preferred use: read it to understand
  *what* behavior/approach is needed, then write our own implementation from
  scratch (ideally cross-checked against public/standard sources), rather than
  transcribing their logic line-for-line. Never copy their strings/URLs/licensing.
