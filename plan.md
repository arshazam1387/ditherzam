# Creative Masking & Selection Expansion

Status: complete
Date: 2026-08-14

## Goal

Extend manual masking with expressive brush tips, capable selection tools, and
non-destructive selection refinement while preserving current render contracts.

## Completed task list

- [x] Audited masking and selection integration seams.
- [x] Added Round, Square, Diamond, and deterministic Texture mask brushes.
- [x] Added brush size, hardness, strength, spacing, and tip controls; Reveal/Hide remains available through the existing X shortcut.
- [x] Added freehand and polygon canvas selections alongside Rectangle and Ellipse.
- [x] Added Qt-free alpha-aware source Color Range selection for future eyedropper workflows.
- [x] Added Select All, Invert, Grow, Shrink, Feather, and Clear actions.
- [x] Preserved Replace/Add/Subtract operations and Transform exclusivity.
- [x] Added focused core, controller, panel, viewport, and integration tests.
- [x] Ran focused and broad bounded Python 3.12 tests with JIT disabled.
- [x] Updated shared project memory.

## Verification

- 112/112 combined creative-mask, legacy-mask, panel, viewport, lifecycle tests passed.
- A wider relevant run reached 148 passed with 5 setup-only errors because the managed Windows sandbox denies pytest temporary-directory creation.
- The earlier broader discovery reached 488 passed and 309 skipped with the same temp ACL issue; its single reported failure was part of the contaminated run and did not reproduce in focused coverage.
- Existing round-brush defaults remain byte-identical.
- Textured brushes are deterministic for a fixed seed and stroke.
- Selection operations are bounded deterministic uint8 operations.
- Transform clears brush, selection, and gradient input modes before taking canvas ownership.
- PySide6 remains confined to approved UI modules.

## Continuation: Color Range workflow

- [x] Audit the existing Qt-free primitive and UI/controller/viewport ownership seams.
- [x] Add exact Qt-free source-to-document selection mapping for transformed layers.
- [x] Add a canvas eyedropper with tolerance and softness controls.
- [x] Keep live preview reversible with explicit Confirm/Cancel.
- [x] Preserve alpha awareness and Replace/Add/Subtract semantics.
- [x] Harden empty layers, out-of-layer clicks, layer switching, document replacement,
  Escape cancellation, and Transform ownership.
- [x] Run focused core/UI/lifecycle/Transform/viewport/Layers tests, `py_compile`,
  and `git diff --check`.
- [x] Record durable completion evidence in shared memory.

## Continuation verification

- 142/142 focused core, brush, selection, controller, lifecycle, viewport, and
  Layers panel tests passed with Python 3.12 and `NUMBA_DISABLE_JIT=1`.
- Changed Python modules passed `py_compile`.
- `git diff --check` passed; Git emitted only existing LF-to-CRLF notices.
- Color Range preview does not enter `LayerDocument` history or serialization.
- Confirm publishes one temporary document-coordinate selection; Cancel and Escape
  restore the exact prior selection overlay.

## Transform handle responsiveness hotfix

- [x] Reproduced zoomed blue handles rendering outside their fixed hit targets.
- [x] Made each resize target contain the full visible handle while retaining the
  16-physical-pixel minimum target.
- [x] Added zoomed-handle containment regression coverage.
- [x] Verified 91 viewport, Transform lifecycle, masking, selection, Layers panel,
  and controller tests; `py_compile` and `git diff --check` passed.

## Layer preview freeze and shutdown hotfix

- [x] Confirmed from the program log that control events and proxy computations
  continued while viewport publication stopped.
- [x] Keep a slow in-flight proxy publishable after the 160 ms settle tick until
  the replacement Layers composite actually arrives.
- [x] Retain Layers and ordinary editor render workers through every terminal
  signal so shutdown cannot delete their Qt signal sources mid-run.
- [x] Keep close nonblocking while active mask/editor workers drain.
- [x] Added slow-proxy, settled-replacement, stale-terminal, and teardown races.
- [x] Verified 105 preview, resilience, Layers, masking lifecycle, and viewport
  tests with Python 3.12 and `NUMBA_DISABLE_JIT=1`.
