# SM-08 implementation report

## Outcome

Implemented the Qt-free, offline ONNX segmentation boundary with a synchronized
lazy session, strict manifest and live-session tensor validation, fixed U-2-Net
RGB preprocessing, primary-output postprocessing, source-resolution immutable
`ProbabilityMap` publication, CPU-only ORT construction, and cancellation checks
at safe boundaries. No production weights, binaries, network access, or runtime
dependency installation were added.

`onnxruntime==1.22.1` is pinned only in the optional `release` dependency group;
the default suite uses an injected fake session.

## Verification

- Focused JIT-off command:
  `.venv/Scripts/python.exe -m pytest -q tests/test_mask_adapter_contract.py tests/test_mask_session.py tests/test_offline_security.py --basetemp=.pytest-tmp-sm08b`
- Result: **13 passed**.
- `git diff --check`: passed.
- A branch-wide run was started only as an additional check but exceeded the
  120-second command window without producing a result; it is not used as the
  task gate. The required foreground suite is fully green, so SM-08 introduces
  zero observed failures beyond the separately documented 71 golden failures.

## Scope notes

- ORT is imported lazily and configured with only `CPUExecutionProvider`.
- The verified local asset path is resolved before the injected/real session is
  created.
- The seven-output contract selects the first (`1959`) output as stable
  candidate `primary`; auxiliary outputs are validated but not exposed.
- Constant/non-finite/incompatible outputs and missing/runtime failures remain
  explicit; no fallback pretends inference succeeded.
