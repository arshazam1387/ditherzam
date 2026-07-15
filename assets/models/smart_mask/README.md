# Smart Mask model assets

This directory is the fixed, application-owned root that
[`ditherzam/masking/model_assets.py`](../../../ditherzam/masking/model_assets.py)
resolves staged Smart Mask model assets under. It never contains real weights
in source control.

## What is (and is not) here

- `manifest.schema.example.yaml` — an annotated example of the manifest
  format `load_manifest` parses. Every value is a placeholder; it does not
  describe a real staged asset and is not used at runtime.
- No `.onnx`, `.pt`, `.pth`, `.bin`, `.ckpt`, or `.safetensors` files are ever
  committed here. `.gitignore` blocks them at this path, and
  `tests/test_offline_security.py` asserts none are present.

Model weights ship inside a release/installer asset, staged locally by a
developer with `tools/stage_smart_mask_model.py` and verified against a real
manifest with `verify_model_asset`. The application never fetches anything.

## Provenance policy this manifest format enforces

- `model_id` must be one of the release's approved logical model IDs
  (currently `u2net`, `u2netp`).
- `upstream_commit` must equal the pinned U-2-Net upstream repository state,
  commit `ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29`. A manifest pinned to any
  other commit is rejected.
- `input_tensor` / `output_tensor` must match the single tensor contract this
  release's adapter supports (`EXPECTED_INPUT_TENSOR` /
  `EXPECTED_OUTPUT_TENSOR` in `model_assets.py`). A mismatch is rejected
  before any inference code runs.
- `relative_path` must be a relative, non-traversing path under this
  directory. Absolute paths and `..` segments are rejected; verification
  resolves strictly under the fixed asset root and never accepts a
  caller-supplied path override.
- `onnx_sha256` and `onnx_byte_count` are checked against the actual staged
  file with a streamed SHA-256 and an exact byte count. Any mismatch, or a
  missing file, fails closed with no fallback.

See `THIRD_PARTY_NOTICES.md` at the repository root for license/attribution
policy and the upstream U-2-Net model's provisional status.
