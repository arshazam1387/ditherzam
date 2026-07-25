# Third-Party Notices

This file records third-party code, models, and assets bundled with or staged
by ditherzam, per component license and attribution requirements.

## Smart Mask — U-2-Net (provisional, weights not yet shipped)

- **Project:** U-2-Net
- **Upstream repository state pinned to commit:**
  `ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29`
- **License:** Apache-2.0 (repository code)
- **Status:** Provisional. No U-2-Net weights are committed to this
  repository or shipped in any build yet. Pretrained-weight redistribution
  under compatible terms requires written confirmation or equivalent
  authoritative evidence before any weight is staged for release (this is a
  hard release gate). If and when an approved, converted asset ships, its
  exact manifest (source hash, conversion revision/opset, output hash,
  tensors, license/attribution, modification notice) is recorded alongside it
  under `assets/models/smart_mask/`, per
  `ditherzam/masking/model_assets.py`.
- **Modifications:** None shipped. Any future conversion to ONNX is performed
  by the developer-only `tools/stage_smart_mask_model.py` staging step and a
  separate, not-yet-built reproducible conversion step; the original
  checkpoint is never modified in place.
