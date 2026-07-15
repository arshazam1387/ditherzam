"""Bounded frozen-release smoke operation; dependency injection keeps tests asset-free."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ditherzam.masking.release_gate import verify_release_bundle


def run(lock: Path, *, adapter_factory=None, output=None, bundle_root: Path | None = None) -> int:
    root = bundle_root.resolve() if bundle_root is not None else lock.resolve().parents[1]
    bundle = verify_release_bundle(root, lock)
    if adapter_factory is None:
        from ditherzam.masking.model_assets import load_manifest
        from ditherzam.masking.ort_adapter import OrtSegmentationAdapter
        adapter_factory = lambda: OrtSegmentationAdapter(load_manifest(bundle["model_manifest"]), asset_root=root)
    rgba = np.asarray(Image.open(bundle["smoke_fixture"]).convert("RGBA"), dtype=np.uint8)
    adapter = adapter_factory()
    result = adapter.infer(rgba)
    confidence = getattr(getattr(result, "probability", result), "values", None)
    if confidence is None or np.asarray(confidence).shape != rgba.shape[:2]:
        raise RuntimeError("offline smoke adapter produced no source-sized mask")
    mask = np.asarray(confidence, np.float32)[..., None]
    masked = np.rint(rgba[..., :3] * mask).astype(np.uint8)
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "masked.png"; jpg = Path(td) / "disabled.jpg"
        Image.fromarray(masked, "RGB").save(png); Image.fromarray(rgba[..., :3], "RGB").save(jpg, quality=95)
        case = {"mode": "offline-smoke"}
        evidence = {"executed": True, "case": case, "outputs": {
            "masked_png": _record(png, "PNG"), "disabled_jpeg": _record(jpg, "JPEG")}}
        canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        evidence["evidence_sha256"] = hashlib.sha256(canonical).hexdigest()
        print(json.dumps(evidence, sort_keys=True), file=output)
    return 0


def _record(path: Path, fmt: str) -> dict[str, object]:
    with Image.open(path) as image: width, height = image.size
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "width": width,
            "height": height, "format": fmt}
