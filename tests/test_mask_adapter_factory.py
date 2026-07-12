"""load_default_segmentation_adapter: fail-closed startup wiring for Smart Mask.

Builds a real (lazy) adapter from a staged manifest, and returns None on any
missing/invalid/mismatched asset so the app stays cleanly disabled. Uses a tiny
dummy asset — the ONNX session is lazy, so no real model or onnxruntime is
needed to exercise the wiring.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from ditherzam.masking.contracts import ModelIdentity
from ditherzam.masking.model_assets import APPROVED_UPSTREAM_COMMIT
from ditherzam.masking.ort_adapter import (
    MANIFEST_ALGORITHM_VERSION,
    MANIFEST_OUTPUT_SEMANTICS,
    MANIFEST_PREPROCESSING,
    OrtSegmentationAdapter,
    load_default_segmentation_adapter,
)


def _stage_valid_model(root: Path) -> str:
    """Write a dummy asset + an adapter-valid manifest.yaml; return the onnx sha."""
    root.mkdir(parents=True, exist_ok=True)
    payload = b"dummy-u2netp-bytes"
    (root / "u2netp.onnx").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    manifest = {
        "model_id": "u2netp",
        "model_version": "2020-05-06",
        "upstream_repository": "U-2-Net",
        "upstream_commit": APPROVED_UPSTREAM_COMMIT,
        "upstream_source_url": "local-test",
        "upstream_source_sha256": "a" * 64,
        "license": "Apache-2.0",
        "attribution": "U-2-Net authors",
        "conversion_revision": "test",
        "conversion_opset": 12,
        "conversion_tool_versions": {"pytorch": "x", "onnx": "y"},
        "onnx_sha256": sha,
        "onnx_byte_count": len(payload),
        "input_tensor": {"name": "input.1", "shape": [1, 3, 320, 320], "dtype": "float32"},
        "output_tensor": {"name": "1959", "shape": [1, 1, 320, 320], "dtype": "float32"},
        "preprocessing": MANIFEST_PREPROCESSING,
        "output_semantics": MANIFEST_OUTPUT_SEMANTICS,
        "algorithm_version": MANIFEST_ALGORITHM_VERSION,
        "relative_path": "u2netp.onnx",
        "output_names": ["1959", "1960", "1961", "1962", "1963", "1964", "1965"],
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return sha


def test_returns_adapter_from_staged_manifest(tmp_path):
    sha = _stage_valid_model(tmp_path)
    adapter = load_default_segmentation_adapter(tmp_path)
    assert isinstance(adapter, OrtSegmentationAdapter)
    identity = adapter.model_identity
    assert isinstance(identity, ModelIdentity)
    assert identity.model_id == "u2netp"
    assert identity.model_hash == sha


def test_none_when_nothing_staged(tmp_path):
    assert load_default_segmentation_adapter(tmp_path) is None


def test_none_on_asset_hash_mismatch(tmp_path):
    _stage_valid_model(tmp_path)
    # Corrupt the asset so the streamed hash no longer matches the manifest.
    (tmp_path / "u2netp.onnx").write_bytes(b"tampered-bytes")
    assert load_default_segmentation_adapter(tmp_path) is None


def test_none_when_manifest_missing_output_names(tmp_path):
    _stage_valid_model(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data.pop("output_names")
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    # load_manifest tolerates absent output_names, but the adapter refuses to
    # build without a finalized 7-name contract -> fail closed to None.
    assert load_default_segmentation_adapter(tmp_path) is None
