"""Offline provenance gate for Smart Mask model assets (SM-01).

No real model weights exist or are committed. Every test generates its own
tiny temporary local asset, computes its real SHA-256/byte count, and asserts
that ``verify_model_asset`` fail-closes on any mismatch. ``onnxruntime`` is not
required by this module.
"""
from __future__ import annotations

import dataclasses
import hashlib
import textwrap
from pathlib import Path

import pytest

from ditherzam.masking.model_assets import (
    APPROVED_MODEL_IDS,
    APPROVED_UPSTREAM_COMMIT,
    EXPECTED_INPUT_TENSOR,
    EXPECTED_OUTPUT_TENSOR,
    ModelAssetError,
    ModelManifest,
    TensorSpec,
    default_asset_root,
    load_manifest,
    verify_model_asset,
)


def _write_asset(directory: Path, name: str, payload: bytes) -> tuple[str, int]:
    path = directory / name
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    return digest, len(payload)


def _valid_manifest_yaml(
    tmp_path: Path,
    *,
    model_id: str = "u2netp",
    upstream_commit: str = APPROVED_UPSTREAM_COMMIT,
    relative_path: str = "u2netp.onnx",
    onnx_sha256: str | None = None,
    onnx_byte_count: int | None = None,
    input_shape: str = "[1, 3, 320, 320]",
    output_shape: str = "[1, 1, 320, 320]",
    extra_fields: str = "",
    omit_field: str | None = None,
) -> tuple[Path, str, int]:
    """Write a manifest YAML file plus a matching local asset; return (path, sha256, bytes)."""
    payload = f"tiny-fixture-for-{model_id}".encode("utf-8")
    digest, byte_count = _write_asset(tmp_path, relative_path, payload)
    if onnx_sha256 is None:
        onnx_sha256 = digest
    if onnx_byte_count is None:
        onnx_byte_count = byte_count

    fields = {
        "model_id": model_id,
        "model_version": "2020-05-06",
        "upstream_repository": "u2net-repository",
        "upstream_commit": upstream_commit,
        "upstream_source_url": "u2net-source-asset",
        "upstream_source_sha256": "a" * 64,
        "license": "Apache-2.0",
        "attribution": "U-2-Net authors",
        "conversion_revision": "conv-rev-1",
        "conversion_opset": 12,
        "conversion_tool_versions": {"pytorch": "1.13.1", "onnx": "1.14.0"},
        "onnx_sha256": onnx_sha256,
        "onnx_byte_count": onnx_byte_count,
        "preprocessing": "resize 320x320, RGB, mean/std normalize",
        "output_semantics": "single-channel foreground probability in [0, 1]",
        "algorithm_version": "1",
        "relative_path": relative_path,
    }
    if omit_field:
        fields.pop(omit_field, None)

    lines = []
    for key, value in fields.items():
        if key == "conversion_tool_versions":
            lines.append(f"{key}:")
            for tool, version in value.items():
                lines.append(f"  {tool}: \"{version}\"")
        else:
            lines.append(f'{key}: "{value}"' if isinstance(value, str) else f"{key}: {value}")

    text = "\n".join(lines) + "\n"
    text += textwrap.dedent(
        f"""
        input_tensor:
          name: "input.1"
          shape: {input_shape}
          dtype: "float32"
        output_tensor:
          name: "1959"
          shape: {output_shape}
          dtype: "float32"
        """
    )
    text += extra_fields

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(text, encoding="utf-8")
    return manifest_path, onnx_sha256, onnx_byte_count


# -- module-level constants ---------------------------------------------------


def test_approved_upstream_commit_is_pinned():
    assert APPROVED_UPSTREAM_COMMIT == "ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29"


def test_approved_model_ids_are_fixed():
    assert "u2netp" in APPROVED_MODEL_IDS
    assert "u2net" in APPROVED_MODEL_IDS
    assert "unapproved-model" not in APPROVED_MODEL_IDS


def test_default_asset_root_is_fixed_and_application_owned():
    root = default_asset_root()
    assert isinstance(root, Path)
    assert root.parts[-3:] == ("assets", "models", "smart_mask")


# -- load_manifest -------------------------------------------------------------


def test_load_manifest_missing_file_fails_closed(tmp_path):
    with pytest.raises(ModelAssetError):
        load_manifest(tmp_path / "does_not_exist.yaml")


def test_load_manifest_rejects_incomplete_metadata(tmp_path):
    manifest_path, _, _ = _valid_manifest_yaml(tmp_path, omit_field="license")
    with pytest.raises(ModelAssetError):
        load_manifest(manifest_path)


def test_load_manifest_rejects_unapproved_model_id(tmp_path):
    manifest_path, _, _ = _valid_manifest_yaml(tmp_path, model_id="unapproved-model")
    with pytest.raises(ModelAssetError):
        load_manifest(manifest_path)


def test_load_manifest_rejects_wrong_upstream_commit(tmp_path):
    manifest_path, _, _ = _valid_manifest_yaml(tmp_path, upstream_commit="0" * 40)
    with pytest.raises(ModelAssetError):
        load_manifest(manifest_path)


def test_load_manifest_rejects_tensor_contract_mismatch(tmp_path):
    manifest_path, _, _ = _valid_manifest_yaml(tmp_path, input_shape="[1, 3, 512, 512]")
    with pytest.raises(ModelAssetError):
        load_manifest(manifest_path)


def test_load_manifest_rejects_malformed_source_hash(tmp_path):
    payload = b"tiny-fixture"
    digest, byte_count = _write_asset(tmp_path, "u2netp.onnx", payload)
    manifest_path, _, _ = _valid_manifest_yaml(
        tmp_path, onnx_sha256=digest, onnx_byte_count=byte_count
    )
    text = manifest_path.read_text(encoding="utf-8").replace(
        'upstream_source_sha256: "' + "a" * 64 + '"',
        'upstream_source_sha256: "not-a-hash"',
    )
    manifest_path.write_text(text, encoding="utf-8")
    with pytest.raises(ModelAssetError):
        load_manifest(manifest_path)


def test_load_manifest_accepts_approved_manifest(tmp_path):
    manifest_path, digest, byte_count = _valid_manifest_yaml(tmp_path)
    manifest = load_manifest(manifest_path)

    assert isinstance(manifest, ModelManifest)
    assert manifest.model_id == "u2netp"
    assert manifest.upstream_commit == APPROVED_UPSTREAM_COMMIT
    assert manifest.onnx_sha256 == digest
    assert manifest.onnx_byte_count == byte_count
    assert manifest.input_tensor == EXPECTED_INPUT_TENSOR
    assert manifest.output_tensor == EXPECTED_OUTPUT_TENSOR
    assert manifest.output_names is None


def test_manifest_rejects_incomplete_or_duplicate_final_output_names(tmp_path):
    manifest_path, _, _ = _valid_manifest_yaml(tmp_path)
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\noutput_names: [primary, aux, aux]\n",
        encoding="utf-8",
    )
    with pytest.raises(ModelAssetError, match="exactly seven unique non-blank"):
        load_manifest(manifest_path)


def test_model_manifest_is_frozen(tmp_path):
    manifest_path, _, _ = _valid_manifest_yaml(tmp_path)
    manifest = load_manifest(manifest_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.model_id = "u2net"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "unapproved-model"),
        ("upstream_commit", "0" * 40),
        ("onnx_sha256", "A" * 64),
        ("upstream_source_sha256", "not-a-hash"),
        ("input_tensor", TensorSpec("input.1", (1, 3, 512, 512), "float32")),
    ],
)
def test_direct_model_manifest_construction_enforces_release_invariants(
    tmp_path, field, value
):
    manifest_path, _, _ = _valid_manifest_yaml(tmp_path)
    valid = load_manifest(manifest_path)
    with pytest.raises(ModelAssetError):
        dataclasses.replace(valid, **{field: value})


# -- verify_model_asset ---------------------------------------------------------


def test_verify_model_asset_missing_asset_fails_closed(tmp_path):
    manifest_path, _, _ = _valid_manifest_yaml(tmp_path)
    manifest = load_manifest(manifest_path)
    (tmp_path / manifest.relative_path).unlink()

    with pytest.raises(ModelAssetError):
        verify_model_asset(tmp_path, manifest)


def test_verify_model_asset_rejects_wrong_byte_count(tmp_path):
    manifest_path, digest, byte_count = _valid_manifest_yaml(tmp_path)
    manifest = load_manifest(manifest_path)
    tampered = dataclasses.replace(manifest, onnx_byte_count=byte_count + 1)

    with pytest.raises(ModelAssetError):
        verify_model_asset(tmp_path, tampered)


def test_verify_model_asset_rejects_wrong_hash(tmp_path):
    manifest_path, _, _ = _valid_manifest_yaml(tmp_path)
    manifest = load_manifest(manifest_path)
    tampered = dataclasses.replace(manifest, onnx_sha256="f" * 64)

    with pytest.raises(ModelAssetError):
        verify_model_asset(tmp_path, tampered)


@pytest.mark.parametrize(
    "traversal_path",
    [
        "../outside.onnx",
        "../../secret/outside.onnx",
        "..\\..\\secret\\outside.onnx",
        "/etc/passwd",
        "C:\\Windows\\System32\\evil.onnx",
    ],
)
def test_verify_model_asset_rejects_absolute_and_traversing_paths(tmp_path, traversal_path):
    manifest_path, digest, byte_count = _valid_manifest_yaml(tmp_path)
    manifest = load_manifest(manifest_path)
    with pytest.raises(ModelAssetError):
        dataclasses.replace(manifest, relative_path=traversal_path)


def test_verify_model_asset_rejects_arbitrary_preset_supplied_path(tmp_path):
    """There is no path parameter a preset could use to override the manifest."""
    manifest_path, _, _ = _valid_manifest_yaml(tmp_path)
    manifest = load_manifest(manifest_path)

    with pytest.raises(TypeError):
        verify_model_asset(tmp_path, manifest, path="C:/evil.onnx")  # type: ignore[call-arg]


def test_verify_model_asset_accepts_valid_local_asset(tmp_path):
    manifest_path, digest, byte_count = _valid_manifest_yaml(tmp_path)
    manifest = load_manifest(manifest_path)

    verified_path = verify_model_asset(tmp_path, manifest)

    assert verified_path == (tmp_path / manifest.relative_path).resolve()
    assert verified_path.is_file()


def test_verify_model_asset_requires_a_loaded_manifest(tmp_path):
    with pytest.raises(ModelAssetError):
        verify_model_asset(tmp_path, {"relative_path": "x.onnx"})


def test_tensor_spec_contract_constants_are_frozen():
    assert isinstance(EXPECTED_INPUT_TENSOR, TensorSpec)
    assert isinstance(EXPECTED_OUTPUT_TENSOR, TensorSpec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        EXPECTED_INPUT_TENSOR.dtype = "float64"
