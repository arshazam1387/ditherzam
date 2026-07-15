"""Offline model provenance gate for Smart Mask.

This module is the legal/provenance/offline-security boundary that must exist
before any segmentation model is ever staged or loaded: a frozen manifest
value object plus a fail-closed local verifier. It never fetches anything —
see ``tools/stage_smart_mask_model.py`` for the developer-only staging step.

No real model weights ship in this repository. ``verify_model_asset`` checks a
locally staged file against its manifest and fails closed on anything absent,
corrupt, mismatched, or attempting to escape the fixed asset root.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml


class ModelAssetError(Exception):
    """Raised when a model manifest or staged asset fails offline verification."""


# Fixed, application-owned location for staged Smart Mask model assets.
# Never derived from user input, presets, or any external response.
ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets" / "models" / "smart_mask"

# Logical model IDs approved for this release. Anything else fails closed.
APPROVED_MODEL_IDS = frozenset({"u2net", "u2netp"})

# Upstream repository state every approved model manifest must be pinned to.
APPROVED_UPSTREAM_COMMIT = "ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29"

_STREAM_CHUNK_BYTES = 1024 * 1024
_SHA256_HEX_LENGTH = 64
_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class TensorSpec:
    """A single named tensor's shape/dtype contract."""

    name: str
    shape: tuple[int, ...]
    dtype: str


# The only tensor contract this release's adapter understands. A manifest
# declaring anything else is rejected before any inference code ever runs.
EXPECTED_INPUT_TENSOR = TensorSpec(name="input.1", shape=(1, 3, 320, 320), dtype="float32")
EXPECTED_OUTPUT_TENSOR = TensorSpec(name="1959", shape=(1, 1, 320, 320), dtype="float32")


@dataclass(frozen=True)
class ModelManifest:
    """Explicit, immutable provenance record for one staged model asset."""

    model_id: str
    model_version: str
    upstream_repository: str
    upstream_commit: str
    upstream_source_url: str
    upstream_source_sha256: str
    license: str
    attribution: str
    conversion_revision: str
    conversion_opset: int
    conversion_tool_versions: tuple[tuple[str, str], ...]
    onnx_sha256: str
    onnx_byte_count: int
    input_tensor: TensorSpec
    output_tensor: TensorSpec
    preprocessing: str
    output_semantics: str
    algorithm_version: str
    relative_path: str
    # Finalized by SM-16 from the approved converted graph. ``None`` means the
    # release asset is intentionally not selected yet; live inference refuses
    # that state instead of guessing exporter-generated names.
    output_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Fail closed even when callers construct the value object directly."""
        if self.model_id not in APPROVED_MODEL_IDS:
            raise ModelAssetError(
                f"Model id is not approved for this release: {self.model_id!r}"
            )
        if self.upstream_commit != APPROVED_UPSTREAM_COMMIT:
            raise ModelAssetError(
                f"Upstream commit {self.upstream_commit!r} does not match the approved "
                f"pinned state {APPROVED_UPSTREAM_COMMIT!r}"
            )
        for field_name in ("upstream_source_sha256", "onnx_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _normalize_sha256(value) != value:
                raise ModelAssetError(
                    f"{field_name} must be a lowercase SHA-256 hex digest: {value!r}"
                )
        if (isinstance(self.onnx_byte_count, bool)
                or not isinstance(self.onnx_byte_count, int)
                or self.onnx_byte_count < 0):
            raise ModelAssetError(
                f"onnx_byte_count must be a non-negative int: {self.onnx_byte_count!r}"
            )
        _validate_relative_path(self.relative_path)
        if (self.input_tensor != EXPECTED_INPUT_TENSOR
                or self.output_tensor != EXPECTED_OUTPUT_TENSOR):
            raise ModelAssetError(
                "Model manifest tensor contract does not match the supported adapter "
                f"contract: input={self.input_tensor!r} output={self.output_tensor!r}"
            )
        if self.output_names is not None:
            names = self.output_names
            if (not isinstance(names, tuple) or len(names) != 7
                    or any(not isinstance(name, str) or not name.strip() for name in names)
                    or len(set(names)) != 7):
                raise ModelAssetError(
                    "output_names must contain exactly seven unique non-blank names"
                )


_REQUIRED_TOP_LEVEL_FIELDS = (
    "model_id",
    "model_version",
    "upstream_repository",
    "upstream_commit",
    "upstream_source_url",
    "upstream_source_sha256",
    "license",
    "attribution",
    "conversion_revision",
    "conversion_opset",
    "conversion_tool_versions",
    "onnx_sha256",
    "onnx_byte_count",
    "input_tensor",
    "output_tensor",
    "preprocessing",
    "output_semantics",
    "algorithm_version",
    "relative_path",
)


def default_asset_root() -> Path:
    """The single, fixed, application-owned root for staged Smart Mask assets."""
    return ASSET_ROOT


def load_manifest(path: str | Path) -> ModelManifest:
    """Parse and validate a manifest YAML file into a frozen ``ModelManifest``.

    Fails closed on a missing file, incomplete metadata, malformed hashes, an
    unapproved model ID, an unpinned upstream commit, or a tensor contract
    that does not match the fixed adapter contract this release supports.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ModelAssetError(f"Model manifest not found: {manifest_path}")

    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ModelAssetError(f"Error parsing model manifest: {exc}") from exc

    if not isinstance(data, dict):
        raise ModelAssetError(f"Model manifest is empty or malformed: {manifest_path}")

    missing = [key for key in _REQUIRED_TOP_LEVEL_FIELDS if key not in data]
    if missing:
        raise ModelAssetError(
            f"Model manifest missing required field(s): {', '.join(missing)}"
        )

    model_id = str(data["model_id"])
    if model_id not in APPROVED_MODEL_IDS:
        raise ModelAssetError(f"Model id is not approved for this release: {model_id!r}")

    upstream_commit = str(data["upstream_commit"])
    if upstream_commit != APPROVED_UPSTREAM_COMMIT:
        raise ModelAssetError(
            f"Upstream commit {upstream_commit!r} does not match the approved "
            f"pinned state {APPROVED_UPSTREAM_COMMIT!r}"
        )

    try:
        manifest = ModelManifest(
            model_id=model_id,
            model_version=str(data["model_version"]),
            upstream_repository=str(data["upstream_repository"]),
            upstream_commit=upstream_commit,
            upstream_source_url=str(data["upstream_source_url"]),
            upstream_source_sha256=_normalize_sha256(data["upstream_source_sha256"]),
            license=str(data["license"]),
            attribution=str(data["attribution"]),
            conversion_revision=str(data["conversion_revision"]),
            conversion_opset=int(data["conversion_opset"]),
            conversion_tool_versions=_load_tool_versions(data["conversion_tool_versions"]),
            onnx_sha256=_normalize_sha256(data["onnx_sha256"]),
            onnx_byte_count=_non_negative_int(data["onnx_byte_count"], "onnx_byte_count"),
            input_tensor=_load_tensor_spec(data["input_tensor"]),
            output_tensor=_load_tensor_spec(data["output_tensor"]),
            preprocessing=str(data["preprocessing"]),
            output_semantics=str(data["output_semantics"]),
            algorithm_version=str(data["algorithm_version"]),
            relative_path=str(data["relative_path"]),
            output_names=_load_output_names(data.get("output_names")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelAssetError(f"Model manifest field is invalid: {exc}") from exc

    _validate_relative_path(manifest.relative_path)

    if manifest.input_tensor != EXPECTED_INPUT_TENSOR or manifest.output_tensor != EXPECTED_OUTPUT_TENSOR:
        raise ModelAssetError(
            "Model manifest tensor contract does not match the supported adapter "
            f"contract: input={manifest.input_tensor!r} output={manifest.output_tensor!r}"
        )

    return manifest


def _load_output_names(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("output_names must be a list")
    names = tuple(str(item).strip() for item in value)
    if len(names) != 7 or any(not name for name in names) or len(set(names)) != 7:
        raise ValueError("output_names must contain exactly seven unique non-blank names")
    return names


def verify_model_asset(asset_root: str | Path, manifest: ModelManifest) -> Path:
    """Fail-closed local verification of a staged asset against its manifest.

    Resolves ``manifest.relative_path`` under ``asset_root`` only — there is no
    caller-supplied path parameter, so nothing (including a future preset) can
    redirect verification to an arbitrary file. Returns the verified path.
    """
    if not isinstance(manifest, ModelManifest):
        raise ModelAssetError("verify_model_asset requires a loaded ModelManifest")

    root = Path(asset_root).resolve()
    asset_path = _resolve_within_root(root, manifest.relative_path)

    if not asset_path.is_file():
        raise ModelAssetError(f"Model asset not found: {asset_path}")

    byte_count, digest = _hash_and_count(asset_path)

    if byte_count != manifest.onnx_byte_count:
        raise ModelAssetError(
            f"Model asset byte count {byte_count} does not match manifest "
            f"{manifest.onnx_byte_count}: {asset_path}"
        )
    if digest != manifest.onnx_sha256:
        raise ModelAssetError(
            f"Model asset SHA-256 {digest} does not match manifest "
            f"{manifest.onnx_sha256}: {asset_path}"
        )

    return asset_path


def _load_tensor_spec(data: object) -> TensorSpec:
    if not isinstance(data, dict):
        raise ModelAssetError(f"Tensor spec is not a mapping: {data!r}")
    try:
        return TensorSpec(
            name=str(data["name"]),
            shape=tuple(int(dim) for dim in data["shape"]),
            dtype=str(data["dtype"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelAssetError(f"Tensor spec is invalid: {exc}") from exc


def _load_tool_versions(data: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(data, dict) or not data:
        raise ModelAssetError(f"conversion_tool_versions must be a non-empty mapping: {data!r}")
    return tuple((str(tool), str(version)) for tool, version in data.items())


def _normalize_sha256(value: object) -> str:
    text = str(value).strip().lower()
    if len(text) != _SHA256_HEX_LENGTH or any(ch not in _SHA256_HEX_DIGITS for ch in text):
        raise ModelAssetError(f"Value is not a valid SHA-256 hex digest: {value!r}")
    return text


def _non_negative_int(value: object, field_name: str) -> int:
    count = int(value)
    if count < 0:
        raise ModelAssetError(f"{field_name} must not be negative: {count}")
    return count


def _validate_relative_path(relative_path: str) -> None:
    if not relative_path or not relative_path.strip():
        raise ModelAssetError("Model asset relative_path must not be empty")
    if PureWindowsPath(relative_path).is_absolute() or PurePosixPath(relative_path).is_absolute():
        raise ModelAssetError(f"Model asset path must be relative: {relative_path!r}")
    normalized = PurePosixPath(relative_path.replace("\\", "/"))
    if ".." in normalized.parts:
        raise ModelAssetError(f"Model asset path may not traverse directories: {relative_path!r}")


def _resolve_within_root(root: Path, relative_path: str) -> Path:
    _validate_relative_path(relative_path)
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ModelAssetError(
            f"Model asset path escapes the approved asset root: {relative_path!r}"
        ) from None
    return candidate


def _hash_and_count(path: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_STREAM_CHUNK_BYTES), b""):
            hasher.update(chunk)
            total += len(chunk)
    return total, hasher.hexdigest()
