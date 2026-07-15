"""Offline CPU-only ONNX adapter for the frozen U-2-Net tensor contract."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ditherzam.masking.adapter import InferenceCancelled, InferenceResult, NoClearSubject
from ditherzam.masking.contracts import InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity, validate_rgba_u8
from ditherzam.masking.model_assets import EXPECTED_INPUT_TENSOR, EXPECTED_OUTPUT_TENSOR, ModelAssetError, ModelManifest, default_asset_root, load_manifest, verify_model_asset
from ditherzam.masking.session import LazySession

PREPROCESSING_VERSION = "u2net-rgb-imagenet-bilinear-v1"
MANIFEST_PREPROCESSING = "resize to 320x320, RGB, scale to [0, 1], normalize by documented mean/std"
MANIFEST_OUTPUT_SEMANTICS = "single-channel foreground probability in [0, 1] at model resolution"
MANIFEST_ALGORITHM_VERSION = "1"
INPUT_NAME = EXPECTED_INPUT_TENSOR.name
OUTPUT_NAME = EXPECTED_OUTPUT_TENSOR.name
INPUT_SIZE = 320
_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


def _cancelled(check: Callable[[], bool] | None) -> None:
    if check is not None and check():
        raise InferenceCancelled("segmentation inference cancelled")


def preprocess_u2net(rgba_u8: np.ndarray) -> np.ndarray:
    """Resize straight RGB to 320 square and apply ImageNet normalization."""
    source = validate_rgba_u8(rgba_u8)
    rgb = Image.fromarray(source[..., :3], mode="RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    chw = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0)
    return np.ascontiguousarray(((chw - _MEAN) / _STD)[None], dtype=np.float32)


def postprocess_probability(output: object, source_shape: tuple[int, int]) -> np.ndarray:
    """Normalize primary output and deterministically resize to source resolution."""
    raw = np.asarray(output)
    if raw.dtype != np.float32 or raw.shape != (1, 1, INPUT_SIZE, INPUT_SIZE):
        raise RuntimeError(f"incompatible model output: expected float32 (1, 1, 320, 320), got {raw.dtype} {raw.shape}")
    if not np.isfinite(raw).all():
        raise RuntimeError("model output contains NaN or Inf")
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo <= np.finfo(np.float32).eps:
        raise NoClearSubject("model returned no clear subject")
    height, width = source_shape
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (height, width)):
        raise ValueError("source_shape must contain positive integer height and width")
    normalized = np.ascontiguousarray((raw[0, 0] - lo) / (hi - lo), dtype=np.float32)
    resized = Image.fromarray(normalized, mode="F").resize((width, height), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0))


def _metadata(item: object) -> tuple[str, tuple[int, ...], str]:
    try:
        shape = tuple(int(value) for value in getattr(item, "shape", None))
    except (TypeError, ValueError):
        shape = ()
    type_name = getattr(item, "type", None)
    dtype = "float32" if type_name in ("tensor(float)", "float32") else str(type_name)
    return str(getattr(item, "name", None)), shape, dtype


def _validate_session_contract(session: object, manifest: ModelManifest) -> None:
    try:
        inputs, outputs = session.get_inputs(), session.get_outputs()
    except Exception as exc:
        raise RuntimeError(f"unable to inspect model tensor contract: {exc}") from exc
    expected_in = (manifest.input_tensor.name, manifest.input_tensor.shape, manifest.input_tensor.dtype)
    expected_out = (manifest.output_tensor.name, manifest.output_tensor.shape, manifest.output_tensor.dtype)
    if len(inputs) != 1 or _metadata(inputs[0]) != expected_in:
        raise RuntimeError("incompatible model input tensor contract")
    output_metadata = [_metadata(item) for item in outputs]
    output_names = [item[0] for item in output_metadata]
    if manifest.output_names is None:
        raise RuntimeError("model manifest has no finalized seven-output name contract")
    if output_names != list(manifest.output_names):
        raise RuntimeError("incompatible model output tensor contract")
    if output_names.count(manifest.output_tensor.name) != 1:
        raise RuntimeError("manifest primary output tensor is absent or duplicated")
    primary = output_metadata[output_names.index(manifest.output_tensor.name)]
    if primary != expected_out:
        raise RuntimeError("incompatible model output tensor contract")
    if any(item[1:] != expected_out[1:] for item in output_metadata):
        raise RuntimeError("incompatible U-2-Net auxiliary output tensor contract")


def create_cpu_session(model_path: Path, *, intra_op_threads: int = 1) -> object:
    """Lazily import ORT and create a local CPU-only session."""
    if not isinstance(intra_op_threads, int) or isinstance(intra_op_threads, bool) or intra_op_threads < 1:
        raise ValueError("intra_op_threads must be a positive int")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ModelAssetError("onnxruntime 1.22.1 is not installed") from exc
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


class OrtSegmentationAdapter:
    def __init__(self, manifest: ModelManifest, *, asset_root: str | Path | None = None,
                 session_factory: Callable[[Path], object] | None = None, intra_op_threads: int = 1) -> None:
        if not isinstance(manifest, ModelManifest):
            raise TypeError("manifest must be a ModelManifest")
        if manifest.input_tensor != EXPECTED_INPUT_TENSOR or manifest.output_tensor != EXPECTED_OUTPUT_TENSOR:
            raise ModelAssetError("manifest tensor contract is incompatible with this adapter")
        if manifest.output_names is None:
            raise ModelAssetError("manifest output names are not finalized for release")
        if (manifest.preprocessing != MANIFEST_PREPROCESSING or
                manifest.output_semantics != MANIFEST_OUTPUT_SEMANTICS or
                manifest.algorithm_version != MANIFEST_ALGORITHM_VERSION):
            raise ModelAssetError("manifest preprocessing/output algorithm contract is incompatible with this adapter")
        self._model_identity = ModelIdentity(manifest.model_id, manifest.model_version, manifest.onnx_sha256)
        root = default_asset_root() if asset_root is None else Path(asset_root)

        def build() -> object:
            path = verify_model_asset(root, manifest)
            session = session_factory(path) if session_factory is not None else create_cpu_session(path, intra_op_threads=intra_op_threads)
            _validate_session_contract(session, manifest)
            return session

        self._session = LazySession(build)

    @property
    def model_identity(self) -> ModelIdentity:
        """Public identity (model id/version/onnx hash) for editor wiring."""
        return self._model_identity

    def infer(self, rgba_u8: np.ndarray, *, should_cancel: Callable[[], bool] | None = None) -> InferenceResult:
        validated = validate_rgba_u8(rgba_u8)
        source = np.array(validated, dtype=np.uint8, order="C", copy=True)
        source.flags.writeable = False
        identity = InferenceIdentity(source_identity(source), self._model_identity, PREPROCESSING_VERSION, "primary")
        _cancelled(should_cancel)
        tensor = preprocess_u2net(source)
        _cancelled(should_cancel)
        session = self._session.get()
        _cancelled(should_cancel)
        try:
            outputs: Sequence[Any] = session.run([OUTPUT_NAME], {INPUT_NAME: tensor})
        except Exception as exc:
            raise RuntimeError(f"segmentation runtime failed: {exc}") from exc
        _cancelled(should_cancel)
        if len(outputs) != 1:
            raise RuntimeError(f"incompatible model result: expected selected primary output, got {len(outputs)} outputs")
        confidence = postprocess_probability(outputs[0], source.shape[:2])
        _cancelled(should_cancel)
        return InferenceResult("primary", ProbabilityMap(identity, confidence))


def load_default_segmentation_adapter(asset_root: str | Path | None = None):
    """Best-effort build of a segmentation adapter from a staged manifest.

    Returns an :class:`OrtSegmentationAdapter` when a valid ``manifest.yaml`` and
    a hash-verified asset are staged under the asset root, otherwise ``None``.
    Never raises: a missing, unstaged, invalid, or contract-incompatible model
    must leave Smart Mask cleanly fail-closed (disabled), not crash startup.
    The returned adapter's ONNX session is lazy — no model is loaded until the
    first inference — so this stays cheap at startup.
    """
    root = default_asset_root() if asset_root is None else Path(asset_root)
    try:
        manifest = load_manifest(root / "manifest.yaml")
        verify_model_asset(root, manifest)
        return OrtSegmentationAdapter(manifest, asset_root=root)
    except Exception:
        return None
