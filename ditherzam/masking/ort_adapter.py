"""Offline CPU-only ONNX adapter for the frozen U-2-Net tensor contract."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ditherzam.masking.adapter import InferenceCancelled, InferenceResult, NoClearSubject
from ditherzam.masking.contracts import InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity, validate_rgba_u8
from ditherzam.masking.model_assets import EXPECTED_INPUT_TENSOR, EXPECTED_OUTPUT_TENSOR, ModelAssetError, ModelManifest, default_asset_root, verify_model_asset
from ditherzam.masking.session import LazySession

PREPROCESSING_VERSION = "u2net-rgb-imagenet-bilinear-v1"
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


def _validate_session_contract(session: object) -> None:
    try:
        inputs, outputs = session.get_inputs(), session.get_outputs()
    except Exception as exc:
        raise RuntimeError(f"unable to inspect model tensor contract: {exc}") from exc
    expected_in = (INPUT_NAME, EXPECTED_INPUT_TENSOR.shape, EXPECTED_INPUT_TENSOR.dtype)
    expected_out = (OUTPUT_NAME, EXPECTED_OUTPUT_TENSOR.shape, EXPECTED_OUTPUT_TENSOR.dtype)
    if len(inputs) != 1 or _metadata(inputs[0]) != expected_in:
        raise RuntimeError("incompatible model input tensor contract")
    if len(outputs) != 7 or _metadata(outputs[0]) != expected_out:
        raise RuntimeError("incompatible model output tensor contract")
    if any(_metadata(item)[1:] != expected_out[1:] for item in outputs):
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
        self._model_identity = ModelIdentity(manifest.model_id, manifest.model_version, manifest.onnx_sha256)
        root = default_asset_root() if asset_root is None else Path(asset_root)

        def build() -> object:
            path = verify_model_asset(root, manifest)
            session = session_factory(path) if session_factory is not None else create_cpu_session(path, intra_op_threads=intra_op_threads)
            _validate_session_contract(session)
            return session

        self._session = LazySession(build)

    def infer(self, rgba_u8: np.ndarray, *, should_cancel: Callable[[], bool] | None = None) -> InferenceResult:
        source = validate_rgba_u8(rgba_u8)
        _cancelled(should_cancel)
        tensor = preprocess_u2net(source)
        _cancelled(should_cancel)
        session = self._session.get()
        _cancelled(should_cancel)
        try:
            outputs: Sequence[Any] = session.run(None, {INPUT_NAME: tensor})
        except Exception as exc:
            raise RuntimeError(f"segmentation runtime failed: {exc}") from exc
        _cancelled(should_cancel)
        if len(outputs) != 7:
            raise RuntimeError(f"incompatible model result: expected 7 outputs, got {len(outputs)}")
        confidence = postprocess_probability(outputs[0], source.shape[:2])
        _cancelled(should_cancel)
        identity = InferenceIdentity(source_identity(source), self._model_identity, PREPROCESSING_VERSION, "primary")
        return InferenceResult("primary", ProbabilityMap(identity, confidence))
