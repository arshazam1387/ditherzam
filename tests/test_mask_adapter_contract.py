from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ditherzam.masking.adapter import InferenceCancelled, NoClearSubject
from ditherzam.masking.model_assets import APPROVED_UPSTREAM_COMMIT, EXPECTED_INPUT_TENSOR, EXPECTED_OUTPUT_TENSOR, ModelManifest
from ditherzam.masking.contracts import source_identity
from ditherzam.masking.ort_adapter import INPUT_NAME, MANIFEST_ALGORITHM_VERSION, MANIFEST_OUTPUT_SEMANTICS, MANIFEST_PREPROCESSING, OUTPUT_NAME, OrtSegmentationAdapter, postprocess_probability, preprocess_u2net


def _manifest(data: bytes) -> ModelManifest:
    digest = hashlib.sha256(data).hexdigest()
    return ModelManifest("u2netp", "1", "repo", APPROVED_UPSTREAM_COMMIT, "source", "b" * 64,
                         "Apache-2.0", "attr", "rev", 17, (("onnx", "1"),),
                         digest, len(data), EXPECTED_INPUT_TENSOR, EXPECTED_OUTPUT_TENSOR,
                         MANIFEST_PREPROCESSING, MANIFEST_OUTPUT_SEMANTICS,
                         MANIFEST_ALGORITHM_VERSION, "model.onnx",
                         ("1959", "0", "1", "2", "3", "4", "5"))


def _meta(name, *, shape=(1, 1, 320, 320)):
    return SimpleNamespace(name=name, shape=shape, type="tensor(float)")


class FakeSession:
    def __init__(self):
        self.calls = []

    def get_inputs(self): return [_meta(INPUT_NAME, shape=(1, 3, 320, 320))]
    def get_outputs(self): return [_meta("1959")] + [_meta(str(i)) for i in range(6)]
    def run(self, names, feeds):
        self.calls.append((names, feeds))
        primary = np.linspace(0, 1, 320 * 320, dtype=np.float32).reshape(1, 1, 320, 320)
        return [primary]


def _adapter(tmp_path: Path):
    data = b"safe local fake model"
    (tmp_path / "model.onnx").write_bytes(data)
    fake = FakeSession()
    calls = []
    adapter = OrtSegmentationAdapter(_manifest(data), asset_root=tmp_path,
                                     session_factory=lambda path: calls.append(path) or fake)
    return adapter, fake, calls


def test_preprocess_exact_shape_dtype_normalization_and_alpha_ignored():
    rgba = np.zeros((2, 3, 4), np.uint8)
    rgba[..., :3] = (255, 0, 127)
    rgba[..., 3] = np.arange(6, dtype=np.uint8).reshape(2, 3)
    tensor = preprocess_u2net(rgba)
    assert tensor.shape == (1, 3, 320, 320) and tensor.dtype == np.float32
    expected = (np.array([1.0, 0.0, 127 / 255], np.float32) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np.testing.assert_allclose(tensor[0, :, 100, 100], expected, rtol=0, atol=1e-6)


def test_primary_output_source_resize_immutable_and_session_reused(tmp_path):
    adapter, fake, creates = _adapter(tmp_path)
    source = np.zeros((7, 11, 4), np.uint8)
    source[..., 3] = 255
    first = adapter.infer(source)
    second = adapter.infer(source)
    assert first.candidate_id == "primary" and first.confidence.shape == (7, 11)
    assert first.confidence.dtype == np.float32 and not first.confidence.flags.writeable
    assert 0 <= first.confidence.min() < first.confidence.max() <= 1
    assert len(creates) == 1 and len(fake.calls) == 2
    assert fake.calls[0][0] == [OUTPUT_NAME] and set(fake.calls[0][1]) == {INPUT_NAME}
    assert second.probability.identity == first.probability.identity


def test_degenerate_output_is_explicit_no_subject():
    with pytest.raises(NoClearSubject, match="no clear subject"):
        postprocess_probability(np.ones((1, 1, 320, 320), np.float32), (2, 2))


def test_safe_boundary_cancellation_avoids_session_creation(tmp_path):
    adapter, fake, creates = _adapter(tmp_path)
    with pytest.raises(InferenceCancelled):
        adapter.infer(np.zeros((2, 2, 4), np.uint8), should_cancel=lambda: True)
    assert creates == [] and fake.calls == []


def test_missing_asset_and_runtime_errors_are_honest(tmp_path):
    adapter = OrtSegmentationAdapter(_manifest(b"absent"), asset_root=tmp_path, session_factory=lambda _: FakeSession())
    with pytest.raises(Exception, match="not found"):
        adapter.infer(np.zeros((2, 2, 4), np.uint8))
    adapter, fake, _ = _adapter(tmp_path)
    fake.run = lambda *_: (_ for _ in ()).throw(MemoryError("oom"))
    with pytest.raises(RuntimeError, match="segmentation runtime failed: oom"):
        adapter.infer(np.zeros((2, 2, 4), np.uint8))


def test_incompatible_session_contract_fails_before_run(tmp_path):
    adapter, fake, _ = _adapter(tmp_path)
    fake.get_inputs = lambda: [_meta("wrong", shape=(1, 3, 320, 320))]
    with pytest.raises(RuntimeError, match="input tensor contract"):
        adapter.infer(np.zeros((2, 2, 4), np.uint8))


def test_session_outputs_must_match_exact_manifest_order(tmp_path):
    adapter, fake, _ = _adapter(tmp_path)
    fake.get_outputs = lambda: [_meta("aux-a"), _meta(OUTPUT_NAME), _meta("aux-b"),
                                _meta("aux-c"), _meta("aux-d"), _meta("aux-e"), _meta("aux-f")]
    with pytest.raises(RuntimeError, match="output tensor contract"):
        adapter.infer(np.zeros((2, 2, 4), np.uint8))

    adapter, fake, _ = _adapter(tmp_path)
    fake.get_outputs = lambda: [_meta(OUTPUT_NAME), _meta("dup"), _meta("dup"),
                                _meta("a"), _meta("b"), _meta("c"), _meta("d")]
    with pytest.raises(RuntimeError, match="output tensor contract"):
        adapter.infer(np.zeros((2, 2, 4), np.uint8))


def test_unfinalized_output_names_fail_before_session_creation(tmp_path):
    data = b"safe local fake model"
    (tmp_path / "model.onnx").write_bytes(data)
    with pytest.raises(Exception, match="output names are not finalized"):
        OrtSegmentationAdapter(replace(_manifest(data), output_names=None), asset_root=tmp_path,
                               session_factory=lambda _: FakeSession())


def test_inference_owns_one_snapshot_before_session_can_mutate_caller(tmp_path):
    adapter, fake, _ = _adapter(tmp_path)
    source = np.zeros((5, 7, 4), np.uint8)
    source[..., :3] = (20, 80, 160)
    source[..., 3] = 255
    before = source.copy()
    expected_tensor = preprocess_u2net(before)
    expected_identity = source_identity(before)
    original_run = fake.run

    def mutating_run(names, feeds):
        source[:] = 255
        return original_run(names, feeds)

    fake.run = mutating_run
    result = adapter.infer(source)
    np.testing.assert_array_equal(fake.calls[0][1][INPUT_NAME], expected_tensor)
    assert result.probability.identity.source == expected_identity
    assert not result.confidence.flags.writeable


@pytest.mark.parametrize("field", ["preprocessing", "output_semantics", "algorithm_version"])
def test_adapter_rejects_noncanonical_manifest_algorithm_fields(tmp_path, field):
    manifest = replace(_manifest(b"x"), **{field: "different"})
    with pytest.raises(Exception, match="algorithm contract"):
        OrtSegmentationAdapter(manifest, asset_root=tmp_path, session_factory=lambda _: FakeSession())
