"""Tests for ditherzam.masking.contracts: content-based identity and value
contracts for source, model, inference, and derived-mask data.

Identity is always content-based (a digest of decoded pixel bytes plus
explicit dimensions/alpha participation), never a filename or path.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from ditherzam.masking.contracts import (
    InferenceIdentity,
    MaskContractError,
    MaskIdentity,
    ModelIdentity,
    ProbabilityMap,
    SourceIdentity,
    source_identity,
    validate_confidence_array,
    validate_rgba_u8,
)
from ditherzam.masking.settings import MaskTarget


VALID_SHA256 = "a" * 64


def _rgba(height=4, width=6, alpha=255):
    array = np.zeros((height, width, 4), dtype=np.uint8)
    array[..., 0] = 10
    array[..., 1] = 20
    array[..., 2] = 30
    array[..., 3] = alpha
    return array


def _confidence(height=4, width=6, fill=0.5):
    return np.full((height, width), fill, dtype=np.float32)


def _model_identity():
    return ModelIdentity(model_id="u2net", model_version="1.0.0", model_hash=VALID_SHA256)


def _inference_identity(src=None):
    return InferenceIdentity(
        source=src or source_identity(_rgba()),
        model=_model_identity(),
        preprocessing_version="pp-1",
        candidate_id="primary",
    )


def _mask_identity():
    return MaskIdentity(
        inference=_inference_identity(),
        sensitivity=50,
        target=MaskTarget.SUBJECT,
        invert=False,
        expansion_px=0,
        feather_px=8,
        feather_algorithm_version="feather-v1",
    )


# -- source_identity: content-based, not path-based --------------------------


def test_source_identity_has_no_path_parameter():
    params = inspect.signature(source_identity).parameters
    assert list(params) == ["rgba_u8"]


def test_source_identity_reports_dimensions_and_hash():
    ident = source_identity(_rgba(height=5, width=9))
    assert ident.height == 5
    assert ident.width == 9
    assert isinstance(ident.content_hash, str)
    assert len(ident.content_hash) == 64


def test_source_identity_same_bytes_same_identity():
    a = source_identity(_rgba())
    b = source_identity(_rgba())
    assert a == b
    assert hash(a) == hash(b)


def test_source_identity_different_bytes_different_identity():
    a = source_identity(_rgba())
    b = source_identity(_rgba(alpha=254))
    assert a != b


def test_source_identity_alpha_participation_true_when_translucent():
    array = _rgba(alpha=255)
    array[0, 0, 3] = 100
    ident = source_identity(array)
    assert ident.has_alpha is True


def test_source_identity_alpha_participation_false_when_fully_opaque():
    ident = source_identity(_rgba(alpha=255))
    assert ident.has_alpha is False


def test_source_identity_rejects_wrong_dtype():
    bad = _rgba().astype(np.float32)
    with pytest.raises(MaskContractError):
        source_identity(bad)


def test_source_identity_rejects_wrong_channel_count():
    bad = np.zeros((4, 6, 3), dtype=np.uint8)
    with pytest.raises(MaskContractError):
        source_identity(bad)


def test_source_identity_rejects_non_3d():
    bad = np.zeros((4, 6), dtype=np.uint8)
    with pytest.raises(MaskContractError):
        source_identity(bad)


def test_validate_rgba_u8_returns_input():
    array = _rgba()
    assert validate_rgba_u8(array) is array


# -- SourceIdentity direct construction / validation --------------------------


def test_source_identity_direct_construction_valid():
    ident = SourceIdentity(content_hash=VALID_SHA256, width=6, height=4, has_alpha=False)
    assert ident.width == 6 and ident.height == 4


def test_source_identity_rejects_bad_hash_format():
    with pytest.raises(MaskContractError):
        SourceIdentity(content_hash="not-a-hash", width=6, height=4, has_alpha=False)


def test_source_identity_rejects_uppercase_hash():
    with pytest.raises(MaskContractError):
        SourceIdentity(content_hash="A" * 64, width=6, height=4, has_alpha=False)


def test_source_identity_rejects_non_positive_dimensions():
    with pytest.raises(MaskContractError):
        SourceIdentity(content_hash=VALID_SHA256, width=0, height=4, has_alpha=False)
    with pytest.raises(MaskContractError):
        SourceIdentity(content_hash=VALID_SHA256, width=6, height=-1, has_alpha=False)


# -- ModelIdentity -------------------------------------------------------------


def test_model_identity_valid():
    m = _model_identity()
    assert m.model_id == "u2net"


def test_model_identity_rejects_empty_id():
    with pytest.raises(MaskContractError):
        ModelIdentity(model_id="", model_version="1.0.0", model_hash=VALID_SHA256)


def test_model_identity_rejects_bad_hash():
    with pytest.raises(MaskContractError):
        ModelIdentity(model_id="u2net", model_version="1.0.0", model_hash="short")


def test_model_identity_equality_and_hash():
    a = _model_identity()
    b = _model_identity()
    assert a == b
    assert hash(a) == hash(b)


# -- InferenceIdentity ----------------------------------------------------------


def test_inference_identity_valid():
    ident = _inference_identity()
    assert ident.candidate_id == "primary"


def test_inference_identity_rejects_wrong_source_type():
    with pytest.raises(MaskContractError):
        InferenceIdentity(
            source="not-a-source-identity",  # type: ignore[arg-type]
            model=_model_identity(),
            preprocessing_version="pp-1",
            candidate_id="primary",
        )


def test_inference_identity_rejects_empty_candidate_id():
    with pytest.raises(MaskContractError):
        InferenceIdentity(
            source=source_identity(_rgba()),
            model=_model_identity(),
            preprocessing_version="pp-1",
            candidate_id="",
        )


def test_inference_identity_equality_and_hash():
    src = source_identity(_rgba())
    a = _inference_identity(src)
    b = _inference_identity(src)
    assert a == b
    assert hash(a) == hash(b)


def test_inference_identity_changes_with_source():
    src_a = source_identity(_rgba())
    src_b = source_identity(_rgba(alpha=254))
    a = _inference_identity(src_a)
    b = _inference_identity(src_b)
    assert a != b


# -- MaskIdentity ----------------------------------------------------------------


def test_mask_identity_valid():
    ident = _mask_identity()
    assert ident.target is MaskTarget.SUBJECT


def test_mask_identity_equality_and_hash():
    a = _mask_identity()
    b = _mask_identity()
    assert a == b
    assert hash(a) == hash(b)


def test_mask_identity_changes_with_sensitivity():
    a = _mask_identity()
    b = MaskIdentity(
        inference=a.inference,
        sensitivity=60,
        target=MaskTarget.SUBJECT,
        invert=False,
        expansion_px=0,
        feather_px=8,
        feather_algorithm_version="feather-v1",
    )
    assert a != b


def test_mask_identity_rejects_out_of_range_sensitivity():
    with pytest.raises(MaskContractError):
        MaskIdentity(
            inference=_inference_identity(),
            sensitivity=101,
            target=MaskTarget.SUBJECT,
            invert=False,
            expansion_px=0,
            feather_px=8,
            feather_algorithm_version="feather-v1",
        )


def test_mask_identity_rejects_out_of_range_expansion():
    with pytest.raises(MaskContractError):
        MaskIdentity(
            inference=_inference_identity(),
            sensitivity=50,
            target=MaskTarget.SUBJECT,
            invert=False,
            expansion_px=65,
            feather_px=8,
            feather_algorithm_version="feather-v1",
        )


def test_mask_identity_rejects_negative_feather():
    with pytest.raises(MaskContractError):
        MaskIdentity(
            inference=_inference_identity(),
            sensitivity=50,
            target=MaskTarget.SUBJECT,
            invert=False,
            expansion_px=0,
            feather_px=-1,
            feather_algorithm_version="feather-v1",
        )


def test_mask_identity_rejects_wrong_target_type():
    with pytest.raises(MaskContractError):
        MaskIdentity(
            inference=_inference_identity(),
            sensitivity=50,
            target="subject",  # type: ignore[arg-type]
            invert=False,
            expansion_px=0,
            feather_px=8,
            feather_algorithm_version="feather-v1",
        )


def test_mask_identity_rejects_empty_feather_algorithm_version():
    with pytest.raises(MaskContractError):
        MaskIdentity(
            inference=_inference_identity(),
            sensitivity=50,
            target=MaskTarget.SUBJECT,
            invert=False,
            expansion_px=0,
            feather_px=8,
            feather_algorithm_version="",
        )


# -- ProbabilityMap / validate_confidence_array ---------------------------------


def test_probability_map_valid_construction():
    src = source_identity(_rgba(height=4, width=6))
    ident = _inference_identity(src)
    pmap = ProbabilityMap(identity=ident, values=_confidence(height=4, width=6))
    assert pmap.values.shape == (4, 6)


def test_probability_map_values_are_read_only():
    src = source_identity(_rgba(height=4, width=6))
    ident = _inference_identity(src)
    pmap = ProbabilityMap(identity=ident, values=_confidence(height=4, width=6))
    assert pmap.values.flags.writeable is False
    with pytest.raises(ValueError):
        pmap.values[0, 0] = 0.9


def test_probability_map_takes_defensive_copy_from_view():
    # A view/slice of a larger writable base buffer must not be able to mutate
    # the stored payload after construction.
    src = source_identity(_rgba(height=4, width=6))
    ident = _inference_identity(src)
    base = np.zeros((5, 6), dtype=np.float32)  # larger base, writable
    view = base[:4, :]  # a view sharing base's memory, correct (4, 6) shape
    assert view.base is base
    pmap = ProbabilityMap(identity=ident, values=view)
    assert pmap.values.flags.owndata is True
    before = pmap.values.copy()
    base[:] = 0.9  # mutate the original base buffer
    np.testing.assert_array_equal(pmap.values, before)
    assert float(pmap.values.max()) == 0.0


def test_probability_map_hash_is_stable_and_identity_keyed():
    src = source_identity(_rgba(height=4, width=6))
    ident = _inference_identity(src)
    a = ProbabilityMap(identity=ident, values=_confidence(height=4, width=6, fill=0.2))
    b = ProbabilityMap(identity=ident, values=_confidence(height=4, width=6, fill=0.8))
    # hash must not raise and must be stable/equal for equal identity, even
    # though the value arrays differ (values are opaque payload).
    assert hash(a) == hash(a)
    assert hash(a) == hash(b)
    assert a == b
    assert not (a != b)


def test_probability_map_differing_identity_compare_unequal():
    src_a = source_identity(_rgba(height=4, width=6))
    src_b = source_identity(_rgba(height=4, width=6, alpha=254))
    values = _confidence(height=4, width=6)
    a = ProbabilityMap(identity=_inference_identity(src_a), values=values)
    b = ProbabilityMap(identity=_inference_identity(src_b), values=values)
    assert a != b
    assert hash(a) != hash(b)


def test_probability_map_rejects_shape_mismatch_with_source():
    src = source_identity(_rgba(height=4, width=6))
    ident = _inference_identity(src)
    with pytest.raises(MaskContractError):
        ProbabilityMap(identity=ident, values=_confidence(height=5, width=6))


def test_probability_map_rejects_wrong_identity_type():
    with pytest.raises(MaskContractError):
        ProbabilityMap(identity="not-an-identity", values=_confidence())  # type: ignore[arg-type]


def test_validate_confidence_array_rejects_wrong_dtype():
    with pytest.raises(MaskContractError):
        validate_confidence_array(_confidence().astype(np.float64))


def test_validate_confidence_array_rejects_non_2d():
    with pytest.raises(MaskContractError):
        validate_confidence_array(np.zeros((4, 6, 1), dtype=np.float32))


def test_validate_confidence_array_rejects_out_of_range():
    bad = _confidence(fill=1.5)
    with pytest.raises(MaskContractError):
        validate_confidence_array(bad)
    bad2 = _confidence(fill=-0.1)
    with pytest.raises(MaskContractError):
        validate_confidence_array(bad2)


def test_validate_confidence_array_rejects_nan():
    bad = _confidence()
    bad[0, 0] = np.nan
    with pytest.raises(MaskContractError):
        validate_confidence_array(bad)


def test_validate_confidence_array_rejects_inf():
    bad = _confidence()
    bad[0, 0] = np.inf
    with pytest.raises(MaskContractError):
        validate_confidence_array(bad)


def test_validate_confidence_array_rejects_non_c_contiguous():
    base = np.ones((6, 4), dtype=np.float32)
    non_contiguous = base.T  # transpose of a non-square array is not C-contiguous
    assert non_contiguous.flags["C_CONTIGUOUS"] is False
    with pytest.raises(MaskContractError):
        validate_confidence_array(non_contiguous)


def test_validate_confidence_array_rejects_non_array():
    with pytest.raises(MaskContractError):
        validate_confidence_array([[0.1, 0.2], [0.3, 0.4]])  # type: ignore[arg-type]


# -- No Qt imports --------------------------------------------------------------


def test_contracts_module_has_no_qt_import():
    import ditherzam.masking.contracts as mod
    from pathlib import Path

    text = Path(mod.__file__).read_text(encoding="utf-8")
    assert "PySide6" not in text
    assert "PyQt" not in text
