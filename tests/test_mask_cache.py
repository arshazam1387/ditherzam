import numpy as np

from ditherzam.masking.cache import (
    CompositeIdentity, DEFAULT_MASK_CACHE_BUDGET_BYTES, MaskCaches,
    editor_cache_allocation,
)
from ditherzam.masking.contracts import (
    InferenceIdentity, MaskIdentity, ModelIdentity, ProbabilityMap, source_identity,
)
from ditherzam.masking.settings import MaskTarget, OutsideMode
from ditherzam.render_cache import DEFAULT_CACHE_BUDGET_BYTES, MIB


def _ids(seed=1, *, preprocessing="pp1", model="a" * 64):
    rgba = np.full((2, 3, 4), seed, np.uint8); rgba[..., 3] = 255
    source = source_identity(rgba)
    inference = InferenceIdentity(source, ModelIdentity("u2", "1", model), preprocessing, "primary")
    mask = MaskIdentity(inference, 50, MaskTarget.SUBJECT, False, 0, 8, "f1")
    return source, inference, mask


def _probability(identity):
    return ProbabilityMap(identity, np.full((2, 3), .5, np.float32))


def test_partitions_use_complete_identities_and_publish_readonly_arrays():
    caches = MaskCaches(1000)
    source, inference, mask_id = _ids()
    probability = _probability(inference)
    derived = np.full((2, 3), .25, np.float32)
    composite_id = CompositeIdentity("render-1", mask_id, OutsideMode.ORIGINAL, source, "alpha1")

    assert caches.put_inference(probability)
    assert caches.put_derived(mask_id, derived)
    assert caches.put_composite(composite_id, np.zeros((2, 3, 3), np.uint8))
    derived[:] = 1

    assert caches.get_inference(inference) is probability
    assert np.all(caches.get_derived(mask_id) == .25)
    assert not caches.get_derived(mask_id).flags.writeable
    assert not caches.get_composite(composite_id).flags.writeable
    assert caches.get_inference(_ids(preprocessing="pp2")[1]) is None


def test_atomic_lru_oversized_and_metrics_are_bounded():
    caches = MaskCaches(30)
    _, inf1, mask1 = _ids(1)
    _, inf2, mask2 = _ids(2)
    assert caches.put_derived(mask1, np.zeros((2, 3), np.float32))
    assert caches.put_derived(mask2, np.ones((2, 3), np.float32))
    assert caches.get_derived(mask1) is None
    assert caches.get_derived(mask2) is not None
    before = caches.metrics
    assert not caches.put_composite(
        CompositeIdentity("r", mask2, OutsideMode.WHITE, inf2.source, "a1"),
        np.zeros((20, 20, 3), np.uint8),
    )
    assert caches.retained_bytes <= caches.budget_bytes
    assert before["entry_count"] == 1
    try:
        before["entry_count"] = 9
        assert False
    except TypeError:
        pass


def test_editor_allocation_preserves_unmasked_default_and_splits_masked_budget():
    assert DEFAULT_CACHE_BUDGET_BYTES == 192 * MIB
    assert editor_cache_allocation(False).render_bytes == 192 * MIB
    assert editor_cache_allocation(False).mask_bytes == 0
    masked = editor_cache_allocation(True)
    assert masked.render_bytes == 128 * MIB
    assert masked.mask_bytes == DEFAULT_MASK_CACHE_BUDGET_BYTES == 64 * MIB
    assert masked.render_bytes + masked.mask_bytes == 192 * MIB


def test_editor_allocation_rejects_invalid_custom_aggregate():
    import pytest
    with pytest.raises(ValueError, match="exceeds"):
        editor_cache_allocation(True, render_bytes=192 * MIB, mask_bytes=64 * MIB)
    with pytest.raises(ValueError, match="zero"):
        editor_cache_allocation(False, render_bytes=128 * MIB, mask_bytes=64 * MIB)
    custom = editor_cache_allocation(True, render_bytes=100 * MIB, mask_bytes=50 * MIB)
    assert custom.render_bytes + custom.mask_bytes == 150 * MIB


def test_inference_payload_is_charged_and_oversized_probability_not_retained():
    _, inference, _ = _ids()
    caches = MaskCaches(23)  # float32 2x3 payload is 24 bytes
    assert not caches.put_inference(_probability(inference))
    assert caches.get_inference(inference) is None
    assert caches.retained_bytes == 0


def test_probability_alias_replacement_is_not_double_charged():
    _, inference, _ = _ids()
    probability = _probability(inference)
    caches = MaskCaches(24)
    assert caches.put_inference(probability)
    assert caches.retained_bytes == probability.values.nbytes
    assert caches.put_inference(probability)
    assert caches.retained_bytes == probability.values.nbytes


def test_clear_source_and_fifty_source_soak():
    caches = MaskCaches(80)
    first = None
    for seed in range(50):
        source, inference, mask = _ids(seed)
        first = first or source
        caches.put_inference(_probability(inference))
        caches.put_derived(mask, np.zeros((2, 3), np.float32))
        assert caches.retained_bytes <= 80
    caches.clear_source(first)
    assert all(key.source != first for key in caches._stores["inference"])
    assert all(key.inference.source != first for key in caches._stores["derived"])


def test_fifty_probability_sources_stay_within_retained_budget():
    caches = MaskCaches(120)  # at most five 24-byte probability payloads
    for seed in range(50):
        _, inference, _ = _ids(seed)
        assert caches.put_inference(_probability(inference))
        assert caches.retained_bytes <= caches.budget_bytes
    assert caches.entry_count == 5
    assert caches.retained_bytes == 120


def test_composite_identity_partitions_outside_source_and_alpha_version():
    source, _, mask = _ids()
    other_source, _, _ = _ids(2)
    base = CompositeIdentity("render", mask, OutsideMode.ORIGINAL, source, "alpha1")
    assert base != CompositeIdentity("render", mask, OutsideMode.BLACK, source, "alpha1")
    assert base != CompositeIdentity("render", mask, OutsideMode.ORIGINAL, other_source, "alpha1")
    assert base != CompositeIdentity("render", mask, OutsideMode.ORIGINAL, source, "alpha2")
