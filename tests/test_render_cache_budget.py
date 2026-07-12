"""Memory-budget contract for retained staged render results.

The cache owns complete dependency groups.  A group may contain signatures,
arrays, and aliases of those arrays; it is admitted or evicted atomically.
"""

from types import MappingProxyType

import numpy as np
import pytest

from ditherzam.render_cache import DEFAULT_CACHE_BUDGET_BYTES, RenderCache


MIB = 1024 * 1024


def _array(nbytes: int) -> np.ndarray:
    return np.zeros(nbytes, dtype=np.uint8)


def test_default_budget_is_192_mib_and_metrics_are_read_only():
    cache = RenderCache()

    assert DEFAULT_CACHE_BUDGET_BYTES == 192 * MIB
    assert cache.budget_bytes == 192 * MIB
    assert cache.retained_bytes == 0
    assert cache.entry_count == 0
    assert cache.eviction_count == 0

    for name in ("budget_bytes", "retained_bytes", "entry_count", "eviction_count"):
        with pytest.raises(AttributeError):
            setattr(cache, name, 1)


def test_group_counts_unique_numpy_backing_storage_once():
    cache = RenderCache(budget_bytes=1_000)
    backing = _array(120)
    left = backing[:80]
    right = backing[40:]
    independent = _array(30)

    admitted = cache.put(
        "chain",
        {
            "backing": backing,
            "left_alias": left,
            "right_alias": right,
            "independent": independent,
            "signatures": ("Floyd-Steinberg", 2),
        },
    )

    assert admitted is True
    assert cache.retained_bytes == 150
    assert cache.entry_count == 1


def test_distinct_views_with_a_shared_ultimate_base_are_not_double_counted():
    cache = RenderCache(budget_bytes=1_000)
    backing = _array(200)
    reshaped = backing.reshape(20, 10)

    cache.put("one", {"a": reshaped[:, :5], "b": reshaped[:, 5:]})

    assert cache.retained_bytes == backing.nbytes


def test_oversized_group_is_returned_to_caller_but_not_retained():
    cache = RenderCache(budget_bytes=100)
    existing = {"g": _array(60)}
    oversized = {"g": _array(101)}
    cache.put("existing", existing)

    admitted = cache.put("oversized", oversized)

    assert admitted is False
    assert cache.get("oversized") is None
    assert cache.get("existing") is existing
    assert cache.retained_bytes == 60
    assert cache.entry_count == 1
    assert cache.eviction_count == 0


def test_get_touches_lru_and_eviction_removes_a_complete_group():
    cache = RenderCache(budget_bytes=100)
    first = {"g": _array(40), "d": _array(10), "sig": "first"}
    second = {"g": _array(50), "sig": "second"}
    third = {"g": _array(45), "sig": "third"}
    cache.put("first", first)
    cache.put("second", second)

    assert cache.get("first") is first  # first is now most recently used
    assert cache.put("third", third) is True

    assert cache.get("second") is None
    assert cache.get("first") is first
    assert cache.get("third") is third
    assert cache.retained_bytes == 95
    assert cache.entry_count == 2
    assert cache.eviction_count == 1
    # No stage from the evicted dependency group remains addressable.
    assert set(cache.keys()) == {"first", "third"}


def test_replacing_a_group_reaccounts_storage_without_counting_an_eviction():
    cache = RenderCache(budget_bytes=100)
    cache.put("chain", {"g": _array(70)})

    assert cache.put("chain", {"g": _array(25)}) is True

    assert cache.retained_bytes == 25
    assert cache.entry_count == 1
    assert cache.eviction_count == 0


def test_clear_drops_groups_and_resets_all_metrics():
    cache = RenderCache(budget_bytes=100)
    cache.put("first", {"g": _array(60)})
    cache.put("second", {"g": _array(60)})  # evicts first
    assert cache.eviction_count == 1

    cache.clear()

    assert cache.retained_bytes == 0
    assert cache.entry_count == 0
    assert cache.eviction_count == 0
    assert tuple(cache.keys()) == ()


def test_metrics_snapshot_is_immutable_and_includes_byte_count_and_evictions():
    cache = RenderCache(budget_bytes=100)
    cache.put("chain", {"g": _array(25)})

    metrics = cache.metrics

    assert isinstance(metrics, MappingProxyType)
    assert metrics == {
        "budget_bytes": 100,
        "retained_bytes": 25,
        "entry_count": 1,
        "eviction_count": 0,
    }
    with pytest.raises(TypeError):
        metrics["retained_bytes"] = 0
