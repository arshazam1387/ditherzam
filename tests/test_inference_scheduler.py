import numpy as np

from ditherzam.masking.contracts import ModelIdentity, source_identity
from ditherzam.masking.inference_request import InferenceRequest
from ditherzam.masking.inference_scheduler import InferenceScheduler


def _request(value=0, model_hash="a" * 64):
    rgba = np.full((2, 2, 4), value, np.uint8); rgba[..., 3] = 255
    return InferenceRequest(source_identity(rgba), ModelIdentity("u2", "1", model_hash), "u2net-320-v1", rgba)


def test_first_launch_then_only_newest_trailing_and_duplicate_terminal_safe():
    scheduler = InferenceScheduler()
    first = scheduler.request(_request(1))
    assert first is not None and scheduler.is_current(first)
    assert scheduler.request(_request(2)) is None
    assert scheduler.request(_request(3)) is None
    assert scheduler.should_cancel(first)
    trailing = scheduler.on_terminal(first)
    assert trailing is not None and trailing.rgba[0, 0, 0] == 3
    assert scheduler.on_terminal(first) is None
    assert scheduler.is_current(trailing)
    assert scheduler.on_terminal(trailing) is None


def test_source_invalidation_rejects_stale_publication_and_recovers():
    scheduler = InferenceScheduler(); first = scheduler.request(_request(1))
    new = _request(2)
    scheduler.invalidate_source(new.source)
    assert not scheduler.is_current(first) and scheduler.should_cancel(first)
    assert scheduler.on_terminal(first) is None
    launched = scheduler.request(new)
    assert launched is not None and scheduler.is_current(launched)


def test_model_and_generation_must_match_publication():
    scheduler = InferenceScheduler(); first = scheduler.request(_request(1))
    scheduler.request(_request(1, "b" * 64))
    assert not scheduler.is_current(first)
    second = scheduler.on_terminal(first)
    assert second is not None and scheduler.is_current(second)


def test_resubmitting_a_value_gets_a_fresh_cancellation_lifecycle():
    scheduler = InferenceScheduler(); value = _request(1)
    first = scheduler.request(value)
    scheduler.request(_request(2))
    assert first.cancellation.is_cancelled
    trailing = scheduler.on_terminal(first)
    scheduler.on_terminal(trailing)
    again = scheduler.request(value)
    assert again is not None and not again.cancellation.is_cancelled
