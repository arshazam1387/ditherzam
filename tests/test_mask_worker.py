import logging

import numpy as np
import pytest

from ditherzam.masking.adapter import InferenceCancelled, InferenceResult, NoClearSubject
from ditherzam.masking.contracts import ModelIdentity, ProbabilityMap, source_identity
from ditherzam.masking.inference_request import InferenceRequest, InferenceTerminal
from ditherzam.masking.model_assets import ModelAssetError
from ditherzam.ui.mask_workers import InferenceWorker


def _request():
    rgba = np.zeros((3, 4, 4), dtype=np.uint8)
    return InferenceRequest(
        source_identity(rgba),
        ModelIdentity("u2netp", "1", "a" * 64),
        "pre-v1",
        rgba,
    )


def _result(request):
    from ditherzam.masking.contracts import InferenceIdentity

    identity = InferenceIdentity(request.source, request.model, request.preprocessing_version, "primary")
    return InferenceResult("primary", ProbabilityMap(identity, np.ones((3, 4), dtype=np.float32)))


class _Adapter:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def infer(self, rgba, *, should_cancel=None):
        self.calls += 1
        if isinstance(self.action, BaseException):
            raise self.action
        return self.action(should_cancel) if callable(self.action) else self.action


def _record(worker):
    events = []
    for name in ("succeeded", "no_subject", "cancelled", "model_unavailable", "failed"):
        getattr(worker.signals, name).connect(lambda outcome, name=name: events.append((name, outcome)))
    return events


def test_success_emits_one_terminal_outcome():
    request = _request()
    worker = InferenceWorker(request, _Adapter(_result(request)))
    events = _record(worker)
    worker.run()
    assert [(name, outcome.terminal) for name, outcome in events] == [("succeeded", InferenceTerminal.SUCCESS)]
    assert events[0][1].request is request


def test_success_progress_is_coarse_monotonic_and_precedes_terminal():
    request = _request()
    worker = InferenceWorker(request, _Adapter(_result(request)))
    progress = []
    worker.signals.progress.connect(lambda req, value: progress.append((req, value)))
    events = _record(worker)
    worker.run()
    assert [value for _, value in progress] == [0, 10, 90, 100]
    assert all(req is request for req, _ in progress)
    assert [name for name, _ in events] == ["succeeded"]


@pytest.mark.parametrize(
    ("error", "signal", "terminal"),
    [
        (NoClearSubject("flat"), "no_subject", InferenceTerminal.NO_SUBJECT),
        (InferenceCancelled("old"), "cancelled", InferenceTerminal.CANCELLED),
        (ModelAssetError("missing"), "model_unavailable", InferenceTerminal.FAILED),
    ],
)
def test_expected_terminal_paths_are_mutually_exclusive(error, signal, terminal):
    worker = InferenceWorker(_request(), _Adapter(error))
    events = _record(worker)
    worker.run()
    assert [(name, outcome.terminal) for name, outcome in events] == [(signal, terminal)]


def test_cancel_before_start_does_not_enter_adapter():
    request = _request()
    adapter = _Adapter(_result(request))
    request.cancellation.cancel()
    worker = InferenceWorker(request, adapter)
    events = _record(worker)
    worker.run()
    assert [name for name, _ in events] == ["cancelled"]
    assert adapter.calls == 0


def test_cancel_after_runtime_safe_boundary_discards_result():
    request = _request()

    def infer(_check):
        request.cancellation.cancel()
        return _result(request)

    worker = InferenceWorker(request, _Adapter(infer))
    events = _record(worker)
    worker.run()
    assert [name for name, _ in events] == ["cancelled"]


@pytest.mark.parametrize(
    "error",
    [NoClearSubject("flat"), ModelAssetError("missing"), RuntimeError("runtime lost")],
)
def test_cancellation_wins_race_with_expected_adapter_terminal(error):
    request = _request()

    def infer(_check):
        request.cancellation.cancel()
        raise error

    worker = InferenceWorker(request, _Adapter(infer))
    events = _record(worker)
    worker.run()
    assert [name for name, _ in events] == ["cancelled"]


@pytest.mark.parametrize("bad_result", [None, object(), "not an inference result"])
def test_invalid_adapter_result_logs_and_fails_once(bad_result, caplog):
    worker = InferenceWorker(_request(), _Adapter(bad_result))
    events = _record(worker)
    with caplog.at_level(logging.ERROR):
        worker.run()
    assert [name for name, _ in events] == ["failed"]
    assert isinstance(events[0][1].error, TypeError)
    assert "Smart Mask inference failed" in caplog.text


@pytest.mark.parametrize("mismatch", ["source", "model", "preprocessing", "candidate"])
def test_result_identity_must_match_the_frozen_request(mismatch, caplog):
    from ditherzam.masking.contracts import InferenceIdentity, SourceIdentity

    request = _request()
    source = request.source
    model = request.model
    preprocessing = request.preprocessing_version
    candidate = "primary"
    if mismatch == "source":
        source = SourceIdentity(
            "f" * 64,
            request.source.width,
            request.source.height,
            request.source.has_alpha,
        )
    elif mismatch == "model":
        model = ModelIdentity("u2netp", "1", "b" * 64)
    elif mismatch == "preprocessing":
        preprocessing = "other-pre-v1"
    else:
        candidate = "secondary"
    identity = InferenceIdentity(source, model, preprocessing, candidate)
    probability = ProbabilityMap(identity, np.ones((3, 4), dtype=np.float32))
    # InferenceResult itself enforces v1 primary, so a candidate mismatch enters
    # through the probability identity while its public candidate remains primary.
    result = InferenceResult("primary", probability)
    worker = InferenceWorker(request, _Adapter(result))
    events = _record(worker)
    with caplog.at_level(logging.ERROR):
        worker.run()
    assert [name for name, _ in events] == ["failed"]
    assert isinstance(events[0][1].error, ValueError)


def test_unexpected_failure_is_logged_and_emitted_once(caplog):
    worker = InferenceWorker(_request(), _Adapter(RuntimeError("boom")))
    events = _record(worker)
    with caplog.at_level(logging.ERROR):
        worker.run()
    assert [name for name, _ in events] == ["failed"]
    assert isinstance(events[0][1].error, RuntimeError)
    assert "Smart Mask inference failed" in caplog.text


def test_terminal_outcome_releases_scheduler_and_promotes_once():
    from ditherzam.masking.inference_scheduler import InferenceScheduler

    scheduler = InferenceScheduler()
    first = scheduler.request(_request())
    assert first is not None
    assert scheduler.request(_request()) is None
    worker = InferenceWorker(first, _Adapter(_result(first)))
    promoted = []
    for name in ("succeeded", "no_subject", "cancelled", "model_unavailable", "failed"):
        getattr(worker.signals, name).connect(
            lambda outcome: promoted.append(scheduler.on_terminal(outcome))
        )
    worker.run()
    assert len(promoted) == 1 and promoted[0] is not None
    assert scheduler.on_terminal(first) is None


def test_constructor_rejects_live_or_incomplete_dependencies():
    with pytest.raises(TypeError, match="request"):
        InferenceWorker(object(), _Adapter(None))
    with pytest.raises(TypeError, match="adapter"):
        InferenceWorker(_request(), object())
