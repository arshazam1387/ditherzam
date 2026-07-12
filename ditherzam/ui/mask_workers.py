"""Resilient Qt worker for one immutable Smart Mask inference request."""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

from ditherzam.masking.adapter import (
    InferenceCancelled,
    InferenceResult,
    NoClearSubject,
    SegmentationAdapter,
)
from ditherzam.masking.inference_request import (
    InferenceOutcome,
    InferenceRequest,
    InferenceTerminal,
)
from ditherzam.masking.model_assets import ModelAssetError


_LOG = logging.getLogger(__name__)


class InferenceSignals(QObject):
    """Mutually exclusive terminal outcomes for an inference run."""

    succeeded = Signal(object)  # InferenceOutcome
    no_subject = Signal(object)  # InferenceOutcome
    cancelled = Signal(object)  # InferenceOutcome
    model_unavailable = Signal(object)  # InferenceOutcome
    failed = Signal(object)  # InferenceOutcome
    progress = Signal(object, int)  # immutable request, coarse safe-boundary percent


class InferenceWorker(QRunnable):
    """Run an adapter using only a frozen request and cooperative cancellation.

    ONNX execution is never forcibly interrupted.  The adapter observes the
    request token at its safe boundaries, and the post-inference check prevents
    obsolete work from being published if cancellation arrived during a runtime
    call.
    """

    def __init__(self, request: InferenceRequest, adapter: SegmentationAdapter) -> None:
        super().__init__()
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        if not callable(getattr(adapter, "infer", None)):
            raise TypeError("adapter must provide infer")
        self._request = request
        self._adapter = adapter
        self.signals = InferenceSignals()

    def _outcome(
        self,
        terminal: InferenceTerminal,
        *,
        result=None,
        error: BaseException | None = None,
    ) -> InferenceOutcome:
        return InferenceOutcome(self._request, terminal, result=result, error=error)

    def run(self) -> None:
        """Emit exactly one terminal signal, including every error path."""
        try:
            self.signals.progress.emit(self._request, 0)
            if self._request.cancellation.should_cancel():
                raise InferenceCancelled("segmentation inference cancelled")
            self.signals.progress.emit(self._request, 10)
            result = self._adapter.infer(
                self._request.rgba,
                should_cancel=self._request.cancellation.should_cancel,
            )
            if self._request.cancellation.should_cancel():
                raise InferenceCancelled("segmentation inference cancelled")
            self.signals.progress.emit(self._request, 90)
            if not isinstance(result, InferenceResult):
                raise TypeError("adapter must return an InferenceResult")
            identity = result.probability.identity
            expected = (
                self._request.source,
                self._request.model,
                self._request.preprocessing_version,
                result.candidate_id,
            )
            actual = (
                identity.source,
                identity.model,
                identity.preprocessing_version,
                identity.candidate_id,
            )
            if result.candidate_id != "primary" or actual != expected:
                raise ValueError("adapter result identity does not match inference request")
            outcome = self._outcome(InferenceTerminal.SUCCESS, result=result)
            signal = self.signals.succeeded
        except InferenceCancelled:
            outcome = self._outcome(InferenceTerminal.CANCELLED)
            signal = self.signals.cancelled
        except NoClearSubject:
            if self._request.cancellation.should_cancel():
                outcome = self._outcome(InferenceTerminal.CANCELLED)
                signal = self.signals.cancelled
            else:
                outcome = self._outcome(InferenceTerminal.NO_SUBJECT)
                signal = self.signals.no_subject
        except ModelAssetError as exc:
            if self._request.cancellation.should_cancel():
                outcome = self._outcome(InferenceTerminal.CANCELLED)
                signal = self.signals.cancelled
            else:
                # This is an expected fail-closed local installation state, not
                # an inference crash. Keep it distinct so rendering stays unmasked.
                outcome = self._outcome(InferenceTerminal.FAILED, error=exc)
                signal = self.signals.model_unavailable
        except Exception as exc:
            if self._request.cancellation.should_cancel():
                outcome = self._outcome(InferenceTerminal.CANCELLED)
                signal = self.signals.cancelled
            else:
                _LOG.exception("Smart Mask inference failed")
                outcome = self._outcome(InferenceTerminal.FAILED, error=exc)
                signal = self.signals.failed
        if outcome.terminal is InferenceTerminal.SUCCESS:
            self.signals.progress.emit(self._request, 100)
        signal.emit(outcome)
