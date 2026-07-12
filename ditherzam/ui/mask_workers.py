"""Resilient Qt worker for one immutable Smart Mask inference request."""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

from ditherzam.masking.adapter import InferenceCancelled, NoClearSubject, SegmentationAdapter
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
            if self._request.cancellation.should_cancel():
                raise InferenceCancelled("segmentation inference cancelled")
            result = self._adapter.infer(
                self._request.rgba,
                should_cancel=self._request.cancellation.should_cancel,
            )
            if self._request.cancellation.should_cancel():
                raise InferenceCancelled("segmentation inference cancelled")
        except InferenceCancelled:
            self.signals.cancelled.emit(self._outcome(InferenceTerminal.CANCELLED))
            return
        except NoClearSubject:
            self.signals.no_subject.emit(self._outcome(InferenceTerminal.NO_SUBJECT))
            return
        except ModelAssetError as exc:
            # This is an expected fail-closed local installation state, not an
            # inference crash.  Keep it distinct so rendering remains unmasked.
            self.signals.model_unavailable.emit(
                self._outcome(InferenceTerminal.FAILED, error=exc)
            )
            return
        except Exception as exc:
            _LOG.exception("Smart Mask inference failed")
            self.signals.failed.emit(self._outcome(InferenceTerminal.FAILED, error=exc))
            return
        self.signals.succeeded.emit(
            self._outcome(InferenceTerminal.SUCCESS, result=result)
        )
