"""Latest-wins scheduling dedicated to offline subject inference."""
from __future__ import annotations

from dataclasses import replace
from threading import Lock

from ditherzam.masking.contracts import SourceIdentity
from ditherzam.masking.inference_request import CancellationToken, InferenceOutcome, InferenceRequest


class InferenceScheduler:
    """Keep at most one active request and one complete newest trailing request."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._generation = 0
        self._active: InferenceRequest | None = None
        self._pending: InferenceRequest | None = None
        self._wanted_source: SourceIdentity | None = None
        self._wanted_model_hash: str | None = None

    def _stamp(self, request: InferenceRequest) -> InferenceRequest:
        self._generation += 1
        # A submitted value may safely be reused by a caller.  Cancellation is
        # lifecycle state, so each scheduled run receives its own one-way token.
        return replace(request, generation=self._generation, cancellation=CancellationToken())

    def request(self, request: InferenceRequest) -> InferenceRequest | None:
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        with self._lock:
            stamped = self._stamp(request)
            self._wanted_source = stamped.source
            self._wanted_model_hash = stamped.model.model_hash
            if self._active is None:
                self._active = stamped
                return stamped
            self._pending = stamped
            self._active.cancellation.cancel()
            return None

    def is_current(self, request: InferenceRequest) -> bool:
        if not isinstance(request, InferenceRequest):
            return False
        with self._lock:
            return self._is_current_unlocked(request)

    def _is_current_unlocked(self, request: InferenceRequest) -> bool:
        return (
            self._active is request
            and request.generation == self._generation
            and request.source == self._wanted_source
            and request.model.model_hash == self._wanted_model_hash
        )

    def should_cancel(self, request: InferenceRequest) -> bool:
        if not isinstance(request, InferenceRequest):
            return True
        with self._lock:
            return request.cancellation.should_cancel() or not self._is_current_unlocked(request)

    def invalidate_source(self, source: SourceIdentity | None = None) -> None:
        """Invalidate publication for the active source without starting new work."""
        if source is not None and not isinstance(source, SourceIdentity):
            raise TypeError("source must be a SourceIdentity or None")
        with self._lock:
            self._generation += 1
            self._wanted_source = source
            self._wanted_model_hash = None
            self._pending = None
            if self._active is not None:
                self._active.cancellation.cancel()

    def on_terminal(self, terminal: InferenceOutcome | InferenceRequest) -> InferenceRequest | None:
        """Release one active request and promote its newest trailing request once.

        Accepting the request directly keeps scheduler recovery independent of the
        worker's terminal kind; workers may instead pass the richer outcome value.
        """
        request = terminal.request if isinstance(terminal, InferenceOutcome) else terminal
        if not isinstance(request, InferenceRequest):
            raise TypeError("terminal must be an InferenceOutcome or InferenceRequest")
        with self._lock:
            # Object identity is intentional: only the exact launched snapshot may
            # release the active slot. Once released, duplicate and out-of-order
            # notifications necessarily fail this same guard without retained history.
            if self._active is not request:
                return None
            self._active = None
            if self._pending is None:
                return None
            next_request = self._pending
            self._pending = None
            self._active = next_request
            return next_request
