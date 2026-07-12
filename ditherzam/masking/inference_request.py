"""Immutable work and terminal values for asynchronous mask inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Event

import numpy as np

from ditherzam.masking.adapter import InferenceResult
from ditherzam.masking.contracts import ModelIdentity, SourceIdentity, source_identity, validate_rgba_u8


class CancellationToken:
    """A small, thread-safe, one-way advisory cancellation flag."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def should_cancel(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True, eq=False)
class InferenceRequest:
    """A complete worker snapshot; no editor or widget state is read later."""

    source: SourceIdentity
    model: ModelIdentity
    preprocessing_version: str
    rgba: np.ndarray
    generation: int = 0
    cancellation: CancellationToken = field(default_factory=CancellationToken, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be a SourceIdentity")
        if not isinstance(self.model, ModelIdentity):
            raise TypeError("model must be a ModelIdentity")
        if not isinstance(self.preprocessing_version, str) or not self.preprocessing_version.strip():
            raise ValueError("preprocessing_version must be a non-empty str")
        if not isinstance(self.generation, int) or isinstance(self.generation, bool) or self.generation < 0:
            raise ValueError("generation must be a non-negative int")
        if not isinstance(self.cancellation, CancellationToken):
            raise TypeError("cancellation must be a CancellationToken")
        array = validate_rgba_u8(self.rgba)
        snapshot = np.array(array, dtype=np.uint8, order="C", copy=True)
        snapshot.flags.writeable = False
        if source_identity(snapshot) != self.source:
            raise ValueError("source identity must exactly match the owned rgba snapshot")
        object.__setattr__(self, "rgba", snapshot)


class InferenceTerminal(Enum):
    SUCCESS = "success"
    NO_SUBJECT = "no-subject"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class InferenceOutcome:
    """Exactly one terminal notification produced by an inference worker."""

    request: InferenceRequest
    terminal: InferenceTerminal
    result: InferenceResult | None = None
    error: BaseException | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        if not isinstance(self.terminal, InferenceTerminal):
            raise TypeError("terminal must be an InferenceTerminal")
        if self.terminal is InferenceTerminal.SUCCESS:
            if not isinstance(self.result, InferenceResult) or self.error is not None:
                raise ValueError("success requires only an InferenceResult")
        elif self.result is not None:
            raise ValueError("non-success outcomes cannot carry a result")
        if self.terminal is InferenceTerminal.FAILED:
            if not isinstance(self.error, BaseException):
                raise ValueError("failure requires an exception")
        elif self.error is not None:
            raise ValueError("only failure outcomes can carry an exception")
