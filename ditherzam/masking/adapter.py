"""Qt-free segmentation boundary and stable terminal vocabulary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from ditherzam.masking.contracts import ProbabilityMap


class InferenceCancelled(Exception):
    """Inference stopped at an adapter safe boundary."""


class NoClearSubject(Exception):
    """The model output did not distinguish a foreground subject."""


CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class InferenceResult:
    candidate_id: str
    probability: ProbabilityMap

    def __post_init__(self) -> None:
        if self.candidate_id != "primary":
            raise ValueError("v1 segmentation candidate_id must be 'primary'")
        if not isinstance(self.probability, ProbabilityMap):
            raise TypeError("probability must be a ProbabilityMap")

    @property
    def confidence(self) -> np.ndarray:
        return self.probability.values


class SegmentationAdapter(Protocol):
    def infer(self, rgba_u8: np.ndarray, *, should_cancel: CancelCheck | None = None) -> InferenceResult: ...
