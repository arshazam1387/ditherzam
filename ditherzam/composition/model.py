from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..animation.timeline import Timeline
from .look import Look

_TRANSITION_KINDS = frozenset({
    "crossfade", "dither-dissolve", "spatial-wipe", "param-morph",
})


def _require_frame(value, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an int frame")


@dataclass(frozen=True)
class LookClip:
    look: Look
    start: int
    end: int
    automation: Timeline | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.look, Look):
            raise ValueError("look must be a Look")
        _require_frame(self.start, "start")
        _require_frame(self.end, "end")
        if self.start < 0 or self.start >= self.end:
            raise ValueError("frames require 0 <= start < end")
        if self.automation is not None and not isinstance(self.automation, Timeline):
            raise ValueError("automation must be a Timeline or None")


@dataclass(frozen=True)
class TransitionSpec:
    kind: str
    duration: int
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dict")
        object.__setattr__(self, "params", copy.deepcopy(self.params))


@dataclass(frozen=True)
class Resolution:
    kind: str
    clip: LookClip
    clip_b: LookClip | None = None
    t: float = 0.0
    spec: TransitionSpec | None = None


class Composition:
    def __init__(self, length: int):
        _require_frame(length, "length")
        if length <= 0:
            raise ValueError("length must be positive")
        self.length = length
        self.clips: list[LookClip] = []
        self.transitions: dict[int, TransitionSpec] = {}

    def add_clip(self, clip: LookClip) -> None:
        if not isinstance(clip, LookClip):
            raise ValueError("clip must be a LookClip")
        self.clips.append(clip)
        self.clips.sort(key=lambda value: value.start)

    def set_transition(self, later_clip_index: int, spec: TransitionSpec) -> None:
        if isinstance(later_clip_index, bool) or not isinstance(later_clip_index, int):
            raise ValueError("later_clip_index must be an int")
        if not 1 <= later_clip_index < len(self.clips):
            raise ValueError("later_clip_index is out of range")
        if not isinstance(spec, TransitionSpec):
            raise ValueError("spec must be a TransitionSpec")
        if spec.kind not in _TRANSITION_KINDS:
            raise ValueError(f"unregistered transition kind: {spec.kind!r}")
        _require_frame(spec.duration, "duration")
        if spec.duration <= 0:
            raise ValueError("duration must be positive")
        before = self.clips[later_clip_index - 1]
        after = self.clips[later_clip_index]
        if spec.duration > before.end - before.start or spec.duration > after.end - after.start:
            raise ValueError("duration exceeds adjacent clip length")
        self.transitions[later_clip_index] = spec

    def validate(self) -> None:
        expected = 0
        for clip in self.clips:
            if clip.start != expected:
                raise ValueError("clips must be contiguous")
            expected = clip.end
        if expected != self.length:
            raise ValueError("clips must be contiguous and cover composition length")
        for index in self.transitions:
            if not 1 <= index < len(self.clips):
                raise ValueError("transition index is invalid")

    def resolve(self, idx: int) -> Resolution:
        _require_frame(idx, "idx")
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        for index, clip in enumerate(self.clips):
            if clip.start <= idx < clip.end:
                spec = self.transitions.get(index)
                if spec is not None and idx < clip.start + spec.duration:
                    return Resolution(
                        "transition",
                        self.clips[index - 1],
                        clip_b=clip,
                        t=(idx - clip.start) / spec.duration,
                        spec=spec,
                    )
                return Resolution("hold", clip)
        raise IndexError(idx)
