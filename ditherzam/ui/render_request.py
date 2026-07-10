"""Immutable, Qt-free description of one preview render.

A ``RenderRequest`` is a frozen snapshot of everything a render worker needs
to know about *what* to render and *how urgently* -- settings, source
identity, target/logical geometry, and the color/effect context -- taken once
at schedule time. Workers must render exactly this snapshot and never reread
mutable UI state (the pipeline's live ``color_engine``/``effect_stack``, the
panel's current control values, etc.) mid-render.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ..render import RenderSettings


class RenderKind(Enum):
    """What triggered this render, in ascending scope/urgency."""
    DRAG = auto()     # debounce tick mid-drag: fast capped proxy
    SETTLE = auto()   # settle tick once idle: exact settled render
    ZOOM = auto()     # settled zoom crossing a quality bucket (task 2.3)
    FULL = auto()     # explicit Full Quality Preview action (task 2.3)


# Priority within one generation: DRAG < {SETTLE, ZOOM} < FULL. Ties (equal
# priority, e.g. two DRAG requests in one drag burst) are broken by recency in
# supersedes() below -- freshest state wins.
_PRIORITY: dict[RenderKind, int] = {
    RenderKind.DRAG: 0,
    RenderKind.SETTLE: 1,
    RenderKind.ZOOM: 1,
    RenderKind.FULL: 2,
}


@dataclass(frozen=True, eq=False)
class RenderRequest:
    """One immutable render ask.

    ``eq=False`` keeps identity-based equality/hash: RenderSettings and the
    snapshotted color/effect objects aren't meaningfully comparable by value,
    and numpy-array-bearing fields would otherwise break dataclass-generated
    ``__eq__``.
    """

    generation: int
    kind: RenderKind
    settings: RenderSettings
    source_id: int                 # id(base_gray) at snapshot time -- identity only
    target_max_side: int           # longest-side cap for this render
    logical_size: tuple[int, int]  # (w, h) of the full source, for display geometry
    color_engine: object = None    # pipeline.color_engine snapshot at request time
    effect_stack: object = None    # pipeline.effect_stack snapshot at request time

    @property
    def mode(self) -> str:
        """Worker render mode: capped proxy for drag, exact for everything else."""
        return "proxy" if self.kind is RenderKind.DRAG else "full"


def supersedes(new: RenderRequest, old: RenderRequest) -> bool:
    """True if ``new`` should replace ``old`` as the pending trailing request.

    Newest generation always wins outright, regardless of kind. Within one
    generation, priority order is DRAG < {SETTLE, ZOOM} < FULL; an
    equal-priority request replaces an older same-tier one so a burst of
    same-kind requests keeps only the freshest state.
    """
    if new.generation != old.generation:
        return new.generation > old.generation
    return _PRIORITY[new.kind] >= _PRIORITY[old.kind]
