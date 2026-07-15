from __future__ import annotations

from dataclasses import dataclass, replace

from ..render import RenderSettings

_INT_FIELDS = frozenset({"scale"})


def ease(t: float, kind: str) -> float:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else float(t)
    if kind == "linear":
        return t
    if kind == "ease-in":
        return t * t
    if kind == "ease-out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if kind == "ease-in-out":
        if t < 0.5:
            return 2.0 * t * t
        return 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0
    raise ValueError(f"Unknown easing kind: {kind!r}")


@dataclass
class Keyframe:
    frame: int
    field: str
    value: float
    kind: str = "linear"


class Timeline:
    def __init__(self, length: int) -> None:
        self.length = int(length)
        self._keys: dict[str, list[Keyframe]] = {}

    def add(self, kf: Keyframe) -> None:
        lst = self._keys.setdefault(kf.field, [])
        for i, existing in enumerate(lst):
            if existing.frame == kf.frame:
                lst[i] = kf
                break
        else:
            lst.append(kf)
        lst.sort(key=lambda k: k.frame)

    def fields(self) -> list[str]:
        return list(self._keys.keys())

    def value_at(self, field: str, frame: int) -> float:
        keys = self._keys.get(field)
        if not keys:
            raise KeyError(field)
        if frame <= keys[0].frame:
            return float(keys[0].value)
        if frame >= keys[-1].frame:
            return float(keys[-1].value)
        for i in range(len(keys) - 1):
            a, b = keys[i], keys[i + 1]
            if a.frame <= frame <= b.frame:
                span = b.frame - a.frame
                if span == 0:
                    return float(b.value)
                t = (frame - a.frame) / span
                e = ease(t, b.kind)
                return float(a.value + (b.value - a.value) * e)
        return float(keys[-1].value)

    def settings_at(self, base: RenderSettings, frame: int) -> RenderSettings:
        updates: dict = {}
        for fld in self._keys:
            v = self.value_at(fld, frame)
            if fld in _INT_FIELDS:
                v = int(round(v))
            updates[fld] = v
        return replace(base, **updates)
