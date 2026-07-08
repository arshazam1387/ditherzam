from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DitherEntry:
    name: str
    category: str
    dims: int
    param_sliders: tuple[str, ...]
    func: Callable
    param_func: Callable | None = None
    supports_levels: bool = False


class DitherRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, DitherEntry] = {}

    def register(self, name, category, dims=2, param_sliders=(), param_func=None,
                 supports_levels=False):
        def decorator(func):
            self._entries[name] = DitherEntry(
                name=name, category=category, dims=dims,
                param_sliders=tuple(param_sliders), func=func, param_func=param_func,
                supports_levels=supports_levels,
            )
            return func
        return decorator

    def get_entry(self, name: str) -> DitherEntry | None:
        return self._entries.get(name)

    def list_dithers(self) -> list[str]:
        return list(self._entries.keys())

    def by_category(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for e in self._entries.values():
            out.setdefault(e.category, []).append(e.name)
        return out
