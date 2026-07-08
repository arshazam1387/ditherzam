from __future__ import annotations

import os
from pathlib import Path

from .palette import Palette, builtin_palettes


def _default_user_dir() -> Path:
    base = os.environ.get("APPDATA")
    if base:
        return Path(base) / "ditherzam" / "palettes"
    return Path.home() / ".config" / "ditherzam" / "palettes"


class PaletteStore:
    """User-owned palette repository. User YAML files shadow read-only builtins."""

    def __init__(self, user_dir: Path | None = None) -> None:
        self.user_dir = Path(user_dir) if user_dir is not None else _default_user_dir()

    # -- discovery ------------------------------------------------------------
    def _user_path(self, name: str) -> Path:
        return self.user_dir / f"{name}.yaml"

    def _user_names(self) -> list[str]:
        if not self.user_dir.is_dir():
            return []
        return [p.stem for p in self.user_dir.glob("*.yaml")]

    def is_user(self, name: str) -> bool:
        return self._user_path(name).is_file()

    def is_builtin(self, name: str) -> bool:
        return name in builtin_palettes()

    def list(self) -> list[str]:
        return sorted(set(builtin_palettes().keys()) | set(self._user_names()))

    # -- access ---------------------------------------------------------------
    def get(self, name: str) -> Palette:
        if self.is_user(name):
            p = Palette.load(self._user_path(name))
        else:
            builtins = builtin_palettes()
            if name not in builtins:
                raise KeyError(name)
            p = builtins[name]
        return Palette(name=p.name, colors=p.colors.copy(), category=p.category)

    def list_by_category(self) -> dict[str, list[str]]:
        cats: dict[str, list[str]] = {}
        for name in self.list():
            key = self.get(name).category or "uncategorized"
            cats.setdefault(key, []).append(name)
        ordered = sorted(k for k in cats if k != "uncategorized")
        if "uncategorized" in cats:
            ordered.append("uncategorized")
        return {k: sorted(cats[k]) for k in ordered}

    # -- mutation -------------------------------------------------------------
    def save(self, palette: Palette) -> None:
        self.user_dir.mkdir(parents=True, exist_ok=True)
        palette.to_yaml(self._user_path(palette.name))

    def delete(self, name: str) -> None:
        path = self._user_path(name)
        if path.is_file():
            path.unlink()

    def reset_to_builtin(self, name: str) -> Palette:
        if name not in builtin_palettes():
            raise KeyError(name)
        self.delete(name)
        return self.get(name)
