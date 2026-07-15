from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ThemeData:
    name: str
    stylesheet: str
    glow_color: str
    idle_gif: str | None = None
    labels: dict = field(default_factory=dict)


def find_themes(root) -> list[str]:
    """Return the names of every subfolder of ``root`` that has a theme.yaml."""
    root = Path(root)
    if not root.is_dir():
        return []
    return [
        sub.name
        for sub in sorted(root.iterdir())
        if sub.is_dir() and (sub / "theme.yaml").is_file()
    ]


def load_theme(root, name) -> ThemeData:
    """Load ``<root>/<name>/theme.yaml`` into a ThemeData.

    Resolves the idle GIF to an absolute-ish path (theme-local override, else a
    sibling ``idle.gif``, else None).
    """
    root = Path(root)
    path = root / name / "theme.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Theme not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    idle = data.get("idle_gif")
    gif_path: str | None = None
    if idle:
        gif_path = str(root / name / idle)
    elif (root / name / "idle.gif").is_file():
        gif_path = str(root / name / "idle.gif")
    return ThemeData(
        name=name,
        stylesheet=data.get("app_stylesheet", ""),
        glow_color=data.get("glow_color", "#5e89ed"),
        idle_gif=gif_path,
        labels=data.get("labels", {}) or {},
    )
