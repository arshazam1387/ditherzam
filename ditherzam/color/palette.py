from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Palette:
    """An RGB palette: ``colors`` is float32[K, 3] in the 0..255 range."""

    name: str
    colors: np.ndarray

    @classmethod
    def from_list(cls, name: str, rgb_list) -> "Palette":
        arr = np.asarray(rgb_list, dtype=np.float32).reshape(-1, 3)
        return cls(name=name, colors=arr)

    def to_yaml(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "colors": [[int(round(c)) for c in row] for row in self.colors.tolist()],
        }
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Palette":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Palette file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = data.get("name", path.stem)
        return cls.from_list(name, data["colors"])


def builtin_palettes() -> dict[str, "Palette"]:
    """Load every bundled palette from ``ditherzam/color/builtin/*.yaml``."""
    directory = Path(__file__).parent / "builtin"
    out: dict[str, Palette] = {}
    for f in sorted(directory.glob("*.yaml")):
        out[f.stem] = Palette.load(f)
    return out
