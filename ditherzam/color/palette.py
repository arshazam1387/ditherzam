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
    category: str = ""

    @classmethod
    def from_list(cls, name: str, rgb_list, category: str = "") -> "Palette":
        arr = np.asarray(rgb_list, dtype=np.float32).reshape(-1, 3)
        return cls(name=name, colors=arr, category=category)

    def to_yaml(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"name": self.name}
        if self.category:
            data["category"] = self.category
        data["colors"] = [[int(round(c)) for c in row] for row in self.colors.tolist()]
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Palette":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Palette file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = data.get("name", path.stem)
        return cls.from_list(name, data["colors"], category=data.get("category", ""))

    def shuffle(self, locked, rng) -> "Palette":
        """Return a copy where every swatch not in ``locked`` is randomized."""
        locked = set(locked)
        new = self.colors.copy()
        for i in range(new.shape[0]):
            if i not in locked:
                new[i] = rng.integers(0, 256, size=3).astype(np.float32)
        return Palette(name=self.name, colors=new, category=self.category)


def _median_cut(pixels: np.ndarray, depth: int) -> list[np.ndarray]:
    """Recursively split ``pixels`` (N,3 float) into 2**depth buckets."""
    if depth == 0 or pixels.shape[0] <= 1:
        return [pixels]
    ranges = pixels.max(axis=0) - pixels.min(axis=0)
    axis = int(np.argmax(ranges))
    order = np.argsort(pixels[:, axis], kind="stable")
    pixels = pixels[order]
    mid = pixels.shape[0] // 2
    left = _median_cut(pixels[:mid], depth - 1)
    right = _median_cut(pixels[mid:], depth - 1)
    return left + right


def extract_palette(rgb_u8: np.ndarray, k: int = 16, name: str = "source", category: str = "user") -> "Palette":
    """Median-cut palette extraction. Returns exactly ``k`` colors."""
    k = max(1, int(k))
    pixels = np.asarray(rgb_u8, dtype=np.float32).reshape(-1, 3)
    depth = 0
    while (1 << depth) < k:
        depth += 1
    buckets = [b for b in _median_cut(pixels, depth) if b.shape[0] > 0]
    means = [b.mean(axis=0) for b in buckets]
    # normalize to exactly k rows (pad by repeating the last, or trim)
    if len(means) >= k:
        means = means[:k]
    else:
        means = means + [means[-1]] * (k - len(means))
    colors = np.asarray(means, dtype=np.float32).reshape(k, 3)
    return Palette(name=name, colors=colors, category=category)


def source_palette(rgb_u8: np.ndarray, completeness: float = 1.0,
                   name: str = "source", category: str = "user") -> "Palette":
    """Extract a 'source' palette; completeness in [0,1] maps to k in [2,256]."""
    c = min(1.0, max(0.0, float(completeness)))
    k = int(round(2 + c * (256 - 2)))
    return extract_palette(rgb_u8, k=k, name=name, category=category)


def generate_palette(rgb_u8: np.ndarray, unit: str, value: int,
                     name: str = "from image", category: str = "user") -> "Palette":
    """Extract a palette from an image. ``unit`` is 'k' (exact colors) or 'pct'."""
    if unit == "k":
        return extract_palette(rgb_u8, k=max(1, int(value)), name=name, category=category)
    if unit == "pct":
        return source_palette(rgb_u8, completeness=float(value) / 100.0, name=name, category=category)
    raise ValueError(f"unknown unit: {unit!r}")


def builtin_palettes() -> dict[str, "Palette"]:
    """Load every bundled palette from ``ditherzam/color/builtin/*.yaml``."""
    directory = Path(__file__).parent / "builtin"
    out: dict[str, Palette] = {}
    for f in sorted(directory.glob("*.yaml")):
        out[f.stem] = Palette.load(f)
    return out
