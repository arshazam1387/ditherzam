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
    coverages: np.ndarray | None = None

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


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert 0..255 sRGB rows to OKLab for perceptual distance checks."""
    srgb = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    r, g, b = linear.T
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.column_stack((
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ))


def _sample_pixels(rgb_u8: np.ndarray, limit: int = 65_536) -> np.ndarray:
    pixels = np.asarray(rgb_u8, dtype=np.uint8).reshape(-1, 3)
    if pixels.shape[0] <= limit:
        return pixels
    # Evenly spaced deterministic sampling keeps extraction responsive and
    # includes the whole image rather than privileging its top-left region.
    indices = np.linspace(0, pixels.shape[0] - 1, limit, dtype=np.int64)
    return pixels[indices]


def distinct_palette(
    rgb_u8: np.ndarray,
    k: int = 5,
    min_coverage: float = 0.005,
    diversity: float = 1.0,
    name: str = "from image",
    category: str = "user",
) -> "Palette":
    """Choose up to ``k`` perceptually separated, sufficiently common colors.

    ``min_coverage`` is a 0..1 share of sampled image pixels. Unlike median-cut,
    this extractor intentionally returns fewer than ``k`` colors when too few
    candidates meet the threshold.
    """
    requested = max(1, int(k))
    minimum = min(1.0, max(0.0, float(min_coverage)))
    diversity_weight = float(diversity)
    if diversity_weight > 1.0:
        diversity_weight /= 100.0
    diversity_weight = min(1.0, max(0.0, diversity_weight))

    sample_rgb = _sample_pixels(rgb_u8)
    if sample_rgb.shape[0] == 0:
        raise ValueError("cannot extract a palette from an empty image")
    unique_rgb, counts = np.unique(sample_rgb, axis=0, return_counts=True)
    unique_lab = _srgb_to_oklab(unique_rgb)
    candidate_count = min(unique_rgb.shape[0], max(16, requested * 4), 64)

    # Deterministic weighted farthest-point initialization. Dominant colors seed
    # the clusters, while perceptually remote colors remain eligible.
    chosen = [int(np.argmax(counts))]
    nearest_d2 = np.sum((unique_lab - unique_lab[chosen[0]]) ** 2, axis=1)
    while len(chosen) < candidate_count:
        score = nearest_d2 * np.sqrt(counts.astype(np.float64))
        score[chosen] = -1.0
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        d2 = np.sum((unique_lab - unique_lab[nxt]) ** 2, axis=1)
        nearest_d2 = np.minimum(nearest_d2, d2)

    sample_lab = _srgb_to_oklab(sample_rgb)
    centers = unique_lab[chosen].copy()
    labels = np.zeros(sample_lab.shape[0], dtype=np.int32)
    for _ in range(12):
        d2 = np.sum(
            (sample_lab[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        new_labels = np.argmin(d2, axis=1).astype(np.int32)
        new_centers = centers.copy()
        for i in range(candidate_count):
            members = sample_lab[new_labels == i]
            if members.size:
                new_centers[i] = members.mean(axis=0)
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers

    cluster_counts = np.bincount(labels, minlength=candidate_count)
    coverage = cluster_counts.astype(np.float64) / sample_rgb.shape[0]
    candidate_rgb = np.empty((candidate_count, 3), dtype=np.float64)
    for i in range(candidate_count):
        members = sample_rgb[labels == i]
        candidate_rgb[i] = members.mean(axis=0) if members.size else unique_rgb[chosen[i]]

    eligible = np.flatnonzero((cluster_counts > 0) & (coverage >= minimum))
    if eligible.size == 0:
        return Palette(
            name=name,
            colors=np.empty((0, 3), dtype=np.float32),
            category=category,
            coverages=np.empty((0,), dtype=np.float32),
        )

    eligible_lab = centers[eligible]
    eligible_coverage = coverage[eligible]
    selected_local = [int(np.argmax(eligible_coverage))]
    nearest = np.linalg.norm(
        eligible_lab - eligible_lab[selected_local[0]], axis=1)
    while len(selected_local) < min(requested, eligible.size):
        distance_score = nearest / max(float(nearest.max()), 1e-12)
        coverage_score = eligible_coverage / max(float(eligible_coverage.max()), 1e-12)
        score = (
            diversity_weight * distance_score
            + (1.0 - diversity_weight) * coverage_score
        )
        score[selected_local] = -1.0
        nxt = int(np.argmax(score))
        selected_local.append(nxt)
        nearest = np.minimum(
            nearest,
            np.linalg.norm(eligible_lab - eligible_lab[nxt], axis=1),
        )

    selected = eligible[np.asarray(selected_local, dtype=np.int64)]
    return Palette(
        name=name,
        colors=candidate_rgb[selected].astype(np.float32),
        category=category,
        coverages=coverage[selected].astype(np.float32),
    )


def generate_palette(rgb_u8: np.ndarray, unit: str, value: int,
                     name: str = "from image", category: str = "user",
                     algorithm: str = "balanced", min_coverage: float = 0.0,
                     diversity: float = 100.0) -> "Palette":
    """Extract a palette from an image. ``unit`` is 'k' (exact colors) or 'pct'."""
    if algorithm == "distinct":
        k = max(1, int(value))
        if unit == "pct":
            k = int(round(2 + (float(value) / 100.0) * (256 - 2)))
        elif unit != "k":
            raise ValueError(f"unknown unit: {unit!r}")
        return distinct_palette(
            rgb_u8, k=k, min_coverage=min_coverage, diversity=diversity,
            name=name, category=category,
        )
    if algorithm != "balanced":
        raise ValueError(f"unknown palette algorithm: {algorithm!r}")
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
