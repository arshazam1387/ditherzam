from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
import math

import numpy as np

from ..composition import Look
from ..masking.contracts import ProbabilityMap, SourceIdentity, source_identity


BLEND_MODES = frozenset(
    {"normal", "multiply", "screen", "overlay", "difference"}
)
_UNCHANGED = object()


def _index(value, length: int, label: str = "index") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if not 0 <= value < length:
        raise IndexError(value)
    return value


@dataclass(frozen=True)
class CanvasSpec:
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive non-bool integer")


@dataclass(frozen=True)
class LayerSource:
    gray: np.ndarray
    rgba: np.ndarray
    probability: ProbabilityMap | None = None
    source_identity: SourceIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.gray, np.ndarray) or self.gray.dtype != np.float32:
            raise ValueError("gray must be a float32 ndarray")
        if not isinstance(self.rgba, np.ndarray) or self.rgba.dtype != np.uint8:
            raise ValueError("rgba must be a uint8 ndarray")
        if self.gray.ndim != 2 or not self.gray.size:
            raise ValueError("gray must be a non-empty 2-D array")
        if (
            self.rgba.ndim != 3
            or self.rgba.shape[2] != 4
            or not self.rgba.shape[0]
            or not self.rgba.shape[1]
        ):
            raise ValueError("rgba must be a non-empty straight-RGBA array")
        if self.gray.shape != self.rgba.shape[:2]:
            raise ValueError("gray and rgba dimensions must match")
        gray = np.array(self.gray, dtype=np.float32, order="C", copy=True)
        rgba = np.array(self.rgba, dtype=np.uint8, order="C", copy=True)
        gray.flags.writeable = False
        rgba.flags.writeable = False
        identity = self.source_identity
        if identity is None:
            identity = source_identity(rgba)
        elif not isinstance(identity, SourceIdentity):
            raise ValueError("source_identity must be a SourceIdentity")
        if self.probability is not None:
            if not isinstance(self.probability, ProbabilityMap):
                raise ValueError("probability must be a ProbabilityMap")
            if self.probability.identity.source != identity:
                raise ValueError("probability source identity does not match rgba")
        object.__setattr__(self, "gray", gray)
        object.__setattr__(self, "rgba", rgba)
        object.__setattr__(self, "source_identity", identity)


class RasterLayerMask:
    """Immutable source-coordinate coverage for one or more spatial layers."""

    __slots__ = ("_pixels", "_enabled", "_density", "_revision")
    __hash__ = None

    def __init__(
        self,
        pixels: np.ndarray,
        enabled: bool = True,
        density: int = 100,
        revision: int = 0,
    ) -> None:
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8:
            raise ValueError("pixels must be a uint8 ndarray")
        if pixels.ndim != 2 or not pixels.size:
            raise ValueError("pixels must be a non-empty 2-D array")
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        if isinstance(density, bool) or not isinstance(density, int):
            raise ValueError("density must be an integer")
        if not 0 <= density <= 100:
            raise ValueError("density must be within 0..100")
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 0
        ):
            raise ValueError("revision must be a non-negative integer")
        contiguous = np.array(pixels, dtype=np.uint8, order="C", copy=True)
        immutable = np.frombuffer(
            contiguous.tobytes(order="C"), dtype=np.uint8).reshape(contiguous.shape)
        object.__setattr__(self, "_pixels", immutable)
        object.__setattr__(self, "_enabled", enabled)
        object.__setattr__(self, "_density", density)
        object.__setattr__(self, "_revision", revision)

    def __setattr__(self, _name, _value) -> None:
        raise AttributeError("RasterLayerMask is immutable")

    @property
    def pixels(self) -> np.ndarray:
        return self._pixels

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def density(self) -> int:
        return self._density

    @property
    def revision(self) -> int:
        return self._revision

    def __eq__(self, other) -> bool:
        if not isinstance(other, RasterLayerMask):
            return NotImplemented
        return (
            self.enabled == other.enabled
            and self.density == other.density
            and self.revision == other.revision
            and np.array_equal(self.pixels, other.pixels)
        )

    @classmethod
    def _reuse_pixels(
        cls, pixels: np.ndarray, enabled: bool, density: int, revision: int
    ) -> RasterLayerMask:
        value = object.__new__(cls)
        object.__setattr__(value, "_pixels", pixels)
        object.__setattr__(value, "_enabled", enabled)
        object.__setattr__(value, "_density", density)
        object.__setattr__(value, "_revision", revision)
        return value

    def evolve(
        self,
        *,
        pixels=_UNCHANGED,
        enabled=_UNCHANGED,
        density=_UNCHANGED,
    ) -> RasterLayerMask:
        """Return one validated replacement and advance its revision once."""
        if pixels is _UNCHANGED:
            pixels = self.pixels
        if enabled is _UNCHANGED:
            enabled = self.enabled
        if density is _UNCHANGED:
            density = self.density
        if pixels is self.pixels:
            if type(enabled) is not bool:
                raise ValueError("enabled must be a bool")
            if (
                isinstance(density, bool)
                or not isinstance(density, int)
                or not 0 <= density <= 100
            ):
                raise ValueError("density must be an integer within 0..100")
            return self._reuse_pixels(
                self.pixels, enabled, density, self.revision + 1)
        return RasterLayerMask(
            pixels, enabled=enabled, density=density, revision=self.revision + 1)


@dataclass(frozen=True)
class LayerTransform:
    x: float = 0.0
    y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation_degrees: float = 0.0
    flip_x: bool = False
    flip_y: bool = False

    def __post_init__(self) -> None:
        for name in ("x", "y", "scale_x", "scale_y", "rotation_degrees"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if name in ("scale_x", "scale_y") and value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if type(self.flip_x) is not bool or type(self.flip_y) is not bool:
            raise ValueError("flip values must be bools")


@dataclass(frozen=True)
class Layer:
    id: str
    name: str
    look: Look
    visible: bool = True
    opacity: int = 100
    blend_mode: str = "normal"
    source: LayerSource | None = None
    transform: LayerTransform = LayerTransform()
    mask_revision: int = 0
    raster_mask: RasterLayerMask | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("id must be a non-blank string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-blank string")
        if not isinstance(self.look, Look):
            raise ValueError("look must be a Look")
        if type(self.visible) is not bool:
            raise ValueError("visible must be a bool")
        if isinstance(self.opacity, bool) or not isinstance(self.opacity, int):
            raise ValueError("opacity must be an integer")
        if not 0 <= self.opacity <= 100:
            raise ValueError("opacity must be within 0..100")
        if self.blend_mode not in BLEND_MODES:
            raise ValueError("blend_mode is invalid")
        if self.source is not None and not isinstance(self.source, LayerSource):
            raise ValueError("source must be a LayerSource or None")
        if not isinstance(self.transform, LayerTransform):
            raise ValueError("transform must be a LayerTransform")
        if (
            isinstance(self.mask_revision, bool)
            or not isinstance(self.mask_revision, int)
            or self.mask_revision < 0
        ):
            raise ValueError("mask_revision must be a non-negative integer")
        if self.raster_mask is not None:
            if not isinstance(self.raster_mask, RasterLayerMask):
                raise ValueError("raster_mask must be a RasterLayerMask or None")
            if self.source is None:
                raise ValueError("raster_mask requires a layer source")
            if self.raster_mask.pixels.shape != self.source.gray.shape:
                raise ValueError("raster_mask dimensions must match the layer source")


@dataclass(frozen=True)
class LayerStack:
    layers: tuple[Layer, ...] = ()

    def __post_init__(self) -> None:
        try:
            owned = tuple(self.layers)
        except TypeError as exc:
            raise ValueError("layers must be an iterable of Layer values") from exc
        if any(not isinstance(layer, Layer) for layer in owned):
            raise ValueError("layers must contain only Layer values")
        ids = [layer.id for layer in owned]
        if len(ids) != len(set(ids)):
            raise ValueError("layer ids must be unique")
        object.__setattr__(self, "layers", owned)

    def add(self, layer: Layer) -> LayerStack:
        if not isinstance(layer, Layer):
            raise ValueError("layer must be a Layer")
        return LayerStack((*self.layers, layer))

    def replace(self, index: int, layer: Layer) -> LayerStack:
        index = _index(index, len(self.layers))
        if not isinstance(layer, Layer):
            raise ValueError("layer must be a Layer")
        values = list(self.layers)
        values[index] = layer
        return LayerStack(values)

    def remove(self, index: int) -> LayerStack:
        index = _index(index, len(self.layers))
        return LayerStack(self.layers[:index] + self.layers[index + 1:])

    def duplicate(
        self, index: int, new_id: str, name: str | None = None
    ) -> LayerStack:
        """Insert a copy directly above the selected layer.

        IDs are deliberately caller-supplied so the headless model never chooses
        persistence or UI identity policy.
        """
        index = _index(index, len(self.layers))
        original = self.layers[index]
        duplicate = dataclass_replace(
            original,
            id=new_id,
            name=original.name if name is None else name,
        )
        values = list(self.layers)
        values.insert(index + 1, duplicate)
        return LayerStack(values)

    def move(self, index: int, new_index: int) -> LayerStack:
        index = _index(index, len(self.layers))
        new_index = _index(new_index, len(self.layers), "new_index")
        if index == new_index:
            return self
        values = list(self.layers)
        values.insert(new_index, values.pop(index))
        return LayerStack(values)


@dataclass(frozen=True)
class LayerDocument:
    canvas: CanvasSpec
    layers: tuple[Layer, ...] = ()
    selected_ids: tuple[str, ...] = ()
    revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.canvas, CanvasSpec):
            raise ValueError("canvas must be a CanvasSpec")
        try:
            layers = tuple(self.layers)
            selected = tuple(self.selected_ids)
        except TypeError as exc:
            raise ValueError("layers and selected_ids must be iterable") from exc
        if any(not isinstance(layer, Layer) for layer in layers):
            raise ValueError("layers must contain only Layer values")
        if any(layer.source is None for layer in layers):
            raise ValueError("document layers must own sources")
        ids = [layer.id for layer in layers]
        if len(ids) != len(set(ids)):
            raise ValueError("layer ids must be unique")
        if (
            len(selected) > 1
            or any(not isinstance(value, str) or value not in ids for value in selected)
        ):
            raise ValueError("selected_ids must contain at most one live layer id")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("revision must be a non-negative integer")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "selected_ids", selected)

    def _next(self, layers, selected_ids=None) -> LayerDocument:
        return LayerDocument(
            self.canvas,
            tuple(layers),
            self.selected_ids if selected_ids is None else tuple(selected_ids),
            self.revision + 1,
        )

    def add(
        self, layer: Layer, *, index: int | None = None, select: bool = True
    ) -> LayerDocument:
        if not isinstance(layer, Layer):
            raise ValueError("layer must be a Layer")
        values = list(self.layers)
        insertion = len(values) if index is None else _index(index, len(values)) + 1
        values.insert(insertion, layer)
        return self._next(values, (layer.id,) if select else self.selected_ids)

    def replace(self, index: int, layer: Layer) -> LayerDocument:
        index = _index(index, len(self.layers))
        if not isinstance(layer, Layer):
            raise ValueError("layer must be a Layer")
        values = list(self.layers)
        old_id = values[index].id
        values[index] = layer
        selected = (layer.id,) if self.selected_ids == (old_id,) else self.selected_ids
        return self._next(values, selected)

    def remove(self, index: int) -> LayerDocument:
        index = _index(index, len(self.layers))
        values = list(self.layers)
        removed = values.pop(index)
        selected = self.selected_ids
        if selected == (removed.id,):
            selected = () if not values else (values[min(index, len(values) - 1)].id,)
        return self._next(values, selected)

    def duplicate(
        self, index: int, new_id: str, name: str | None = None
    ) -> LayerDocument:
        index = _index(index, len(self.layers))
        original = self.layers[index]
        duplicate = dataclass_replace(
            original, id=new_id, name=original.name if name is None else name)
        return self.add(duplicate, index=index, select=True)

    def move(self, index: int, new_index: int) -> LayerDocument:
        index = _index(index, len(self.layers))
        new_index = _index(new_index, len(self.layers), "new_index")
        values = list(self.layers)
        values.insert(new_index, values.pop(index))
        return self._next(values)

    def select(self, index: int | None) -> LayerDocument:
        selected = () if index is None else (self.layers[_index(index, len(self.layers))].id,)
        return self._next(self.layers, selected)
