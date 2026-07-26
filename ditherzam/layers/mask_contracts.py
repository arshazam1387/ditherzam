"""Qt-free executable contracts for manual raster-layer masking."""

from dataclasses import dataclass
from enum import Enum
import math

MIB = 1024 * 1024
HISTORY_MAX_ENTRIES = 64
HISTORY_MAX_RETAINED_BYTES = 128 * MIB
PROXY_TYPICAL_ROI_P95_MS = 33
PROXY_LARGE_ROI_P95_MS = 50
PREVIEW_COMPOSITE_720_P95_MS = 50
PREVIEW_COMPOSITE_1440_P95_MS = 120
EXACT_MASK_COMPOSITE_1080P_TARGET_P95_MS = 250
EXACT_MASK_COMPOSITE_4K_TARGET_P95_MS = 750
STROKE_INCREMENTAL_1080P_BYTES = 6 * MIB
STROKE_INCREMENTAL_4K_BYTES = 18 * MIB
THUMBNAIL_COLD_MS = 10
THUMBNAIL_WARM_P95_MS = 2
LOOK_RENDERS_PER_MASK_STAMP = 0
EXACT_RENDERS_PER_COMPLETED_STROKE = 1
HISTORY_OPERATIONS = frozenset({
    "layer_add", "layer_remove", "layer_reorder", "layer_properties",
    "raster_mask_replace", "raster_mask_stroke", "transform_confirm",
})
NON_HISTORY_OPERATIONS = frozenset({
    "selection", "transform_cancel", "look_slider",
})
NATIVE_PROJECT_PERSISTENCE_EPIC = "separate"
# Metrics become measured gates only when their implementation seam exists.
MEASUREMENT_OWNER = {
    "mask_composite": "MT-03",
    "cache_and_thumbnail": "MT-04",
    "history_retained_bytes": "MT-06",
    "dirty_proxy_and_stroke_peak": "MT-10",
}


class MaskRenderStage(Enum):
    COMPLETE_LOOK = 1
    SOURCE_ALPHA = 2
    RASTER_MASK = 3
    LAYER_COMPOSITE = 4


MASK_RENDER_ORDER = tuple(MaskRenderStage)
SUPPORTED_MASK_TRANSFORMS = frozenset({"x", "y", "scale_x", "scale_y"})
UNSUPPORTED_MASK_TRANSFORMS = frozenset(
    {"rotation_degrees", "flip_x", "flip_y"})


def _integer(value, lower, upper, label):
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{label} must be an integer within {lower}..{upper}")
    return value


def effective_mask(mask: int, density: int) -> int:
    mask = _integer(mask, 0, 255, "mask")
    density = _integer(density, 0, 100, "density")
    return 255 - ((density * (255 - mask) + 50) // 100)


def masked_alpha(source_alpha: int, mask: int, density: int) -> int:
    source_alpha = _integer(source_alpha, 0, 255, "source_alpha")
    return (source_alpha * effective_mask(mask, density) + 127) // 255


def source_mask_bytes(width: int, height: int) -> int:
    return (
        _integer(width, 1, 2**31 - 1, "width")
        * _integer(height, 1, 2**31 - 1, "height")
    )


def document_to_mask_point(
    document_x: float,
    document_y: float,
    *,
    layer_x: float,
    layer_y: float,
    scale_x: float,
    scale_y: float,
    rotation_degrees: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> tuple[float, float]:
    """Map document coordinates into source mask coordinates.

    Rotation and flips are rejected because document rendering does not yet
    implement them and therefore cannot certify matching brush geometry.
    """
    values = (document_x, document_y, layer_x, layer_y, scale_x, scale_y)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for value in values):
        raise ValueError("coordinates and scales must be numeric")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("coordinates and scales must be finite")
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("scales must be positive")
    if (
        isinstance(rotation_degrees, bool)
        or not isinstance(rotation_degrees, (int, float))
        or not math.isfinite(float(rotation_degrees))
    ):
        raise ValueError("rotation must be finite")
    if type(flip_x) is not bool or type(flip_y) is not bool:
        raise ValueError("flip values must be bools")
    if rotation_degrees != 0.0 or flip_x or flip_y:
        raise ValueError("rotation and flip mask mapping is unsupported")
    return (
        (float(document_x) - float(layer_x)) / float(scale_x),
        (float(document_y) - float(layer_y)) / float(scale_y),
    )


@dataclass(frozen=True)
class LookInputsKey:
    """Typed Look-only inputs; raster-mask state has no admission field."""

    smart_mask_signature: object
    render_context_signature: object

    def __post_init__(self):
        hash(self.smart_mask_signature)
        hash(self.render_context_signature)


@dataclass(frozen=True)
class RenderedLookKey:
    """Expensive Look identity; raster-mask state is intentionally absent."""

    source_identity: object
    look_signature: object
    target: object
    look_inputs: LookInputsKey

    def __post_init__(self):
        if not isinstance(self.look_inputs, LookInputsKey):
            raise ValueError("look_inputs must be a LookInputsKey")
        hash(self.source_identity)
        hash(self.look_signature)
        hash(self.target)
        hash(self.look_inputs)


@dataclass(frozen=True)
class MaskPublicationKey:
    """Cheap alpha/composite/thumbnail publication identity."""

    layer_id: str
    source_identity: object
    mask_identity: object
    mask_revision: int
    mask_enabled: bool
    mask_density: int
    transform: object
    document_revision: int
    target: object
    worker_generation: int

    def __post_init__(self):
        if not isinstance(self.layer_id, str) or not self.layer_id.strip():
            raise ValueError("layer_id must be a non-empty string")
        for value, label in (
            (self.mask_revision, "mask_revision"),
            (self.document_revision, "document_revision"),
            (self.worker_generation, "worker_generation"),
        ):
            _integer(value, 0, 2**63 - 1, label)
        if type(self.mask_enabled) is not bool:
            raise ValueError("mask_enabled must be bool")
        _integer(self.mask_density, 0, 100, "mask_density")
        for value, label in (
            (self.source_identity, "source_identity"),
            (self.mask_identity, "mask_identity"),
            (self.transform, "transform"),
            (self.target, "target"),
        ):
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError(f"{label} must be hashable") from exc


class MaskObjectIdentity:
    """Hashable strong object-identity token for immutable raster masks."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return id(self.value)

    def __eq__(self, other):
        if not isinstance(other, MaskObjectIdentity):
            return NotImplemented
        return self.value is other.value
