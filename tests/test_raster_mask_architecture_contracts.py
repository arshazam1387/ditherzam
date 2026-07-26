import pytest

from ditherzam.layers.mask_contracts import (
    EXACT_MASK_COMPOSITE_1080P_TARGET_P95_MS,
    EXACT_MASK_COMPOSITE_4K_TARGET_P95_MS,
    EXACT_RENDERS_PER_COMPLETED_STROKE,
    HISTORY_MAX_ENTRIES,
    HISTORY_MAX_RETAINED_BYTES,
    HISTORY_OPERATIONS,
    LOOK_RENDERS_PER_MASK_STAMP,
    LookInputsKey,
    MASK_RENDER_ORDER,
    MEASUREMENT_OWNER,
    NATIVE_PROJECT_PERSISTENCE_EPIC,
    NON_HISTORY_OPERATIONS,
    PREVIEW_COMPOSITE_1440_P95_MS,
    PREVIEW_COMPOSITE_720_P95_MS,
    PROXY_LARGE_ROI_P95_MS,
    PROXY_TYPICAL_ROI_P95_MS,
    STROKE_INCREMENTAL_1080P_BYTES,
    STROKE_INCREMENTAL_4K_BYTES,
    THUMBNAIL_COLD_MS,
    THUMBNAIL_WARM_P95_MS,
    MaskPublicationKey,
    MaskObjectIdentity,
    MaskRenderStage,
    RenderedLookKey,
    SUPPORTED_MASK_TRANSFORMS,
    UNSUPPORTED_MASK_TRANSFORMS,
    effective_mask,
    document_to_mask_point,
    masked_alpha,
    source_mask_bytes,
)


def test_two_mask_render_order_is_frozen():
    assert MASK_RENDER_ORDER == (
        MaskRenderStage.COMPLETE_LOOK,
        MaskRenderStage.SOURCE_ALPHA,
        MaskRenderStage.RASTER_MASK,
        MaskRenderStage.LAYER_COMPOSITE,
    )


@pytest.mark.parametrize(("mask", "density", "expected"), [
    (0, 0, 255), (0, 100, 0), (255, 100, 255),
    (127, 50, 191), (254, 50, 254),
])
def test_effective_mask_uses_integer_half_up(mask, density, expected):
    assert effective_mask(mask, density) == expected


@pytest.mark.parametrize(("alpha", "mask", "density", "expected"), [
    (0, 255, 100, 0), (255, 255, 100, 255), (255, 0, 100, 0),
    (127, 127, 100, 63), (1, 128, 100, 1),
])
def test_source_alpha_is_applied_before_raster_coverage(
    alpha, mask, density, expected
):
    assert masked_alpha(alpha, mask, density) == expected


@pytest.mark.parametrize("value", [-1, 256, True, 1.5])
def test_formula_rejects_non_byte_values(value):
    with pytest.raises(ValueError):
        masked_alpha(value, 255, 100)


def test_expensive_key_cannot_contain_raster_mask_state():
    inputs = LookInputsKey("smart-mask", "context")
    key = RenderedLookKey("source", "look", "preview", inputs)
    assert not hasattr(key, "mask_revision")
    cheap = MaskPublicationKey(
        "layer", "source", "mask-a", 7, True, 80,
        ("x", "y", "sx", "sy"), 11, "preview", 4)
    assert (cheap.mask_revision, cheap.document_revision,
            cheap.worker_generation) == (7, 11, 4)
    assert cheap != MaskPublicationKey(
        "layer", "source", "mask-b", 7, True, 80,
        ("x", "y", "sx", "sy"), 11, "preview", 4)
    assert key == RenderedLookKey("source", "look", "preview", inputs)
    # Raster mask replacement changes only the cheap publication key.
    assert key == RenderedLookKey("source", "look", "preview", inputs)
    with pytest.raises((TypeError, ValueError)):
        RenderedLookKey("source", "look", "preview", ("raster-mask", 7))
    with pytest.raises(TypeError):
        RenderedLookKey([], "look", "preview", inputs)


def test_coordinate_certification_is_truthful():
    assert SUPPORTED_MASK_TRANSFORMS == {"x", "y", "scale_x", "scale_y"}
    assert UNSUPPORTED_MASK_TRANSFORMS == {
        "rotation_degrees", "flip_x", "flip_y"}
    assert document_to_mask_point(
        11, 14, layer_x=5, layer_y=2, scale_x=2, scale_y=4) == (3, 3)
    for unsupported in (
        {"rotation_degrees": 1},
        {"flip_x": True},
        {"flip_y": True},
    ):
        with pytest.raises(ValueError, match="unsupported"):
            document_to_mask_point(
                11, 14, layer_x=5, layer_y=2, scale_x=2, scale_y=4,
                **unsupported)
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            document_to_mask_point(
                value, 14, layer_x=5, layer_y=2, scale_x=2, scale_y=4)


def test_mask_history_and_render_count_contracts():
    assert source_mask_bytes(1920, 1080) == 2_073_600
    assert source_mask_bytes(3840, 2160) == 8_294_400
    assert (HISTORY_MAX_ENTRIES, HISTORY_MAX_RETAINED_BYTES) == (
        64, 128 * 1024 * 1024)
    assert LOOK_RENDERS_PER_MASK_STAMP == 0
    assert EXACT_RENDERS_PER_COMPLETED_STROKE == 1


def test_numeric_release_budgets_are_frozen():
    assert (PROXY_TYPICAL_ROI_P95_MS, PROXY_LARGE_ROI_P95_MS) == (33, 50)
    assert (PREVIEW_COMPOSITE_720_P95_MS,
            PREVIEW_COMPOSITE_1440_P95_MS) == (50, 120)
    assert (EXACT_MASK_COMPOSITE_1080P_TARGET_P95_MS,
            EXACT_MASK_COMPOSITE_4K_TARGET_P95_MS) == (250, 750)
    assert (STROKE_INCREMENTAL_1080P_BYTES,
            STROKE_INCREMENTAL_4K_BYTES) == (
                6 * 1024 * 1024, 18 * 1024 * 1024)
    assert (THUMBNAIL_COLD_MS, THUMBNAIL_WARM_P95_MS) == (10, 2)
    assert MEASUREMENT_OWNER == {
        "mask_composite": "MT-03",
        "cache_and_thumbnail": "MT-04",
        "history_retained_bytes": "MT-06",
        "dirty_proxy_and_stroke_peak": "MT-10",
    }


def test_history_and_persistence_boundaries_are_explicit():
    assert "raster_mask_stroke" in HISTORY_OPERATIONS
    assert "transform_confirm" in HISTORY_OPERATIONS
    assert {"selection", "look_slider"} <= NON_HISTORY_OPERATIONS
    assert NATIVE_PROJECT_PERSISTENCE_EPIC == "separate"


def test_publication_key_validates_live_raster_state():
    identity = MaskObjectIdentity(object())
    key = MaskPublicationKey(
        "layer", "source", identity, 2, True, 75, ("transform",),
        4, ("frame", 480), 6)
    assert hash(key)
    bad_values = (
        {"layer_id": ""},
        {"mask_revision": True},
        {"mask_enabled": 1},
        {"mask_density": 101},
        {"transform": []},
        {"document_revision": -1},
        {"target": []},
        {"worker_generation": True},
    )
    values = dict(
        layer_id="layer", source_identity="source",
        mask_identity=identity, mask_revision=2,
        mask_enabled=True, mask_density=75, transform=("transform",),
        document_revision=4, target=("frame", 480), worker_generation=6)
    for change in bad_values:
        with pytest.raises(ValueError):
            MaskPublicationKey(**(values | change))
