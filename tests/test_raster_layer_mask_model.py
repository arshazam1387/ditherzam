from dataclasses import replace
import dataclasses

import numpy as np
import pytest

from ditherzam.composition import Look
from ditherzam.layers import (
    CanvasSpec,
    Layer,
    LayerDocument,
    LayerSource,
    LayerStack,
    RasterLayerMask,
)


def _look() -> Look:
    return Look("look", {"dither": {"style": "None"}})


def _source(height: int = 2, width: int = 3) -> LayerSource:
    gray = np.zeros((height, width), dtype=np.float32)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    return LayerSource(gray, rgba)


def test_mask_owns_immutable_c_order_uint8_pixels():
    original = np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::-1]
    mask = RasterLayerMask(original)

    assert mask.pixels.dtype == np.uint8
    assert mask.pixels.shape == (2, 3)
    assert mask.pixels.flags.c_contiguous
    assert not mask.pixels.flags.writeable

    original[0, 0] = 99
    assert mask.pixels[0, 0] != 99
    with pytest.raises(ValueError):
        mask.pixels[0, 0] = 10
    with pytest.raises(ValueError):
        mask.pixels.flags.writeable = True


@pytest.mark.parametrize(
    "pixels",
    [
        [[0, 255]],
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2, 1), dtype=np.uint8),
        np.zeros((0, 2), dtype=np.uint8),
    ],
)
def test_mask_rejects_invalid_pixels(pixels):
    with pytest.raises(ValueError):
        RasterLayerMask(pixels)


@pytest.mark.parametrize("enabled", [0, 1, np.bool_(True), "yes"])
def test_mask_enabled_is_strict_bool(enabled):
    with pytest.raises(ValueError):
        RasterLayerMask(np.zeros((1, 1), np.uint8), enabled=enabled)


@pytest.mark.parametrize("density", [True, 1.0, -1, 101])
def test_mask_density_is_strict_bounded_int(density):
    with pytest.raises(ValueError):
        RasterLayerMask(np.zeros((1, 1), np.uint8), density=density)


@pytest.mark.parametrize("revision", [True, 1.0, -1])
def test_mask_revision_is_strict_nonnegative_int(revision):
    with pytest.raises(ValueError):
        RasterLayerMask(np.zeros((1, 1), np.uint8), revision=revision)


def test_mask_evolve_replaces_values_and_increments_revision_exactly_once():
    initial = RasterLayerMask(
        np.zeros((2, 3), np.uint8), enabled=True, density=100, revision=7
    )
    replacement_pixels = np.full((2, 3), 123, np.uint8)

    evolved = initial.evolve(
        pixels=replacement_pixels, enabled=False, density=40
    )

    assert evolved is not initial
    assert evolved.revision == 8
    assert evolved.enabled is False
    assert evolved.density == 40
    np.testing.assert_array_equal(evolved.pixels, replacement_pixels)
    assert initial.revision == 7
    assert initial.enabled is True
    assert initial.density == 100
    np.testing.assert_array_equal(initial.pixels, 0)


def test_mask_evolve_with_defaults_still_creates_one_new_revision():
    initial = RasterLayerMask(np.zeros((1, 1), np.uint8), revision=2)
    evolved = initial.evolve()

    assert evolved.revision == 3
    assert evolved.pixels is initial.pixels


def test_mask_has_semantic_array_safe_equality_and_is_unhashable():
    pixels = np.array([[0, 255]], np.uint8)
    left = RasterLayerMask(pixels, density=50, revision=3)
    right = RasterLayerMask(pixels.copy(), density=50, revision=3)
    assert left == right
    assert left != right.evolve()
    with pytest.raises(TypeError):
        hash(left)


def test_dataclass_replace_cannot_bypass_revision_evolution():
    mask = RasterLayerMask(np.zeros((1, 1), np.uint8))
    with pytest.raises(TypeError):
        dataclasses.replace(mask, density=20)


def test_mask_preserves_coverage_endpoints():
    mask = RasterLayerMask(np.array([[0, 255]], np.uint8))
    assert mask.pixels.tolist() == [[0, 255]]


def test_layer_validates_mask_dimensions_against_source():
    source = _source(2, 3)
    mask = RasterLayerMask(np.zeros((3, 2), np.uint8))

    with pytest.raises(ValueError, match="dimensions"):
        Layer("a", "A", _look(), source=source, raster_mask=mask)


def test_layer_without_source_cannot_attach_a_raster_mask():
    mask = RasterLayerMask(np.zeros((2, 3), np.uint8))

    with pytest.raises(ValueError, match="source"):
        Layer("a", "A", _look(), raster_mask=mask)


def test_duplicate_layers_share_mask_until_one_layer_is_replaced():
    mask = RasterLayerMask(np.zeros((2, 3), np.uint8), revision=4)
    layer = Layer(
        "a", "A", _look(), source=_source(), mask_revision=19, raster_mask=mask
    )

    stack = LayerStack((layer,)).duplicate(0, "b")
    assert stack.layers[0].raster_mask is stack.layers[1].raster_mask

    edited_mask = stack.layers[1].raster_mask.evolve(density=25)
    changed = stack.replace(1, replace(stack.layers[1], raster_mask=edited_mask))

    assert changed.layers[0].raster_mask is mask
    assert changed.layers[0].raster_mask.revision == 4
    assert changed.layers[1].raster_mask is edited_mask
    assert changed.layers[1].raster_mask.revision == 5
    assert changed.layers[0].mask_revision == 19
    assert changed.layers[1].mask_revision == 19


def test_document_duplicate_also_shares_immutable_mask_object():
    mask = RasterLayerMask(np.full((2, 3), 255, np.uint8))
    layer = Layer("a", "A", _look(), source=_source(), raster_mask=mask)
    document = LayerDocument(CanvasSpec(3, 2), (layer,), ("a",))

    duplicate = document.duplicate(0, "b")

    assert duplicate.layers[0].raster_mask is mask
    assert duplicate.layers[1].raster_mask is mask


def test_appended_mask_field_preserves_legacy_layer_positional_abi():
    layer = Layer("a", "A", _look(), True, 90, "normal")
    assert layer.visible is True
    assert layer.opacity == 90
    assert layer.blend_mode == "normal"
    assert layer.raster_mask is None
