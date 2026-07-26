import sys

import numpy as np
import pytest


def _look(name="Layer"):
    from ditherzam.composition import Look

    return Look(name, {
        "adjustments": {"contrast": 50},
        "dither": {"style": "None", "scale": 1, "params": {}},
    })


def _source(value, *, alpha=255, shape=(2, 3)):
    from ditherzam.layers import LayerSource

    gray = np.full(shape, value, dtype=np.float32)
    rgba = np.full((*shape, 4), value, dtype=np.uint8)
    rgba[..., 3] = alpha
    return LayerSource(gray, rgba)


def test_layer_source_defensively_owns_read_only_pixels():
    from ditherzam.layers import LayerSource

    gray = np.arange(6, dtype=np.float32).reshape(2, 3)
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    source = LayerSource(gray, rgba)
    gray[0, 0] = 99
    rgba[0, 0] = 99
    assert source.gray[0, 0] == 0
    assert np.array_equal(source.rgba[0, 0], [0, 0, 0, 255])
    assert source.gray.flags.c_contiguous and not source.gray.flags.writeable
    assert source.rgba.flags.c_contiguous and not source.rgba.flags.writeable
    with pytest.raises(ValueError):
        LayerSource(np.zeros((2, 2), np.float32),
                    np.zeros((3, 2, 4), np.uint8))


def test_document_canvas_survives_removing_final_layer():
    from ditherzam.layers import CanvasSpec, Layer, LayerDocument

    layer = Layer("a", "Layer 1", _look(), source=_source(20))
    document = LayerDocument(CanvasSpec(3, 2), (layer,), ("a",))
    empty = document.remove(0)
    assert empty.canvas == CanvasSpec(3, 2)
    assert empty.layers == ()
    assert empty.selected_ids == ()
    assert empty.revision == document.revision + 1


def test_document_duplicate_shares_pixels_but_look_edits_are_independent():
    from dataclasses import replace
    from ditherzam.layers import CanvasSpec, Layer, LayerDocument

    layer = Layer("a", "Layer 1", _look("One"), source=_source(30))
    original = LayerDocument(CanvasSpec(3, 2), (layer,), ("a",))
    duplicated = original.duplicate(0, "b", "Layer 1 copy")
    assert duplicated.layers[0].source is duplicated.layers[1].source
    edited = duplicated.replace(
        1, replace(duplicated.layers[1], look=_look("Changed")))
    assert edited.layers[0].look.name == "One"
    assert edited.layers[1].look.name == "Changed"
    assert edited.layers[0].source is original.layers[0].source


def test_document_renderer_uses_each_layer_source_and_reveals_lower_alpha():
    from ditherzam.dithering import registry
    from ditherzam.layers import (
        CanvasSpec, Layer, LayerDocument, render_layer_document,
    )

    lower = _source(25)
    upper_gray = np.full((2, 3), 200, dtype=np.float32)
    upper_rgba = np.full((2, 3, 4), 200, dtype=np.uint8)
    upper_rgba[..., 3] = 255
    upper_rgba[:, 0, 3] = 0
    from ditherzam.layers import LayerSource
    upper = LayerSource(upper_gray, upper_rgba)
    document = LayerDocument(
        CanvasSpec(3, 2),
        (
            Layer("a", "Lower", _look("Lower"), source=lower),
            Layer("b", "Upper", _look("Upper"), source=upper),
        ),
        ("b",),
    )
    result = render_layer_document(document, registry)
    assert result.shape == (2, 3, 4)
    assert np.all(result[:, 0, :3] == 25)
    assert np.all(result[:, 1:, :3] == 200)
    assert np.all(result[..., 3] == 255)


def test_blank_layer_is_composite_noop_and_zero_document_is_transparent():
    from ditherzam.dithering import registry
    from ditherzam.layers import (
        CanvasSpec, Layer, LayerDocument, render_layer_document,
    )

    base = Layer("a", "Base", _look(), source=_source(80))
    canvas = CanvasSpec(3, 2)
    one = LayerDocument(canvas, (base,), ("a",))
    blank = Layer(
        "b", "Blank", _look("Blank"),
        source=_source(0, alpha=0),
    )
    two = one.add(blank)
    assert np.array_equal(
        render_layer_document(one, registry),
        render_layer_document(two, registry),
    )
    empty = LayerDocument(canvas)
    rendered = render_layer_document(empty, registry)
    assert rendered.shape == (2, 3, 4)
    assert np.count_nonzero(rendered) == 0


def test_pending_subject_mask_can_render_preview_but_exact_remains_fail_closed():
    from ditherzam.composition import Look
    from ditherzam.composition.rendering import CompositionMaskError
    from ditherzam.dithering import registry
    from ditherzam.layers import (
        CanvasSpec, Layer, LayerDocument, render_layer_document,
    )

    preset = _look().preset
    preset["smart_mask"] = {
        "enabled": True,
        "target": "subject",
        "sensitivity": 50,
        "feather_px": 0,
        "expansion_px": 0,
        "invert": False,
        "outside": "transparent",
        "bake_fill": False,
    }
    layer = Layer(
        "pending", "Pending mask", Look("Pending", preset), source=_source(90))
    document = LayerDocument(CanvasSpec(3, 2), (layer,), ("pending",))
    with pytest.raises(CompositionMaskError):
        render_layer_document(document, registry)
    preview = render_layer_document(
        document, registry, allow_pending_masks=True)
    assert preview.shape == (2, 3, 4)
    assert np.all(preview[..., 3] == 255)


def test_document_places_native_sources_and_clips_at_canvas_edges():
    from ditherzam.dithering import registry
    from ditherzam.layers import (
        CanvasSpec, Layer, LayerDocument, LayerTransform, render_layer_document,
    )

    small = Layer(
        "small", "Small", _look(), source=_source(60, shape=(2, 2)),
        transform=LayerTransform(x=2, y=1),
    )
    large = Layer(
        "large", "Large", _look(), source=_source(180, shape=(6, 8)),
        transform=LayerTransform(x=-2, y=-2),
    )
    document = LayerDocument(CanvasSpec(6, 4), (large, small), ("small",))
    rendered = render_layer_document(document, registry)
    assert rendered.shape == (4, 6, 4)
    assert np.all(rendered[0, 0] == [180, 180, 180, 255])
    assert np.all(rendered[1:3, 2:4] == [60, 60, 60, 255])
    assert np.all(rendered[:, 4:] == [180, 180, 180, 255])

    preview = render_layer_document(document, registry, target_max_side=3)
    assert preview.shape == (2, 3, 4)
    assert np.all(preview[0, 0] == [180, 180, 180, 255])
    assert np.all(preview[0, 1] == [60, 60, 60, 255])


def test_document_renderer_resizes_layer_before_placing_it():
    from ditherzam.dithering import registry
    from ditherzam.layers import (
        CanvasSpec, Layer, LayerDocument, LayerTransform, render_layer_document,
    )

    layer = Layer(
        "scaled",
        "Scaled",
        _look(),
        source=_source(140, shape=(2, 3)),
        transform=LayerTransform(x=1, y=2, scale_x=2.0, scale_y=1.5),
    )
    document = LayerDocument(CanvasSpec(8, 6), (layer,), ("scaled",))
    rendered = render_layer_document(document, registry)
    assert np.all(rendered[2:5, 1:7, :3] == 140)
    outside = rendered.copy()
    outside[2:5, 1:7] = 0
    assert np.count_nonzero(outside) == 0


def test_layers_core_imports_without_qt():
    import subprocess

    script = """
import sys
sys.modules["PySide6"] = None
import ditherzam.layers
assert hasattr(ditherzam.layers, "LayerDocument")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
