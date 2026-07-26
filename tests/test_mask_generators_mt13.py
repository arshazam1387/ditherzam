import numpy as np
import pytest

from ditherzam.layers import (
    GradientKind,
    GradientSpec,
    LUMINANCE_PRESETS,
    LuminancePreset,
    LuminanceRange,
    MaskGeneratorCandidate,
    MaskGeneratorError,
    gradient_mask,
    luminance_mask,
    source_luminance_u8,
)


def test_source_rgba_luminance_golden_and_alpha_is_not_luminance():
    rgba = np.array([[
        [0, 0, 0, 255], [255, 255, 255, 255],
        [255, 0, 0, 1], [0, 255, 0, 0], [0, 0, 255, 127],
    ]], dtype=np.uint8)
    assert source_luminance_u8(rgba).tolist() == [[0, 255, 54, 182, 18]]


def test_luminance_hard_range_is_independent_of_source_alpha():
    rgba = np.array([[
        [49, 49, 49, 255], [50, 50, 50, 255],
        [100, 100, 100, 128], [150, 150, 150, 255],
        [151, 151, 151, 255],
    ]], dtype=np.uint8)
    candidate = luminance_mask(rgba, LuminanceRange(50, 150, 0))
    assert candidate.pixels.tolist() == [[0, 255, 255, 255, 0]]
    assert candidate.pixels.flags.writeable is False


def test_luminance_soft_bounds_are_smoothstep_golden():
    values = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200], np.uint8)
    rgba = np.concatenate([
        np.repeat(values[None, :, None], 3, axis=2),
        np.full((1, values.size, 1), 255, np.uint8),
    ], axis=2)
    result = luminance_mask(rgba, LuminanceRange(50, 150, 50)).pixels
    assert result.tolist() == [[0, 128, 255, 255, 255, 255, 255, 128, 0]]


def test_exact_named_presets_are_frozen():
    assert LUMINANCE_PRESETS == {
        LuminancePreset.SHADOWS: LuminanceRange(0, 64, 64),
        LuminancePreset.MIDTONES: LuminanceRange(96, 159, 64),
        LuminancePreset.HIGHLIGHTS: LuminanceRange(191, 255, 64),
    }


def test_linear_gradient_golden_horizontal_and_vertical():
    horizontal = gradient_mask(
        np.zeros((3, 5, 4), np.uint8),
        GradientSpec(GradientKind.LINEAR, 0, 0, 1, 0))
    assert horizontal.pixels.tolist() == [
        [0, 64, 128, 191, 255],
        [0, 64, 128, 191, 255],
        [0, 64, 128, 191, 255],
    ]
    vertical = gradient_mask(
        np.zeros((3, 2, 4), np.uint8),
        GradientSpec(GradientKind.LINEAR, 0, 0, 0, 1))
    assert vertical.pixels.tolist() == [[0, 0], [128, 128], [255, 255]]


def test_linear_gradient_clamps_before_start_and_after_end():
    result = gradient_mask(
        np.zeros((1, 5, 4), np.uint8),
        GradientSpec(GradientKind.LINEAR, .25, 0, .75, 0))
    assert result.pixels.tolist() == [[0, 0, 128, 255, 255]]


def test_radial_gradient_golden():
    result = gradient_mask(
        np.zeros((3, 3, 4), np.uint8),
        GradientSpec(GradientKind.RADIAL, .5, .5, 1, .5))
    assert result.pixels.tolist() == [
        [255, 255, 255],
        [255, 0, 255],
        [255, 255, 255],
    ]


def test_numeric_geometry_validation_excludes_unsupported_kinds():
    with pytest.raises(MaskGeneratorError):
        GradientSpec("angular", 0, 0, 1, 1)
    with pytest.raises(MaskGeneratorError):
        GradientSpec(GradientKind.LINEAR, 0, 0, 0, 0)
    with pytest.raises(MaskGeneratorError):
        GradientSpec(GradientKind.LINEAR, -0.1, 0, 1, 1)


def test_candidate_is_owned_immutable_and_typed():
    from ditherzam.masking.contracts import source_identity

    source = np.array([[1, 2]], np.uint8)
    rgba = np.zeros((1, 2, 4), np.uint8)
    candidate = MaskGeneratorCandidate(
        source, "  Generated  ", source_identity(rgba))
    source[:] = 9
    assert candidate.label == "Generated"
    assert candidate.pixels.tolist() == [[1, 2]]
    with pytest.raises(ValueError):
        candidate.pixels[0, 0] = 3
    with pytest.raises(MaskGeneratorError):
        MaskGeneratorCandidate(source, "Generated", object())


def test_panel_exposes_presets_and_numeric_gradient_fallback(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    presets = []
    gradients = []
    panel.luminance_mask_requested.connect(presets.append)
    panel.gradient_mask_requested.connect(
        lambda *values: gradients.append(values))
    panel.shadows_mask_action.trigger()
    panel.show_gradient_editor("radial")
    panel.set_gradient_geometry(.25, .5, .75, 1.0)
    panel.confirm_gradient_btn.click()
    assert presets == ["shadows"]
    assert gradients == [("radial", .25, .5, .75, 1.0)]
    assert not panel.gradient_editor.isVisible()


def test_controller_canvas_gradient_uses_inverse_layer_geometry(qapp_fixture):
    from dataclasses import replace
    from ditherzam.dithering import registry
    from ditherzam.layers import LayerTransform
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    rgba = np.zeros((2, 3, 4), np.uint8)
    controller = LayersController(
        LayersPanel(), registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (np.zeros((2, 3), np.float32), rgba, None),
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
    )
    controller.initialize_source_layer()
    layer = controller.document.layers[0]
    controller._document = replace(
        controller.document,
        layers=(replace(
            layer, transform=LayerTransform(
                x=10, y=20, scale_x=2, scale_y=3)),),
    )
    assert controller.generate_gradient_from_document_points(
        GradientKind.LINEAR, 10, 20, 14, 20)
    assert controller.document.layers[0].raster_mask.pixels.tolist() == [
        [0, 128, 255], [0, 128, 255]]
