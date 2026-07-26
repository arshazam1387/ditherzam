import numpy as np
import pytest

from ditherzam.layers import (
    MaskGeneratorError,
    MaskPatternKind,
    MaskPatternSpec,
    dither_mask,
    pattern_mask,
    pattern_thresholds,
)


def test_default_bayer_is_canonical_4x4_half_up_thresholds():
    assert pattern_thresholds((4, 4)).tolist() == [
        [8, 135, 40, 167],
        [199, 72, 231, 104],
        [56, 183, 24, 151],
        [247, 120, 215, 88],
    ]


def test_bayer_scale_rotation_and_signed_offsets_are_before_modulo():
    scaled = pattern_thresholds((2, 4), MaskPatternSpec(scale=2))
    assert scaled.tolist() == [[8, 8, 135, 135], [8, 8, 135, 135]]
    shifted = pattern_thresholds(
        (2, 2), MaskPatternSpec(orientation=90, offset_x=-1, offset_y=-1))
    assert shifted.shape == (2, 2)
    assert not np.array_equal(shifted, pattern_thresholds((2, 2)))


def test_lines_golden_for_meaningful_orientation_and_period():
    horizontal = pattern_thresholds(
        (2, 4), MaskPatternSpec(MaskPatternKind.LINES, scale=4))
    assert horizontal.tolist() == [[0, 85, 170, 255], [0, 85, 170, 255]]
    vertical = pattern_thresholds(
        (4, 2), MaskPatternSpec(
            MaskPatternKind.LINES, scale=4, orientation=90))
    assert vertical.tolist() == [[0, 0], [85, 85], [170, 170], [255, 255]]


def test_seeded_noise_is_reproducible_grained_and_seed_sensitive():
    spec = MaskPatternSpec(MaskPatternKind.NOISE, scale=2, seed=8675309)
    first = pattern_thresholds((6, 8), spec)
    second = pattern_thresholds((6, 8), spec)
    other = pattern_thresholds(
        (6, 8), MaskPatternSpec(MaskPatternKind.NOISE, scale=2, seed=7))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.all(first[0:2, 0:2] == first[0, 0])
    assert np.unique(first).size > 4


def test_pattern_candidate_uses_shared_owned_typed_boundary():
    rgba = np.zeros((3, 5, 4), np.uint8)
    candidate = pattern_mask(
        rgba, MaskPatternSpec(MaskPatternKind.LINES, scale=3))
    assert candidate.label == "Lines Pattern"
    assert candidate.pixels.shape == (3, 5)
    assert candidate.pixels.flags.writeable is False


def test_dither_threshold_golden_and_forces_soft_mask_endpoints():
    from ditherzam.masking.contracts import source_identity

    source = np.array([
        [0, 7, 8, 134],
        [198, 199, 230, 255],
    ], np.uint8)
    identity = source_identity(np.zeros((2, 4, 4), np.uint8))
    result = dither_mask(source, source_identity=identity).pixels
    assert result.tolist() == [[0, 0, 0, 0], [0, 255, 0, 255]]


def test_dither_mix_uses_frozen_integer_half_up_rule():
    from ditherzam.masking.contracts import source_identity

    source = np.array([[0, 1, 127, 128, 254, 255]], np.uint8)
    identity = source_identity(np.zeros((1, 6, 4), np.uint8))
    zero = dither_mask(source, MaskPatternSpec(mix=0), identity).pixels
    half = dither_mask(source, MaskPatternSpec(mix=50), identity).pixels
    assert zero.tolist() == source.tolist()
    assert half.tolist() == [[0, 1, 191, 64, 255, 255]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "dots"}, {"scale": 0}, {"orientation": 30},
        {"mix": -1}, {"mix": 101}, {"seed": 2 ** 31},
        {"offset_x": 1_000_001},
    ],
)
def test_pattern_spec_rejects_broad_families_and_invalid_controls(kwargs):
    with pytest.raises(MaskGeneratorError):
        MaskPatternSpec(**kwargs)


def test_pattern_inputs_are_strict_and_default_is_deterministic():
    assert MaskPatternSpec() == MaskPatternSpec()
    with pytest.raises(MaskGeneratorError):
        pattern_thresholds([2, 2])
    with pytest.raises(MaskGeneratorError):
        dither_mask(np.zeros((2, 2), np.float32))


def test_panel_exposes_only_frozen_pattern_controls_and_dither_mix(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    requests = []
    panel.pattern_mask_requested.connect(lambda *values: requests.append(values))
    assert [
        panel.pattern_family.itemData(index)
        for index in range(panel.pattern_family.count())
    ] == ["bayer", "lines", "noise"]
    assert [
        panel.pattern_orientation.itemData(index)
        for index in range(panel.pattern_orientation.count())
    ] == [0, 90, 180, 270]
    panel.show_pattern_editor(False)
    assert panel.pattern_mix.isHidden()
    panel._confirm_pattern()
    panel.pattern_family.setCurrentIndex(2)
    panel.pattern_scale.setValue(3)
    panel.pattern_seed.setValue(42)
    panel.show_pattern_editor(True)
    assert panel.pattern_seed.isEnabled()
    assert not panel.pattern_orientation.isEnabled()
    assert not panel.pattern_mix.isHidden()
    panel.pattern_mix.setValue(50)
    panel._confirm_pattern()
    assert requests == [
        ("bayer", 1, 0, 0, 0, 100, 0, False),
        ("noise", 3, 0, 0, 0, 50, 42, True),
    ]


def test_controller_panel_pattern_and_dither_follow_typed_confirmation(
    qapp_fixture,
):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., 3] = 255
    panel = LayersPanel()
    controller = LayersController(
        panel, registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (
            np.zeros((4, 4), np.float32), rgba, None),
        cap_provider=lambda: 480, frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None)
    controller.initialize_source_layer()
    panel.pattern_mask_requested.emit("bayer", 1, 0, 0, 0, 100, 0, False)
    assert controller.document.layers[0].raster_mask is not None
    before = controller.document.layers[0].raster_mask.pixels.copy()
    panel.pattern_mask_requested.emit("lines", 4, 90, 0, 0, 100, 0, True)
    proposal = controller.mask_combination_preview
    assert proposal is not None
    assert np.array_equal(
        controller.document.layers[0].raster_mask.pixels, before)
    assert controller.confirm_mask_replacement(proposal.token)
    assert set(np.unique(
        controller.document.layers[0].raster_mask.pixels)) <= {0, 255}
