import numpy as np
import pytest
from ditherzam.masking.contracts import source_identity

from ditherzam.layers import (
    MaskCandidate,
    MaskCandidateOrigin,
    MaskCombinationError,
    MaskCombinationMode,
    capped_combination_preview,
    combine_mask_candidate,
)


def _candidate(values, origin=MaskCandidateOrigin.PATTERN):
    rgba = np.zeros((1, len(values), 4), np.uint8)
    return MaskCandidate(
        np.array([values], np.uint8), origin, "Candidate",
        source_identity(rgba))


def test_exact_public_vocabulary_and_origins():
    assert [mode.value for mode in MaskCombinationMode] == [
        "Replace", "Add", "Subtract", "Intersect"]
    assert {origin.value for origin in MaskCandidateOrigin} == {
        "Smart", "Imported", "Luminance", "Gradient", "Pattern"}


@pytest.mark.parametrize(("mode", "expected"), [
    (MaskCombinationMode.REPLACE, [0, 100, 255, 128]),
    (MaskCombinationMode.ADD, [0, 255, 255, 255]),
    (MaskCombinationMode.SUBTRACT, [0, 100, 0, 72]),
    (MaskCombinationMode.INTERSECT, [0, 78, 255, 100]),
])
def test_soft_coverage_golden_formulas(mode, expected):
    current = np.array([[0, 200, 255, 200]], np.uint8)
    result = combine_mask_candidate(
        current, _candidate([0, 100, 255, 128]), mode)
    assert result.pixels.tolist() == [expected]
    assert not result.pixels.flags.writeable


def test_candidate_is_owned_typed_and_rejects_shape_mismatch():
    source = np.array([[1, 2]], np.uint8)
    rgba = np.zeros((1, 2, 4), np.uint8)
    candidate = MaskCandidate(
        source, MaskCandidateOrigin.SMART, " Smart ", source_identity(rgba))
    source[:] = 9
    assert candidate.pixels.tolist() == [[1, 2]]
    assert candidate.label == "Smart"
    with pytest.raises(MaskCombinationError):
        combine_mask_candidate(
            np.zeros((2, 2), np.uint8), candidate,
            MaskCombinationMode.INTERSECT)


def test_controller_combination_requires_confirmation_and_is_one_history_step(
    qapp_fixture,
):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    rgba = np.zeros((1, 4, 4), np.uint8)
    controller = LayersController(
        LayersPanel(), registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (np.zeros((1, 4), np.float32), rgba, None),
        cap_provider=lambda: 480, frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None)
    controller.initialize_source_layer()
    controller.propose_active_raster_mask(
        np.array([[100, 200, 255, 1]], np.uint8), "base")
    before = controller.document
    candidate = MaskCandidate(
        np.array([[100, 100, 1, 255]], np.uint8),
        MaskCandidateOrigin.PATTERN, "Candidate",
        controller.document.layers[0].source.source_identity)
    assert not controller.propose_mask_candidate(
        candidate, MaskCombinationMode.ADD)
    assert controller.document is before
    transaction = controller.mask_combination_preview
    assert transaction is not None
    assert controller.confirm_mask_replacement(transaction.token)
    assert controller.document.layers[0].raster_mask.pixels.tolist() == [
        [200, 255, 255, 255]]
    assert controller.undo()
    assert controller.document.layers[0].raster_mask.pixels.tolist() == [
        [100, 200, 255, 1]]


def test_stale_combination_confirmation_is_rejected(qapp_fixture):
    from tests.test_raster_mask_lifecycle import _controller
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    controller.propose_active_raster_mask(
        np.zeros((3, 4), np.uint8), "base")
    controller.propose_mask_candidate(
        MaskCandidate(
            np.ones((3, 4), np.uint8), MaskCandidateOrigin.IMPORTED, "PNG",
            controller.document.layers[0].source.source_identity),
        MaskCombinationMode.ADD)
    proposal = controller.mask_replacement_proposal
    controller.invert_active_raster_mask()
    assert not controller.confirm_mask_replacement(proposal.token)


def test_panel_exposes_only_exact_combination_vocabulary(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    assert [
        panel.mask_combination_combo.itemText(index)
        for index in range(panel.mask_combination_combo.count())
    ] == ["Replace", "Add", "Subtract", "Intersect"]
    assert panel.mask_combination_combo.accessibleName() == (
        "Mask candidate combination mode")


def test_new_plain_replacement_supersedes_old_combination_token(qapp_fixture):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    rgba = np.zeros((3, 4, 4), np.uint8)
    clears = []
    controller = LayersController(
        LayersPanel(), registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (np.zeros((3, 4), np.float32), rgba, None),
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
        refinement_preview_clear=lambda: clears.append(True),
    )
    controller.initialize_source_layer()
    controller.propose_active_raster_mask(
        np.full((3, 4), 100, np.uint8), "base")
    source = controller.document.layers[0].source
    candidate = MaskCandidate(
        np.full((3, 4), 50, np.uint8),
        MaskCandidateOrigin.PATTERN, "old", source.source_identity)
    controller.propose_mask_candidate(candidate, MaskCombinationMode.ADD)
    old_token = controller.mask_replacement_proposal.token
    controller.propose_active_raster_mask(
        np.full((3, 4), 25, np.uint8), "new")
    assert clears
    new_token = controller.mask_replacement_proposal.token
    assert new_token != old_token
    assert not controller.confirm_mask_replacement(old_token)
    assert controller.mask_replacement_proposal.token == new_token
    assert controller.confirm_mask_replacement(new_token)
    assert np.all(controller.document.layers[0].raster_mask.pixels == 25)


def test_combination_preview_is_published_and_cancel_restores(qapp_fixture):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    rgba = np.zeros((3, 4, 4), np.uint8)
    previews, clears = [], []
    controller = LayersController(
        LayersPanel(), registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (np.zeros((3, 4), np.float32), rgba, None),
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
        refinement_preview_sink=lambda pixels, token: previews.append(
            (pixels.copy(), token)),
        refinement_preview_clear=lambda: clears.append(True),
    )
    controller.initialize_source_layer()
    controller.propose_active_raster_mask(
        np.full((3, 4), 100, np.uint8), "base")
    source = controller.document.layers[0].source
    controller.propose_mask_candidate(
        MaskCandidate(
            np.full((3, 4), 50, np.uint8),
            MaskCandidateOrigin.PATTERN, "preview", source.source_identity),
        MaskCombinationMode.ADD,
    )
    token = controller.mask_replacement_proposal.token
    assert previews and previews[-1][1] == token
    assert np.all(previews[-1][0] == 150)
    assert controller.cancel_mask_replacement(token)
    assert clears


def test_preview_is_capped_immutable_and_matches_exact_sampling():
    rgba = np.zeros((1080, 1920, 4), np.uint8)
    identity = source_identity(rgba)
    current = np.full((1080, 1920), 100, np.uint8)
    candidate = MaskCandidate(
        np.full((1080, 1920), 200, np.uint8),
        MaskCandidateOrigin.GRADIENT, "large", identity)
    preview = capped_combination_preview(
        current, candidate, MaskCombinationMode.ADD)
    assert preview.shape == (405, 720)
    assert np.all(preview == 255)
    assert not preview.flags.writeable


def _live_candidate(controller, pixels):
    return MaskCandidate(
        np.asarray(pixels, dtype=np.uint8),
        MaskCandidateOrigin.LUMINANCE, "live",
        controller.document.layers[0].source.source_identity)


def test_preview_transaction_cancel_no_history_and_exact_confirm(qapp_fixture):
    from tests.test_raster_mask_lifecycle import _controller
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    controller.propose_active_raster_mask(
        np.full((3, 4), 100, np.uint8), "base")
    document = controller.document
    history_count = controller._history.entry_count
    token = controller.begin_mask_combination(
        _live_candidate(controller, np.full((3, 4), 80, np.uint8)),
        MaskCombinationMode.ADD)
    assert token and controller.document is document
    assert controller.cancel_mask_combination(token)
    assert controller.document is document
    assert controller._history.entry_count == history_count

    token = controller.begin_mask_combination(
        _live_candidate(controller, np.full((3, 4), 80, np.uint8)),
        MaskCombinationMode.ADD)
    assert controller.confirm_mask_combination(token)
    assert np.all(controller.document.layers[0].raster_mask.pixels == 180)
    assert controller.undo()
    assert np.all(controller.document.layers[0].raster_mask.pixels == 100)


def test_latest_wins_late_a_cannot_clear_b_and_stale_confirm_fails(qapp_fixture):
    from tests.test_raster_mask_lifecycle import _controller
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    controller.propose_active_raster_mask(
        np.full((3, 4), 100, np.uint8), "base")
    candidate = _live_candidate(controller, np.full((3, 4), 10, np.uint8))
    token_a = controller.begin_mask_combination(
        candidate, MaskCombinationMode.ADD)
    token_b = controller.begin_mask_combination(
        candidate, MaskCombinationMode.SUBTRACT)
    assert not controller.cancel_mask_combination(token_a)
    assert controller.mask_combination_preview.token == token_b
    controller.invert_active_raster_mask()
    assert not controller.confirm_mask_combination(token_b)
    assert controller.mask_combination_preview is None


def test_noop_confirm_publishes_nothing(qapp_fixture):
    from tests.test_raster_mask_lifecycle import _controller
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    pixels = np.full((3, 4), 77, np.uint8)
    controller.propose_active_raster_mask(pixels, "base")
    document = controller.document
    token = controller.begin_mask_combination(
        _live_candidate(controller, pixels), MaskCombinationMode.REPLACE)
    assert not controller.confirm_mask_combination(token)
    assert controller.document is document
    assert controller.mask_combination_preview is None
