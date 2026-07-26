import numpy as np
import pytest
import time

from ditherzam.layers import (
    MaskRefinementError,
    MaskRefinementKind,
    SmartRefinementSpec,
    capped_refinement_shape,
    derive_refined_smart_mask,
    derive_refined_smart_preview,
    scaled_refinement_radius,
)
from ditherzam.masking.contracts import (
    InferenceIdentity,
    ModelIdentity,
    ProbabilityMap,
    source_identity,
)
from ditherzam.masking.settings import MaskTarget


def _rgba(shape):
    result = np.zeros((*shape, 4), dtype=np.uint8)
    result[..., 3] = 255
    return result


def _probability(values, rgba):
    identity = InferenceIdentity(
        source_identity(rgba), ModelIdentity("test", "1", "0" * 64),
        "v1", "subject")
    return ProbabilityMap(identity, np.asarray(values, dtype=np.float32))


def test_float_threshold_is_inclusive_at_exact_and_nextafter_boundaries():
    rgba = _rgba((1, 5))
    threshold = np.float32(0.5)
    values = np.array([[
        0.0, np.nextafter(threshold, np.float32(0)),
        threshold, np.nextafter(threshold, np.float32(1)), 1.0,
    ]], dtype=np.float32)
    mask = derive_refined_smart_mask(
        _probability(values, rgba),
        SmartRefinementSpec(MaskTarget.SUBJECT, 50), rgba=rgba)
    assert mask.tolist() == [[0, 0, 255, 255, 255]]


def test_target_background_whole_and_invert_are_exact():
    rgba = _rgba((1, 2))
    probability = _probability([[0.25, 0.75]], rgba)
    assert derive_refined_smart_mask(
        probability, SmartRefinementSpec(MaskTarget.BACKGROUND, 50),
        rgba=rgba).tolist() == [[255, 0]]
    assert derive_refined_smart_mask(
        probability,
        SmartRefinementSpec(MaskTarget.SUBJECT, 50, invert=True),
        rgba=rgba).tolist() == [[255, 0]]
    assert np.all(derive_refined_smart_mask(
        None, SmartRefinementSpec(MaskTarget.WHOLE_IMAGE, 0, invert=True),
        rgba=rgba) == 0)


def test_morphology_uses_invert_then_square_chebyshev_and_black_border():
    rgba = _rgba((3, 3))
    probability = _probability([
        [0, 0, 0], [0, 1, 0], [0, 0, 0]], rgba)
    grown = derive_refined_smart_mask(
        probability,
        SmartRefinementSpec(
            MaskTarget.SUBJECT, 50,
            morphology=MaskRefinementKind.GROW, morphology_radius=1),
        rgba=rgba)
    assert np.all(grown == 255)
    shrunk = derive_refined_smart_mask(
        _probability(np.ones((3, 3), np.float32), rgba),
        SmartRefinementSpec(
            MaskTarget.SUBJECT, 50,
            morphology=MaskRefinementKind.SHRINK, morphology_radius=1),
        rgba=rgba)
    assert shrunk.tolist() == [[0, 0, 0], [0, 255, 0], [0, 0, 0]]


@pytest.mark.parametrize(
    "kind", (MaskRefinementKind.NONE, MaskRefinementKind.GROW,
             MaskRefinementKind.SHRINK))
def test_radius_zero_is_identity_for_every_refinement(kind):
    rgba = _rgba((2, 3))
    probability = _probability([[0, 0.5, 1], [1, 0, 0.5]], rgba)
    actual = derive_refined_smart_mask(
        probability, SmartRefinementSpec(
            MaskTarget.SUBJECT, 50, morphology=kind,
            morphology_radius=0), rgba=rgba)
    assert actual.tolist() == [[0, 255, 255], [255, 0, 255]]


def test_feather_has_pillow_soft_edges_and_half_up_u8():
    rgba = _rgba((1, 5))
    probability = _probability([[0, 0, 1, 0, 0]], rgba)
    result = derive_refined_smart_mask(
        probability,
        SmartRefinementSpec(
            MaskTarget.SUBJECT, 50,
            feather_radius=1),
        rgba=rgba)
    assert result.dtype == np.uint8
    assert 0 < result[0, 1] < result[0, 2] < 255


def test_combined_order_is_invert_then_morph_then_feather_and_r64_is_bounded():
    rgba = _rgba((9, 9))
    values = np.zeros((9, 9), dtype=np.float32)
    values[4, 4] = 1.0
    probability = _probability(values, rgba)
    combined = derive_refined_smart_mask(
        probability,
        SmartRefinementSpec(
            MaskTarget.SUBJECT, 50, invert=True,
            morphology=MaskRefinementKind.SHRINK,
            morphology_radius=1, feather_radius=1),
        rgba=rgba)
    assert combined.dtype == np.uint8
    assert np.any((combined > 0) & (combined < 255))
    maximum = derive_refined_smart_mask(
        probability,
        SmartRefinementSpec(
            MaskTarget.SUBJECT, 50,
            morphology=MaskRefinementKind.GROW, morphology_radius=64),
        rgba=rgba)
    assert np.all(maximum == 255)


def test_derivation_observes_cancellation_between_stages():
    rgba = _rgba((5, 5))
    probability = _probability(np.eye(5, dtype=np.float32), rgba)
    checks = iter((False, True))
    from ditherzam.render import RenderCancelled
    with pytest.raises(RenderCancelled):
        derive_refined_smart_mask(
            probability,
            SmartRefinementSpec(
                MaskTarget.SUBJECT, 50,
                morphology=MaskRefinementKind.GROW, morphology_radius=1),
            rgba=rgba, is_cancelled=lambda: next(checks, True))


def test_preview_cap_never_upscales_and_radius_scaling_is_half_up():
    assert capped_refinement_shape((100, 200)) == (100, 200)
    assert capped_refinement_shape((1000, 2000)) == (360, 720)
    assert scaled_refinement_radius(1, (1440, 720), (720, 360)) == 1
    assert scaled_refinement_radius(1, (2160, 1080), (720, 360)) == 0
    rgba = _rgba((1000, 2000))
    probability = _probability(np.ones((1000, 2000), np.float32), rgba)
    assert derive_refined_smart_preview(
        probability, SmartRefinementSpec(MaskTarget.SUBJECT, 50),
        rgba=rgba).shape == (360, 720)


def test_exact_confirm_derivation_is_independent_of_preview_shape():
    rgba = _rgba((721, 2))
    probability = _probability(np.ones((721, 2), np.float32), rgba)
    spec = SmartRefinementSpec(
        MaskTarget.SUBJECT, 50, morphology=MaskRefinementKind.SHRINK,
        morphology_radius=1)
    preview = derive_refined_smart_preview(probability, spec, rgba=rgba)
    exact = derive_refined_smart_mask(probability, spec, rgba=rgba)
    assert preview.shape == (720, 2)
    assert exact.shape == (721, 2)
    assert exact[360, 0] == 0  # constant-black border at source resolution


def test_strict_radius_and_stale_probability_validation():
    with pytest.raises(MaskRefinementError):
        SmartRefinementSpec(
            MaskTarget.SUBJECT, 50, morphology_radius=65)
    rgba = _rgba((2, 2))
    other = _rgba((2, 2))
    other[0, 0, 0] = 1
    with pytest.raises(MaskRefinementError, match="does not match"):
        derive_refined_smart_mask(
            _probability(np.ones((2, 2)), other),
            SmartRefinementSpec(MaskTarget.SUBJECT, 50), rgba=rgba)


def _smart_controller(panel, *, previews):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController

    rgba = _rgba((4, 4))
    gray = np.zeros((4, 4), np.float32)
    probability = _probability(np.eye(4, dtype=np.float32), rgba)
    preset = {
        "adjustments": {"contrast": 50},
        "dither": {"style": "None", "scale": 1, "params": {}},
        "smart_mask": {
            "enabled": True, "target": "subject", "sensitivity": 50,
            "feather_px": 0, "expansion_px": 0, "invert": False,
            "outside": "original", "bake_fill": False,
        },
    }
    controller = LayersController(
        panel, registry, preset_provider=lambda: preset,
        source_provider=lambda: (gray, rgba, probability),
        cap_provider=lambda: 720, frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
        refinement_preview_sink=lambda pixels, token:
            previews.append((pixels.copy(), token)))
    controller.initialize_source_layer()
    return controller


def _wait_until(predicate, timeout=3.0):
    from PySide6.QtWidgets import QApplication
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def test_controller_transaction_preview_is_display_only_cancel_is_exact(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    previews = []
    panel = LayersPanel()
    controller = _smart_controller(panel, previews=previews)
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    before = controller.document
    assert controller.start_smart_refinement()
    token = controller._smart_refinement[0]
    _wait_until(lambda: bool(previews))
    assert previews[-1][0].shape == (4, 4)
    for radius in range(5):
        assert controller.update_smart_refinement(
            50, "none", 0, radius, False)
    _wait_until(lambda: len(previews) >= 2)
    assert controller._smart_refinement[2].feather_radius == 4
    assert controller.document is before
    assert controller.undo_label is None
    assert controller.cancel_smart_refinement(token)
    assert controller.document is before


def test_controller_exact_confirm_commits_first_mask_once_and_undoes(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _smart_controller(LayersPanel(), previews=[])
    requests = []
    monkeypatch.setattr(
        controller, "request_preview", lambda: requests.append(1))
    assert controller.start_smart_refinement()
    token = controller._smart_refinement[0]
    assert controller.confirm_smart_refinement(token)
    _wait_until(
        lambda: controller.document.layers[0].raster_mask is not None)
    assert controller.document.layers[0].raster_mask is not None
    assert controller.undo_label == "Edit Layer Mask"
    assert requests == [1]
    assert controller.undo()
    assert controller.document.layers[0].raster_mask is None


def test_controller_existing_mask_uses_only_replacement_proposal(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _smart_controller(LayersPanel(), previews=[])
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    assert controller.reveal_all_raster_mask(replace_existing=False)
    existing = controller.document.layers[0].raster_mask
    assert controller.start_smart_refinement()
    token = controller._smart_refinement[0]
    assert controller.update_smart_refinement(50, "none", 0, 0, True)
    assert controller.confirm_smart_refinement(token) is True
    _wait_until(lambda: controller.mask_replacement_proposal is not None)
    proposal = controller.mask_replacement_proposal
    assert proposal is not None
    assert controller.document.layers[0].raster_mask is existing
    assert controller.confirm_mask_replacement(proposal.token)
    assert controller.document.layers[0].raster_mask.revision == 1


def test_smart_replacement_rejects_document_generation_change(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _smart_controller(LayersPanel(), previews=[])
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    assert controller.reveal_all_raster_mask(replace_existing=False)
    assert controller.start_smart_refinement()
    token = controller._smart_refinement[0]
    assert controller.update_smart_refinement(50, "none", 0, 0, True)
    assert controller.confirm_smart_refinement(token)
    _wait_until(lambda: controller.mask_replacement_proposal is not None)
    proposal = controller.mask_replacement_proposal
    controller._generation += 1
    assert controller.confirm_mask_replacement(proposal.token) is False


def test_late_worker_terminals_cannot_clear_or_publish_newer_request(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_controller import _RefinementWorker
    from ditherzam.ui.layers_panel import LayersPanel

    previews = []
    controller = _smart_controller(LayersPanel(), previews=previews)
    monkeypatch.setattr(controller._refinement_pool, "start", lambda _w: None)
    assert controller.start_smart_refinement()
    old = controller._refinement_worker
    token, _authority, spec = controller._smart_refinement
    layer = controller.document.layers[0]
    newer = _RefinementWorker(
        token, layer.source.probability, spec, layer.source.rgba, exact=False)
    controller._refinement_workers[newer.request_id] = newer
    controller._refinement_worker = newer

    controller._refinement_cancelled(token, False, old.request_id)
    assert controller._refinement_worker is newer
    controller._refinement_failed(
        "late failure", token, False, old.request_id)
    assert controller._refinement_worker is newer
    controller._refinement_finished(
        np.full((4, 4), 17, np.uint8), token, False, old.request_id)
    assert controller._refinement_worker is newer
    assert previews == []
    assert old.request_id not in controller._refinement_workers
    assert newer.request_id in controller._refinement_workers


def test_controller_rejects_stale_document_and_panel_locks_transaction(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _smart_controller(panel, previews=[])
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    assert controller.start_smart_refinement()
    token, authority, spec = controller._smart_refinement
    controller._document = controller.document.select(0)
    assert controller.confirm_smart_refinement(token) is False
    assert controller.document.layers[0].raster_mask is None

    assert controller.start_smart_refinement()
    assert panel.smart_refinement.isVisible() is False  # parent is not shown
    assert panel.smart_threshold.accessibleName() == "Smart mask threshold"
    assert not panel.layer_list.isEnabled()
    panel._cancel_smart_refinement()
