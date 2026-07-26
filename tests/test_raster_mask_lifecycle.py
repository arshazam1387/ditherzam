import numpy as np
import pytest


def _preset():
    return {
        "adjustments": {"contrast": 50},
        "dither": {"style": "None", "scale": 1, "params": {}},
    }


def _source(alpha=None):
    gray = np.zeros((3, 4), np.float32)
    rgba = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)
    rgba[..., 3] = (
        np.full((3, 4), 255, np.uint8) if alpha is None else alpha
    )
    return gray, rgba, None


def _controller(panel, source_provider=_source, **overrides):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController

    values = dict(
        preset_provider=_preset,
        source_provider=source_provider,
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
    )
    values.update(overrides)
    return LayersController(panel, registry, **values)


def _confirm_proposal(controller):
    proposal = controller.mask_replacement_proposal
    assert proposal is not None
    assert controller.confirm_mask_replacement(proposal.token)


def test_lifecycle_endpoints_and_from_transparency_preserve_source(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    unusual_alpha = np.array(
        [[0, 1, 17, 254], [255, 128, 63, 9], [4, 88, 199, 240]],
        dtype=np.uint8,
    )
    gray, rgba, probability = _source(unusual_alpha)
    original_rgba = rgba.copy()
    controller = _controller(
        LayersPanel(), source_provider=lambda: (gray, rgba, probability)
    )
    controller.initialize_source_layer()

    assert controller.reveal_all_raster_mask(replace_existing=True)
    assert np.all(controller.document.layers[0].raster_mask.pixels == 255)
    assert controller.hide_all_raster_mask(replace_existing=True) is False
    _confirm_proposal(controller)
    assert np.all(controller.document.layers[0].raster_mask.pixels == 0)
    assert controller.raster_mask_from_transparency(
        replace_existing=True) is False
    _confirm_proposal(controller)
    np.testing.assert_array_equal(
        controller.document.layers[0].raster_mask.pixels, unusual_alpha
    )
    np.testing.assert_array_equal(controller.document.layers[0].source.rgba, original_rgba)


def test_replacement_refuses_then_resets_metadata_and_advances_once(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    assert controller.hide_all_raster_mask(replace_existing=False)
    created = controller.document.layers[0].raster_mask
    assert (created.enabled, created.density, created.revision) == (True, 100, 0)

    assert controller.reveal_all_raster_mask(replace_existing=False) is False
    assert controller.cancel_mask_replacement(
        controller.mask_replacement_proposal.token)
    assert controller.document.layers[0].raster_mask is created
    assert controller.set_active_raster_mask_enabled(False)
    assert controller.set_active_raster_mask_density(23)
    before = controller.document.layers[0].raster_mask
    assert controller.reveal_all_raster_mask(replace_existing=True) is False
    _confirm_proposal(controller)
    after = controller.document.layers[0].raster_mask
    assert (after.enabled, after.density, after.revision) == (
        True, 100, before.revision + 1)


def test_selected_layer_isolation_and_duplicate_sharing(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    assert controller.hide_all_raster_mask(replace_existing=True)
    controller.duplicate_layer(0)
    shared = controller.document.layers[1].raster_mask
    assert controller.document.layers[0].raster_mask is shared

    controller.activate_layer(0)
    assert controller.reveal_all_raster_mask(replace_existing=True) is False
    _confirm_proposal(controller)
    assert np.all(controller.document.layers[0].raster_mask.pixels == 255)
    assert controller.document.layers[1].raster_mask is shared
    assert np.all(shared.pixels == 0)


def test_metadata_requires_mask_is_strict_and_shares_pixels(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    assert controller.set_active_raster_mask_enabled(False) is False
    assert controller.set_active_raster_mask_density(0) is False
    with pytest.raises(ValueError):
        controller.set_active_raster_mask_enabled(1)
    with pytest.raises(ValueError):
        controller.set_active_raster_mask_density(True)
    with pytest.raises(ValueError):
        controller.set_active_raster_mask_density(101)

    controller.reveal_all_raster_mask(replace_existing=True)
    first = controller.document.layers[0].raster_mask
    assert controller.set_active_raster_mask_enabled(False)
    second = controller.document.layers[0].raster_mask
    assert second.pixels is first.pixels
    assert second.revision == first.revision + 1
    assert controller.set_active_raster_mask_density(0)
    third = controller.document.layers[0].raster_mask
    assert third.pixels is first.pixels
    assert third.revision == second.revision + 1


def test_one_preview_and_thumbnail_invalidation_per_success_zero_on_refusal(
    qapp_fixture, monkeypatch
):
    from PySide6.QtGui import QPixmap
    from ditherzam.ui.layers_controller import _thumbnail_key
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    calls = []
    monkeypatch.setattr(controller, "request_preview", lambda: calls.append(True))
    old_key = _thumbnail_key(controller.document.layers[0])
    controller._thumbnail_cache[old_key] = QPixmap(1, 1)

    assert controller.hide_all_raster_mask(replace_existing=True)
    assert calls == [True]
    assert old_key not in controller._thumbnail_cache
    assert controller.reveal_all_raster_mask(replace_existing=False) is False
    assert calls == [True]


def test_replace_active_mask_requires_keyword_and_exact_shape(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    pixels = np.zeros((3, 4), np.uint8)
    with pytest.raises(TypeError):
        controller.replace_active_raster_mask(pixels, True)
    with pytest.raises(ValueError):
        controller.replace_active_raster_mask(
            np.zeros((4, 3), np.uint8), replace_existing=True)


def test_lifecycle_mutations_reuse_completed_look_cache(
    qapp_fixture, monkeypatch
):
    from ditherzam.layers import render_layer_document
    from ditherzam.ui.layers_panel import LayersPanel

    calls = []

    class StubLookRenderer:
        def __init__(self, registry, gray, rgba, **_kwargs):
            self.rgba = rgba

        def render(self, _look, **_kwargs):
            calls.append(True)
            return self.rgba[..., :3]

    monkeypatch.setattr(
        "ditherzam.layers.render.LookRenderer", StubLookRenderer)
    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    monkeypatch.setattr(controller, "request_preview", lambda: None)

    render_layer_document(
        controller.document, object(), look_cache=controller._look_cache)
    controller.hide_all_raster_mask(replace_existing=True)
    render_layer_document(
        controller.document, object(), look_cache=controller._look_cache)
    controller.reveal_all_raster_mask(replace_existing=True)
    _confirm_proposal(controller)
    render_layer_document(
        controller.document, object(), look_cache=controller._look_cache)
    controller.set_active_raster_mask_density(37)
    render_layer_document(
        controller.document, object(), look_cache=controller._look_cache)
    controller.set_active_raster_mask_enabled(False)
    render_layer_document(
        controller.document, object(), look_cache=controller._look_cache)

    assert len(calls) == 1


def test_mt08_edit_delete_and_history_are_single_selected_mutations(
    qapp_fixture, monkeypatch
):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    controller.reveal_all_raster_mask(replace_existing=False)
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    first = controller.document.layers[0].raster_mask
    assert controller.invert_active_raster_mask()
    inverted = controller.document.layers[0].raster_mask
    assert np.all(inverted.pixels == 0)
    assert inverted.revision == first.revision + 1
    assert (inverted.enabled, inverted.density) == (True, 100)
    assert controller.fill_active_raster_mask(255)
    assert np.all(controller.document.layers[0].raster_mask.pixels == 255)
    assert controller.delete_active_raster_mask()
    assert controller.document.layers[0].raster_mask is None
    assert controller.undo()
    assert controller.document.layers[0].raster_mask is not None


def test_mt08_import_first_confirm_replace_cancel_stale_and_export(
    qapp_fixture, tmp_path, monkeypatch
):
    from PIL import Image
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    monkeypatch.setattr(controller, "request_preview", lambda: None)
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.fromarray(np.full((3, 4), 17, np.uint8), "L").save(first_path)
    Image.fromarray(np.full((3, 4), 99, np.uint8), "L").save(second_path)
    assert controller.import_active_raster_mask(first_path)
    assert np.all(controller.document.layers[0].raster_mask.pixels == 17)
    assert controller.import_active_raster_mask(second_path) is False
    proposal = controller.mask_replacement_proposal
    assert proposal is not None
    assert controller.cancel_mask_replacement(proposal.token)
    assert np.all(controller.document.layers[0].raster_mask.pixels == 17)
    assert controller.import_active_raster_mask(second_path) is False
    proposal = controller.mask_replacement_proposal
    controller.set_active_raster_mask_density(90)
    assert controller.confirm_mask_replacement(proposal.token) is False
    assert np.all(controller.document.layers[0].raster_mask.pixels == 17)
    exported = tmp_path / "export.png"
    assert controller.export_active_raster_mask(exported)
    np.testing.assert_array_equal(
        np.asarray(Image.open(exported)), np.full((3, 4), 17, np.uint8))


def test_mt08_missing_mask_and_transform_guard_are_honest(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    allowed = [True]
    panel = LayersPanel()
    controller = _controller(panel, mutation_guard=lambda: allowed[0])
    controller.initialize_source_layer()
    assert controller.invert_active_raster_mask() is False
    assert panel.status_label.property("error") is True
    allowed[0] = False
    assert controller.reveal_all_raster_mask(replace_existing=False) is False
    assert controller.import_active_raster_mask("missing.png") is False


def test_mt08_smart_freeze_obeys_explicit_live_readiness(qapp_fixture):
    from ditherzam.masking.contracts import (
        InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity,
    )
    from ditherzam.ui.layers_panel import LayersPanel

    gray, rgba, _ = _source()
    identity = InferenceIdentity(
        source_identity(rgba),
        ModelIdentity("test", "1", "0" * 64),
        "v1",
        "subject",
    )
    probability = ProbabilityMap(
        identity, np.full(gray.shape, 0.9, np.float32))
    preset = _preset()
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
    authority = [True, ""]
    panel = LayersPanel()
    controller = _controller(
        panel,
        source_provider=lambda: (gray, rgba, probability),
        preset_provider=lambda: preset,
        smart_mask_readiness_provider=lambda _layer: tuple(authority),
    )
    controller.initialize_source_layer()
    assert controller.raster_mask_from_smart()
    controller.delete_active_raster_mask()

    for reason in (
        "Enable Smart Mask before freezing it.",
        "Smart Mask is not ready; wait for detection to finish.",
        "Smart Mask model is unavailable.",
        "Smart Mask result is stale for the selected layer.",
    ):
        authority[:] = [False, reason]
        assert controller.raster_mask_from_smart() is False
        assert panel.status_label.text() == reason
        assert controller.document.layers[0].raster_mask is None
