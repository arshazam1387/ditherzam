import numpy as np
import pytest

from ditherzam.masking.cache import MaskCaches
from ditherzam.masking.composite import MaskCompositeError, bake_outside_base
from ditherzam.masking.contracts import InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity
from ditherzam.masking.render import bake_fill_active, render_with_mask
from ditherzam.masking.settings import MaskSettingsError, OutsideMode, SmartMaskSettings
from ditherzam.ui.render_request import MaskContext


def _context(probability, *, outside=OutsideMode.ORIGINAL, **setting_changes):
    probability = np.asarray(probability, np.float32)
    rgba = np.zeros((*probability.shape, 4), np.uint8)
    rgba[..., :3] = (10, 20, 30); rgba[..., 3] = 255; rgba.flags.writeable = False
    source = source_identity(rgba)
    identity = InferenceIdentity(source, ModelIdentity("m", "1", "a" * 64), "p", "primary")
    prob = ProbabilityMap(identity, np.asarray(probability, np.float32))
    return MaskContext(source, rgba, prob,
                       SmartMaskSettings(enabled=True, feather_px=0, outside=outside,
                                         **setting_changes))


def test_bake_fill_defaults_off_and_rejects_non_bool():
    assert SmartMaskSettings().bake_fill is False
    with pytest.raises(MaskSettingsError):
        SmartMaskSettings(bake_fill=1)


def test_bake_fill_active_only_for_white_and_black():
    for outside, active in ((OutsideMode.WHITE, True), (OutsideMode.BLACK, True),
                            (OutsideMode.ORIGINAL, False), (OutsideMode.TRANSPARENT, False)):
        settings = SmartMaskSettings(enabled=True, outside=outside, bake_fill=True)
        assert bake_fill_active(settings) is active
        assert bake_fill_active(SmartMaskSettings(enabled=True, outside=outside)) is False


def test_bake_outside_base_blends_toward_fill():
    base = np.array([[100.0, 200.0]], np.float32)
    mask = np.array([[1.0, 0.0]], np.float32)
    out = bake_outside_base(base, mask, 255.0)
    assert out.tolist() == [[100.0, 255.0]]
    soft = bake_outside_base(np.array([[100.0]], np.float32),
                             np.array([[0.5]], np.float32), 255.0)
    assert soft.tolist() == [[177.5]]
    assert base.tolist() == [[100.0, 200.0]]  # input untouched


def test_bake_outside_base_rejects_shape_mismatch():
    with pytest.raises(MaskCompositeError):
        bake_outside_base(np.zeros((2, 2), np.float32), np.zeros((2, 3), np.float32), 0.0)


def test_baked_render_fills_base_and_skips_composite(monkeypatch):
    from ditherzam.masking import render as module
    monkeypatch.setattr(module, "composite_masked",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("composited")))
    context = _context([[1, 0], [0, 1]], outside=OutsideMode.WHITE, bake_fill=True)
    seen = {}

    def renderer(bake):
        baked = bake(np.array([[10, 20], [30, 40]], np.float32))
        seen["baked"] = baked
        return np.full((2, 2, 3), 7, np.uint8)

    result = render_with_mask(renderer, context, target_shape=(2, 2))
    assert np.all(result == 7)
    assert seen["baked"].tolist() == [[10.0, 255.0], [255.0, 40.0]]


def test_bake_ignored_for_original_outside():
    context = _context([[1, 0], [0, 1]], outside=OutsideMode.ORIGINAL, bake_fill=True)
    calls = []
    rendered = np.full((2, 2, 3), 200, np.uint8)
    result = render_with_mask(lambda *args: calls.append(args) or rendered, context)
    assert calls == [()]  # plain zero-argument renderer call, composite applied
    assert np.array_equal(result[..., 0], [[200, 10], [10, 200]])


def test_baked_and_composited_results_cache_separately():
    caches = MaskCaches(1024 * 1024)
    calls = {"baked": 0, "plain": 0}

    def renderer(bake=None):
        calls["baked" if bake is not None else "plain"] += 1
        return np.full((2, 2, 3), 90 if bake is not None else 200, np.uint8)

    baked_ctx = _context([[1, 0], [0, 1]], outside=OutsideMode.BLACK, bake_fill=True)
    plain_ctx = _context([[1, 0], [0, 1]], outside=OutsideMode.BLACK)
    kwargs = dict(caches=caches, rendered_identity="branch", target_shape=(2, 2))
    first = render_with_mask(renderer, baked_ctx, **kwargs)
    second = render_with_mask(renderer, baked_ctx, **kwargs)
    plain = render_with_mask(renderer, plain_ctx, **kwargs)
    assert calls == {"baked": 1, "plain": 1}
    assert np.array_equal(first, second)
    assert np.all(first == 90)
    assert not np.array_equal(first, plain)
    assert caches.metrics["composite_entries"] == 2


def test_render_preview_dithers_the_baked_fill():
    from ditherzam.dithering import registry
    from ditherzam.render import RenderPipeline, RenderSettings
    from ditherzam.ui.preview import render_preview

    base = np.full((4, 6), 100.0, np.float32)
    pipeline = RenderPipeline(registry)
    settings = RenderSettings(style="None", scale=1, invert=True)
    # Probability 0 everywhere: the subject mask is empty, the whole frame is
    # outside. Invert runs LAST in the pipeline, so a fill processed by the
    # pipeline comes out black while a post-composite stamp would stay white.
    baked_ctx = _context(np.zeros((4, 6), np.float32),
                         outside=OutsideMode.WHITE, bake_fill=True)
    stamped_ctx = _context(np.zeros((4, 6), np.float32), outside=OutsideMode.WHITE)
    baked = render_preview(pipeline, base, settings, 6, mask_context=baked_ctx)
    stamped = render_preview(pipeline, base, settings, 6, mask_context=stamped_ctx)
    assert np.all(baked == 0)
    assert np.all(stamped == 255)


def test_panel_bake_checkbox_follows_outside_mode(qapp_fixture):
    from ditherzam.ui.smart_mask_panel import SmartMaskPanel

    panel = SmartMaskPanel()
    panel.enabled_check.setChecked(True)
    assert not panel.bake_check.isEnabled()  # Original outside
    panel.outside_combo.setCurrentIndex(panel.outside_combo.findData(OutsideMode.WHITE))
    assert panel.bake_check.isEnabled()
    panel.bake_check.setChecked(True)
    assert panel.settings.bake_fill is True
    panel.outside_combo.setCurrentIndex(panel.outside_combo.findData(OutsideMode.TRANSPARENT))
    assert not panel.bake_check.isEnabled()
    panel.set_settings(SmartMaskSettings(enabled=True, outside=OutsideMode.BLACK,
                                         bake_fill=True))
    assert panel.bake_check.isChecked() and panel.bake_check.isEnabled()


def test_preset_round_trip_and_legacy_default():
    from ditherzam.presets import preset_to_settings, settings_to_preset
    from ditherzam.render import RenderSettings

    smart = SmartMaskSettings(enabled=True, outside=OutsideMode.WHITE, bake_fill=True)
    preset = settings_to_preset(RenderSettings(), smart_mask=smart)
    assert preset["smart_mask"]["bake_fill"] is True
    assert preset_to_settings(preset).smart_mask.bake_fill is True
    del preset["smart_mask"]["bake_fill"]  # legacy preset written before the option
    assert preset_to_settings(preset).smart_mask.bake_fill is False
