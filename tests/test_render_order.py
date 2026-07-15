import numpy as np
from unittest import mock

import ditherzam.render as R
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine
from ditherzam.effects.stack import EffectStack

FROZEN_ORDER = (
    "contrast", "midtones", "highlights", "blur", "dither",
    "color", "saturation", "effects", "invert",
)


def _recorder(calls, tag):
    def _rec(*args, **kwargs):
        calls.append(tag)
        return mock.DEFAULT           # fall through to wraps -> real function result
    return _rec


def test_stage_order_constant_is_frozen():
    # The declared contract must match the spec (§8.1 + color/effects/saturation insert).
    assert RenderPipeline.STAGE_ORDER == FROZEN_ORDER


def test_render_calls_stages_in_frozen_order():
    calls: list[str] = []

    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    stack = EffectStack()
    stack.add("Chromatic Aberration", shift=1)
    p = RenderPipeline(registry, color_engine=eng, effect_stack=stack)

    base = np.tile(np.linspace(0, 255, 8, np.float32), (8, 1))
    settings = RenderSettings(
        style="Floyd-Steinberg", scale=1, invert=True,
        contrast=60, midtones=60, highlights=60, blur=60, saturation=60,
    )

    with mock.patch.object(R, "apply_contrast", side_effect=_recorder(calls, "contrast"), wraps=R.apply_contrast), \
         mock.patch.object(R, "apply_midtones", side_effect=_recorder(calls, "midtones"), wraps=R.apply_midtones), \
         mock.patch.object(R, "apply_highlights", side_effect=_recorder(calls, "highlights"), wraps=R.apply_highlights), \
         mock.patch.object(R, "apply_blur", side_effect=_recorder(calls, "blur"), wraps=R.apply_blur), \
         mock.patch.object(R, "apply_dither", side_effect=_recorder(calls, "dither"), wraps=R.apply_dither), \
         mock.patch.object(R, "apply_saturation", side_effect=_recorder(calls, "saturation"), wraps=R.apply_saturation), \
         mock.patch.object(R, "apply_invert", side_effect=_recorder(calls, "invert"), wraps=R.apply_invert), \
         mock.patch.object(eng, "map", side_effect=_recorder(calls, "color"), wraps=eng.map), \
         mock.patch.object(stack, "apply", side_effect=_recorder(calls, "effects"), wraps=stack.apply):
        out = p.render(base, settings)

    assert calls == list(FROZEN_ORDER)
    assert tuple(calls) == RenderPipeline.STAGE_ORDER
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8


def test_invert_is_strictly_last():
    calls: list[str] = []
    p = RenderPipeline(registry)
    settings = RenderSettings(style="None", invert=True)
    with mock.patch.object(R, "apply_invert", side_effect=_recorder(calls, "invert"), wraps=R.apply_invert), \
         mock.patch.object(R, "apply_saturation", side_effect=_recorder(calls, "saturation"), wraps=R.apply_saturation):
        p.render(np.zeros((8, 8), np.float32), settings)
    assert calls[-1] == "invert"                     # invert runs after saturation


def test_color_before_saturation_before_effects():
    calls: list[str] = []
    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    stack = EffectStack()
    stack.add("Blur", radius=0)
    p = RenderPipeline(registry, color_engine=eng, effect_stack=stack)
    with mock.patch.object(eng, "map", side_effect=_recorder(calls, "color"), wraps=eng.map), \
         mock.patch.object(R, "apply_saturation", side_effect=_recorder(calls, "saturation"), wraps=R.apply_saturation), \
         mock.patch.object(stack, "apply", side_effect=_recorder(calls, "effects"), wraps=stack.apply):
        p.render(np.full((8, 8), 100.0, np.float32), RenderSettings(style="None"))
    assert calls == ["color", "saturation", "effects"]
