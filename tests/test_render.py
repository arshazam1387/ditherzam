import numpy as np
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine
from ditherzam.effects.stack import EffectStack


def test_render_grayscale_none_returns_rgb():
    p = RenderPipeline(registry)
    out = p.render(np.full((8, 8), 100.0, np.float32), RenderSettings())
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8


def test_render_none_style_is_gray_broadcast_to_rgb():
    # style "None", no color engine, blur uniform -> flat 100 across all 3 channels
    p = RenderPipeline(registry)
    out = p.render(np.full((8, 8), 100.0, np.float32),
                   RenderSettings(style="None"))
    assert out.shape == (8, 8, 3)
    assert np.all(out[..., 0] == out[..., 1]) and np.all(out[..., 1] == out[..., 2])


def test_render_applies_color_palette():
    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    p = RenderPipeline(registry, color_engine=eng)
    out = p.render(np.tile(np.linspace(0, 255, 8, np.float32), (8, 1)),
                   RenderSettings(style="Floyd-Steinberg", scale=1))
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert uniq <= {(0, 0, 0), (255, 255, 255)}


def test_render_effects_applied():
    s = EffectStack()
    s.add("Chromatic Aberration", shift=1)
    p = RenderPipeline(registry, effect_stack=s)
    out = p.render(np.full((8, 8), 128.0, np.float32), RenderSettings())
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8


def test_render_invert_is_last():
    # black base, style None, invert True -> pure white output
    p = RenderPipeline(registry)
    out = p.render(np.zeros((8, 8), np.float32), RenderSettings(invert=True))
    assert np.all(out == 255)


def test_render_preview_disabled_skips_dither():
    # preview_disabled makes apply_dither a passthrough; uniform stays uniform gray
    p = RenderPipeline(registry)
    out = p.render(np.full((8, 8), 77.0, np.float32),
                   RenderSettings(style="Floyd-Steinberg", scale=1, preview_disabled=True))
    assert np.all(out == 77)


def test_render_accepts_temporal_field_kwarg():
    # temporal_field defaults to None and is forwarded to apply_dither(threshold_field=...)
    p = RenderPipeline(registry)
    out = p.render(np.full((8, 8), 100.0, np.float32), RenderSettings(), temporal_field=None)
    assert out.shape == (8, 8, 3)
