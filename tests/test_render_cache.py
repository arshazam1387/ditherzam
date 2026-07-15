"""render_cached must be bit-identical to render(), and must actually skip
recomputing stages whose inputs did not change."""
import numpy as np
from unittest import mock

import ditherzam.render as R
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.color.palette import Palette, builtin_palettes
from ditherzam.color.engine import ColorEngine
from ditherzam.effects.stack import EffectStack


def _base(seed=0, h=40, w=52):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 255, size=(h, w)).astype(np.float32)


def _fresh_pipeline(color=None, effects=None):
    return RenderPipeline(registry, color, effects)


def _duo():
    return ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")


def _stack():
    s = EffectStack()
    s.add("Chromatic Aberration", shift=2)
    s.add("Epsilon Glow", threshold=40.0, radius=3.0, intensity=1.0)
    return s


# ---- equivalence: cached output == fresh render() output, always -------------

def _assert_identical(pipe_cached, color, effects, base, settings, temporal=None):
    ref = _fresh_pipeline(color, effects).render(base, settings, temporal)
    got = pipe_cached.render_cached(base, settings, temporal)
    np.testing.assert_array_equal(got, ref)


def test_cached_matches_render_over_a_settings_sequence():
    color, effects = _duo(), _stack()
    pipe = _fresh_pipeline(color, effects)
    base = _base()

    seq = [
        RenderSettings(style="Floyd-Steinberg", scale=3, saturation=50),
        RenderSettings(style="Floyd-Steinberg", scale=3, saturation=50),   # full hit
        RenderSettings(style="Floyd-Steinberg", scale=3, saturation=80),   # sat only
        RenderSettings(style="Floyd-Steinberg", scale=3, saturation=20),   # sat only
        RenderSettings(style="Floyd-Steinberg", scale=3, saturation=20, invert=True),
        RenderSettings(style="Atkinson", scale=3, saturation=20),          # dither change
        RenderSettings(style="Atkinson", scale=3, saturation=20, contrast=70),
        RenderSettings(style="Atkinson", scale=3, saturation=20, luminance_threshold=70),
        RenderSettings(style="Atkinson", scale=5, saturation=20),          # scale change
        RenderSettings(style="None"),
    ]
    for s in seq:
        _assert_identical(pipe, color, effects, base, s)


def test_cached_matches_when_palette_and_effects_mutate():
    color, effects = _duo(), _stack()
    pipe = _fresh_pipeline(color, effects)
    base = _base(1)
    s = RenderSettings(style="Bayer-Matrix 4x4", scale=2, saturation=60)
    _assert_identical(pipe, color, effects, base, s)

    # swap palette on the SAME pipeline (UI rebuilds engines each render)
    pipe.color_engine = ColorEngine(builtin_palettes()["gameboy"], "nearest")
    _assert_identical(pipe, pipe.color_engine, effects, base, s)

    # change an effect param
    s2 = EffectStack(); s2.add("Chromatic Aberration", shift=5)
    pipe.effect_stack = s2
    _assert_identical(pipe, pipe.color_engine, s2, base, s)


def test_cached_matches_with_no_color_and_no_effects():
    pipe = _fresh_pipeline(None, None)
    base = _base(2)
    for sat in (50, 90, 10, 50):
        _assert_identical(pipe, None, None, base, RenderSettings(style="Floyd-Steinberg", saturation=sat))


def test_no_color_render_passes_gray_directly_to_fused_saturation(monkeypatch):
    pipe = _fresh_pipeline(None, None)
    base = _base(22)
    seen = []

    original = R.apply_saturation

    def record(gray_or_rgb, value, **kwargs):
        seen.append(gray_or_rgb.ndim)
        return original(gray_or_rgb, value, **kwargs)

    monkeypatch.setattr(R, "apply_saturation", record)
    settings = RenderSettings(style="None", saturation=73)
    pipe.render(base, settings)
    pipe.render_cached(base, settings)
    assert seen == [2, 2]


def test_cached_invalidates_on_new_base():
    color, effects = _duo(), _stack()
    pipe = _fresh_pipeline(color, effects)
    s = RenderSettings(style="Floyd-Steinberg", scale=2)
    _assert_identical(pipe, color, effects, _base(3), s)
    _assert_identical(pipe, color, effects, _base(4), s)   # different array object + content


def test_cached_matches_with_temporal_field():
    color, effects = _duo(), _stack()
    pipe = _fresh_pipeline(color, effects)
    base = _base(5)
    s = RenderSettings(style="Floyd-Steinberg", scale=2)
    for k in range(3):
        tf = (np.sin(np.linspace(0, 3.14 * (k + 1), base.size)).reshape(base.shape) * 40).astype(np.float32)
        _assert_identical(pipe, color, effects, base, s, temporal=tf)
    # a following non-temporal render must not reuse a temporal dither result
    _assert_identical(pipe, color, effects, base, s)


# ---- effectiveness: unchanged upstream stages are NOT recomputed -------------

def test_saturation_only_change_skips_dither_and_color():
    color, effects = _duo(), _stack()
    pipe = _fresh_pipeline(color, effects)
    base = _base(6)
    s1 = RenderSettings(style="Floyd-Steinberg", scale=3, saturation=50)
    pipe.render_cached(base, s1)          # warm the cache

    s2 = RenderSettings(style="Floyd-Steinberg", scale=3, saturation=90)
    with mock.patch.object(R, "apply_dither", wraps=R.apply_dither) as dith, \
         mock.patch.object(color, "map", wraps=color.map) as cmap, \
         mock.patch.object(R, "apply_saturation", wraps=R.apply_saturation) as sat, \
         mock.patch.object(effects, "apply", wraps=effects.apply) as fx:
        pipe.render_cached(base, s2)
    assert dith.call_count == 0           # dither reused
    assert cmap.call_count == 0           # color reused
    assert sat.call_count == 1            # saturation recomputed
    assert fx.call_count == 1             # effects recomputed (downstream of sat)


def test_effects_only_change_skips_everything_above():
    color = _duo()
    pipe = _fresh_pipeline(color, EffectStack())
    base = _base(7)
    s = RenderSettings(style="Floyd-Steinberg", scale=3, saturation=50)
    e1 = EffectStack(); e1.add("Chromatic Aberration", shift=1)
    pipe.effect_stack = e1
    pipe.render_cached(base, s)

    e2 = EffectStack(); e2.add("Chromatic Aberration", shift=9)
    pipe.effect_stack = e2
    with mock.patch.object(R, "apply_dither", wraps=R.apply_dither) as dith, \
         mock.patch.object(color, "map", wraps=color.map) as cmap, \
         mock.patch.object(R, "apply_saturation", wraps=R.apply_saturation) as sat:
        pipe.render_cached(base, s)
    assert dith.call_count == 0 and cmap.call_count == 0 and sat.call_count == 0
