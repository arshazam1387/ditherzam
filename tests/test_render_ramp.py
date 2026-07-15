import numpy as np
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.dithering.kernels import error_diffusion as ed
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine

R = ed.registry
GRAD = np.tile(np.linspace(0, 255, 128, np.float32), (128, 1))
QUAD = Palette.from_list("quad", [[0, 0, 0], [90, 0, 0], [0, 160, 0], [255, 255, 255]])


def _settings(**kw):
    base = dict(style="Floyd-Steinberg", scale=1, depth=4, color_mapping="match")
    base.update(kw)
    return RenderSettings(**base)


def test_depth_field_default():
    s = RenderSettings()
    assert s.depth == 2 and s.color_mapping == "match"


def test_render_depth_produces_multi_tone_color():
    eng = ColorEngine(QUAD, mode="ramp", depth=4, mapping="match")
    p = RenderPipeline(R, color_engine=eng)
    out = p.render(GRAD, _settings(depth=4))
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert 2 < len(uniq) <= 4


def test_render_cached_matches_render_with_depth():
    eng = ColorEngine(QUAD, mode="ramp", depth=5, mapping="interpolated")
    p = RenderPipeline(R, color_engine=eng)
    s = _settings(depth=5, color_mapping="interpolated")
    a = p.render(GRAD, s)
    b = p.render_cached(GRAD, s)
    np.testing.assert_array_equal(a, b)


def test_render_cached_invalidates_on_depth_change():
    eng = ColorEngine(QUAD, mode="ramp", depth=3, mapping="match")
    p = RenderPipeline(R, color_engine=eng)
    a = p.render_cached(GRAD, _settings(depth=3))
    eng.depth = 6
    b = p.render_cached(GRAD, _settings(depth=6))
    assert not np.array_equal(a, b)


def test_render_cached_invalidates_on_mapping_change_only():
    # depth stays fixed at 5 for both renders; only color_mapping differs, so
    # this isolates color_mapping's presence in render_cached's col_sig
    # (depth-change tests above would pass even if mapping were absent).
    eng = ColorEngine(QUAD, mode="ramp", depth=5, mapping="match")
    p = RenderPipeline(R, color_engine=eng)
    s_match = _settings(depth=5, color_mapping="match")
    s_glitch = _settings(depth=5, color_mapping="glitch")

    a = p.render_cached(GRAD, s_match)
    b = p.render_cached(GRAD, s_glitch)  # depth unchanged; only mapping differs
    assert not np.array_equal(a, b)  # col_sig must include color_mapping

    np.testing.assert_array_equal(b, p.render(GRAD, s_glitch))
    np.testing.assert_array_equal(p.render_cached(GRAD, s_match), p.render(GRAD, s_match))
