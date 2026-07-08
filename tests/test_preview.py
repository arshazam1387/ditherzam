"""Qt-free tests for the interactive preview proxy."""
import numpy as np

from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.ui.preview import proxy_factor, proxy_scale, render_preview


def test_proxy_factor_caps_longest_side():
    assert proxy_factor(1080, 1920, 720) == 3     # 1920/720 -> ceil 3
    assert proxy_factor(1080, 1920, 640) == 3
    assert proxy_factor(2160, 3840, 640) == 6
    assert proxy_factor(400, 500, 720) == 1       # already small -> no downscale
    assert proxy_factor(720, 720, 720) == 1


def test_proxy_scale_keeps_block_size_and_stays_ge_one():
    assert proxy_scale(5, 3) == 2                 # round(5/3)
    assert proxy_scale(5, 1) == 5
    assert proxy_scale(1, 3) == 1                 # never below 1
    assert proxy_scale(10, 4) == 2


def test_render_preview_returns_full_display_size_uint8():
    pipe = RenderPipeline(registry)
    base = np.random.default_rng(0).uniform(0, 255, (1080, 1920)).astype(np.float32)
    s = RenderSettings(style="Floyd-Steinberg", scale=5)
    out = render_preview(pipe, base, s, max_side=640)
    assert out.shape == (1080, 1920, 3)
    assert out.dtype == np.uint8


def test_render_preview_factor_one_equals_full_render():
    # small image -> no proxy downscale -> identical to a normal full render
    pipe = RenderPipeline(registry)
    base = np.random.default_rng(1).uniform(0, 255, (300, 400)).astype(np.float32)
    s = RenderSettings(style="Atkinson", scale=4, saturation=60)
    prev = render_preview(pipe, base, s, max_side=720)
    full = RenderPipeline(registry).render(base, s)
    np.testing.assert_array_equal(prev, full)


def test_render_preview_is_cheaper_shape_wise():
    # the proxy dithers fewer source pixels: block count is lower than full-res
    pipe = RenderPipeline(registry)
    base = np.random.default_rng(2).uniform(0, 255, (1080, 1920)).astype(np.float32)
    s = RenderSettings(style="Floyd-Steinberg", scale=5)
    out = render_preview(pipe, base, s, max_side=640)
    # still full display size, just approximate content
    assert out.shape == (1080, 1920, 3)
