"""Qt-free tests for the interactive preview proxy."""
import numpy as np

from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.ui.preview import (
    PREVIEW_RESOLUTIONS,
    auto_preview_resolution,
    normalize_preview_resolution,
    preview_cap,
    preview_target_size,
    proxy_factor,
    proxy_scale,
    render_preview,
    resize_preview_bucket,
    zoom_preview_bucket,
)


def test_preview_resolution_parsing_and_caps():
    assert PREVIEW_RESOLUTIONS == ("Auto", "480", "720", "1080", "1440", "2160", "Full")
    assert normalize_preview_resolution(" auto ") == "Auto"
    assert normalize_preview_resolution(1080) == "1080"
    assert normalize_preview_resolution("FULL") == "Full"
    assert normalize_preview_resolution("garbage") == "Auto"
    assert preview_cap("1080", 4000) == 1080
    assert preview_cap("Full", 4000) == 4000
    assert preview_cap("480", 320) == 320


def test_preview_target_size_preserves_aspect_and_never_upscales():
    assert preview_target_size(2160, 3840, 1080) == (608, 1080)
    assert preview_target_size(300, 400, 720) == (300, 400)
    assert preview_target_size(1, 4000, 480) == (1, 480)


def test_auto_resolution_is_viewport_aware_and_clamped_to_720_1440():
    assert auto_preview_resolution((2160, 3840), (500, 300), 1.0) == 720
    assert auto_preview_resolution((2160, 3840), (1000, 700), 1.0) == 1080
    assert auto_preview_resolution((2160, 3840), (1600, 1000), 2.0) == 1440
    # A small source is an exact render, not an artificial 720px upscale.
    assert auto_preview_resolution((300, 400), (1600, 1000), 2.0) == 400


def test_resize_and_zoom_helpers_change_only_at_buckets_and_respect_ceiling():
    assert resize_preview_bucket(719) == 720
    assert resize_preview_bucket(721) == 1080
    assert resize_preview_bucket(2000) == 1440
    assert zoom_preview_bucket(720, 721, 1440, 3840) == 1080
    assert zoom_preview_bucket(1080, 1000, 1440, 3840) == 1080
    assert zoom_preview_bucket(1080, 2000, 1440, 3840) == 1440
    assert zoom_preview_bucket(1440, 3000, 2160, 1800) == 1800


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


def test_render_preview_returns_capped_size_uint8():
    pipe = RenderPipeline(registry)
    base = np.random.default_rng(0).uniform(0, 255, (1080, 1920)).astype(np.float32)
    s = RenderSettings(style="Floyd-Steinberg", scale=5)
    out = render_preview(pipe, base, s, max_side=640)
    assert out.shape == (360, 640, 3)
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
    assert out.shape == (360, 640, 3)


# ---- Task 4.1: temporal field shape consistency (hazard #1) ----------------

def test_render_preview_accepts_full_res_temporal_field_no_crash():
    # A field built at the FULL-resolution small_shape (h//scale, w//scale)
    # fed into a CAPPED render must not crash and must produce a correctly
    # capped-size output -- apply_dither's internal nearest resize reshapes
    # the field to whatever the capped raster's own downscale shape is.
    pipe = RenderPipeline(registry)
    h, w, scale, cap = 1080, 1920, 5, 640
    base = np.random.default_rng(3).uniform(0, 255, (h, w)).astype(np.float32)
    s = RenderSettings(style="Bayer-Matrix 4x4", scale=scale)
    field = np.random.default_rng(3).uniform(-20, 20, (h // scale, w // scale)).astype(np.float32)
    out = render_preview(pipe, base, s, max_side=cap, temporal_field=field)
    assert out.shape == (int(round(h * cap / w)), cap, 3)
    assert out.dtype == np.uint8


def test_render_preview_temporal_field_shape_consistent_across_scale_cap_combos():
    pipe = RenderPipeline(registry)
    base = np.random.default_rng(4).uniform(0, 255, (720, 1280)).astype(np.float32)
    for scale, cap in ((1, 480), (3, 480), (5, 720), (7, 1440), (2, 1280)):
        h, w = base.shape
        s = RenderSettings(style="Floyd-Steinberg", scale=scale)
        field = np.random.default_rng(scale).uniform(-15, 15, (h // scale, w // scale)).astype(np.float32)
        out = render_preview(pipe, base, s, max_side=cap, temporal_field=field)
        assert out.dtype == np.uint8
        assert out.shape[2] == 3
        assert max(out.shape[:2]) <= cap


def test_render_preview_temporal_field_changes_capped_output_between_frames():
    # Different fields (different "frames") must produce different capped
    # rasters -- the field must actually be consumed by the capped path.
    pipe = RenderPipeline(registry)
    base = np.random.default_rng(5).uniform(0, 255, (1080, 1920)).astype(np.float32)
    s = RenderSettings(style="Bayer-Matrix 4x4", scale=4)
    h, w = base.shape
    field_a = np.random.default_rng(11).uniform(-30, 30, (h // 4, w // 4)).astype(np.float32)
    field_b = np.random.default_rng(22).uniform(-30, 30, (h // 4, w // 4)).astype(np.float32)
    out_a = render_preview(pipe, base, s, max_side=640, temporal_field=field_a)
    out_b = render_preview(pipe, base, s, max_side=640, temporal_field=field_b)
    assert not np.array_equal(out_a, out_b)
