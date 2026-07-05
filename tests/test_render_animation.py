import numpy as np
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.animation import render_animation
from ditherzam.animation.timeline import Timeline, Keyframe


def _flat():
    return np.full((16, 16), 128.0, dtype=np.float32)


def test_yields_length_rgb_frames():
    p = RenderPipeline(registry)
    tl = Timeline(length=5)
    frames = list(render_animation(
        p, _flat(), RenderSettings(style="Bayer-Matrix 4x4", scale=1),
        tl, "static", 90.0, seed=2))
    assert len(frames) == 5
    for fr in frames:
        assert fr.shape == (16, 16, 3) and fr.dtype == np.uint8


def test_temporal_motion_visible():
    p = RenderPipeline(registry)
    tl = Timeline(length=3)
    frames = list(render_animation(
        p, _flat(), RenderSettings(style="Bayer-Matrix 4x4", scale=1),
        tl, "vhs-jitter", 90.0, seed=0))
    assert not np.array_equal(frames[0], frames[1])


def test_no_pattern_is_static_across_frames():
    p = RenderPipeline(registry)
    tl = Timeline(length=3)                     # no keyframes, no temporal pattern
    frames = list(render_animation(
        p, _flat(), RenderSettings(style="Bayer-Matrix 4x4", scale=1),
        tl, "none", 0.0))
    np.testing.assert_array_equal(frames[0], frames[2])


def test_timeline_animates_settings():
    p = RenderPipeline(registry)
    tl = Timeline(length=5)
    tl.add(Keyframe(0, "luminance_threshold", 10))
    tl.add(Keyframe(4, "luminance_threshold", 90))
    frames = list(render_animation(
        p, _flat(), RenderSettings(style="Floyd-Steinberg", scale=1),
        tl, "none", 0.0))
    assert not np.array_equal(frames[0], frames[-1])
