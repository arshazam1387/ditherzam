import numpy as np
from ditherzam.animation.timeline import Timeline, Keyframe, ease
from ditherzam.render import RenderSettings


def test_ease_endpoints_exact():
    for k in ("linear", "ease-in", "ease-out", "ease-in-out"):
        assert abs(ease(0.0, k) - 0.0) < 1e-9
        assert abs(ease(1.0, k) - 1.0) < 1e-9


def test_ease_shapes():
    assert abs(ease(0.5, "linear") - 0.5) < 1e-9
    assert ease(0.5, "ease-in") < 0.5           # slow start
    assert ease(0.5, "ease-out") > 0.5          # fast start
    assert abs(ease(0.5, "ease-in-out") - 0.5) < 1e-9


def test_ease_clamps_out_of_range():
    assert ease(-1.0, "linear") == 0.0
    assert ease(2.0, "linear") == 1.0


def test_ease_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        ease(0.5, "bounce")


def test_linear_interpolation_midpoint():
    t = Timeline(length=10)
    t.add(Keyframe(0, "contrast", 0))
    t.add(Keyframe(10, "contrast", 100))
    assert abs(t.value_at("contrast", 5) - 50) < 1e-6


def test_value_clamps_outside_range():
    t = Timeline(length=10)
    t.add(Keyframe(2, "contrast", 30))
    t.add(Keyframe(8, "contrast", 90))
    assert t.value_at("contrast", 0) == 30       # before first key
    assert t.value_at("contrast", 20) == 90      # after last key


def test_eased_segment_uses_end_keyframe_kind():
    t = Timeline(length=10)
    t.add(Keyframe(0, "contrast", 0))
    t.add(Keyframe(10, "contrast", 100, kind="ease-in"))
    # ease-in at t=0.5 -> 0.25 -> value 25
    assert abs(t.value_at("contrast", 5) - 25) < 1e-6


def test_add_replaces_same_frame_same_field():
    t = Timeline(length=10)
    t.add(Keyframe(5, "scale", 3))
    t.add(Keyframe(5, "scale", 9))               # overwrite
    assert t.value_at("scale", 5) == 9


def test_settings_at_applies_and_rounds_int_fields():
    t = Timeline(length=10)
    t.add(Keyframe(0, "scale", 2))
    t.add(Keyframe(10, "scale", 12))
    s = t.settings_at(RenderSettings(), 5)
    assert s.scale == 7 and isinstance(s.scale, int)


def test_settings_at_leaves_unkeyed_fields_alone():
    t = Timeline(length=10)
    t.add(Keyframe(0, "contrast", 10))
    t.add(Keyframe(10, "contrast", 90))
    base = RenderSettings(style="Floyd-Steinberg", saturation=42)
    s = t.settings_at(base, 5)
    assert s.style == "Floyd-Steinberg" and s.saturation == 42
    assert abs(s.contrast - 50) < 1e-6
