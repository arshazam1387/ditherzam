from ditherzam.ui.viewport_math import (
    next_zoom, inertia_step, clamp_velocity, zoom_percent,
    viewport_device_size,
)


def test_zoom_in_multiplies_by_1_2():
    assert abs(next_zoom(1.0, +1) - 1.2) < 1e-9


def test_zoom_out_multiplies_by_0_8():
    assert abs(next_zoom(1.0, -1) - 0.8) < 1e-9


def test_zoom_in_capped_returns_none():
    assert next_zoom(90.0, +1) is None          # 90 * 1.2 = 108 > 100


def test_zoom_out_floor_returns_none():
    assert next_zoom(0.012, -1) is None          # 0.012 * 0.8 = 0.0096 < 0.01


def test_inertia_decays_and_clamps_range():
    pos, vel = inertia_step(50.0, 1000.0, 100.0, 0.95)
    assert 0.0 <= pos <= 100.0
    assert abs(vel) < 1000.0                      # friction shrank it


def test_inertia_clamps_low_bound():
    pos, _ = inertia_step(5.0, 1000.0, 100.0, 0.95)   # 5 - 16 = -11 -> 0
    assert pos == 0.0


def test_inertia_clamps_high_bound():
    pos, _ = inertia_step(95.0, -1000.0, 100.0, 0.95)  # 95 + 16 = 111 -> 100
    assert pos == 100.0


def test_clamp_velocity_scales_and_bounds():
    assert clamp_velocity(1000.0, scale=0.5, maximum=2000.0) == 500.0
    assert clamp_velocity(10000.0, scale=0.5, maximum=2000.0) == 2000.0
    assert clamp_velocity(-10000.0, scale=0.5, maximum=2000.0) == -2000.0


def test_zoom_percent_truncates():
    assert zoom_percent(1.239) == 123
    assert zoom_percent(0.01) == 1


def test_viewport_device_size_applies_dpr_and_rounds_up():
    assert viewport_device_size(639, 359, 1.5) == (959, 539)


def test_viewport_device_size_rejects_invalid_inputs():
    import pytest

    with pytest.raises(ValueError):
        viewport_device_size(0, 100, 1.0)
    with pytest.raises(ValueError):
        viewport_device_size(100, 100, 0.0)
