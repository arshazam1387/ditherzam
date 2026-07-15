import numpy as np
from ditherzam.animation.temporal import PATTERNS, temporal_noise


def test_exactly_nine_patterns():
    assert len(PATTERNS) == 9


def test_pattern_names_exact():
    assert PATTERNS == (
        "static", "scanline-drift", "interlace", "rolling-bar", "vhs-jitter",
        "blue-noise", "bayer-cycle", "plasma", "film-grain",
    )


def test_shape_dtype_and_determinism():
    a = temporal_noise(3, (8, 8), "static", 10.0, seed=0)
    b = temporal_noise(3, (8, 8), "static", 10.0, seed=0)
    assert a.shape == (8, 8) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)                       # deterministic


def test_all_patterns_deterministic_and_bounded():
    for p in PATTERNS:
        a = temporal_noise(4, (12, 10), p, 7.0, seed=3)
        b = temporal_noise(4, (12, 10), p, 7.0, seed=3)
        assert a.shape == (12, 10) and a.dtype == np.float32
        np.testing.assert_array_equal(a, b)                   # per-pattern determinism
        assert a.max() <= 7.0 + 1e-3, p                        # amplitude upper bound
        assert a.min() >= -7.0 - 1e-3, p                       # amplitude lower bound


def test_frames_differ_over_time():
    a = temporal_noise(0, (8, 8), "vhs-jitter", 10.0)
    b = temporal_noise(1, (8, 8), "vhs-jitter", 10.0)
    assert not np.array_equal(a, b)                            # animates


def test_analytic_patterns_animate():
    for p in ("scanline-drift", "interlace", "rolling-bar", "bayer-cycle", "plasma"):
        a = temporal_noise(0, (16, 16), p, 4.0)
        b = temporal_noise(3, (16, 16), p, 4.0)
        assert not np.array_equal(a, b), p


def test_different_seed_differs():
    a = temporal_noise(2, (8, 8), "film-grain", 5.0, seed=0)
    b = temporal_noise(2, (8, 8), "film-grain", 5.0, seed=1)
    assert not np.array_equal(a, b)


def test_unknown_pattern_raises():
    import pytest
    with pytest.raises(KeyError):
        temporal_noise(0, (4, 4), "nope", 1.0)
