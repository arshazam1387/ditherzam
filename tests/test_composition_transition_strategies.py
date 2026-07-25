from __future__ import annotations

import numpy as np
import pytest

from ditherzam.composition.transitions import TRANSITIONS


def _frames(shape=(8, 8)):
    a = np.zeros((*shape, 4), dtype=np.uint8)
    a[..., 3] = 255
    b = np.full((*shape, 4), 255, dtype=np.uint8)
    return a, b


@pytest.mark.parametrize("name", ["crossfade", "dither-dissolve", "spatial-wipe"])
def test_rendering_transitions_clamp_t_and_preserve_exact_endpoints(name):
    a, b = _frames((2, 3))
    transition = TRANSITIONS[name]
    assert np.array_equal(transition.render(-4, a, b, None, {}), a)
    assert np.array_equal(transition.render(9, a, b, None, {}), b)
    with pytest.raises(ValueError, match="finite scalar"):
        transition.render(np.nan, a, b, None, {})


def test_bayer_dissolve_uses_fixed_8_by_8_thresholds():
    a, b = _frames()
    out = TRANSITIONS["dither-dissolve"].render(0.5, a, b, None, {})
    assert np.count_nonzero(out[..., 0] == 255) == 32
    assert np.array_equal(out[..., 0] == 255, out[..., 1] == 255)


def test_noise_dissolve_is_deterministic_for_a_seed():
    a, b = _frames((12, 12))
    transition = TRANSITIONS["dither-dissolve"]
    first = transition.render(0.5, a, b, None, {"mask": "noise"}, seed=17)
    second = transition.render(0.5, a, b, None, {"mask": "noise"}, seed=17)
    other = transition.render(0.5, a, b, None, {"mask": "noise"}, seed=18)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_linear_wipe_hard_selects_and_soft_wipe_blends():
    a, b = _frames((1, 3))
    transition = TRANSITIONS["spatial-wipe"]
    hard = transition.render(0.5, a, b, None, {"mode": "linear"})
    assert np.array_equal(hard[0, :, 0], [255, 255, 0])
    soft = transition.render(
        0.5, a, b, None, {"mode": "linear", "softness": 1.0}
    )
    assert np.array_equal(soft[0, :, 0], [255, 128, 0])


def test_right_linear_wipe_complements_coordinate():
    a, b = _frames((1, 3))
    out = TRANSITIONS["spatial-wipe"].render(
        0.25, a, b, None, {"mode": "linear", "direction": "right"}
    )
    assert np.array_equal(out[0, :, 0], [0, 0, 255])


def test_luma_wipe_requires_matching_gray_and_uses_clipped_luma():
    a, b = _frames((1, 3))
    transition = TRANSITIONS["spatial-wipe"]
    gray = np.array([[-20.0, 127.5, 300.0]], dtype=np.float32)
    out = transition.render(0.5, a, b, gray, {"mode": "luma"})
    assert np.array_equal(out[0, :, 0], [255, 255, 0])
    with pytest.raises(ValueError, match="base_gray"):
        transition.render(0.5, a, b, np.zeros((2, 3)), {"mode": "luma"})


def test_radial_wipe_starts_at_center_and_reaches_corners():
    a, b = _frames((3, 3))
    transition = TRANSITIONS["spatial-wipe"]
    out = transition.render(0.01, a, b, None, {"mode": "radial"})
    assert np.array_equal(out[..., 0] == 255, np.eye(3, dtype=bool)[::-1] & np.eye(3, dtype=bool))


def test_invalid_strategy_parameters_are_rejected():
    a, b = _frames()
    with pytest.raises(ValueError, match="mask"):
        TRANSITIONS["dither-dissolve"].render(0.5, a, b, None, {"mask": "bad"})
    with pytest.raises(ValueError, match="softness"):
        TRANSITIONS["spatial-wipe"].render(
            0.5, a, b, None, {"mode": "linear", "softness": 2}
        )
    with pytest.raises(ValueError, match="direction"):
        TRANSITIONS["spatial-wipe"].render(
            0.5, a, b, None, {"mode": "linear", "direction": "up"}
        )


def test_param_morph_is_context_marker():
    a, b = _frames()
    with pytest.raises(RuntimeError, match="param-morph requires compositor contexts"):
        TRANSITIONS["param-morph"].render(0.5, a, b, None, {})
