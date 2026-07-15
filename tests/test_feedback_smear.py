import numpy as np
import pytest

from ditherzam.dithering import registry
from ditherzam.dithering.parameters import parameter_specs

DEFAULTS = (32, 2, 6, 20, 88, 0, 30, 100)
THR = np.float32(128.0)


def test_vnoise_range_smoothness_determinism():
    from ditherzam.dithering.kernels.special import _vnoise
    vals = [_vnoise(x * 0.13, 7.7, 3, 606) for x in range(200)]
    assert all(0.0 <= v < 1.0 for v in vals)
    # deterministic
    assert vals == [_vnoise(x * 0.13, 7.7, 3, 606) for x in range(200)]
    # smooth: neighboring samples move less than lattice-uncorrelated ones would
    steps = [abs(vals[i + 1] - vals[i]) for i in range(199)]
    assert max(steps) < 0.35
    # k separates fields
    assert any(_vnoise(x * 0.13, 7.7, 4, 606) != vals[x] for x in range(200))


def entry():
    e = registry.get_entry("Feedback Smear")
    assert e is not None
    return e


def _pylon(size=128):
    # Dark vertical mast with arms on white: crude reference-like subject.
    img = np.full((size, size), 235.0, dtype=np.float32)
    img[20:118, 58:70] = 20.0
    img[34:42, 34:94] = 20.0
    img[64:72, 40:88] = 20.0
    return img


def test_registered_with_exact_sliders():
    e = entry()
    assert e.category == "Special Effects" and e.dims == 2
    specs = {s.key: (s.label, s.minimum, s.maximum, s.default)
             for s in parameter_specs(e) if not s.key.startswith("creative_")}
    assert specs == {
        "fs_length_slider": ("Trail Length", 4, 64, 32),
        "fs_drift_slider": ("Drift", 1, 8, 2),
        "fs_noise_amount_slider": ("Noise Amount", 0, 24, 6),
        "fs_noise_scale_slider": ("Noise Scale", 1, 100, 20),
        "fs_decay_slider": ("Decay", 50, 100, 88),
        "fs_time_slider": ("Time", 0, 360, 0),
        "fs_erode_slider": ("Erode", 0, 100, 30),
        "fs_density_slider": ("Density", 0, 200, 100),
    }


def test_output_contract_and_determinism():
    img = _pylon()
    a = entry().func(img.copy(), DEFAULTS, THR)
    b = entry().func(img.copy(), DEFAULTS, THR)
    assert a.dtype == np.float32 and a.shape == img.shape
    assert set(np.unique(a)) <= {0.0, 255.0}
    np.testing.assert_array_equal(a, b)


def test_trails_accumulate_on_smear_side_and_fade():
    img = _pylon()
    out = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 0, 100), THR)
    near = out[20:118, 71:95]                    # just right of the mast
    far = out[20:118, 110:128]
    assert (near == 0.0).mean() > (far == 0.0).mean() > 0.0


def test_no_trails_at_zero_density():
    img = _pylon()
    out = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 0, 0), THR)
    expected = np.where(img < 128.0, 0.0, 255.0).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_time_scrubs_the_field():
    img = _pylon()
    a = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 0, 100), THR)
    b = entry().func(img.copy(), (32, 2, 6, 20, 88, 180, 0, 100), THR)
    assert np.any(a != b)


def test_erode_eats_subject_organically():
    img = _pylon()
    solid = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 0, 100), THR)
    eaten = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 70, 100), THR)
    body = slice(20, 118), slice(58, 70)
    assert (eaten[body] == 255.0).sum() > (solid[body] == 255.0).sum()


def test_default_output_not_collapsed():
    gradient = np.tile(np.linspace(0, 255, 96, dtype=np.float32), (96, 1))
    out = entry().func(gradient.copy(), DEFAULTS, THR)
    ink = (out == 0.0).mean()
    assert 0.02 < ink < 0.98


def test_not_a_duplicate_of_echo_smear():
    from tests.golden_harness import default_param
    img = _pylon()
    ours = entry().func(img.copy(), DEFAULTS, THR)
    other = registry.get_entry("Echo Smear")
    theirs = other.func(img.copy(), default_param(other), THR)
    assert np.any(ours != theirs)


def test_each_native_control_changes_pixels():
    img = _pylon()
    e = entry()
    base = e.func(img.copy(), DEFAULTS, THR)
    alternatives = (8, 6, 18, 70, 65, 180, 80, 30)
    assert len(e.param_sliders) == 8
    for index, value in enumerate(alternatives):
        params = list(DEFAULTS)
        params[index] = value
        changed = e.func(img.copy(), tuple(params), THR)
        assert np.any(changed != base), e.param_sliders[index]
