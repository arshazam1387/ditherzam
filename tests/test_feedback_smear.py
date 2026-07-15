import numpy as np
import pytest

from ditherzam.dithering import registry
from ditherzam.dithering.parameters import parameter_specs

DEFAULTS = (32, 2, 6, 20, 88, 0, 30, 100, 60, 48)
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


def _bar(h=128, w=256):
    # Full-height dark bar on white: silhouette edge is exactly x=60 in every
    # row, so line/mark positions are analytically derivable.
    img = np.full((h, w), 235.0, dtype=np.float32)
    img[:, 40:61] = 20.0
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
        "fs_lines_slider": ("Lines", 0, 100, 60),
        "fs_line_spacing_slider": ("Line Spacing", 8, 160, 48),
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
    out = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 0, 100, 0, 48), THR)
    near = out[20:118, 71:95]                    # just right of the mast
    far = out[20:118, 110:128]
    assert (near == 0.0).mean() > (far == 0.0).mean() > 0.0


def test_no_ink_at_zero_density_and_zero_lines():
    img = _pylon()
    out = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 0, 0, 0, 48), THR)
    expected = np.where(img < 128.0, 0.0, 255.0).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_time_scrubs_the_field():
    img = _pylon()
    a = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 0, 100, 0, 48), THR)
    b = entry().func(img.copy(), (32, 2, 6, 20, 88, 180, 0, 100, 0, 48), THR)
    assert np.any(a != b)


def test_erode_eats_subject_organically():
    img = _pylon()
    solid = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 0, 100, 0, 48), THR)
    eaten = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 70, 100, 0, 48), THR)
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
    alternatives = (8, 6, 18, 70, 65, 180, 80, 30, 0, 90)
    assert len(e.param_sliders) == 10
    for index, value in enumerate(alternatives):
        params = list(DEFAULTS)
        params[index] = value
        changed = e.func(img.copy(), tuple(params), THR)
        assert np.any(changed != base), e.param_sliders[index]


# ── Feedback lines: the traveling contour lattice (A/B iteration 2) ──
# All line tests kill trails (density=0), erosion (erode=0) and wobble
# (noise amount=0) so the only ink is the line lattice, whose geometry is
# analytically derivable from q = x - wgt*edge, wgt = exp(-(x-edge)/(2*L)).

_LINES_ONLY = (8, 2, 0, 20, 88, 0, 0, 0, 100, 16)


def _line_columns(out, x0, x1):
    # Columns in [x0, x1) whose full height is ink.
    cols = []
    for x in range(x0, x1):
        if (out[:, x] == 0.0).all():
            cols.append(x)
    return cols


def test_lines_are_continuous_full_height_verticals_far_from_subject():
    # Old kernel: density=0 means the background right of the bar is pure
    # white — no lines existed at all, so this fails on iteration 1.
    out = entry().func(_bar(), _LINES_ONLY, THR)
    far = _line_columns(out, 200, 256)
    assert len(far) >= 3
    # straight verticals on the L=16 lattice: consecutive line groups sit
    # one spacing apart (groups may be 1-2px wide)
    groups = [c for c in far if c - 1 not in far]
    gaps = [b - a for a, b in zip(groups, groups[1:])]
    assert gaps and all(g == 16 for g in gaps)


def test_time_marches_lines_into_the_subject():
    # t = time/360 shifts the lattice by t*L toward the subject (leftward).
    # At time=45, t=0.125 -> exactly 2px with L=16. Old kernel: no lines,
    # and time moved nothing when noise amount was 0.
    p0 = list(_LINES_ONLY)
    p45 = list(_LINES_ONLY)
    p45[5] = 45
    p90 = list(_LINES_ONLY)
    p90[5] = 90
    out0 = entry().func(_bar(), tuple(p0), THR)
    out45 = entry().func(_bar(), tuple(p45), THR)
    out90 = entry().func(_bar(), tuple(p90), THR)

    def centroids(out):
        cols = _line_columns(out, 200, 248)
        groups = []
        for c in cols:
            if groups and c - 1 in cols:
                groups[-1].append(c)
            else:
                groups.append([c])
        return [sum(g) / len(g) for g in groups]

    c0, c45, c90 = centroids(out0), centroids(out45), centroids(out90)
    assert c0 and c45 and c90
    for c in c45:
        assert any(abs((c + 2) - ref) <= 0.75 for ref in c0)
    for c in c90:
        assert any(abs((c + 4) - ref) <= 0.75 for ref in c0)


def test_lines_pile_up_into_waves_near_the_silhouette():
    # The warp gradient dq/dx = 1 + wgt*edge/falloff compresses the lattice
    # near the edge: line pitch shrinks from L to ~L/2.9 at x=61 while the
    # on-screen width stays ~constant, so ink density in the hugging band
    # is well above the far field. Old kernel: both bands are pure white.
    out = entry().func(_bar(), _LINES_ONLY, THR)
    near = (out[:, 61:85] == 0.0).mean()
    far = (out[:, 200:250] == 0.0).mean()
    assert far > 0.0
    assert near > 1.8 * far


def test_lines_stay_out_of_subject_and_left_background():
    out = entry().func(_bar(), _LINES_ONLY, THR)
    # subject untouched (erode 0): solid ink
    assert (out[:, 40:61] == 0.0).all()
    # left of the silhouette there are no lines
    assert (out[:, 0:40] == 255.0).all()


def test_zero_lines_slider_kills_the_lattice():
    p = list(_LINES_ONLY)
    p[8] = 0
    out = entry().func(_bar(), tuple(p), THR)
    assert (out[:, 61:] == 255.0).all()


def test_time_advances_feedback_age_of_trail_marks():
    # Trails only (lines off), noise amount 0 -> the walk is a pure
    # horizontal march: after k steps px = x - (k + t)*drift, so marks sit
    # where x - (k + t)*drift lands on the trailing edge {59, 60}. decay=100
    # makes every mark solid. At time=90 (t=0.25, drift=4) the whole mark
    # family shifts right by exactly (k + 0.25)*4 - 4k = 1px. Old kernel:
    # time changed nothing at noise amount 0 (walk ignored tshift).
    p0 = (32, 4, 0, 20, 100, 0, 0, 100, 0, 48)
    p90 = (32, 4, 0, 20, 100, 90, 0, 100, 0, 48)
    out0 = entry().func(_bar(), p0, THR)
    out90 = entry().func(_bar(), p90, THR)
    assert np.any(out0[:, 62:] == 0.0)
    np.testing.assert_array_equal(out90[:, 64:180], out0[:, 63:179])
