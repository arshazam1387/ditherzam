import numpy as np
from ditherzam.dithering.kernels import ordered as od
from tests.golden_harness import STD_INPUT


def test_ordered_levels2_identity():
    base = od._ordered(STD_INPUT.copy(), od._BAYER4)
    got = od._ordered(STD_INPUT.copy(), od._BAYER4, 2)
    np.testing.assert_array_equal(base, got)


def test_ordered_nlevel_tone_count():
    out = od._ordered(STD_INPUT.copy(), od._BAYER4, 4)
    uniq = np.unique(np.round(out))
    assert len(uniq) <= 4


def test_ordered_nlevel_preserves_mean_region():
    flat = np.full((32, 32), 120.0, np.float32)
    out = od._ordered(flat.copy(), od._BAYER4, 5)
    assert abs(out.mean() - 120.0) < 20.0
