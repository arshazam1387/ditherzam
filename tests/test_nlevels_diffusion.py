import numpy as np
from ditherzam.dithering.nlevels import quantize_to_levels
from ditherzam.dithering.kernels import error_diffusion as ed
from tests.golden_harness import STD_INPUT


def test_quantize_levels_values():
    assert quantize_to_levels(0.0, 4) == 0.0
    assert quantize_to_levels(255.0, 4) == 255.0
    # 4 levels -> {0, 85, 170, 255}
    assert abs(quantize_to_levels(80.0, 4) - 85.0) < 1e-3
    assert abs(quantize_to_levels(200.0, 4) - 170.0) < 1e-3


def test_levels2_identical_to_current_fs():
    # levels<=2 must reproduce the exact current output at any threshold
    for thr in (60.0, 127.5, 200.0):
        base = ed._floyd_steinberg(STD_INPUT.copy(), thr)          # current 2-arg core is now 3-arg default
        got = ed._floyd_steinberg(STD_INPUT.copy(), thr, 2)
        np.testing.assert_array_equal(base, got)


def test_nlevel_fs_limits_distinct_tones():
    out = ed._floyd_steinberg(STD_INPUT.copy(), 127.5, 4)
    uniq = np.unique(np.round(out))
    assert len(uniq) <= 4


def test_nlevel_fs_preserves_mean():
    flat = np.full((32, 32), 120.0, np.float32)
    out = ed._floyd_steinberg(flat.copy(), 127.5, 5)
    assert abs(out.mean() - 120.0) < 8.0


def test_diffuse_core_levels2_identity():
    # jjn uses _diffuse; verify the shared core keeps identity at levels=2
    from ditherzam.dithering.kernels.error_diffusion import _JJN_OFF, _JJN_W, _JJN_DIV
    base = ed._diffuse(STD_INPUT.copy(), 127.5, _JJN_OFF, _JJN_W, _JJN_DIV)
    got = ed._diffuse(STD_INPUT.copy(), 127.5, _JJN_OFF, _JJN_W, _JJN_DIV, 2)
    np.testing.assert_array_equal(base, got)
