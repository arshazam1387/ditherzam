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
