import pathlib
import numpy as np
import pytest
from ditherzam.dithering import registry
from tests.golden_harness import STD_INPUT, default_param

GOLD = pathlib.Path(__file__).parent / "golden"
GOLD.mkdir(exist_ok=True)


@pytest.mark.parametrize("name", sorted(registry.list_dithers()))
def test_kernel_matches_golden(name):
    e = registry.get_entry(name)
    out = e.func(STD_INPUT.copy(), default_param(e), 128.0)
    assert out.dtype == np.float32, name
    assert out.shape == STD_INPUT.shape, name
    assert out.min() >= 0.0 and out.max() <= 255.0, name
    f = GOLD / f"{name.replace('/', '_')}.npy"
    if not f.exists():
        np.save(f, out)             # first run seeds the golden fixture
    np.testing.assert_array_equal(out, np.load(f))


def test_total_count_at_least_63():
    assert len(registry.list_dithers()) >= 63


def test_category_grouping_present():
    cats = registry.by_category()
    for expected in ("Error Diffusion", "Ordered Dither", "Patterned",
                     "Glitch Effects", "Special Effects"):
        assert expected in cats, expected


def test_param_order_matches_sliders():
    # Multi-slider kernels must accept a tuple whose length == len(param_sliders).
    multi = [n for n in registry.list_dithers()
             if len(registry.get_entry(n).param_sliders) > 1]
    for name in multi:
        e = registry.get_entry(name)
        arity = len(e.param_sliders)
        out = e.func(STD_INPUT.copy(), tuple(3 for _ in range(arity)), 128.0)
        assert out.shape == STD_INPUT.shape, name
