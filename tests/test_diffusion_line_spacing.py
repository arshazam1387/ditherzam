import numpy as np
import pytest

from ditherzam.dithering import registry
from ditherzam.dithering.pipeline import apply_dither


def _render(style, spacing=None):
    img = np.tile(np.linspace(20, 230, 256, dtype=np.float32), (48, 1))
    params = {}
    if spacing is not None:
        params["diffusion_line_spacing_slider"] = spacing
    return apply_dither(img, style=style, scale=1, luminance_threshold=50,
                        params=params, registry=registry, levels=2)


@pytest.mark.parametrize("style", [
    "Modulated Diffuse Y", "Modulated Diffuse X",
    "Contrast Aware Y", "Contrast Aware X",
    "Uniform Modulation Y", "Uniform Modulation X",
])
def test_spacing_default_preserves_legacy_output(style):
    np.testing.assert_array_equal(_render(style), _render(style, 100))


@pytest.mark.parametrize("style", [
    "Modulated Diffuse Y", "Modulated Diffuse X",
    "Contrast Aware Y", "Contrast Aware X",
    "Uniform Modulation Y", "Uniform Modulation X",
])
def test_spacing_spreads_marks_apart(style):
    """Raising spacing divides the ink debt, so the emergent lines land
    farther apart: ink coverage falls monotonically but never vanishes."""
    inks = [int(np.count_nonzero(_render(style, s) == 0)) for s in (100, 400, 800)]
    assert inks[0] > inks[1] > inks[2]
    assert inks[2] > 0
