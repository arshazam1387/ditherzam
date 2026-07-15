import numpy as np
import pytest

from ditherzam.dithering import registry
from ditherzam.dithering.pipeline import apply_dither


def _render(style, spacing=None):
    img = np.tile(np.linspace(20, 230, 32, dtype=np.float32), (24, 1))
    params = {}
    if spacing is not None:
        params["diffusion_line_spacing_slider"] = spacing
    return apply_dither(img, style=style, scale=1, luminance_threshold=50,
                        params=params, registry=registry, levels=2)


@pytest.mark.parametrize("style", [
    "Modulated Diffuse Y", "Modulated Diffuse X",
    "Contrast Aware Y", "Contrast Aware X",
])
def test_spacing_one_preserves_legacy_output(style):
    np.testing.assert_array_equal(_render(style), _render(style, 1))


@pytest.mark.parametrize("style,horizontal", [
    ("Modulated Diffuse Y", True), ("Modulated Diffuse X", False),
    ("Contrast Aware Y", True), ("Contrast Aware X", False),
])
def test_spacing_creates_real_white_gaps_between_lines(style, horizontal):
    out = _render(style, 4)
    if horizontal:
        assert np.all(out[1::4] == 255) and np.all(out[2::4] == 255)
        assert np.any(out[::4] == 0)
    else:
        assert np.all(out[:, 1::4] == 255) and np.all(out[:, 2::4] == 255)
        assert np.any(out[:, ::4] == 0)
