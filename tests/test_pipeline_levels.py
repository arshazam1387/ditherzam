import numpy as np
import pytest
from ditherzam.dithering import registry as reg
from ditherzam.dithering.pipeline import apply_dither
from ditherzam.dithering.kernels import error_diffusion  # noqa: F401 (registers kernels)
from ditherzam.dithering.kernels import ordered  # noqa: F401 (registers kernels)
from ditherzam.dithering.kernels import pattern  # noqa: F401 (registers kernels)
from ditherzam.dithering.kernels import glitch  # noqa: F401 (registers kernels)
from ditherzam.dithering.kernels import special  # noqa: F401 (registers kernels)

R = error_diffusion.registry  # the module-level shared registry instance


def test_entry_has_supports_levels_flag():
    e = R.get_entry("Floyd-Steinberg")
    assert e.supports_levels is True
    e2 = R.get_entry("Ostromukhov")
    assert e2.supports_levels is False


def test_apply_dither_passes_levels_to_capable_kernel():
    g = np.tile(np.linspace(0, 255, 64, np.float32), (32, 1))
    out2 = apply_dither(g, style="Floyd-Steinberg", scale=1,
                        luminance_threshold=50, params={}, registry=R, levels=2)
    out4 = apply_dither(g, style="Floyd-Steinberg", scale=1,
                        luminance_threshold=50, params={}, registry=R, levels=4)
    assert len(np.unique(np.round(out2))) <= 2
    assert 2 < len(np.unique(np.round(out4))) <= 4


def test_apply_dither_levels_promotes_incapable_kernel():
    # A binary-only kernel (supports_levels False) is promoted to multi-tone by
    # the pipeline so a colour palette can span its whole tonal range instead of
    # collapsing to 2 colours. See the "styles only show one/two colours" bug.
    g = np.tile(np.linspace(0, 255, 64, np.float32), (32, 1))
    out2 = apply_dither(g, style="Dot Screen", scale=1, luminance_threshold=50,
                        params={}, registry=R, levels=2)
    out6 = apply_dither(g, style="Dot Screen", scale=1, luminance_threshold=50,
                        params={}, registry=R, levels=6)
    assert len(np.unique(np.round(out2))) <= 2      # default stays binary
    assert 2 < len(np.unique(np.round(out6))) <= 6  # higher depth -> more tones


def test_apply_dither_levels_two_unchanged_for_incapable_kernel():
    # The default depth (levels<=2) must be byte-identical to the raw binary
    # kernel, so existing goldens / look at default settings are untouched.
    from ditherzam.dithering.pipeline import _build_param, _luminance_to_255
    g = np.tile(np.linspace(0, 255, 64, np.float32), (32, 1))
    entry = R.get_entry("Dot Screen")
    piped = apply_dither(g, style="Dot Screen", scale=1, luminance_threshold=50,
                         params={}, registry=R, levels=2)
    raw = entry.func(g.astype(np.float32), _build_param(entry, {}),
                     _luminance_to_255(50))
    np.testing.assert_array_equal(piped, raw)


@pytest.mark.parametrize("style", R.list_dithers())
def test_apply_dither_levels_safe_for_every_style(style):
    """Guards against a supports_levels flag mismatched with the kernel's real
    signature: every registered style must accept levels=4 without raising."""
    g = np.tile(np.linspace(0, 255, 64, np.float32), (32, 1))
    apply_dither(g, style=style, scale=1, luminance_threshold=50,
                params={}, registry=R, levels=4)
