from __future__ import annotations

import numpy as np
import pytest

from ditherzam.dithering import registry


def _valid_source():
    gray = np.zeros((2, 3), np.float32)
    rgba = np.zeros((2, 3, 4), np.uint8)
    rgba[..., 3] = 255
    return gray, rgba


@pytest.mark.parametrize("cap", [True, 0, -1, 1.5, "480"])
def test_renderer_rejects_invalid_preview_cap(cap):
    from ditherzam.composition import Look, LookRenderer

    gray, rgba = _valid_source()
    renderer = LookRenderer(registry, gray, rgba)
    look = Look("plain", {"dither": {"style": "None", "scale": 1}})
    with pytest.raises(ValueError, match="target_max_side"):
        renderer.render(look, target_max_side=cap)


def test_renderer_rejects_source_dtype_coercion():
    from ditherzam.composition import LookRenderer

    gray, rgba = _valid_source()
    with pytest.raises(ValueError, match="float32"):
        LookRenderer(registry, gray.astype(np.float64), rgba)
    with pytest.raises(ValueError, match="uint8"):
        LookRenderer(registry, gray, rgba.astype(np.float32))


def test_renderer_owns_source_arrays_and_makes_them_read_only():
    from ditherzam.composition import LookRenderer

    gray, rgba = _valid_source()
    renderer = LookRenderer(registry, gray, rgba)
    gray[0, 0] = 200
    rgba[0, 0] = [1, 2, 3, 4]
    assert renderer.source_gray[0, 0] == 0
    assert np.array_equal(renderer.source_rgba[0, 0], [0, 0, 0, 255])
    assert renderer.source_gray.flags.writeable is False
    assert renderer.source_rgba.flags.writeable is False
