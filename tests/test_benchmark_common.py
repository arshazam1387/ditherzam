import numpy as np

from benchmarks.common import heavy_effects


def test_heavy_effects_applies_to_small_rgb_image():
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    result = heavy_effects().apply(rgb)

    assert result.shape == rgb.shape
    assert result.dtype == rgb.dtype
