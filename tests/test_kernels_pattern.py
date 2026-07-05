import numpy as np
from ditherzam.dithering import registry

NAMES = [
    "Checkers - Small", "Checkers - Medium", "Checkers - Large", "Diamond",
    "Gridlock/Traffic", "Print Pattern", "Block Tone", "Stippling",
    "Crosshatch", "Dot Screen", "Line Screen",
]


def test_pattern_registered_binary_shape():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    for n in NAMES:
        e = registry.get_entry(n)
        assert e is not None, n
        param = 6 if e.param_sliders else 0
        out = e.func(img.copy(), param, 128.0)
        assert out.shape == img.shape and out.dtype == np.float32, n
        assert set(np.unique(out).tolist()) <= {0.0, 255.0}, n


def test_stippling_is_deterministic():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    a = registry.get_entry("Stippling").func(img.copy(), 5, 128.0)
    b = registry.get_entry("Stippling").func(img.copy(), 5, 128.0)
    np.testing.assert_array_equal(a, b)


def test_block_tone_white_stays_white():
    white = np.full((24, 24), 255.0, dtype=np.float32)
    out = registry.get_entry("Block Tone").func(white.copy(), 6, 128.0)
    assert np.all(out == 255.0)
