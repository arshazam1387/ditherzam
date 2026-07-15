import numpy as np

from ditherzam.dithering import registry
from ditherzam.dithering.pipeline import apply_dither


def _render(style, params):
    img = np.full((96, 128), 127.0, np.float32)
    return apply_dither(
        img, style=style, scale=1, luminance_threshold=50,
        params=params, registry=registry, levels=2,
    )


def _transitions(img):
    return (np.count_nonzero(img[:, 1:] != img[:, :-1]) +
            np.count_nonzero(img[1:] != img[:-1]))


def test_wave_line_spacing_default_preserves_legacy_output():
    native = {
        "wave_x_frequency_slider": 15, "wave_y_frequency_slider": 15,
        "special_wave_phase_slider": 0, "wave_x_weight_slider": 50,
        "wave_amplitude_slider": 128,
    }
    np.testing.assert_array_equal(
        _render("Wave", native),
        _render("Wave", {**native, "wave_line_spacing_slider": 100}),
    )


def test_increasing_spacing_reduces_wave_line_density():
    styles = {
        "Wave": {
            "wave_x_frequency_slider": 15, "wave_y_frequency_slider": 15,
            "special_wave_phase_slider": 0, "wave_x_weight_slider": 50,
            "wave_amplitude_slider": 128,
        },
        "Sine Wave Modulation": {
            "wave_frequency_slider": 5, "wave_threshold_slider": 10,
            "sine_y_frequency_slider": 10, "sine_phase_slider": 0,
            "sine_contrast_slider": 100,
        },
    }
    for style, native in styles.items():
        close = _render(style, {**native, "wave_line_spacing_slider": 100})
        far = _render(style, {**native, "wave_line_spacing_slider": 300})
        assert _transitions(far) < _transitions(close)
