"""Colored Dither 'Brighten marks': marks lift toward white (screen) instead of
darkening (multiply), while the field stays the base color. Default (off) must be
byte-identical to the existing darken behavior."""
import numpy as np

from ditherzam.color.engine import ColorEngine
from ditherzam.color.palette import Palette


def _palette() -> Palette:
    return Palette("p", np.array(
        [[0, 0, 0], [255, 255, 255], [200, 60, 60], [60, 60, 200]], np.float32))


def _engine(brighten: bool, source_rgb, dither_amt: float = 100.0) -> ColorEngine:
    return ColorEngine(_palette(), "source", source_dither=dither_amt,
                       source_rgb=source_rgb, source_dither_brighten=brighten)


def _scene():
    h, w = 8, 8
    src = np.empty((h, w, 3), np.float32)
    src[:] = np.array([200, 60, 60], np.float32)     # solid base color
    dither = np.zeros((h, w), np.float32)            # left half = marks (0)
    dither[:, w // 2:] = 255.0                        # right half = field (255)
    return src, dither, w


def test_brighten_lifts_marks_above_darken():
    src, dither, w = _scene()
    dark = _engine(False, src).map(dither).astype(np.int32)
    bright = _engine(True, src).map(dither).astype(np.int32)
    mark_dark = dark[:, : w // 2].mean()
    mark_bright = bright[:, : w // 2].mean()
    assert mark_bright > mark_dark
    # At full influence a mark lifts to white under brighten, sinks to black under darken.
    assert bright[0, 0].tolist() == [255, 255, 255]
    assert dark[0, 0].tolist() == [0, 0, 0]


def test_brighten_leaves_field_as_base_color():
    src, dither, w = _scene()
    bright = _engine(True, src).map(dither)
    # Field (no mark) keeps the base color — the image is not altered there.
    assert bright[0, w - 1].tolist() == [200, 60, 60]


def test_off_is_byte_identical_to_engine_without_flag():
    src, dither, _ = _scene()
    with_flag_off = _engine(False, src).map(dither)
    without_flag = ColorEngine(_palette(), "source", source_dither=100.0,
                               source_rgb=src).map(dither)
    np.testing.assert_array_equal(with_flag_off, without_flag)


def test_zero_influence_ignores_brighten():
    src, dither, _ = _scene()
    off = _engine(False, src, dither_amt=0.0).map(dither)
    on = _engine(True, src, dither_amt=0.0).map(dither)
    np.testing.assert_array_equal(off, on)


def test_with_settings_round_trips_brighten_flag():
    src, _, _ = _scene()
    base = ColorEngine(_palette(), "source", source_rgb=src)
    assert base.source_dither_brighten is False
    derived = base.with_settings(source_dither_brighten=True)
    assert derived.source_dither_brighten is True
