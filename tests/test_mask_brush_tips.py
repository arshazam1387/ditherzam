"""Creative tip and spacing tests for the Qt-free masking brush."""

import numpy as np
import pytest

import ditherzam.layers.mask_brush as mask_brush_module
from ditherzam.layers import BrushTip
from ditherzam.layers.mask_brush import (
    BrushMode, BrushSettings, BrushStroke, DirtyRect, stamp_mask_brush,
)


def test_round_defaults_preserve_existing_settings_and_pixels():
    settings = BrushSettings(3, 100, 100, BrushMode.REVEAL)
    assert settings.tip is BrushTip.ROUND
    assert settings.spacing_percent == 25
    assert settings.texture_seed == 0
    assert settings.spacing == 1.0
    pixels = np.zeros((5, 5), np.uint8)
    assert stamp_mask_brush(pixels, 2.5, 2.5, settings) == DirtyRect(1, 1, 4, 4)
    assert pixels.tolist() == [
        [0, 0, 0, 0, 0], [0, 255, 255, 255, 0],
        [0, 255, 255, 255, 0], [0, 255, 255, 255, 0],
        [0, 0, 0, 0, 0],
    ]


@pytest.mark.parametrize("percent, expected", [(1, 1.0), (10, 2.0), (25, 5.0), (100, 20.0)])
def test_spacing_percent_controls_center_distance(percent, expected):
    settings = BrushSettings(20, 100, 100, BrushMode.REVEAL, spacing_percent=percent)
    assert settings.spacing == expected


@pytest.mark.parametrize("bad", [0, -1, 101, True, 2.5])
def test_spacing_percent_is_strict(bad):
    with pytest.raises(ValueError, match="spacing_percent"):
        BrushSettings(8, 100, 100, BrushMode.REVEAL, spacing_percent=bad)


def test_square_and_diamond_have_exact_distinct_footprints():
    square = np.zeros((5, 5), np.uint8)
    diamond = np.zeros_like(square)
    assert stamp_mask_brush(square, 2.5, 2.5, BrushSettings(5, 100, 100, BrushMode.REVEAL, tip=BrushTip.SQUARE)) == DirtyRect(0, 0, 5, 5)
    assert stamp_mask_brush(diamond, 2.5, 2.5, BrushSettings(5, 100, 100, BrushMode.REVEAL, tip=BrushTip.DIAMOND)) == DirtyRect(0, 0, 5, 5)
    assert square.tolist() == [[255] * 5 for _ in range(5)]
    assert diamond.tolist() == [
        [0, 0, 255, 0, 0], [0, 255, 255, 255, 0],
        [255, 255, 255, 255, 255], [0, 255, 255, 255, 0],
        [0, 0, 255, 0, 0],
    ]


def test_texture_is_repeatable_seed_sensitive_and_binary_when_hard():
    def painted(seed):
        pixels = np.zeros((17, 17), np.uint8)
        dirty = stamp_mask_brush(pixels, 8.5, 8.5, BrushSettings(15, 100, 100, BrushMode.REVEAL, tip=BrushTip.TEXTURE, texture_seed=seed))
        return pixels, dirty
    first, first_dirty = painted(12345)
    repeated, repeated_dirty = painted(12345)
    changed, _ = painted(12346)
    assert first_dirty == repeated_dirty
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert 0 < np.count_nonzero(first) < first.size
    assert set(np.unique(first)).issubset({0, 255})


@pytest.mark.parametrize(
    "tip", [BrushTip.SQUARE, BrushTip.DIAMOND, BrushTip.TEXTURE]
)
def test_creative_tips_do_not_enter_round_only_native_seam(monkeypatch, tip):
    monkeypatch.setattr(mask_brush_module, "_brush", object())

    def reject_native(*_args, **_kwargs):
        raise AssertionError("creative brush tip entered round-only native seam")

    monkeypatch.setattr(
        mask_brush_module, "_stamp_mask_brush_native", reject_native
    )
    pixels = np.zeros((9, 9), dtype=np.uint8)
    dirty = stamp_mask_brush(
        pixels,
        4.5,
        4.5,
        BrushSettings(7, 100, 100, BrushMode.REVEAL, tip=tip),
    )
    assert dirty is not None
    assert pixels.any()


def test_seeded_texture_stroke_is_event_segmentation_independent():
    settings = BrushSettings(5, 70, 80, BrushMode.REVEAL, tip=BrushTip.TEXTURE, texture_seed=77, spacing_percent=40)
    whole = np.zeros((20, 24), np.uint8)
    split = np.zeros_like(whole)
    a, b = BrushStroke(whole, settings), BrushStroke(split, settings)
    a.start(2.5, 4.5); a.add_point(20.5, 13.5)
    b.start(2.5, 4.5); b.add_point(8.5, 7.5); b.add_point(14.5, 10.5); b.add_point(20.5, 13.5)
    assert np.array_equal(whole, split)
    assert a.dirty_rect == b.dirty_rect


@pytest.mark.parametrize("bad", [-1, 2**32, True, 1.5])
def test_texture_seed_is_strict_uint32(bad):
    with pytest.raises(ValueError, match="texture_seed"):
        BrushSettings(8, 100, 100, BrushMode.REVEAL, tip=BrushTip.TEXTURE, texture_seed=bad)


def test_tip_requires_enum():
    with pytest.raises(ValueError, match="tip"):
        BrushSettings(8, 100, 100, BrushMode.REVEAL, tip="square")
