import dataclasses
import tracemalloc

import numpy as np
import pytest

from ditherzam.layers.mask_brush import (
    BrushMode,
    BrushSettings,
    BrushStroke,
    DirtyRect,
    stamp_mask_brush,
)


def test_settings_and_dirty_rect_are_frozen_and_strict():
    settings = BrushSettings(3.0, 50, 100, BrushMode.REVEAL)
    assert settings.spacing == 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.size = 4.0
    with pytest.raises(ValueError):
        BrushSettings(True, 50, 100, BrushMode.REVEAL)
    for bad in (-1, 101, True, 2.5):
        with pytest.raises(ValueError):
            BrushSettings(3, bad, 100, BrushMode.REVEAL)
        with pytest.raises(ValueError):
            BrushSettings(3, 50, bad, BrushMode.REVEAL)
    assert DirtyRect(1, 2, 4, 6).union(DirtyRect(0, 3, 2, 8)) == DirtyRect(0, 2, 4, 8)


def test_hard_reveal_and_hide_have_hardcoded_golden_and_exact_dirty():
    reveal = np.zeros((5, 5), np.uint8)
    dirty = stamp_mask_brush(
        reveal, 2.5, 2.5, BrushSettings(3, 100, 100, BrushMode.REVEAL)
    )
    assert dirty == DirtyRect(1, 1, 4, 4)
    assert reveal.tolist() == [
        [0, 0, 0, 0, 0],
        [0, 255, 255, 255, 0],
        [0, 255, 255, 255, 0],
        [0, 255, 255, 255, 0],
        [0, 0, 0, 0, 0],
    ]
    assert stamp_mask_brush(
        reveal, 2.5, 2.5, BrushSettings(3, 100, 100, BrushMode.HIDE)
    ) == DirtyRect(1, 1, 4, 4)
    assert not reveal.any()


def test_soft_falloff_strength_and_saturation_are_exact():
    pixels = np.zeros((1, 3), np.uint8)
    dirty = stamp_mask_brush(
        pixels, 1.5, .5, BrushSettings(3, 0, 50, BrushMode.REVEAL)
    )
    # Distances 1, 0, 1 => tip coverage 85, 255, 85; strength => 43, 128, 43.
    assert pixels.tolist() == [[43, 128, 43]]
    assert dirty == DirtyRect(0, 0, 3, 1)
    assert stamp_mask_brush(
        pixels, 1.5, .5, BrushSettings(3, 0, 0, BrushMode.REVEAL)
    ) is None
    full = np.full((3, 3), 255, np.uint8)
    assert stamp_mask_brush(
        full, 1.5, 1.5, BrushSettings(1, 100, 100, BrushMode.REVEAL)
    ) is None


def test_intermediate_hardness_and_hide_falloff_have_hardcoded_rounding():
    reveal = np.zeros((1, 5), np.uint8)
    assert stamp_mask_brush(
        reveal, 2.5, .5, BrushSettings(5, 50, 100, BrushMode.REVEAL)
    ) == DirtyRect(0, 0, 5, 1)
    assert reveal.tolist() == [[102, 255, 255, 255, 102]]

    hide = np.full((1, 5), 200, np.uint8)
    assert stamp_mask_brush(
        hide, 2.5, .5, BrushSettings(5, 50, 50, BrushMode.HIDE)
    ) == DirtyRect(0, 0, 5, 1)
    # coverage [102,255,255,255,102] -> strength amount [51,128,128,128,51]
    # hide = old - half_up(old * amount / 255)
    assert hide.tolist() == [[160, 100, 100, 100, 160]]


def test_nonuniform_document_mapping_and_clipping():
    pixels = np.zeros((4, 5), np.uint8)
    dirty = stamp_mask_brush(
        pixels, 13.0, 24.0, BrushSettings(2, 100, 100, BrushMode.REVEAL),
        layer_x=10, layer_y=20, scale_x=2, scale_y=4,
    )
    # Pixel (1,0) has document center (13,22), outside; (1,1) is (13,26), outside.
    assert dirty is None
    dirty = stamp_mask_brush(
        pixels, 13.0, 22.0, BrushSettings(2, 100, 100, BrushMode.REVEAL),
        layer_x=10, layer_y=20, scale_x=2, scale_y=4,
    )
    assert dirty == DirtyRect(1, 0, 2, 1)
    assert pixels[0, 1] == 255
    clipped = np.zeros((2, 2), np.uint8)
    assert stamp_mask_brush(
        clipped, 0, 0, BrushSettings(4, 100, 100, BrushMode.REVEAL)
    ) == DirtyRect(0, 0, 2, 2)


@pytest.mark.parametrize(
    ("point", "edge"),
    [
        ((0, 2), "left"),
        ((5, 2), "right"),
        ((2, 0), "top"),
        ((2, 5), "bottom"),
    ],
)
def test_each_source_edge_clips_to_minimal_changed_bounds(point, edge):
    pixels = np.zeros((5, 5), np.uint8)
    dirty = stamp_mask_brush(
        pixels, *point, BrushSettings(4, 100, 100, BrushMode.REVEAL)
    )
    assert dirty is not None
    if edge == "left":
        assert dirty.x0 == 0 and dirty.x1 < 5
    elif edge == "right":
        assert dirty.x1 == 5 and dirty.x0 > 0
    elif edge == "top":
        assert dirty.y0 == 0 and dirty.y1 < 5
    else:
        assert dirty.y1 == 5 and dirty.y0 > 0


def test_fully_outside_source_is_exact_noop():
    pixels = np.arange(20, dtype=np.uint8).reshape(4, 5)
    before = pixels.copy()
    assert stamp_mask_brush(
        pixels, -100, 200, BrushSettings(20, 0, 100, BrushMode.HIDE)
    ) is None
    assert np.array_equal(pixels, before)


def _viewport_path_to_document(points, *, zoom, pan, proxy_scale):
    """Representative fit/zoom/pan/capped-preview affine inverse."""
    px, py = pan
    return [
        ((vx - px) / (zoom * proxy_scale), (vy - py) / (zoom * proxy_scale))
        for vx, vy in points
    ]


def _document_path_to_viewport(points, *, zoom, pan, proxy_scale):
    px, py = pan
    return [
        (x * zoom * proxy_scale + px, y * zoom * proxy_scale + py)
        for x, y in points
    ]


def test_fit_zoom_pan_and_preview_cap_affines_hit_identical_source_pixels():
    document_path = [(3.5, 3.5), (9.5, 5.5), (14.5, 9.5)]
    mappings = [
        dict(zoom=.4, pan=(17, 29), proxy_scale=.25),  # fit + capped proxy
        dict(zoom=1.0, pan=(0, 0), proxy_scale=1.0),
        dict(zoom=3.25, pan=(-83, 41), proxy_scale=.5),  # zoom + pan + cap
    ]
    results = []
    for mapping in mappings:
        view_path = _document_path_to_viewport(document_path, **mapping)
        recovered = _viewport_path_to_document(view_path, **mapping)
        pixels = np.zeros((16, 20), np.uint8)
        stroke = BrushStroke(
            pixels, BrushSettings(6, 50, 75, BrushMode.REVEAL)
        )
        stroke.start(*recovered[0])
        for point in recovered[1:]:
            stroke.add_point(*point)
        results.append((pixels, stroke.dirty_rect))
    first_pixels, first_dirty = results[0]
    assert first_dirty is not None and first_dirty.width >= 6
    for pixels, dirty in results[1:]:
        assert np.array_equal(pixels, first_pixels)
        assert dirty == first_dirty


def test_stroke_click_drag_residual_and_segmentation_are_deterministic():
    settings = BrushSettings(4, 100, 100, BrushMode.REVEAL)  # spacing 1
    click = np.zeros((3, 8), np.uint8)
    stroke = BrushStroke(click, settings)
    assert stroke.start(1.5, 1.5) == DirtyRect(0, 0, 4, 3)
    assert stroke.add_point(1.5, 1.5) is None

    whole = np.zeros((3, 8), np.uint8)
    split = np.zeros_like(whole)
    a = BrushStroke(whole, settings)
    b = BrushStroke(split, settings)
    a.start(.5, 1.5)
    assert a.add_point(6.7, 1.5) == DirtyRect(2, 0, 8, 3)
    b.start(.5, 1.5)
    b.add_point(2.2, 1.5)
    b.add_point(4.8, 1.5)
    b.add_point(6.7, 1.5)
    assert np.array_equal(whole, split)
    assert a.dirty_rect == b.dirty_rect == DirtyRect(0, 0, 8, 3)


def test_diagonal_residual_is_identical_when_collinear_events_are_split():
    settings = BrushSettings(3, 25, 63, BrushMode.HIDE)
    whole = np.full((16, 16), 211, np.uint8)
    split = whole.copy()
    a = BrushStroke(whole, settings)
    b = BrushStroke(split, settings)
    a.start(1.25, 2.0)
    a.add_point(13.25, 11.0)
    b.start(1.25, 2.0)
    # These are exact fractions along the same diagonal geometry.
    b.add_point(4.25, 4.25)
    b.add_point(8.25, 7.25)
    b.add_point(13.25, 11.0)
    assert np.array_equal(whole, split)
    assert a.dirty_rect == b.dirty_rect


@pytest.mark.parametrize("bad", [np.zeros((2, 2), np.float32), np.zeros((2, 2, 1), np.uint8)])
def test_rejects_invalid_buffers(bad):
    with pytest.raises(ValueError):
        BrushStroke(bad, BrushSettings(1, 100, 100, BrushMode.REVEAL))


def test_rejects_readonly_nonfinite_bool_scale_rotation_and_flip():
    readonly = np.zeros((2, 2), np.uint8)
    readonly.flags.writeable = False
    settings = BrushSettings(1, 100, 100, BrushMode.REVEAL)
    with pytest.raises(ValueError):
        BrushStroke(readonly, settings)
    pixels = np.zeros((2, 2), np.uint8)
    for kwargs in (
        {"document_x": float("nan"), "document_y": 0},
        {"document_x": True, "document_y": 0},
        {"document_x": 0, "document_y": 0, "scale_x": 0},
        {"document_x": 0, "document_y": 0, "rotation_degrees": 1},
        {"document_x": 0, "document_y": 0, "flip_x": True},
    ):
        with pytest.raises(ValueError):
            stamp_mask_brush(pixels, settings=settings, **kwargs)


@pytest.mark.parametrize(
    ("shape", "budget"),
    [((1080, 1920), 6 * 1024 * 1024), ((2160, 3840), 18 * 1024 * 1024)],
)
def test_large_brush_working_buffer_and_incremental_peak_are_accounted(shape, budget):
    # Exactly one caller-owned uint8 buffer. Scratch is measured separately because
    # the frozen 6/18 MiB budget is incremental beyond that required work buffer.
    pixels = np.zeros(shape, np.uint8)
    assert pixels.nbytes == shape[0] * shape[1]
    tracemalloc.start()
    try:
        stamp_mask_brush(
            pixels, shape[1] / 2, shape[0] / 2,
            BrushSettings(4000, 25, 50, BrushMode.REVEAL),
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= budget
    assert pixels.nbytes + peak <= pixels.nbytes + budget
