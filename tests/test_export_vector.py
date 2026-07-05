import numpy as np
import pytest
from ditherzam.export.vector import raster_to_svg


def test_svg_header_and_size():
    a = np.array([[0, 255], [0, 255]], np.uint8)
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.startswith("<svg")
    assert 'width="2"' in svg and 'height="2"' in svg
    assert 'xmlns="http://www.w3.org/2000/svg"' in svg
    assert svg.rstrip().endswith("</svg>")


def test_background_colors_and_invert():
    a = np.array([[0, 255]], np.uint8)
    normal = raster_to_svg(a, threshold=128, invert=False)
    assert 'fill="#ffffff"' in normal            # white background
    assert 'fill="#000000"' in normal            # black filled rect
    inverted = raster_to_svg(a, threshold=128, invert=True)
    assert '<rect width="100%" height="100%" fill="#000000"/>' in inverted
    assert 'fill="#ffffff"' in inverted          # white filled rect


def test_vertical_run_merged_into_one_rect():
    a = np.zeros((4, 1), np.uint8)               # one filled column, 4 tall
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.count("<rect") == 2               # background + one merged run
    assert 'height="4"' in svg                    # the whole column is one rect


def test_gap_splits_column_into_two_runs():
    # column pattern filled/empty/filled/filled -> two separate runs, not merged across the gap
    a = np.array([[0], [255], [0], [0]], np.uint8)
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.count("<rect") == 3               # background + 2 runs
    assert 'y="0" width="1" height="1"' in svg   # first run: single pixel at top
    assert 'y="2" width="1" height="2"' in svg   # second run: 2 pixels merged


def test_runs_not_merged_across_columns():
    # two adjacent full columns must stay as two rects (horizontal merge is NOT done)
    a = np.zeros((3, 2), np.uint8)
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.count("<rect") == 3               # background + one rect per column
    assert 'x="0" y="0" width="1" height="3"' in svg
    assert 'x="1" y="0" width="1" height="3"' in svg


def test_empty_image_has_only_background():
    a = np.full((3, 3), 255, np.uint8)           # nothing below threshold
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.count("<rect") == 1               # background only


def test_threshold_boundary_is_exclusive():
    # value == threshold is NOT filled (strictly < threshold)
    a = np.array([[128]], np.uint8)
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.count("<rect") == 1               # background only


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        raster_to_svg(np.zeros((2, 2, 3), np.uint8), threshold=128)
