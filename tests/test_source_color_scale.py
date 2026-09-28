import numpy as np

from ditherzam.color.engine import ColorEngine
from ditherzam.color.palette import Palette
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings


def test_source_colors_follow_dither_scale_blocks():
    """Color-by-source must use the same pixel blocks as the scaled dither."""
    height = width = 8
    source = np.zeros((height, width, 3), np.float32)
    source[:, 0::2] = [255, 0, 0]
    source[:, 1::2] = [0, 0, 255]
    palette = Palette.from_list("red-blue", [[255, 0, 0], [0, 0, 255]])
    engine = ColorEngine(
        palette,
        mode="source",
        source_rgb=source,
        source_dither=0,
    )
    pipeline = RenderPipeline(registry, engine)
    gray = np.full((height, width), 127, np.float32)

    settings_1 = RenderSettings(style="Bayer-Matrix 4x4", scale=1)
    settings_4 = RenderSettings(style="Bayer-Matrix 4x4", scale=4)
    scale_1 = pipeline.render(gray, settings_1)
    scale_4 = pipeline.render(gray, settings_4)

    cached_pipeline = RenderPipeline(registry, engine)
    cached_scale_1 = cached_pipeline.render_cached(gray, settings_1)
    cached_scale_4 = cached_pipeline.render_cached(gray, settings_4)

    assert not np.array_equal(scale_1, scale_4)
    np.testing.assert_array_equal(cached_scale_1, scale_1)
    np.testing.assert_array_equal(cached_scale_4, scale_4)
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            block = scale_4[y : y + 4, x : x + 4]
            np.testing.assert_array_equal(block, np.broadcast_to(block[0, 0], block.shape))
