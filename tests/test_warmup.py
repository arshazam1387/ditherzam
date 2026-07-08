"""Warmup must run headlessly without error and leave renders working."""
import numpy as np

from ditherzam.warmup import warmup_render, start_warmup_thread
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings


def test_warmup_render_completes():
    warmup_render()  # should not raise


def test_warmup_render_tolerates_unknown_styles():
    warmup_render(styles=("No Such Style", "Floyd-Steinberg"))


def test_render_works_after_warmup():
    warmup_render()
    pipe = RenderPipeline(registry)
    out = pipe.render(np.full((16, 16), 100.0, np.float32),
                      RenderSettings(style="Floyd-Steinberg", scale=2))
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8


def test_start_warmup_thread_joins():
    t = start_warmup_thread()
    t.join(timeout=30)
    assert not t.is_alive()
