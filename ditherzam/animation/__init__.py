"""ditherzam.animation — temporal noise, keyframe timeline, animated rendering."""
from __future__ import annotations

import os
import tempfile
from typing import Callable, Iterator, Optional

import numpy as np

from .temporal import PATTERNS, temporal_noise
from .timeline import Keyframe, Timeline, ease

__all__ = [
    "PATTERNS", "temporal_noise", "ease", "Keyframe", "Timeline",
    "render_animation", "export_animation",
]

_INACTIVE_PATTERNS = {None, "", "none", "None", "off"}


def render_animation(pipeline, base_gray_f32, base_settings, timeline,
                     temporal_pattern, temporal_amplitude, seed=0) -> Iterator[np.ndarray]:
    """Yield timeline.length rendered uint8[H, W, 3] frames."""
    h, w = base_gray_f32.shape[:2]
    active = (temporal_pattern not in _INACTIVE_PATTERNS
              and float(temporal_amplitude) > 0.0)
    for f in range(timeline.length):
        settings = timeline.settings_at(base_settings, f)
        factor = max(1, int(settings.scale))
        small_shape = (max(1, h // factor), max(1, w // factor))
        field = None
        if active:
            field = temporal_noise(f, small_shape, temporal_pattern,
                                   float(temporal_amplitude), int(seed))
        yield pipeline.render(base_gray_f32, settings, temporal_field=field)


def export_animation(pipeline, base_gray_f32, base_settings, timeline,
                     temporal_pattern, temporal_amplitude, out_path,
                     fps: int = 24, seed: int = 0,
                     progress: Optional[Callable[[int, int], None]] = None) -> str:
    """Render every frame to PNG, then encode to MP4 via Phase-7's assemble_video.

    Qt-free. Raises ImportError only if the Phase-7 video module is absent.
    """
    from PIL import Image
    from ..video.ffmpeg import assemble_video

    frames_dir = tempfile.mkdtemp(prefix="ditherzam_anim_")
    n = int(timeline.length)
    for i, frame in enumerate(render_animation(
            pipeline, base_gray_f32, base_settings, timeline,
            temporal_pattern, temporal_amplitude, seed)):
        Image.fromarray(frame).save(os.path.join(frames_dir, f"frame{i:06d}.png"))
        if progress is not None:
            progress(i + 1, n)
    assemble_video(frames_dir, fps, None, out_path)
    return out_path
