from __future__ import annotations

import numpy as np
import pytest


def _preset(*, enabled, target="subject"):
    return {
        "dither": {
            "style": "None", "scale": 1, "depth": 2,
            "color_mapping": "match", "preview_disabled": False, "params": {},
        },
        "smart_mask": {
            "enabled": enabled, "target": target, "sensitivity": 50,
            "feather_px": 0, "expansion_px": 0, "invert": False,
            "outside": "transparent", "bake_fill": False,
        },
    }


def test_media_gate_rejects_masked_video_and_animation_before_work():
    from ditherzam.composition import (
        Look, UnsupportedCompositionMaskError, validate_composition_media,
    )

    masked = [Look("masked", _preset(enabled=True))]
    for kind in ("video", "animation"):
        with pytest.raises(UnsupportedCompositionMaskError, match="not supported"):
            validate_composition_media(kind, masked)


def test_media_gate_allows_stills_disabled_masks_and_whole_image():
    from ditherzam.composition import Look, validate_composition_media

    validate_composition_media("still", [Look("masked", _preset(enabled=True))])
    validate_composition_media("video", [Look("off", _preset(enabled=False))])
    validate_composition_media(
        "animation", [Look("whole", _preset(enabled=True, target="whole_image"))],
    )


def test_export_frame_png_preserves_rgba_and_jpeg_flattens_once(tmp_path, monkeypatch):
    from PIL import Image

    from ditherzam.composition.export import export_frame
    import ditherzam.composition.export as export_module

    rgba = np.array([[[255, 0, 0, 128], [0, 0, 255, 0]]], np.uint8)
    calls = {"flatten": 0}
    real_flatten = export_module.flatten_rgba_white

    def counted(value):
        calls["flatten"] += 1
        return real_flatten(value)

    monkeypatch.setattr(export_module, "flatten_rgba_white", counted)
    png = tmp_path / "frame.png"
    jpg = tmp_path / "frame.jpg"
    export_frame(rgba, png)
    assert np.array_equal(np.asarray(Image.open(png).convert("RGBA")), rgba)
    assert calls["flatten"] == 0
    export_frame(rgba, jpg)
    assert calls["flatten"] == 1
    assert Image.open(jpg).mode == "RGB"


def test_render_frames_is_streaming_and_checks_cancellation_between_frames():
    from ditherzam.composition import render_frames
    from ditherzam.render import RenderCancelled

    class FakeCompositor:
        def __init__(self):
            self.calls = []

        def render_frame(self, idx):
            self.calls.append(idx)
            return np.full((1, 1, 3), idx, np.uint8)

    compositor = FakeCompositor()
    cancelled = {"value": False}
    stream = render_frames(compositor, 3, is_cancelled=lambda: cancelled["value"])
    assert not isinstance(stream, list)
    assert next(stream)[0, 0, 0] == 0
    cancelled["value"] = True
    with pytest.raises(RenderCancelled):
        next(stream)
    assert compositor.calls == [0]
