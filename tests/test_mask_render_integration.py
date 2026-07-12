import numpy as np

from ditherzam.masking.contracts import InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity
from ditherzam.masking.render import render_with_mask
from ditherzam.masking.settings import OutsideMode, SmartMaskSettings
from ditherzam.ui.render_request import MaskContext


def _context(probability, *, outside=OutsideMode.ORIGINAL):
    probability = np.asarray(probability, np.float32)
    rgba = np.zeros((*probability.shape, 4), np.uint8)
    rgba[..., :3] = (10, 20, 30); rgba[..., 3] = 255; rgba.flags.writeable = False
    source = source_identity(rgba)
    identity = InferenceIdentity(source, ModelIdentity("m", "1", "a" * 64), "p", "primary")
    prob = ProbabilityMap(identity, np.asarray(probability, np.float32))
    return MaskContext(source, rgba, prob,
                       SmartMaskSettings(enabled=True, feather_px=0, outside=outside))


def test_disabled_is_literal_direct_historical_call(monkeypatch):
    expected = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    calls = []
    monkeypatch.setattr("ditherzam.masking.render.derive_master_mask",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("mask work")))
    assert render_with_mask(lambda: calls.append(1) or expected, None) is expected
    assert calls == [1]


def test_enabled_renders_complete_branch_once_then_composites():
    context = _context([[1, 0], [0, 1]], outside=OutsideMode.BLACK)
    calls = []
    rendered = np.full((2, 2, 3), 200, np.uint8)
    result = render_with_mask(lambda: calls.append(1) or rendered, context)
    assert calls == [1]
    assert np.array_equal(result[..., 0], [[200, 0], [0, 200]])


def test_capped_composite_resizes_source_and_mask_to_render_geometry():
    context = _context(np.ones((4, 6), np.float32))
    out = render_with_mask(lambda: np.full((2, 3, 3), 77, np.uint8), context)
    assert out.shape == (2, 3, 3)
    assert np.all(out == 77)
