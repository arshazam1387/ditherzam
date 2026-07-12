import numpy as np
import pytest

from ditherzam.masking.contracts import InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity
from ditherzam.masking.settings import SmartMaskSettings
from ditherzam.render import RenderSettings
from ditherzam.ui.render_request import MaskContext, RenderKind, RenderRequest


def _context():
    rgba = np.zeros((2, 3, 4), np.uint8); rgba[..., 3] = 255; rgba.flags.writeable = False
    source = source_identity(rgba)
    model = ModelIdentity("m", "1", "a" * 64)
    probability = ProbabilityMap(InferenceIdentity(source, model, "p", "primary"),
                                 np.zeros((2, 3), np.float32))
    return MaskContext(source, rgba, probability, SmartMaskSettings(enabled=True))


def test_render_request_preserves_old_callers_with_no_mask_context():
    request = RenderRequest(0, RenderKind.DRAG, RenderSettings(), 1, 2, (2, 2))
    assert request.mask_context is None


def test_mask_context_is_one_frozen_snapshot():
    context = _context()
    request = RenderRequest(0, RenderKind.FULL, RenderSettings(), 1, 3, (3, 2),
                            mask_context=context)
    assert request.mask_context is context
    with pytest.raises(Exception):
        context.settings = SmartMaskSettings()


def test_mask_context_rejects_writable_source():
    context = _context()
    rgba = context.source_rgba.copy()
    with pytest.raises(ValueError, match="read-only"):
        MaskContext(context.source, rgba, context.probability, context.settings)
