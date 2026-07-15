import numpy as np

from ditherzam.dithering import registry
from ditherzam.masking.contracts import InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity
from ditherzam.masking.settings import OutsideMode, SmartMaskSettings
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.ui.convert import numpy_to_qimage, qimage_to_numpy
from ditherzam.ui.preview import render_preview
from ditherzam.ui.render_request import MaskContext


def _context(shape):
    rgba = np.zeros((*shape, 4), np.uint8)
    rgba[..., :3] = (12, 24, 36)
    rgba[..., 3] = 255
    rgba.flags.writeable = False
    source = source_identity(rgba)
    identity = InferenceIdentity(
        source, ModelIdentity("m", "1", "a" * 64), "p", "primary")
    probability = ProbabilityMap(identity, np.ones(shape, np.float32))
    settings = SmartMaskSettings(
        enabled=True, feather_px=0, outside=OutsideMode.BLACK)
    return MaskContext(source, rgba, probability, settings)


def test_rgba_qimage_conversion_owns_pixels_and_preserves_alpha():
    rgba = np.zeros((2, 3, 4), np.uint8)
    rgba[..., :3] = (9, 18, 27); rgba[..., 3] = 63
    image = numpy_to_qimage(rgba)
    rgba[:] = 255
    assert image.hasAlphaChannel()
    assert image.pixelColor(0, 0).alpha() == 63
    assert np.array_equal(qimage_to_numpy(image)[0, 0], [9, 18, 27])


def test_capped_preview_composites_at_exact_proxy_geometry():
    base = np.full((12, 18), 200, np.float32)
    result = render_preview(
        RenderPipeline(registry), base, RenderSettings(style="None", scale=1),
        max_side=6, mask_context=_context(base.shape),
    )
    assert result.shape == (4, 6, 3)
    assert np.all(result == 200)
