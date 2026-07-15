import numpy as np
import pytest

from ditherzam.masking.contracts import ModelIdentity, SourceIdentity, source_identity
from ditherzam.masking.inference_request import CancellationToken, InferenceRequest


def _request():
    rgba = np.zeros((2, 3, 4), np.uint8)
    rgba[..., 3] = 255
    return rgba, InferenceRequest(source_identity(rgba), ModelIdentity("u2", "1", "a" * 64), "u2net-320-v1", rgba)


def test_request_owns_frozen_source_snapshot():
    rgba, request = _request()
    rgba[0, 0, 0] = 99
    assert request.rgba[0, 0, 0] == 0
    assert not request.rgba.flags.writeable
    with pytest.raises(ValueError):
        request.rgba[0, 0, 0] = 1


def test_cancellation_is_idempotent_and_callable():
    token = CancellationToken()
    assert not token.is_cancelled and not token.should_cancel()
    token.cancel(); token.cancel()
    assert token.is_cancelled and token.should_cancel()


def test_request_rejects_same_dimensions_with_different_content_identity():
    rgba = np.zeros((2, 3, 4), np.uint8); rgba[..., 3] = 255
    other = rgba.copy(); other[0, 0, 0] = 1
    with pytest.raises(ValueError, match="exactly match"):
        InferenceRequest(source_identity(other), ModelIdentity("u2", "1", "a" * 64), "v1", rgba)


def test_request_rejects_alpha_participation_identity_mismatch():
    rgba = np.zeros((2, 3, 4), np.uint8); rgba[..., 3] = 255
    identity = source_identity(rgba)
    forged = SourceIdentity(identity.content_hash, identity.width, identity.height, True)
    with pytest.raises(ValueError, match="exactly match"):
        InferenceRequest(forged, ModelIdentity("u2", "1", "a" * 64), "v1", rgba)
