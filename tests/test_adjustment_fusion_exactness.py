"""Frozen dither-facing float32 output for future adjustment fusion work."""
import hashlib

import numpy as np

from ditherzam.adjustments import apply_contrast, apply_highlights, apply_midtones


def _chain(image, contrast, midtones, highlights):
    return apply_highlights(
        apply_midtones(apply_contrast(image, contrast), midtones), highlights)


def _fixture():
    rng = np.random.default_rng(20260709)
    boundaries = np.array(
        [0.0, np.nextafter(np.float32(0), np.float32(1)), 1.0,
         127.5, 254.99998, 255.0], dtype=np.float32)
    random = rng.uniform(0, 255, 4090).astype(np.float32)
    return np.concatenate((boundaries, random)).reshape(64, 64)


def _sha256(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def test_tonal_chain_reference_hash_primary_settings():
    result = _chain(_fixture(), 70.0, 30.0, 80.0)
    assert result.dtype == np.float32
    assert _sha256(result) == "2ab563dd74e3f662e722cfd3788664b56b1449c5efbe2c7efd5376afdf07da20"


def test_tonal_chain_reference_hash_fraction_sensitive_settings():
    result = _chain(_fixture(), 13.0, 91.0, 47.0)
    assert result.dtype == np.float32
    assert _sha256(result) == "cecd580a09d8f6c9788167dccb27bfe4598bf5a5f173dc0346e28e82ea85e154"
