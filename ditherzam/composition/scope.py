"""Qt-free media-scope validation for composed Looks."""
from __future__ import annotations

from ditherzam.masking.scope import mask_allows_media, unsupported_mask_message

from .look import Look


class UnsupportedCompositionMaskError(ValueError):
    """Raised when a composition mask is unsupported for a media kind."""


def validate_composition_media(kind: str, looks) -> None:
    """Reject unsupported masked media before any rendering or I/O begins."""
    materialized = list(looks)
    for look in materialized:
        if not isinstance(look, Look):
            raise TypeError("looks must contain only Look instances")
        if not mask_allows_media(kind, look.smart_mask):
            raise UnsupportedCompositionMaskError(unsupported_mask_message(kind))
