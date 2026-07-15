"""Qt-free v1 policy for media types that may use Smart Mask."""
from __future__ import annotations

from .settings import MaskTarget, SmartMaskSettings

_STILL_MEDIA = frozenset({"png", "jpeg", "jpg", "raster", "still"})


def mask_allows_media(kind: str, settings: SmartMaskSettings) -> bool:
    """Return whether *kind* can proceed without silently dropping a mask."""
    if not isinstance(settings, SmartMaskSettings):
        raise TypeError("settings must be SmartMaskSettings")
    normalized = str(kind).strip().lower()
    return (not settings.enabled
            or settings.target is MaskTarget.WHOLE_IMAGE
            or normalized in _STILL_MEDIA)


def unsupported_mask_message(kind: str) -> str:
    return (f"Smart Mask Subject/Background is not supported for {kind} export yet. "
            "Disable Smart Mask or select Whole Image to continue.")
