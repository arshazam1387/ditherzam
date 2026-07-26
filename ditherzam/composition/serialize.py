from __future__ import annotations

import math

from ..animation.timeline import Keyframe, Timeline
from .look import Look
from .model import Composition, LookClip, TransitionSpec


_EASING_KINDS = frozenset({"linear", "ease-in", "ease-out", "ease-in-out"})


def _yaml_safe(value, *, label: str):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain only YAML-safe finite values")
        return value
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item, label=label) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} must contain only YAML-safe string keys")
            result[key] = _yaml_safe(item, label=label)
        return result
    raise ValueError(f"{label} must contain only YAML-safe values")


def _validate_automation_length(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("automation length must be a positive integer")
    return value


def _validated_keyframe(frame, field, value, kind, length: int) -> dict:
    if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < length:
        raise ValueError("keyframe frame must be an integer within automation length")
    if not isinstance(field, str) or not field:
        raise ValueError("keyframe field must be a non-empty string")
    if (
        isinstance(value, bool)
        or type(value) not in (int, float)
        or not math.isfinite(value)
    ):
        raise ValueError("keyframe value must be finite numeric and non-bool")
    if kind not in _EASING_KINDS:
        raise ValueError("keyframe kind is invalid")
    return {"frame": frame, "field": field, "value": value, "kind": kind}


def composition_to_dict(composition: Composition) -> dict:
    if not isinstance(composition, Composition):
        raise ValueError("composition must be a Composition")
    clips = []
    for clip in composition.clips:
        automation = None
        if clip.automation is not None:
            length = _validate_automation_length(clip.automation.length)
            automation = {
                "length": length,
                "keys": [
                    _validated_keyframe(
                        key.frame, key.field, key.value, key.kind, length,
                    )
                    for key in clip.automation.keyframes()
                ],
            }
        clips.append({
            "name": clip.look.name,
            "preset": _yaml_safe(clip.look.preset, label="preset YAML"),
            "start": clip.start,
            "end": clip.end,
            "automation": automation,
        })
    return {
        "version": 1,
        "length": composition.length,
        "clips": clips,
        "transitions": {
            str(index): {
                "kind": spec.kind,
                "duration": spec.duration,
                "params": _yaml_safe(spec.params, label="transition params YAML"),
            }
            for index, spec in sorted(composition.transitions.items())
        },
    }


def composition_from_dict(payload: dict) -> Composition:
    try:
        if (
            not isinstance(payload, dict)
            or type(payload.get("version")) is not int
            or payload.get("version") != 1
        ):
            raise ValueError("composition version must be 1")
        if not isinstance(payload.get("clips"), list):
            raise ValueError("clips must be a list")
        transitions = payload.get("transitions", {})
        if not isinstance(transitions, dict):
            raise ValueError("transitions must be a dict")

        composition = Composition(payload["length"])
        for raw in payload["clips"]:
            if not isinstance(raw, dict):
                raise ValueError("clip must be a dict")
            automation_data = raw.get("automation")
            automation = None
            if automation_data is not None:
                if not isinstance(automation_data, dict):
                    raise ValueError("automation must be a dict or None")
                keys = automation_data.get("keys")
                if not isinstance(keys, list):
                    raise ValueError("automation keys must be a list")
                length = _validate_automation_length(automation_data["length"])
                automation = Timeline(length)
                for raw_key in keys:
                    if not isinstance(raw_key, dict):
                        raise ValueError("keyframe must be a dict")
                    validated = _validated_keyframe(
                        raw_key["frame"],
                        raw_key["field"],
                        raw_key["value"],
                        raw_key["kind"],
                        length,
                    )
                    automation.add(Keyframe(**validated))
            preset = _yaml_safe(raw["preset"], label="preset YAML")
            composition.add_clip(LookClip(
                Look(raw["name"], preset),
                raw["start"],
                raw["end"],
                automation,
            ))
        for raw_index, raw_spec in transitions.items():
            if not isinstance(raw_spec, dict):
                raise ValueError("transition must be a dict")
            composition.set_transition(int(raw_index), TransitionSpec(
                raw_spec["kind"],
                raw_spec["duration"],
                _yaml_safe(raw_spec["params"], label="transition params YAML"),
            ))
        composition.validate()
        return composition
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("malformed composition data") from exc
