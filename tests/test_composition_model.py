from __future__ import annotations

import pytest
import yaml

from ditherzam.animation.timeline import Keyframe, Timeline
from ditherzam.composition import (
    Composition,
    Look,
    LookClip,
    TransitionSpec,
    composition_from_dict,
    composition_to_dict,
)


def _look(name="look"):
    return Look(name, {"dither": {"style": "None", "params": {}}})


def test_model_rejects_bool_frames_and_invalid_coverage():
    with pytest.raises(ValueError):
        Composition(True)
    with pytest.raises(ValueError):
        LookClip(_look(), False, 2)

    composition = Composition(3)
    composition.add_clip(LookClip(_look(), 0, 2))
    with pytest.raises(ValueError, match="contiguous"):
        composition.validate()


def test_transition_validation_and_resolution():
    composition = Composition(8)
    composition.add_clip(LookClip(_look("a"), 0, 4))
    composition.add_clip(LookClip(_look("b"), 4, 8))
    params = {"nested": {"value": 1}}
    spec = TransitionSpec("crossfade", 2, params)
    params["nested"]["value"] = 9
    composition.set_transition(1, spec)
    composition.validate()

    assert spec.params == {"nested": {"value": 1}}
    assert composition.resolve(3).kind == "hold"
    assert composition.resolve(4).t == 0.0
    assert composition.resolve(5).t == 0.5
    assert composition.resolve(6).kind == "hold"


def test_serialization_roundtrip_uses_public_sorted_keyframes():
    timeline = Timeline(4)
    timeline.add(Keyframe(3, "scale", 4, "ease-in"))
    timeline.add(Keyframe(1, "contrast", 20))
    composition = Composition(4)
    composition.add_clip(LookClip(_look(), 0, 4, timeline))
    composition.validate()

    payload = composition_to_dict(composition)
    assert [key["field"] for key in payload["clips"][0]["automation"]["keys"]] == [
        "contrast", "scale",
    ]
    restored = composition_from_dict(payload)
    assert restored.clips[0].automation.keyframes() == timeline.keyframes()


@pytest.mark.parametrize("payload", [None, {}, {"version": 2}, {"version": True}])
def test_deserialization_rejects_malformed_payload(payload):
    with pytest.raises(ValueError):
        composition_from_dict(payload)


def test_serialization_rejects_non_yaml_safe_transition_params():
    composition = Composition(4)
    composition.add_clip(LookClip(_look("a"), 0, 2))
    composition.add_clip(LookClip(_look("b"), 2, 4))
    composition.set_transition(1, TransitionSpec("crossfade", 1, {"bad": object()}))
    with pytest.raises(ValueError, match="YAML"):
        composition_to_dict(composition)


def test_deserialization_rejects_malformed_keyframe_types():
    composition = Composition(2)
    composition.add_clip(LookClip(_look(), 0, 2))
    payload = composition_to_dict(composition)
    payload["clips"][0]["automation"] = {
        "length": 2,
        "keys": [{"frame": "bad", "field": 4, "value": object(), "kind": None}],
    }
    with pytest.raises(ValueError, match="keyframe"):
        composition_from_dict(payload)


def test_serialized_payload_is_yaml_safe():
    composition = Composition(2)
    composition.add_clip(LookClip(_look(), 0, 2))
    payload = composition_to_dict(composition)
    assert yaml.safe_load(yaml.safe_dump(payload)) == payload
