import copy

import pytest

from ditherzam.composition import Look
from ditherzam.layers import (
    Layer,
    LayerStack,
    layer_stack_from_dict,
    layer_stack_to_dict,
)


def test_layer_stack_serialization_v1_round_trip_preserves_creative_data():
    preset = {"dither": {"style": "None"}, "extra": {"items": [1, 2.5, True]}}
    stack = LayerStack((
        Layer(
            "stable-id",
            "Ink",
            Look("Captured Look", preset),
            visible=False,
            opacity=37,
            blend_mode="screen",
        ),
    ))
    payload = layer_stack_to_dict(stack)
    restored = layer_stack_from_dict(copy.deepcopy(payload))
    assert payload["version"] == 1
    assert restored.layers[0].id == "stable-id"
    assert restored.layers[0].name == "Ink"
    assert restored.layers[0].look.name == "Captured Look"
    assert restored.layers[0].look.preset == preset
    assert restored.layers[0].visible is False
    assert restored.layers[0].opacity == 37
    assert restored.layers[0].blend_mode == "screen"


@pytest.mark.parametrize("payload", [
    {},
    {"version": True, "layers": []},
    {"version": 2, "layers": []},
    {"version": 1, "layers": "bad"},
    {"version": 1, "layers": [{}]},
])
def test_layer_stack_deserialization_rejects_malformed_data(payload):
    with pytest.raises(ValueError):
        layer_stack_from_dict(payload)


def test_layer_stack_serialization_rejects_non_yaml_preset_value():
    look = Look("x", {"dither": {"style": "None"}})
    look._preset["bad"] = object()
    with pytest.raises(ValueError):
        layer_stack_to_dict(LayerStack((Layer("x", "X", look),)))
