def test_layer_stack_is_immutable_and_bottom_to_top():
    from ditherzam.composition import Look
    from ditherzam.layers import Layer, LayerStack
    a = Layer("a", "Layer 1", Look("a", {"dither": {"style": "None"}}))
    b = Layer("b", "Layer 2", Look("b", {"dither": {"style": "None"}}), opacity=50)
    stack = LayerStack().add(a).add(b)
    assert [x.name for x in stack.layers] == ["Layer 1", "Layer 2"]
    assert stack.move(1, 0).layers == (b, a)
    assert stack.layers == (a, b)


def test_layer_validation():
    import pytest
    from ditherzam.composition import Look
    from ditherzam.layers import Layer
    look = Look("a", {"dither": {"style": "None"}})
    with pytest.raises(ValueError): Layer("a", "", look)
    with pytest.raises(ValueError): Layer("a", "x", look, opacity=101)
    with pytest.raises(ValueError): Layer("a", "x", look, blend_mode="bad")


def test_stack_edit_operations_and_caller_owned_duplicate_identity():
    from ditherzam.composition import Look
    from ditherzam.layers import Layer, LayerStack
    a = Layer("a", "A", Look("a", {"dither": {"style": "None"}}))
    b = Layer("b", "B", Look("b", {"dither": {"style": "None"}}))
    replacement = Layer(
        "a", "A renamed", Look("a", {"dither": {"style": "None"}})
    )
    stack = LayerStack((a, b))
    assert stack.replace(0, replacement).layers == (replacement, b)
    assert stack.remove(0).layers == (b,)
    duplicated = stack.duplicate(0, "copy", "A copy")
    assert [item.id for item in duplicated.layers] == ["a", "copy", "b"]
    assert duplicated.layers[1].look is a.look
    assert duplicated.layers[1].name == "A copy"


def test_layer_stack_rejects_duplicate_ids_and_bool_opacity():
    import pytest
    from ditherzam.composition import Look
    from ditherzam.layers import Layer, LayerStack
    look = Look("a", {"dither": {"style": "None"}})
    with pytest.raises(ValueError):
        Layer("a", "A", look, opacity=True)
    with pytest.raises(ValueError):
        LayerStack((Layer("same", "A", look), Layer("same", "B", look)))
