import numpy as np
import pytest
from ditherzam.effects.stack import EffectStack


def img():
    return np.zeros((8, 8, 3), np.uint8)


def test_new_stack_is_empty():
    s = EffectStack()
    assert s.items == []


def test_add_appends_name_and_params():
    s = EffectStack()
    s.add("Chromatic Aberration", shift=1)
    s.add("Blur", radius=0)
    assert [name for name, _ in s.items] == ["Chromatic Aberration", "Blur"]
    assert s.items[0][1] == {"shift": 1}
    assert s.items[1][1] == {"radius": 0}


def test_apply_runs_in_order_shape_preserved():
    s = EffectStack()
    s.add("Chromatic Aberration", shift=1)
    s.add("Blur", radius=0)                       # identity
    out = s.apply(img())
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8


def test_apply_actually_uses_the_effect():
    x = np.zeros((4, 8, 3), np.uint8)
    x[:, 3, 0] = 255
    s = EffectStack()
    s.add("Chromatic Aberration", shift=2)
    out = s.apply(x)
    assert out[:, 5, 0].max() == 255             # red shifted right by the stack


def test_empty_stack_apply_is_identity():
    x = np.random.RandomState(1).randint(0, 256, (8, 8, 3), np.uint8)
    np.testing.assert_array_equal(EffectStack().apply(x), x)


def test_move_reorders():
    s = EffectStack()
    s.add("Blur", radius=0)
    s.add("Sharpen", amount=1)
    s.move(1, 0)
    assert [name for name, _ in s.items] == ["Sharpen", "Blur"]


def test_remove_deletes_index():
    s = EffectStack()
    s.add("Blur", radius=0)
    s.add("Sharpen", amount=1)
    s.remove(0)
    assert [name for name, _ in s.items] == ["Sharpen"]


def test_unknown_effect_raises_keyerror():
    with pytest.raises(KeyError):
        EffectStack().add("Nope")
