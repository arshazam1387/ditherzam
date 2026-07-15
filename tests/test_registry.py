from ditherzam.dithering.registry import DitherRegistry, DitherEntry

def test_register_and_lookup():
    reg = DitherRegistry()

    @reg.register("Demo", "Error Diffusion", dims=2, param_sliders=("p",))
    def demo(image_array, parameter, luminance_threshold_value):
        return image_array

    e = reg.get_entry("Demo")
    assert isinstance(e, DitherEntry)
    assert e.category == "Error Diffusion" and e.dims == 2
    assert e.param_sliders == ("p",)
    assert e.func is demo
    assert "Demo" in reg.list_dithers()

def test_by_category_groups():
    reg = DitherRegistry()
    reg.register("A", "Cat1")(lambda *a: a[0])
    reg.register("B", "Cat1")(lambda *a: a[0])
    reg.register("C", "Cat2")(lambda *a: a[0])
    cats = reg.by_category()
    assert cats["Cat1"] == ["A", "B"] and cats["Cat2"] == ["C"]

def test_unknown_returns_none():
    assert DitherRegistry().get_entry("nope") is None
