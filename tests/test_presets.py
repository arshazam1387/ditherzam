def test_prereqs_import():
    from ditherzam.render import RenderSettings, RenderPipeline
    from ditherzam.color.palette import Palette
    from ditherzam.effects.stack import EffectStack
    s = RenderSettings()
    assert s.contrast == 50 and s.scale == 5 and s.style == "None"
    assert s.saturation == 50 and s.invert is False
    assert isinstance(s.params, dict)
