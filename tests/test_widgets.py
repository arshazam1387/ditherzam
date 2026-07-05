import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent


def _wheel(widget, dy=120):
    ev = QWheelEvent(
        QPointF(1, 1), QPointF(1, 1), QPoint(0, 0), QPoint(0, dy),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    widget.wheelEvent(ev)


def test_resettable_slider_double_click_resets(qapp_fixture):
    from ditherzam.ui.widgets import ResettableGlowSlider
    s = ResettableGlowSlider(default=5)
    s.setRange(1, 20)
    s.setValue(17)
    assert s.value() == 17
    s.reset()
    assert s.value() == 5
    assert s.default == 5


def test_invisible_spinbox_friendly_display(qapp_fixture):
    from ditherzam.ui.widgets import InvisibleSpinBox
    sb = InvisibleSpinBox(max_display=250)
    sb.setValue(50)
    assert sb.textFromValue(50) == "125"     # 50/100 * 250
    assert sb.textFromValue(100) == "250"


def test_noscroll_combo_ignores_wheel(qapp_fixture):
    from ditherzam.ui.widgets import NoScrollComboBox
    cb = NoScrollComboBox()
    cb.addItems(["a", "b", "c"])
    cb.setCurrentIndex(1)
    _wheel(cb)
    assert cb.currentIndex() == 1            # unchanged


def test_glow_slider_ignores_wheel(qapp_fixture):
    from ditherzam.ui.widgets import GlowSlider
    s = GlowSlider()
    s.setRange(0, 100)
    s.setValue(40)
    _wheel(s)
    assert s.value() == 40


def test_clickable_label_emits(qapp_fixture):
    from PySide6.QtGui import QMouseEvent
    from ditherzam.ui.widgets import ClickableLabel
    lbl = ClickableLabel("hi", font_size=17)
    seen = []
    lbl.clicked.connect(lambda: seen.append(True))
    ev = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, QPointF(1, 1),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    lbl.mousePressEvent(ev)
    assert seen == [True]
    assert lbl.text() == "hi"
