import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"


def test_window_constructs_and_renders(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor
    w = ImageEditor()
    ramp = np.tile(np.linspace(0, 255, 64, dtype=np.float32), (64, 1))
    w.load_array(ramp)
    w.set_style("Floyd-Steinberg")
    img = w.render_now()
    assert img.width() == 64 and img.height() == 64


def test_none_style_renders_grayscale(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor
    w = ImageEditor()
    w.load_array(np.full((32, 48), 128.0, np.float32))
    w.set_style("None")
    img = w.render_now()
    assert img.width() == 48 and img.height() == 32


def test_main_entry_is_callable():
    from ditherzam import app
    assert callable(app.main)


def test_debounced_render_produces_image(qapp_fixture):
    import time

    from PySide6.QtCore import QCoreApplication
    from ditherzam.ui.main_window import ImageEditor
    w = ImageEditor()
    w.load_array(np.full((16, 16), 200.0, np.float32))
    w.set_style("Atkinson")
    w.schedule_render()
    # let the debounce timer fire and the worker finish (wall time must elapse
    # for the single-shot QTimer + threadpool worker, so wait a little per tick)
    for _ in range(50):
        QCoreApplication.processEvents()
        time.sleep(0.005)
    assert w.last_qimage is not None
    assert w.last_qimage.width() == 16
