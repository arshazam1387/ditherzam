import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp_fixture():
    """A single offscreen QApplication for all Qt smoke tests."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
