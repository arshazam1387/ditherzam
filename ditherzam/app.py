from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from ditherzam.ui.main_window import ImageEditor
    from ditherzam.ui.theme import find_themes, load_theme

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("ditherzam")

    themes_root = Path(__file__).resolve().parent.parent / "themes"
    if "default" in find_themes(themes_root):
        app.setStyleSheet(load_theme(themes_root, "default").stylesheet)

    window = ImageEditor()
    window.resize(1100, 720)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
