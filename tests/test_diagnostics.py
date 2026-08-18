from __future__ import annotations

import logging


def test_configure_logging_writes_rotating_program_log(tmp_path):
    from ditherzam.diagnostics import configure_logging

    log_path = configure_logging(tmp_path)
    logging.getLogger("ditherzam.test").error("diagnostic sentinel")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_path == tmp_path / "ditherzam.log"
    assert "diagnostic sentinel" in log_path.read_text(encoding="utf-8")


def test_diagnostic_report_includes_runtime_and_log_tail(tmp_path):
    from ditherzam.diagnostics import build_diagnostic_report

    log_path = tmp_path / "ditherzam.log"
    log_path.write_text("old line\nlast useful line\n", encoding="utf-8")
    report = build_diagnostic_report(log_path, tail_chars=32)

    assert "ditherzam diagnostic report" in report
    assert "Version: 0.3.0" in report
    assert "Python:" in report
    assert "Platform:" in report
    assert "last useful line" in report


def test_debug_menu_is_available(qapp_fixture, tmp_path):
    from ditherzam.ui.main_window import ImageEditor

    window = ImageEditor(diagnostic_log_path=tmp_path / "ditherzam.log")

    assert window.debug_menu.title() == "&Debug"
    assert window.debug_log_action.text() == "Program Notes…"
