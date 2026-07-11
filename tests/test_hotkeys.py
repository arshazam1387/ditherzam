from ditherzam.ui.hotkeys import get_hotkeys

ACTIONS = {
    "change_theme", "export_image", "copy_to_clipboard", "import_image",
    "restart_application", "zoom_in", "zoom_out", "zoom_reset", "show_help",
    "cycle_theme", "open_image", "save", "help", "full_quality_preview",
}


def test_windows_bindings():
    hk = get_hotkeys("win32")
    assert hk["import_image"] == "Ctrl+I"
    assert hk["zoom_in"] == "Ctrl+="
    assert hk["zoom_out"] == "Ctrl+-"
    assert hk["zoom_reset"] == "Ctrl+0"
    assert hk["export_image"] == "Ctrl+Shift+S"
    assert hk["full_quality_preview"] == "Ctrl+Return"


def test_macos_uses_meta():
    hk = get_hotkeys("darwin")
    assert hk["import_image"] == "Meta+I"
    assert hk["zoom_in"] == "Meta+="
    assert hk["change_theme"] == "Meta+Shift+T"
    assert hk["restart_application"] == "Meta+Alt+R"
    assert hk["full_quality_preview"] == "Meta+Return"


def test_all_actions_present_and_differ_per_platform():
    win = get_hotkeys("win32")
    mac = get_hotkeys("darwin")
    assert set(win) == ACTIONS and set(mac) == ACTIONS
    assert win != mac


def test_linux_defaults_to_ctrl():
    assert get_hotkeys("linux")["import_image"] == "Ctrl+I"
