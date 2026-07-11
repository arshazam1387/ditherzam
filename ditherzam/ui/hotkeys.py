from __future__ import annotations

# Logical bindings using the primary modifier ("mod"). On Windows/Linux mod=Ctrl;
# on macOS mod=Meta (Qt maps QKeySequence "Meta" to the Command key).
_BINDINGS = {
    "change_theme": "{mod}+Shift+T",
    "export_image": "{mod}+Shift+S",
    "copy_to_clipboard": "{mod}+Shift+C",
    "import_image": "{mod}+I",
    "restart_application": "{mod}+{alt}+R",
    "zoom_in": "{mod}+=",
    "zoom_out": "{mod}+-",
    "zoom_reset": "{mod}+0",
    "show_help": "{mod}+Shift+/",
    "cycle_theme": "{mod}+T",
    "open_image": "{mod}+O",
    "save": "{mod}+S",
    "help": "{mod}+H",
    "full_quality_preview": "{mod}+Return",
}


def get_hotkeys(platform: str) -> dict[str, str]:
    is_mac = platform == "darwin"
    mod = "Meta" if is_mac else "Ctrl"
    alt = "Alt"  # Qt names the Option key "Alt" on macOS too
    return {action: tpl.format(mod=mod, alt=alt) for action, tpl in _BINDINGS.items()}
