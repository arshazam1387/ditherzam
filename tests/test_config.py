from pathlib import Path
import pytest
from ditherzam.config import load_config, AppConfig, ConfigError

def test_load_config_reads_values():
    cfg = load_config(Path("config/config.yaml"))
    assert isinstance(cfg, AppConfig)
    assert cfg.default_dither_style == "None"
    assert cfg.default_dither_scale == 5
    assert cfg.viewport_bg_color == "#1f1f1f"
    assert cfg.app_style == "Fusion"
    assert cfg.enable_inertia is True
    assert cfg.friction == 0.95

def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("does/not/exist.yaml")

def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.yaml"; p.write_text("")
    with pytest.raises(ConfigError):
        load_config(p)
