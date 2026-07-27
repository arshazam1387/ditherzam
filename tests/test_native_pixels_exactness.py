from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from benchmarks.native_pixels import (
    compare_exact_outputs,
    deterministic_u8,
    measure,
    output_digest,
    speedup,
)


def test_native_smoke_import_and_reference_are_exact():
    from ditherzam import _native

    assert _native.native_available()
    for left, right in ((0, 0), (-4, 9), (127, 128)):
        assert _native.smoke_add(left, right) == _native.smoke_add_reference(left, right)


def test_native_fallback_is_independently_selectable(monkeypatch):
    import ditherzam._native as native

    monkeypatch.setenv("DITHERZAM_DISABLE_NATIVE", "1")
    fallback = importlib.reload(native)
    try:
        assert fallback.native_available() is False
        assert fallback.smoke_add(20, 22) == fallback.smoke_add_reference(20, 22) == 42
        source = deterministic_u8((3, 5, 4))
        result = fallback.smoke_copy_u8(source)
        assert np.array_equal(result, fallback.smoke_copy_u8_reference(source))
        assert result is not source and result.flags.owndata and result.flags.c_contiguous
    finally:
        monkeypatch.delenv("DITHERZAM_DISABLE_NATIVE")
        importlib.reload(native)


def test_differential_input_and_digest_conventions():
    first = deterministic_u8((3, 5, 4))
    second = deterministic_u8((3, 5, 4))
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8
    assert first.flags.owndata and first.flags.c_contiguous
    assert set((0, 1, 127, 128, 254, 255)).issubset(set(first.reshape(-1)))
    assert output_digest(first) == output_digest(second)
    assert output_digest(first) != output_digest(first[..., :3])


def test_reference_and_native_array_seams_obey_exact_output_contract():
    from ditherzam import _native

    for shape in ((1, 1, 4), (3, 5, 4), (7, 2, 4)):
        source = deterministic_u8(shape)
        comparison = compare_exact_outputs(
            _native.smoke_copy_u8_reference,
            _native.smoke_copy_u8_native,
            source,
        )
        assert comparison["shape"] == shape
        assert comparison["dtype"] == np.dtype(np.uint8).str
        assert comparison["digest"] == output_digest(source)


def test_benchmark_convention_excludes_warmup_and_reports_required_fields():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        return np.array([calls], dtype=np.uint8)

    result = measure(operation, warmup=2, samples=3)
    assert calls == 5
    assert result["samples"] == 3
    assert result["median_ms"] >= 0
    assert result["p95_ms"] >= result["median_ms"]
    assert len(result["digest"]) == 64
    assert speedup(10.0, 2.0) == 5.0


def test_frozen_spec_collects_native_smoke_extension():
    spec = (
        Path(__file__).resolve().parents[1]
        / "packaging"
        / "ditherzam-smart-mask.spec"
    ).read_text(encoding="utf-8")
    assert '"ditherzam._native._smoke"' in spec


def test_windows_build_script_checks_every_external_exit_and_pins_pytest():
    root = Path(__file__).resolve().parents[1]
    script = (root / "tools" / "build_windows_release.ps1").read_text(encoding="utf-8")
    requirements = (
        root / "packaging" / "requirements-windows-build.txt"
    ).read_text(encoding="utf-8")
    assert script.count("& $FilePath @ArgumentList") == 1
    assert "& $Python" not in script
    assert "if ($LASTEXITCODE -ne 0)" in script
    assert script.count("Invoke-External $Python") == 7
    assert "dist\\ditherzam\\ditherzam.exe" in script
    assert '@("--native-smoke")' in script
    assert "pytest==9.1.1" in requirements
