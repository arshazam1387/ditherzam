"""Fast, JIT-free coverage of the thread-scaling benchmark's parent-side logic.

The measurement subprocesses need JIT and are exercised by running the benchmark
directly; here we lock the pure aggregation/clamping/divergence logic that could
otherwise break silently.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from benchmarks import thread_scaling as ts


def test_thread_list_clamps_to_cpu_and_always_includes_one(monkeypatch):
    monkeypatch.setattr(ts.os, "cpu_count", lambda: 4)
    assert ts._thread_list(None) == [1, 2, 4]          # 8 dropped (> cpu)
    assert ts._thread_list([8]) == [1]                 # over-cpu dropped, 1 kept
    assert ts._thread_list([2, 4]) == [1, 2, 4]        # 1 injected


def test_thread_list_single_cpu():
    # Even on a hypothetical 1-CPU host, 1 is always probed.
    import types
    monkey = types.SimpleNamespace(cpu_count=lambda: 1)
    saved = ts.os.cpu_count
    ts.os.cpu_count = monkey.cpu_count
    try:
        assert ts._thread_list((1, 2, 4, 8)) == [1]
    finally:
        ts.os.cpu_count = saved


def test_parse_args_defaults():
    args = ts.parse_args([])
    assert args.tier == "quick"
    assert args.section == "all"
    assert args.repeats == 3
    assert args.child is False


def _key(section, case, size):
    return (section, case, size)


def test_report_flags_hash_divergence():
    by_key = {
        _key("kernels", "nearest/k16", "1080p"): {
            1: {"warm_ms": 100.0, "peak_mb": 10.0, "sha16": "aaaa"},
            2: {"warm_ms": 60.0, "peak_mb": 10.0, "sha16": "bbbb"},  # differs!
        },
    }
    out = io.StringIO()
    with redirect_stdout(out):
        diverged = ts._report([1, 2], by_key)
    text = out.getvalue()
    assert "DIVERGED!" in text
    assert diverged == 1                   # surfaced for the non-zero exit path
    assert "1.67" in text  # 100/60 speedup rendered


def test_report_marks_consistent_hashes_ok_and_skips_heartbeat():
    by_key = {
        _key("kernels", "ordered/k16", "1080p"): {
            1: {"warm_ms": 100.0, "peak_mb": 10.0, "sha16": "aaaa"},
            2: {"warm_ms": 50.0, "peak_mb": 10.0, "sha16": "aaaa"},
        },
        _key("heartbeat", "nearest/k16 x3", "1080p"): {
            1: {"max_stall_ms": 20.0, "render_ms": 300.0, "sha16": ""},
            2: {"max_stall_ms": 18.0, "render_ms": 250.0, "sha16": ""},
        },
    }
    out = io.StringIO()
    with redirect_stdout(out):
        diverged = ts._report([1, 2], by_key)
    text = out.getvalue()
    assert "DIVERGED!" not in text
    assert diverged == 0
    assert "max stall" in text            # heartbeat reported separately
    # heartbeat never appears as a warm-ms table row (skipped by section)
    assert "nearest/k16 x3" not in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
