"""JIT-on vs JIT-off equivalence: the compiled kernels must match the pure-Python
fallback bit-for-bit. Runs in subprocesses because NUMBA_DISABLE_JIT is read once,
at numba import time, and cannot be flipped inside a running interpreter."""
import os
import sys
import subprocess

import numpy as np

# Script prints the raveled kernel output as a comma-separated list of ints.
_SCRIPT = r"""
import numpy as np
from ditherzam.dithering import registry
img = np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))
name = __import__("os").environ["DZ_KERNEL"]
out = registry.get_entry(name).func(img.copy(), 0, 128.0)
import sys
sys.stdout.write(",".join(str(int(v)) for v in out.ravel().tolist()))
"""


def _run(kernel_name: str, disable_jit: str) -> np.ndarray:
    env = dict(os.environ)
    env["NUMBA_DISABLE_JIT"] = disable_jit
    env["DZ_KERNEL"] = kernel_name
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    return np.array([int(x) for x in proc.stdout.strip().split(",")], dtype=np.int64)


def test_floyd_steinberg_jit_on_equals_jit_off():
    off = _run("Floyd-Steinberg", "1")   # pure-Python fallback
    on = _run("Floyd-Steinberg", "0")    # JIT-compiled
    assert off.shape == on.shape and off.size == 256
    np.testing.assert_array_equal(on, off)
    assert set(np.unique(off).tolist()) <= {0, 255}


def test_bayer4_jit_on_equals_jit_off():
    off = _run("Bayer-Matrix 4x4", "1")
    on = _run("Bayer-Matrix 4x4", "0")   # exercises the parallel=True prange path
    np.testing.assert_array_equal(on, off)
    assert set(np.unique(off).tolist()) <= {0, 255}
