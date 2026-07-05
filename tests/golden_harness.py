import numpy as np

# Deterministic gradient input shared by every golden test.
STD_INPUT = np.tile(np.linspace(0, 255, 32, dtype=np.float32), (32, 1))


def default_param(entry):
    """Build a representative parameter for a kernel from its slider count."""
    n = len(entry.param_sliders)
    if n == 0:
        return 0
    if n == 1:
        return 4
    return tuple(4 for _ in range(n))
