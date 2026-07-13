import numpy as np

from ditherzam.dithering import parameters

# Deterministic gradient input shared by every golden test.
STD_INPUT = np.tile(np.linspace(0, 255, 32, dtype=np.float32), (32, 1))


def default_param(entry):
    """The kernel parameter the app sends at default slider positions."""
    native = [s.default for s in parameters.parameter_specs(entry)
              if not s.key.startswith("creative_")]
    if not native:
        return 0
    if len(native) == 1:
        return native[0]
    return tuple(native)
