from .registry import DitherRegistry, DitherEntry

registry = DitherRegistry()

# Import kernel modules for their registration side-effects (registry must exist first).
# Order matters: `ordered` before `glitch` (glitch imports `_BAYER4` from `ordered`).
from .kernels import error_diffusion as _error_diffusion  # noqa: E402,F401
from .kernels import ordered as _ordered_kernels          # noqa: E402,F401
