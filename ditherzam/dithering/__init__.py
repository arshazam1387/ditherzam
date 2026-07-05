from .registry import DitherRegistry, DitherEntry

registry = DitherRegistry()

# import kernel modules for their registration side-effects (after `registry` exists)
from .kernels import error_diffusion as _error_diffusion  # noqa: E402,F401
