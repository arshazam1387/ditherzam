"""Build-only smoke extension. Production kernels live in separate modules."""

cimport numpy as cnp


cpdef int smoke_add(int left, int right):
    return left + right


def smoke_copy_u8(cnp.ndarray[cnp.uint8_t, ndim=3, mode="c"] source):
    """Return an owned C-order copy for differential infrastructure tests."""
    return source.copy(order="C")


def native_available():
    return True
