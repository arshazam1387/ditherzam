"""Build-only smoke extension. Production kernels live in separate modules."""


cpdef int smoke_add(int left, int right):
    return left + right


def native_available():
    return True
