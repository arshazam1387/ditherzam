# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
from libc.math cimport floor
from cython.parallel cimport prange

import numpy as np
cimport numpy as cnp


cnp.import_array()

cdef double _overlay_lut[65536]
cdef int _thread_budget = 2


def get_thread_budget():
    """Return the OpenMP worker budget used by compositor operations."""
    return _thread_budget


def set_thread_budget(int threads):
    """Set the compositor OpenMP worker budget, returning the applied value."""
    global _thread_budget
    if threads < 1 or threads > 8:
        raise ValueError("native thread budget must be within 1..8")
    _thread_budget = threads
    return _thread_budget


cdef void initialize_overlay_lut() noexcept:
    cdef int back_value, source_value, index
    cdef double cb, cs
    for back_value in range(256):
        cb = back_value / 255.0
        for source_value in range(256):
            cs = source_value / 255.0
            index = (back_value << 8) | source_value
            if cb <= 0.5:
                _overlay_lut[index] = 2.0 * cb * cs
            else:
                _overlay_lut[index] = 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs)


initialize_overlay_lut()


cdef inline unsigned char quantize(double value) noexcept nogil:
    cdef double rounded = floor(value + 0.5)
    if rounded <= 0.0:
        return 0
    if rounded >= 255.0:
        return 255
    return <unsigned char>rounded


cdef inline unsigned char resolve_straight_channel(
    double premul,
    double out_alpha,
    double scale,
    double hidden,
) noexcept nogil:
    """Resolve one premultiplied channel, including zero-alpha hidden RGB."""
    if out_alpha > 0.0:
        return quantize((premul / out_alpha) * scale)
    return quantize(hidden)


cdef inline unsigned char composite_channel(
    double cb,
    double cs,
    double blended,
    double one_minus_ass,
    double ab,
    double ass,
    double ao,
) noexcept nogil:
    cdef double premul = (
        one_minus_ass * ab * cb
        + (1.0 - ab) * ass * cs
        + ab * ass * blended
    )
    return resolve_straight_channel(premul, ao, 255.0, cs * 255.0)


cdef inline void transition_pixel(
    const unsigned char* a,
    const unsigned char* b,
    unsigned char* output,
    Py_ssize_t offset,
    double weight,
) noexcept nogil:
    cdef double one_minus_weight = 1.0 - weight
    cdef double a_alpha = a[offset + 3]
    cdef double b_alpha = b[offset + 3]
    cdef double out_alpha = a_alpha * one_minus_weight + b_alpha * weight
    cdef double premul, hidden
    cdef Py_ssize_t channel
    for channel in range(3):
        premul = (
            a[offset + channel] * a_alpha * one_minus_weight
            + b[offset + channel] * b_alpha * weight
        )
        hidden = (
            a[offset + channel] * one_minus_weight
            + b[offset + channel] * weight
        )
        output[offset + channel] = resolve_straight_channel(
            premul, out_alpha, 1.0, hidden
        )
    output[offset + 3] = quantize(out_alpha)


cdef tuple transition_arrays(object first, object second):
    cdef cnp.ndarray a_array = np.asarray(first)
    cdef cnp.ndarray b_array = np.asarray(second)
    if (
        a_array.dtype != np.uint8
        or b_array.dtype != np.uint8
        or a_array.ndim != 3
        or b_array.ndim != 3
        or a_array.shape[2] != 4
        or b_array.shape[2] != 4
        or not a_array.flags.c_contiguous
        or not b_array.flags.c_contiguous
    ):
        raise ValueError("inputs must be C-contiguous RGBA uint8 arrays")
    if (
        a_array.shape[0] != b_array.shape[0]
        or a_array.shape[1] != b_array.shape[1]
    ):
        raise ValueError("inputs must have equal height and width")
    return a_array, b_array


def blend_transition_scalar_u8(object first, object second, double weight):
    """Straight-alpha transition blend with one scalar weight."""
    cdef cnp.ndarray a_array
    cdef cnp.ndarray b_array
    a_array, b_array = transition_arrays(first, second)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("weight must be within [0,1]")
    cdef Py_ssize_t height = a_array.shape[0]
    cdef Py_ssize_t width = a_array.shape[1]
    cdef cnp.ndarray output_array = np.empty(
        (height, width, 4), dtype=np.uint8, order="C"
    )
    cdef const unsigned char* a = <const unsigned char*>a_array.data
    cdef const unsigned char* b = <const unsigned char*>b_array.data
    cdef unsigned char* output = <unsigned char*>output_array.data
    cdef Py_ssize_t pixel
    cdef Py_ssize_t pixels = height * width
    cdef int thread_budget = _thread_budget
    with nogil:
        for pixel in prange(pixels, schedule="static", num_threads=thread_budget):
            transition_pixel(a, b, output, pixel * 4, weight)
    return output_array


def blend_transition_plane_u8(object first, object second, object weight_plane):
    """Straight-alpha transition blend with a C-contiguous float64 plane."""
    cdef cnp.ndarray a_array
    cdef cnp.ndarray b_array
    a_array, b_array = transition_arrays(first, second)
    cdef cnp.ndarray weights_array = np.asarray(weight_plane)
    if (
        weights_array.dtype != np.float64
        or weights_array.ndim != 2
        or not weights_array.flags.c_contiguous
        or weights_array.shape[0] != a_array.shape[0]
        or weights_array.shape[1] != a_array.shape[1]
    ):
        raise ValueError("weights must be a matching C-contiguous float64 plane")
    cdef Py_ssize_t height = a_array.shape[0]
    cdef Py_ssize_t width = a_array.shape[1]
    cdef cnp.ndarray output_array = np.empty(
        (height, width, 4), dtype=np.uint8, order="C"
    )
    cdef const unsigned char* a = <const unsigned char*>a_array.data
    cdef const unsigned char* b = <const unsigned char*>b_array.data
    cdef const double* weights = <const double*>weights_array.data
    cdef unsigned char* output = <unsigned char*>output_array.data
    cdef Py_ssize_t pixel
    cdef Py_ssize_t pixels = height * width
    cdef int thread_budget = _thread_budget
    with nogil:
        for pixel in prange(pixels, schedule="static", num_threads=thread_budget):
            transition_pixel(a, b, output, pixel * 4, weights[pixel])
    return output_array


def blend_layer_u8(object backdrop, object source, int mode, int opacity):
    """Composite validated C-contiguous RGBA uint8 arrays in one pixel pass."""
    cdef cnp.ndarray back_array = np.asarray(backdrop)
    cdef cnp.ndarray source_array = np.asarray(source)
    if (
        back_array.dtype != np.uint8
        or source_array.dtype != np.uint8
        or back_array.ndim != 3
        or source_array.ndim != 3
        or back_array.shape[2] != 4
        or source_array.shape[2] != 4
        or not back_array.flags.c_contiguous
        or not source_array.flags.c_contiguous
    ):
        raise ValueError("inputs must be C-contiguous RGBA uint8 arrays")
    if (
        back_array.shape[0] != source_array.shape[0]
        or back_array.shape[1] != source_array.shape[1]
    ):
        raise ValueError("inputs must have equal height and width")
    if mode < 0 or mode > 4:
        raise ValueError("mode code is invalid")
    if opacity < 0 or opacity > 100:
        raise ValueError("opacity must be within 0..100")

    cdef Py_ssize_t height = back_array.shape[0]
    cdef Py_ssize_t width = back_array.shape[1]
    cdef cnp.ndarray output_array = np.empty(
        (height, width, 4), dtype=np.uint8, order="C"
    )
    cdef const unsigned char* back = <const unsigned char*>back_array.data
    cdef const unsigned char* src = <const unsigned char*>source_array.data
    cdef unsigned char* output = <unsigned char*>output_array.data
    cdef Py_ssize_t pixel, offset, channel
    cdef Py_ssize_t pixels = height * width
    cdef int thread_budget = _thread_budget
    cdef double opacity_factor = opacity / 100.0
    cdef double source_alpha, ass, ab, cb, cs, blended
    cdef double one_minus_ass, ao

    with nogil:
        if mode == 0:
            for pixel in prange(pixels, schedule="static", num_threads=thread_budget):
                offset = pixel * 4
                source_alpha = floor(src[offset + 3] * opacity_factor + 0.5)
                ass = source_alpha / 255.0
                ab = back[offset + 3] / 255.0
                one_minus_ass = 1.0 - ass
                ao = ass + ab * one_minus_ass
                for channel in range(3):
                    cb = back[offset + channel] / 255.0
                    cs = src[offset + channel] / 255.0
                    blended = cs
                    output[offset + channel] = composite_channel(
                        cb, cs, blended, one_minus_ass, ab, ass, ao
                    )
                output[offset + 3] = quantize(ao * 255.0)
        elif mode == 3:
            for pixel in prange(pixels, schedule="static", num_threads=thread_budget):
                offset = pixel * 4
                source_alpha = floor(src[offset + 3] * opacity_factor + 0.5)
                ass = source_alpha / 255.0
                ab = back[offset + 3] / 255.0
                one_minus_ass = 1.0 - ass
                ao = ass + ab * one_minus_ass
                for channel in range(3):
                    cb = back[offset + channel] / 255.0
                    cs = src[offset + channel] / 255.0
                    blended = _overlay_lut[
                        (back[offset + channel] << 8) | src[offset + channel]
                    ]
                    output[offset + channel] = composite_channel(
                        cb, cs, blended, one_minus_ass, ab, ass, ao
                    )
                output[offset + 3] = quantize(ao * 255.0)
        else:
            for pixel in prange(pixels, schedule="static", num_threads=thread_budget):
                offset = pixel * 4
                source_alpha = floor(src[offset + 3] * opacity_factor + 0.5)
                ass = source_alpha / 255.0
                ab = back[offset + 3] / 255.0
                one_minus_ass = 1.0 - ass
                ao = ass + ab * one_minus_ass
                for channel in range(3):
                    cb = back[offset + channel] / 255.0
                    cs = src[offset + channel] / 255.0
                    if mode == 1:
                        blended = cb * cs
                    elif mode == 2:
                        blended = 1.0 - (1.0 - cb) * (1.0 - cs)
                    else:
                        if cb >= cs:
                            blended = cb - cs
                        else:
                            blended = cs - cb
                    output[offset + channel] = composite_channel(
                        cb, cs, blended, one_minus_ass, ab, ass, ao
                    )
                output[offset + 3] = quantize(ao * 255.0)
    return output_array
