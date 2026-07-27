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
    if ao > 0.0:
        return quantize((premul / ao) * 255.0)
    return quantize(cs * 255.0)


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
    cdef double opacity_factor = opacity / 100.0
    cdef double source_alpha, ass, ab, cb, cs, blended
    cdef double one_minus_ass, ao

    with nogil:
        if mode == 0:
            for pixel in prange(pixels, schedule="static", num_threads=2):
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
            for pixel in prange(pixels, schedule="static", num_threads=2):
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
            for pixel in prange(pixels, schedule="static", num_threads=2):
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
