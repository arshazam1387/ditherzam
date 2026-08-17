# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from libc.stdlib cimport free, malloc

import numpy as np
cimport numpy as cnp


def selection_to_source_u8(
    const unsigned char[:, ::1] selection,
    Py_ssize_t output_h,
    Py_ssize_t output_w,
    double layer_x,
    double layer_y,
    double scale_x,
    double scale_y,
):
    """Map document selection coverage into source coordinates."""
    cdef:
        cnp.ndarray result = np.empty((output_h, output_w), dtype=np.uint8)
        unsigned char[:, ::1] output = result
        Py_ssize_t document_h = selection.shape[0]
        Py_ssize_t document_w = selection.shape[1]
        Py_ssize_t y, x
        Py_ssize_t y0, y1, x0, x1
        double dy, dx, fy, fx, wy0, wy1, wx0, wx1, value
        Py_ssize_t* x0_values = NULL
        double* fx_values = NULL

    if layer_x == 0.0 and layer_y == 0.0 and scale_x == 1.0 and scale_y == 1.0:
        with nogil:
            for y in range(output_h):
                for x in range(output_w):
                    if y < document_h and x < document_w:
                        output[y, x] = selection[y, x]
                    else:
                        output[y, x] = 0
        return result

    x0_values = <Py_ssize_t*>malloc(output_w * sizeof(Py_ssize_t))
    fx_values = <double*>malloc(output_w * sizeof(double))
    if x0_values == NULL or fx_values == NULL:
        free(x0_values)
        free(fx_values)
        raise MemoryError()
    try:
        with nogil:
            for x in range(output_w):
                dx = layer_x + (x + 0.5) * scale_x - 0.5
                if dx < -1.0:
                    x0_values[x] = -2
                    fx_values[x] = 0.0
                elif dx >= document_w:
                    x0_values[x] = document_w
                    fx_values[x] = 0.0
                else:
                    x0 = <Py_ssize_t>dx
                    if dx < x0:
                        x0 -= 1
                    x0_values[x] = x0
                    fx_values[x] = dx - x0
            for y in range(output_h):
                dy = layer_y + (y + 0.5) * scale_y - 0.5
                if dy < -1.0 or dy >= document_h:
                    for x in range(output_w):
                        output[y, x] = 0
                    continue
                y0 = <Py_ssize_t>dy
                if dy < y0:
                    y0 -= 1
                y1 = y0 + 1
                fy = dy - y0
                wy0 = 1.0 - fy
                wy1 = fy
                for x in range(output_w):
                    x0 = x0_values[x]
                    x1 = x0 + 1
                    fx = fx_values[x]
                    wx0 = 1.0 - fx
                    wx1 = fx
                    value = 0.0
                    if y0 >= 0 and y0 < document_h:
                        if x0 >= 0 and x0 < document_w:
                            value += selection[y0, x0] * wy0 * wx0
                        if x1 >= 0 and x1 < document_w:
                            value += selection[y0, x1] * wy0 * wx1
                    if y1 >= 0 and y1 < document_h:
                        if x0 >= 0 and x0 < document_w:
                            value += selection[y1, x0] * wy1 * wx0
                        if x1 >= 0 and x1 < document_w:
                            value += selection[y1, x1] * wy1 * wx1
                    output[y, x] = <unsigned char>(value + 0.5)
    finally:
        free(x0_values)
        free(fx_values)
    return result
