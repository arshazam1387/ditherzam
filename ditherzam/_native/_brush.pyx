# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from libc.math cimport sqrt


def stamp_mask_brush_u8(
    unsigned char[:, :] mask,
    Py_ssize_t x0,
    Py_ssize_t x1,
    Py_ssize_t y0,
    Py_ssize_t y1,
    double document_x,
    double document_y,
    double layer_x,
    double layer_y,
    double scale_x,
    double scale_y,
    double radius,
    int hardness,
    int strength,
    int mode_code,
):
    """Mutate a validated mask ROI and return changed half-open bounds or None."""
    cdef:
        Py_ssize_t x, y
        Py_ssize_t changed_x0 = mask.shape[1]
        Py_ssize_t changed_y0 = mask.shape[0]
        Py_ssize_t changed_x1 = 0
        Py_ssize_t changed_y1 = 0
        double px, py, dx, dy, distance, coverage_value
        double hard_radius = radius * hardness / 100.0
        double falloff_width = radius - hard_radius
        int coverage, amount, old, new

    with nogil:
        for y in range(y0, y1):
            py = layer_y + (y + 0.5) * scale_y
            dy = py - document_y
            for x in range(x0, x1):
                px = layer_x + (x + 0.5) * scale_x
                dx = px - document_x
                distance = sqrt(dx * dx + dy * dy)
                if distance > radius:
                    continue
                if hardness == 100 or (hard_radius > 0.0 and distance <= hard_radius):
                    coverage = 255
                else:
                    coverage_value = (radius - distance) / falloff_width
                    if coverage_value < 0.0:
                        coverage_value = 0.0
                    elif coverage_value > 1.0:
                        coverage_value = 1.0
                    coverage = <int>(coverage_value * 255.0 + 0.5)
                amount = (coverage * strength + 50) // 100
                old = mask[y, x]
                if mode_code == 0:
                    new = old + ((255 - old) * amount + 127) // 255
                else:
                    new = old - (old * amount + 127) // 255
                if new == old:
                    continue
                mask[y, x] = <unsigned char>new
                if x < changed_x0:
                    changed_x0 = x
                if x + 1 > changed_x1:
                    changed_x1 = x + 1
                if y < changed_y0:
                    changed_y0 = y
                changed_y1 = y + 1

    if changed_x1 == 0:
        return None
    return changed_x0, changed_y0, changed_x1, changed_y1
