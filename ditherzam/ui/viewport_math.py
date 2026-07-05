from __future__ import annotations


def next_zoom(current, direction, factor_in=1.2, factor_out=0.8,
              zmax=100.0, zmin=0.01):
    """Proposed new zoom scale, or None if it would breach the [zmin, zmax] clamp."""
    factor = factor_in if direction > 0 else factor_out
    proposed = current * factor
    if proposed > zmax or proposed < zmin:
        return None
    return proposed


def inertia_step(pos, velocity, maximum, friction, dt=0.016):
    """One fling tick: advance a scrollbar position and decay the velocity.

    Returns (clamped_position, decayed_velocity).
    """
    new_pos = pos - velocity * dt
    if new_pos < 0.0:
        new_pos = 0.0
    elif new_pos > maximum:
        new_pos = maximum
    return new_pos, velocity * friction


def clamp_velocity(v, scale=0.5, maximum=2000.0):
    """Scale a raw pan velocity and clamp its magnitude to +/- maximum."""
    v = v * scale
    if v > maximum:
        return maximum
    if v < -maximum:
        return -maximum
    return v


def zoom_percent(m11) -> int:
    """Integer zoom percent from a view transform's horizontal scale (m11)."""
    return int(m11 * 100)
