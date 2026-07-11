"""Bounded render/export threading policy (Qt-free).

Numba's ``set_num_threads`` is a process-global setting, so ditherzam bounds
thread use with two levers chosen from the 2026-07-10 thread-scaling baseline
(``benchmarks/THREAD_SCALING_2026-07-10.md``):

* **Interactive budget** — the process default. Parallel color kernels plateau by
  ~4 threads, so interactive renders are capped at ``min(4, cpu)`` (snapped to a
  supported count). This is installed once at startup.
* **Export budget** — what a long async export (video) drops the process-global
  count to while it runs: ``cpu`` minus the interactive reserve, snapped, floored
  at 1. Because the count is process-global, this lowers *any* concurrent render
  (including a UI-triggered one) rather than partitioning cores between them — the
  goal is bounding total oversubscription during export, not a hard reservation.
  On an 8-CPU host interactive and export are both 4, so it is moot there.

Diffusion stays sequential regardless (its kernel is not parallel), and the heavy
effect stack is GIL-bound and does not benefit from extra threads — neither needs
a policy knob. ``numba_threads`` is a best-effort context manager that restores the
prior count, so nested/concurrent use never leaks a lowered budget.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

SUPPORTED = (1, 2, 4, 8)
INTERACTIVE_CAP = 4  # parallel kernels plateau past this (4.3 baseline)


def cpu_threads() -> int:
    return os.cpu_count() or 1


def snap_to_supported(n: int) -> int:
    """Largest supported thread count <= n (never below 1, never above max)."""
    best = SUPPORTED[0]
    for count in SUPPORTED:
        if count <= n:
            best = count
    return best


def interactive_budget(cpu: int | None = None) -> int:
    cpu = cpu_threads() if cpu is None else cpu
    return snap_to_supported(min(INTERACTIVE_CAP, cpu))


def export_budget(cpu: int | None = None) -> int:
    """Threads an async export may use while reserving the interactive budget."""
    cpu = cpu_threads() if cpu is None else cpu
    return snap_to_supported(max(1, cpu - interactive_budget(cpu)))


def _numba():
    """Return the numba module, or None if it cannot provide thread controls."""
    try:
        import numba  # local import keeps this module Qt-free and import-cheap
        numba.get_num_threads  # noqa: B018 - presence check
        numba.set_num_threads
        return numba
    except Exception:  # noqa: BLE001 - numba missing/incompatible -> no-op policy
        return None


@contextmanager
def numba_threads(n: int):
    """Best-effort: run the block with ``n`` numba threads, then restore.

    A no-op if numba is unavailable. Restores in ``finally`` so an exception or a
    concurrent export never leaves the process at a lowered budget.
    """
    nb = _numba()
    if nb is None:
        yield
        return
    prev = nb.get_num_threads()
    try:
        nb.set_num_threads(max(1, n))
        yield
    finally:
        nb.set_num_threads(prev)


def install_interactive_budget() -> int | None:
    """Set the process-default numba thread count to the interactive budget.

    Returns the value applied, or None if numba is unavailable.
    """
    nb = _numba()
    budget = interactive_budget()
    if nb is None:
        return None
    nb.set_num_threads(budget)
    return budget
