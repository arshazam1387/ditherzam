"""Reproducible render thread-scaling matrix (JIT enabled).

Characterises how ditherzam's render core scales across 1/2/4/8 Numba threads so
Task 4.4 can pick a bounded threading policy from *measured* results rather than a
guess. Numba reads ``NUMBA_NUM_THREADS`` once at import, so each thread count runs
in its own child subprocess with that variable set; the parent aggregates.

Sections:
  * ``kernels``    — parallel color modes (nearest/ordered/ramp) through the full
                     pipeline; the prange kernels that should scale.
  * ``preview``    — capped ``render_preview`` from a 4K source.
  * ``effects``    — the heavy post-effect stack (mostly PIL/NumPy, GIL-bound).
  * ``diffusion``  — sequential RGB Floyd–Steinberg control; MUST stay flat.
  * ``heartbeat``  — max GUI-thread stall while a render runs on a worker thread.

Correctness: every deterministic case must produce a byte-identical output hash at
every thread count; the parent flags any case whose hash diverges. Diffusion is
labelled sequential and is expected to show ~1x speedup — that is the point.

Thread settings are isolated per child process, so the parent environment is never
mutated; the optional in-process ``--set-threads`` path restores the prior count in
a ``finally`` block.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import tracemalloc

import numpy as np

THREAD_COUNTS = (1, 2, 4, 8)
SIZES = {"480p": (480, 854), "1080p": (1080, 1920), "4K": (2160, 3840)}


def _make_gray(h: int, w: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    base = 0.5 * yy + 0.5 * xx
    noise = rng.normal(0.0, 18.0, size=(h, w)).astype(np.float32)
    return np.clip(base + noise, 0, 255).astype(np.float32)


def _palette(k: int):
    from ditherzam.color.palette import Palette

    i = np.arange(k, dtype=np.uint32)
    colors = np.column_stack(
        ((i * 67) % 256, (i * 151 + 31) % 256, (i * 211 + 97) % 256)
    ).astype(np.float32)
    return Palette(f"bench-{k}", colors)


def _pipeline(mode: str, k: int = 16):
    from ditherzam.color.engine import ColorEngine
    from ditherzam.render import RenderPipeline
    from ditherzam.dithering import registry as REGISTRY

    return RenderPipeline(REGISTRY, ColorEngine(_palette(k), mode), None)


def _settings():
    from ditherzam.render import RenderSettings

    return RenderSettings(style="Bayer-Matrix 4x4", scale=5, depth=4)


def _measure(fn, repeats: int):
    """first_ms, median warm_ms, tracemalloc peak MB, sha256/16 of the output."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    first = (time.perf_counter() - t0) * 1000.0
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    digest = hashlib.sha256(np.ascontiguousarray(result).tobytes()).hexdigest()[:16]
    return first, float(np.median(samples)), peak / 1048576, digest


# --- workloads -------------------------------------------------------------

def _kernel_cases(tier: str):
    sizes = ("1080p", "4K") if tier == "full" else ("1080p",)
    for size in sizes:
        h, w = SIZES[size]
        base = _make_gray(h, w)
        for mode, k in (("nearest", 16), ("ordered", 16), ("ramp", 4)):
            pipe = _pipeline(mode, k)
            settings = _settings()
            yield f"{mode}/k{k}", size, (lambda p=pipe, b=base, s=settings: p.render(b, s))


def _preview_cases(tier: str):
    from ditherzam.ui.preview import render_preview

    h, w = SIZES["4K"]
    base = _make_gray(h, w)
    caps = (720, 1080, 1440) if tier == "full" else (1080,)
    for cap in caps:
        pipe = _pipeline("nearest", 16)
        settings = _settings()
        yield (f"preview/cap{cap}", "4K",
               lambda p=pipe, b=base, s=settings, c=cap: render_preview(p, b, s, c))


def _effect_cases(_tier: str):
    from ditherzam.effects.stack import EffectStack

    h, w = SIZES["1080p"]
    gray = np.clip(_make_gray(h, w), 0, 255).astype(np.uint8)
    source = np.stack((gray, np.roll(gray, w // 7, axis=1), np.flipud(gray)), axis=-1)

    def build():
        stack = EffectStack()
        stack.add("Chromatic Aberration", shift=2)
        stack.add("Epsilon Glow", threshold=64.0, smoothing=32.0, radius=8.0,
                  intensity=1.0, epsilon=0.4, falloff=0.5,
                  distance_scale=1.0, aspect=1.0)
        return stack

    stack = build()
    yield "stack/heavy", "1080p", lambda s=stack, src=source: s.apply(src)


def _diffusion_cases(_tier: str):
    # Sequential control — labelled sequential, expected to stay flat.
    h, w = SIZES["480p"]
    base = _make_gray(h, w)
    pipe = _pipeline("diffused", 4)
    settings = _settings()
    yield "diffused/k4 (sequential)", "480p", lambda p=pipe, b=base, s=settings: p.render(b, s)


SECTIONS = {
    "kernels": _kernel_cases,
    "preview": _preview_cases,
    "effects": _effect_cases,
    "diffusion": _diffusion_cases,
}


def _heartbeat(threads: int, repeats: int):
    """Max GUI-thread stall (ms) while a parallel render runs on a worker thread.

    Numba prange regions release the GIL, so a healthy scaling story keeps the
    main-thread tick smooth; GIL-bound stretches show up as large stalls.
    """
    h, w = SIZES["1080p"]
    base = _make_gray(h, w)
    pipe = _pipeline("nearest", 16)
    settings = _settings()
    pipe.render(base, settings)  # warm the JIT before timing

    done = threading.Event()

    def work():
        for _ in range(max(3, repeats)):
            pipe.render(base, settings)
        done.set()

    worker = threading.Thread(target=work)
    t0 = time.perf_counter()
    worker.start()
    last = time.perf_counter()
    max_gap = 0.0
    while not done.is_set():
        now = time.perf_counter()
        gap = now - last
        if gap > max_gap:
            max_gap = gap
        last = now
        time.sleep(0.001)
    worker.join()
    render_ms = (time.perf_counter() - t0) * 1000.0
    return {"section": "heartbeat", "case": "nearest/k16 x3", "size": "1080p",
            "threads": threads, "max_stall_ms": max_gap * 1000.0,
            "render_ms": render_ms, "sha16": ""}


def _run_child(args) -> None:
    if args.set_threads:  # optional in-process path; env is the default mechanism
        from numba import get_num_threads, set_num_threads
        prev = get_num_threads()
        set_num_threads(args.threads)
    try:
        sections = list(SECTIONS) if args.section == "all" else [args.section]
        for name in sections:
            if name == "heartbeat":
                continue
            for case, size, fn in SECTIONS[name](args.tier):
                first, warm, peak, sha16 = _measure(fn, args.repeats)
                print("RESULT " + json.dumps({
                    "section": name, "case": case, "size": size,
                    "threads": args.threads, "first_ms": first, "warm_ms": warm,
                    "peak_mb": peak, "sha16": sha16}), flush=True)
        if args.section in ("all", "heartbeat"):
            print("RESULT " + json.dumps(_heartbeat(args.threads, args.repeats)),
                  flush=True)
    finally:
        if args.set_threads:
            set_num_threads(prev)


def _spawn(threads: int, args) -> list[dict]:
    env = dict(os.environ)
    env["NUMBA_NUM_THREADS"] = str(threads)
    env.pop("NUMBA_DISABLE_JIT", None)  # perf numbers require JIT on
    cmd = [sys.executable, "-m", "benchmarks.thread_scaling", "--child",
           "--threads", str(threads), "--section", args.section,
           "--tier", args.tier, "--repeats", str(args.repeats)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"child (threads={threads}) failed: rc={proc.returncode}")
    rows = []
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            rows.append(json.loads(line[len("RESULT "):]))
    return rows


def _thread_list(requested) -> list[int]:
    cpu = os.cpu_count() or 1
    counts = [t for t in (requested or THREAD_COUNTS) if t <= cpu]
    if 1 not in counts:
        counts.insert(0, 1)
    return sorted(set(counts))


def _report(threads_used: list[int], by_key: dict) -> int:
    """Print the scaling table; return the number of hash-diverged cases."""
    diverged = 0
    base_n = threads_used[0]
    width = max((len(k[1]) for k in by_key), default=12) + 4
    print(f"\n== thread scaling: warm ms and speedup vs {base_n} thread(s) ==")
    hdr = f"{'section':<10}{'case':<{width}}{'size':>6} " + \
        " ".join(f"{f'{t}t ms':>9}" for t in threads_used) + \
        " " + " ".join(f"{f'{t}t x':>7}" for t in threads_used[1:]) + \
        f"  {'peak MB':>8}  hash"
    print(hdr)
    for (section, case, size) in sorted(by_key):
        if section == "heartbeat":
            continue
        row = by_key[(section, case, size)]
        base = row.get(base_n, {}).get("warm_ms")
        cells = []
        for t in threads_used:
            v = row.get(t, {}).get("warm_ms")
            cells.append(f"{v:>9.1f}" if v is not None else f"{'--':>9}")
        speed = []
        for t in threads_used[1:]:
            v = row.get(t, {}).get("warm_ms")
            speed.append(f"{base / v:>7.2f}" if (v and base) else f"{'--':>7}")
        peak = row.get(base_n, {}).get("peak_mb", 0.0)
        hashes = {r["sha16"] for r in row.values() if r.get("sha16")}
        ok = len(hashes) <= 1
        hstate = "ok" if ok else "DIVERGED!"
        if not ok:
            diverged += 1
        print(f"{section:<10}{case:<{width}}{size:>6} " +
              " ".join(cells) + " " + " ".join(speed) +
              f"  {peak:>8.1f}  {hstate}")

    hb = {t: by_key.get(("heartbeat", "nearest/k16 x3", "1080p"), {}).get(t)
          for t in threads_used}
    if any(hb.values()):
        print("\n== GUI heartbeat: max main-thread stall while rendering (ms) ==")
        for t in threads_used:
            r = hb.get(t)
            if r:
                print(f"  {t} thread(s): max stall {r['max_stall_ms']:>7.1f} ms "
                      f"over {r['render_ms']:.0f} ms render")
    return diverged


def _run_parent(args) -> None:
    threads_used = _thread_list(args.threads_list)
    print(f"thread scaling matrix; threads={threads_used}; tier={args.tier}; "
          f"repeats={args.repeats}; section={args.section}")
    by_key: dict = {}
    for t in threads_used:
        for r in _spawn(t, args):
            key = (r["section"], r["case"], r["size"])
            by_key.setdefault(key, {})[r["threads"]] = r
    diverged = _report(threads_used, by_key)
    if diverged:
        # A kernel produced different output at different thread counts — a real
        # correctness regression. Fail loudly so CI/automation can catch it.
        raise SystemExit(f"FAIL: {diverged} case(s) diverged across thread counts")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--threads", type=int, default=1, help="child: active thread count")
    p.add_argument("--set-threads", action="store_true",
                   help="child: also call numba.set_num_threads (restored after)")
    p.add_argument("--threads-list", type=int, nargs="+", default=None,
                   help="parent: thread counts to probe (default 1 2 4 8, capped to CPUs)")
    p.add_argument("--tier", choices=("quick", "full"), default="quick")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--section",
                   choices=("all", "kernels", "preview", "effects", "diffusion",
                            "heartbeat"),
                   default="all")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.child:
        _run_child(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
