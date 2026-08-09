#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Do CPU and GPU bandwidth add up on Apple Silicon, or do they fight?

This is the measurement that decides whether KTransformers-style CPU offload has
any headroom to exploit on a unified-memory Mac. Peak specs and solo benchmarks
cannot answer it: what matters is whether recruiting CPU cores while the GPU is
already streaming produces MORE aggregate bandwidth, or merely redistributes a
fixed budget.

Three methodology points are load-bearing. Each of them was learned by getting a
confident, plausible, wrong answer first.

1. WINDOWS MUST BE ALIGNED, and the alignment must be measured rather than
   assumed. Two benchmark processes each timing their own window will happily
   report an aggregate above the hardware's peak -- the arithmetic counts bytes
   the CPU moved while the GPU was not running. A first version of this script
   reported "+75 GB/s of real headroom" that way. membw now emits its timed
   window as CLOCK_MONOTONIC timestamps, so the CPU rate during the genuinely
   contended sub-window is solved for rather than approximated.

2. CONFIGURATIONS ARE INTERLEAVED, not batched. Running all CPU trials, then all
   GPU trials, then all contention trials confounds machine drift with
   configuration. One repeat runs all three back to back.

3. THE MACHINE MUST BE QUIET. A background workload was observed changing the
   same binary's result by 3x -- 99 GB/s versus 290 GB/s for identical arguments
   seconds apart. Load is checked, and a high-variance run refuses to render a
   verdict rather than rendering a plausible-looking lie.
"""
import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

MEMBW = Path(__file__).parent / "membw"


def mono():
    """Same clock base as membw.c's now_sec(). Must stay CLOCK_MONOTONIC."""
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def gpu_stream(arr, seconds, nbytes):
    """Sustained read-only GPU streaming. Returns (GB/s, t_start, t_end)."""
    mx.eval(mx.sum(arr))  # warm up outside the timed window
    n, t0 = 0, mono()
    while mono() - t0 < seconds:
        mx.eval(mx.sum(arr))
        n += 1
    t1 = mono()
    return n * nbytes / (t1 - t0) / 1e9, t0, t1


def membw_cmd(threads, secs, qos, gb):
    return [str(MEMBW), "--threads", str(threads), "--gb", str(gb),
            "--secs", str(secs), "--qos", qos, "--quiet"]


def parse_membw(out):
    """quiet format: threads qos gbps t0 t1"""
    f = out.split()
    return float(f[2]), float(f[3]), float(f[4])


def run_membw(threads, secs, qos="ui", gb=16.0):
    out = subprocess.run(membw_cmd(threads, secs, qos, gb),
                         capture_output=True, text=True).stdout
    return parse_membw(out)[0]


def start_membw(threads, secs, qos="ui", gb=16.0):
    return subprocess.Popen(membw_cmd(threads, secs, qos, gb),
                            stdout=subprocess.PIPE, text=True)


def contended_cpu_rate(reported, t0, t1, g0, g1, solo_rate):
    """Solve for the CPU rate during the overlap with the GPU's window.

    membw reports an average over its whole window, part of which may have run
    with the GPU idle. Bytes split into a contended and an uncontended portion:

        reported * (t1-t0) = solo_rate * uncontended + contended * overlap

    Returns (contended_rate, overlap_fraction). Without this correction the CPU
    number is biased upward by however much of its window ran unopposed.
    """
    total = t1 - t0
    overlap = max(0.0, min(t1, g1) - max(t0, g0))
    if overlap <= 0 or total <= 0:
        return float("nan"), 0.0
    uncontended = total - overlap
    contended = (reported * total - solo_rate * uncontended) / overlap
    return contended, overlap / total


def summarize(label, xs):
    med = statistics.median(xs)
    spread = (max(xs) - min(xs)) / med * 100 if med else 0.0
    return med, spread, f"{label:26s} {med:>8.1f}   (n={len(xs)}, spread {spread:>4.1f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6, help="CPU threads (6 = the Super tier)")
    ap.add_argument("--secs", type=float, default=6.0, help="GPU window; CPU window is longer")
    ap.add_argument("--gb", type=float, default=4.0, help="GPU operand size")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-load", type=float, default=2.5)
    ap.add_argument("--settle", type=float, default=180.0,
                    help="seconds to wait for load to fall below --max-load")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # The 1-minute load average is a trailing figure that includes any benchmark
    # run moments ago, so a bare threshold check makes back-to-back runs
    # impossible. Wait for it to decay instead of refusing outright.
    load0 = os.getloadavg()[0]
    if load0 > args.max_load and not args.force:
        print(f"load average is {load0:.1f} (limit {args.max_load}); waiting up to "
              f"{args.settle:.0f}s", end="", flush=True)
        deadline = time.time() + args.settle
        while time.time() < deadline and load0 > args.max_load:
            time.sleep(5)
            load0 = os.getloadavg()[0]
            print(".", end="", flush=True)
        print()
        if load0 > args.max_load:
            sys.exit(f"load is still {load0:.1f}. Something else is using this machine;\n"
                     f"measuring now would produce a plausible-looking lie.")

    nbytes = int(args.gb * (1 << 30))
    arr = mx.random.uniform(shape=(nbytes // 4,), dtype=mx.float32)
    mx.eval(arr)

    print(f"CPU threads={args.threads}   GPU operand={args.gb:.0f} GiB   "
          f"GPU window={args.secs:.0f}s   reps={args.reps}   load={load0:.2f}\n")

    cpu_solo, gpu_solo, cpu_cont, gpu_cont, overlaps = [], [], [], [], []

    for rep in range(args.reps):
        print(f"  rep {rep + 1}/{args.reps} ...", end="", flush=True)

        cs = run_membw(args.threads, args.secs)
        gs, _, _ = gpu_stream(arr, args.secs, nbytes)
        cpu_solo.append(cs)
        gpu_solo.append(gs)

        # CPU window deliberately brackets the GPU's, so the GPU runs fully
        # contended; the CPU's partially-unopposed portion is corrected for below.
        proc = start_membw(args.threads, args.secs + 4.0)
        time.sleep(1.5)
        gb_, g0, g1 = gpu_stream(arr, args.secs, nbytes)
        out, _ = proc.communicate()
        cb_reported, t0, t1 = parse_membw(out)
        cb, frac = contended_cpu_rate(cb_reported, t0, t1, g0, g1, cs)

        cpu_cont.append(cb)
        gpu_cont.append(gb_)
        overlaps.append(frac)
        print(f" cpu {cs:.0f} | gpu {gs:.0f} | contended {cb:.0f}+{gb_:.0f} "
              f"(overlap {frac * 100:.0f}%)")

    cs_m, cs_sp, cs_l = summarize("CPU alone", cpu_solo)
    gs_m, gs_sp, gs_l = summarize("GPU alone", gpu_solo)
    cb_m, cb_sp, cb_l = summarize("CPU while GPU runs", cpu_cont)
    gb_m, gb_sp, gb_l = summarize("GPU while CPU runs", gpu_cont)

    print(f"\n{'configuration':26s} {'GB/s':>8s}")
    for line in (cs_l, gs_l, cb_l, gb_l):
        print(line)

    agg = cb_m + gb_m
    headroom = agg - gs_m
    print(f"\nmean GPU-window overlap       : {statistics.mean(overlaps) * 100:.0f}%")
    print(f"aggregate while contended     : {agg:.1f} GB/s")
    print(f"GPU retained                  : {gb_m / gs_m * 100:.0f}% of solo")
    print(f"CPU retained                  : {cb_m / cs_m * 100:.0f}% of solo")
    print(f"aggregate vs GPU alone        : {agg / gs_m:.2f}x  ({headroom:+.1f} GB/s)")
    print(f"exchange rate                 : CPU gains {cb_m:.0f}, GPU loses "
          f"{gs_m - gb_m:.0f} GB/s")

    worst = max(cs_sp, gs_sp, cb_sp, gb_sp)
    if worst > 20:
        print(f"\nUNRELIABLE: run-to-run spread reached {worst:.0f}%. Something else is using\n"
              "the machine. Do not trust a verdict from this run; re-run when quiet.")
        return

    if agg < gs_m * 1.05:
        print("\nVERDICT: no headroom. The GPU alone effectively saturates the bus; CPU\n"
              "streaming redistributes bandwidth rather than adding any.")
    elif agg < gs_m * 1.25:
        print(f"\nVERDICT: marginal headroom ({headroom:+.0f} GB/s, {agg / gs_m:.2f}x). Not enough to\n"
              "justify heterogeneous execution for bandwidth-bound work.")
    else:
        print(f"\nVERDICT: real headroom ({headroom:+.0f} GB/s). Heterogeneous execution has\n"
              "something to exploit on this chip.")


if __name__ == "__main__":
    if not MEMBW.exists():
        sys.exit(f"build membw first: cc -O3 -o {MEMBW} {MEMBW.with_suffix('.c')}")
    main()
