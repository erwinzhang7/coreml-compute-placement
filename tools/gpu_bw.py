#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""GPU-side streaming bandwidth on Apple Silicon, via MLX.

Counterpart to membw.c. Together they answer whether CPU cores are worth
recruiting for compute on a unified-memory Mac, or whether the GPU already owns
the bus.

Several op shapes are tried because no single MLX kernel is guaranteed to be
bandwidth-optimal; the best observed figure is the honest estimate of what the
GPU can pull, and a low number for one op says more about that kernel than about
the hardware.

Traffic accounting is explicit per op rather than assumed: a reduction reads
once, a binary elementwise reads twice and writes once. Getting this wrong is
the easiest way to report a bandwidth figure that is off by 3x.
"""
import argparse
import time

import mlx.core as mx


def timed(fn, seconds, traffic_bytes):
    """Run fn in a loop for `seconds`, return GB/s.

    mx.eval() forces the lazy graph to actually execute; without it this measures
    graph construction and reports absurd numbers.
    """
    fn()  # warm up: compiles the kernel and faults the buffers in
    mx.eval(mx.array(0))

    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        out = fn()
        mx.eval(out)
        n += 1
    dt = time.perf_counter() - t0
    return n * traffic_bytes / dt / 1e9, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=4.0, help="size of each operand")
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--quiet", action="store_true", help="print only the best GB/s")
    args = ap.parse_args()

    nbytes = int(args.gb * (1 << 30))
    n = nbytes // 4  # float32

    a = mx.random.uniform(shape=(n,), dtype=mx.float32)
    b = mx.random.uniform(shape=(n,), dtype=mx.float32)
    mx.eval(a, b)

    ops = [
        # (label, thunk, bytes of DRAM traffic per invocation)
        ("sum (read 1x)", lambda: mx.sum(a), nbytes),
        ("max (read 1x)", lambda: mx.max(a), nbytes),
        ("a+b (read 2x, write 1x)", lambda: a + b, 3 * nbytes),
        ("a*2 (read 1x, write 1x)", lambda: a * 2.0, 2 * nbytes),
    ]

    best = 0.0
    rows = []
    for label, fn, traffic in ops:
        gbps, iters = timed(fn, args.secs, traffic)
        rows.append((label, gbps, iters))
        best = max(best, gbps)

    if args.quiet:
        print(f"{best:.1f}")
    else:
        print(f"operand={args.gb:.1f} GiB each, {args.secs:.0f}s per op\n")
        print(f"{'op':28s} {'GB/s':>8s} {'iters':>7s}")
        for label, gbps, iters in rows:
            print(f"{label:28s} {gbps:>8.1f} {iters:>7d}")
        print(f"\nbest observed GPU bandwidth: {best:.1f} GB/s")


if __name__ == "__main__":
    main()
