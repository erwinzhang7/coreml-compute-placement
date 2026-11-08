#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""STREAM-style GPU memory bandwidth (copy, scale, add, triad) through MLX custom Metal kernels.

Custom kernels rather than MLX ops so each kernel is exactly the STREAM loop and the traffic
is known: copy and scale move 2 arrays per call, add and triad move 3 (STREAM's convention:
no read-for-ownership counted on the store). float4 loads and stores, one element group per
thread. Arrays are allocated once; each repeat times ITERS back-to-back calls, each forced
with mx.eval, and the reported figure is the median repeat.

Published peak figures to compare against are given on the command line (--published).
"""
import argparse
import json
import platform
import statistics
import subprocess
import time

import mlx.core as mx

SRC = {
    "copy":  "c4[i] = a4[i];",
    "scale": "c4[i] = 3.0f * a4[i];",
    "add":   "c4[i] = a4[i] + b4[i];",
    "triad": "c4[i] = a4[i] + 3.0f * b4[i];",
}
ARRAYS = {"copy": 2, "scale": 2, "add": 3, "triad": 3}


def kernel(op):
    body = ("uint i = thread_position_in_grid.x;\n"
            "const device float4* a4 = (const device float4*)a;\n"
            "const device float4* b4 = (const device float4*)b;\n"
            "device float4* c4 = (device float4*)c;\n" + SRC[op])
    return mx.fast.metal_kernel(name=f"stream_{op}", input_names=["a", "b"], output_names=["c"],
                                source=body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=1.0, help="size of EACH array")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--published", type=float, help="published peak GB/s, for the fraction")
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args()

    n = int(a_.gib * (1 << 30)) // 4
    n -= n % 4
    nbytes = n * 4
    a = mx.random.uniform(shape=(n,))
    b = mx.random.uniform(shape=(n,))
    mx.eval(a, b)

    res = {}
    for op in ("copy", "scale", "add", "triad"):
        k = kernel(op)
        call = lambda: k(inputs=[a, b], grid=(n // 4, 1, 1), threadgroup=(256, 1, 1),
                         output_shapes=[(n,)], output_dtypes=[mx.float32])[0]
        c = call()
        mx.eval(c)
        # correctness on a slice, so a broken kernel cannot report a bandwidth
        ref = {"copy": a, "scale": 3 * a, "add": a + b, "triad": a + 3 * b}[op]
        assert mx.allclose(c[:4096], ref[:4096]).item() and mx.allclose(c[-4096:], ref[-4096:]).item(), op
        del c
        for _ in range(3):
            mx.eval(call())
        traffic = ARRAYS[op] * nbytes
        runs = []
        for _ in range(a_.repeats):
            t0 = time.perf_counter()
            for _ in range(a_.iters):
                mx.eval(call())
            runs.append(traffic * a_.iters / (time.perf_counter() - t0) / 1e9)
        med = statistics.median(runs)
        res[op] = {"median_GBps": med, "runs": runs,
                   "spread_pct": (max(runs) - min(runs)) / med * 100}
        frac = f"  {med / a_.published:.1%} of published {a_.published:g}" if a_.published else ""
        print(f"  {op:6s} {med:7.1f} GB/s  spread {res[op]['spread_pct']:4.1f}%{frac}", flush=True)

    best = max(r["median_GBps"] for r in res.values())
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                          text=True).stdout.strip()
    out = {"chip": chip, "macos": platform.mac_ver()[0], "mlx": mx.__version__,
           "gib_per_array": a_.gib, "iters": a_.iters, "repeats": a_.repeats,
           "ops": res, "best_GBps": best, "published_GBps": a_.published,
           "best_fraction_of_published": best / a_.published if a_.published else None}
    json.dump(out, open(a_.out, "w"), indent=2)
    print(f"best {best:.1f} GB/s" + (f" = {best / a_.published:.1%} of published" if a_.published else ""))


if __name__ == "__main__":
    main()
