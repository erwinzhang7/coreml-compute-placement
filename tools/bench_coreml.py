#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure real Core ML throughput for a model under different compute units.

MLComputePlan tells you where ops are *planned* to run. This tells you what actually
happens on the clock, which is the only thing the spec's kill criterion can be decided on.
"""
import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import coremltools as ct
import numpy as np


def bench(path, units, batch, warmup, iters):
    model = ct.models.MLModel(path, compute_units=units)
    spec = model.get_spec()
    name = spec.description.input[0].name
    shape = tuple(spec.description.input[0].type.multiArrayType.shape)
    x = {name: np.random.rand(*shape).astype(np.float32)}

    for _ in range(warmup):
        model.predict(x)

    t0 = time.perf_counter()
    for _ in range(iters):
        model.predict(x)
    dt = time.perf_counter() - t0

    per_call = dt / iters
    return {
        "units": units.name,
        "shape": shape,
        "iters": iters,
        "seconds": dt,
        "ms_per_call": per_call * 1000,
        "images_per_s": batch / per_call,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--batch", type=int, default=1, help="images per call, for the rate")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--units", nargs="+",
                    default=["CPU_AND_NE", "CPU_AND_GPU", "ALL", "CPU_ONLY"])
    args = ap.parse_args()

    print(f"{'compute units':>14}  {'ms/call':>9}  {'img/s':>9}")
    results = []
    for u in args.units:
        r = bench(args.model, ct.ComputeUnit[u], args.batch, args.warmup, args.iters)
        results.append(r)
        print(f"{r['units']:>14}  {r['ms_per_call']:>8.2f}  {r['images_per_s']:>9.1f}")

    ne = next((r for r in results if r["units"] == "CPU_AND_NE"), None)
    gpu = next((r for r in results if r["units"] == "CPU_AND_GPU"), None)
    if ne and gpu:
        ratio = ne["images_per_s"] / gpu["images_per_s"]
        print(f"\nANE / GPU throughput ratio: {ratio:.2f}x")
        print("Spec kill criterion is ANE < ~0.60x of Metal GPU on the same encoder.")
        print("  -> " + ("ANE is competitive; concurrency is upside on top."
                         if ratio >= 0.60 else
                         "below threshold; concurrency is the only remaining argument."))


if __name__ == "__main__":
    main()
