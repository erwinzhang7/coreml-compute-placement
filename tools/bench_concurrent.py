#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure ANE and GPU running the same encoder at the same time.

Every "use both units" claim, including the one in this repo's README, assumes the two
add up. They share memory bandwidth, so they may not. This runs each unit alone, then
both together, and reports what the combination actually delivers against the sum of the
parts.

Separate *processes*, not threads: the Python driving loop is a real cost (it was 45% of
the ANE arm's power draw), and the GIL would serialise it.
"""
import argparse
import multiprocessing as mp
import time
import warnings


def _worker(model_path, units, seconds, barrier, q):
    warnings.filterwarnings("ignore")
    import coremltools as ct
    import numpy as np

    model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit[units])
    spec = model.get_spec()
    name = spec.description.input[0].name
    shape = tuple(spec.description.input[0].type.multiArrayType.shape)
    x = {name: np.random.rand(*shape).astype(np.float32)}

    for _ in range(5):
        model.predict(x)

    barrier.wait()  # start both arms together
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        model.predict(x)
        n += 1
    dt = time.perf_counter() - t0
    q.put({"units": units, "calls": n, "seconds": dt, "batch": shape[0]})


def run(model_paths, unit_list, seconds):
    """model_paths: dict unit -> path, so each unit can run its own model variant."""
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(unit_list))
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(model_paths[u], u, seconds, barrier, q))
        for u in unit_list
    ]
    for p in procs:
        p.start()
    results = [q.get() for _ in unit_list]
    for p in procs:
        p.join()
    for r in results:
        r["img_per_s"] = r["calls"] * r["batch"] / r["seconds"]
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="model for both units, unless overridden below")
    ap.add_argument("--model-ane", help="separate model variant for the ANE")
    ap.add_argument("--model-gpu", help="separate model variant for the GPU")
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    paths = {"CPU_AND_NE": args.model_ane or args.model,
             "CPU_AND_GPU": args.model_gpu or args.model}
    solo = {}
    for u in ("CPU_AND_NE", "CPU_AND_GPU"):
        r = run(paths, [u], args.seconds)[0]
        solo[u] = r["img_per_s"]
        print(f"  alone  {u:14s} {r['img_per_s']:8.1f} img/s")

    together = run(paths, ["CPU_AND_NE", "CPU_AND_GPU"], args.seconds)
    combined = sum(r["img_per_s"] for r in together)
    for r in together:
        share = r["img_per_s"] / solo[r["units"]]
        print(f"  both   {r['units']:14s} {r['img_per_s']:8.1f} img/s   "
              f"({share:.0%} of its solo rate)")

    ideal = solo["CPU_AND_NE"] + solo["CPU_AND_GPU"]
    best_solo = max(solo.values())
    print(f"\n  sum of solo rates      {ideal:8.1f} img/s")
    print(f"  actual combined        {combined:8.1f} img/s   "
          f"({combined / ideal:.0%} of ideal)")
    print(f"  best single unit alone {best_solo:8.1f} img/s")
    print(f"  real uplift from using both: {combined / best_solo:.2f}x")


if __name__ == "__main__":
    main()
