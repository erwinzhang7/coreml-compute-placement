#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Repeated measurement runner. Emits JSON with per-run values, not just a mean.

Single-run numbers are how benchmark claims go wrong. Everything here is repeated and the
spread is reported, so a reader can see whether a difference is real.
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time


def chip():
    out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                         capture_output=True, text=True).stdout.strip()
    model = subprocess.run(["sysctl", "-n", "hw.model"],
                           capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                         capture_output=True, text=True).stdout
    cores = next((l.split(":")[1].strip() for l in gpu.splitlines()
                  if "Total Number of Cores" in l), "?")
    return {"cpu": out, "model": model, "gpu_cores": cores,
            "macos": platform.mac_ver()[0]}


def bench_once(py, model, units, batch, iters, warmup):
    import numpy as np
    import coremltools as ct
    m = ct.models.MLModel(model, compute_units=ct.ComputeUnit[units])
    spec = m.get_spec()
    name = spec.description.input[0].name
    shape = tuple(spec.description.input[0].type.multiArrayType.shape)
    x = {name: np.random.rand(*shape).astype(np.float32)}
    for _ in range(warmup):
        m.predict(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        m.predict(x)
    dt = time.perf_counter() - t0
    return batch * iters / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--units", nargs="+", default=["CPU_AND_NE", "CPU_AND_GPU", "ALL"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="sweep.json")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    host = chip()
    print(f"host: {host['model']}  {host['cpu']}  {host['gpu_cores']} GPU cores  "
          f"macOS {host['macos']}")
    results = []
    for model in args.models:
        for units in args.units:
            runs = [bench_once(sys.executable, model, units, args.batch,
                               args.iters, args.warmup)
                    for _ in range(args.repeats)]
            rec = {
                "host": host, "model": model, "units": units, "batch": args.batch,
                "iters": args.iters, "repeats": args.repeats, "runs": runs,
                "median": statistics.median(runs),
                "min": min(runs), "max": max(runs),
                "stdev": statistics.stdev(runs) if len(runs) > 1 else 0.0,
            }
            results.append(rec)
            spread = (rec["max"] - rec["min"]) / rec["median"] * 100
            print(f"  {model:26s} {units:12s} median {rec['median']:7.1f} img/s  "
                  f"spread {spread:4.1f}%  (n={args.repeats})")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
