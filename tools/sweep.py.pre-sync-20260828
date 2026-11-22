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


def power_and_thermal():
    """Power source, Low Power Mode, and thermal pressure at sweep time.

    None of this was recorded for the two sweeps already in results/, and it
    should have been: Mac17,7 is a LAPTOP. A sweep on battery is not comparable
    to one on AC, and nothing in the JSON said which it was. Measured with
    tools/thermal_soak.py, GPU throughput on this machine falls ~18% within 30
    seconds on battery while thermal pressure stays nominal, so the difference is
    large enough to reorder a comparison.

    The burst numbers in those sweeps are still sound as PEAK measurements -- 30
    iterations is over before any of this bites, and their spread is 0.0-3.5%.
    The gap is that a reader cannot tell peak from sustained, or AC from battery,
    without this field.

    Thermal pressure comes from notify(3), which needs no root, so this cannot
    turn a clone-and-run into a sudo prompt.
    """
    out = {}
    ps = subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True).stdout
    out["source"] = ("AC" if "AC Power" in ps else
                     "battery" if "Battery Power" in ps else "unknown")
    out["battery"] = next((l.strip() for l in ps.splitlines() if "%" in l), "")
    # macOS 26 does not list lowpowermode in `pmset -g` or `pmset -g custom` at
    # all, so absence here is the OS not exposing it rather than a parse miss.
    # Said explicitly, because a field that reads "unknown" on every machine is
    # indistinguishable from one that is quietly broken.
    g = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    lpm = next((l for l in g.splitlines() if "lowpowermode" in l), "")
    out["lowpowermode"] = lpm.split()[-1] if lpm else "not-exposed-by-pmset"
    try:
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("System"))
        tok = ctypes.c_int(0)
        if lib.notify_register_check(
                ctypes.c_char_p(b"com.apple.system.thermalpressurelevel"),
                ctypes.byref(tok)) == 0:
            st = ctypes.c_uint64(0)
            if lib.notify_get_state(tok, ctypes.byref(st)) == 0:
                out["thermal"] = int(st.value)
    except OSError:
        pass
    return out


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
            "macos": platform.mac_ver()[0],
            "power": power_and_thermal()}


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
