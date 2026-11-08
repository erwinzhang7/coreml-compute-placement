#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does a dual Neural Engine double throughput, and for which kind of work?

M6 has two 16-core engines behind a kernel load balancer (H1xANELoadBalancer). One model
instance in a closed loop issues one request at a time, so if the balancer schedules whole
requests it can only ever use one engine. Two independent callers could use both. This
measures, on CPU_AND_NE:

  procs-K    K separate processes, one MLModel each (no shared state at all)
  inproc-K   one process, K MLModel objects, K threads (Swift, so no GIL)
  shared-K   one process, ONE MLModel, K threads

and, for energy per image, GPU alone. Each phase is sampled with macmon (no sudo).
"""
import argparse
import json
import multiprocessing as mp
import os
import statistics
import subprocess
import threading
import time
import warnings

MACMON = "/opt/homebrew/bin/macmon"  # absolute: a launchd job's PATH need not include Homebrew


class Power:
    """macmon pipe in the background; mean power over a wall-clock window."""

    def __init__(self, interval_ms=250):
        self.samples = []
        self.p = None
        if os.path.exists(MACMON):
            self.p = subprocess.Popen([MACMON, "pipe", "-i", str(interval_ms)],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      text=True)
            threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            self.samples.append((time.time(), d))

    def window(self, t0, t1):
        # A sample read at time t covers roughly (t - interval, t]; keep those wholly inside.
        rows = [d for t, d in self.samples if t0 + 0.3 <= t <= t1]
        if not rows:
            return None
        keys = ("all_power", "cpu_power", "gpu_power", "ane_power", "ram_power", "sys_power")
        out = {k: statistics.mean(r.get(k, 0.0) for r in rows) for k in keys}
        out["n_samples"] = len(rows)
        return out

    def stop(self):
        if self.p:
            self.p.terminate()
            self.p.wait()


def _worker(model_path, units, seconds, barrier, q):
    warnings.filterwarnings("ignore")
    import coremltools as ct
    import numpy as np

    # Load the stable compiled .mlmodelc when there is one. ct.models.MLModel(<.mlpackage>)
    # compiles into a fresh temp directory on every load, and on macOS 27.2 every such load
    # paid a full ANE recompile (30-70 s for SigLIP, seen in ANECompilerService's log).
    pkg = model_path
    compiled = model_path[: -len(".mlpackage")] + ".mlmodelc"
    inp = ct.utils.load_spec(pkg).description.input[0]
    shape = tuple(inp.type.multiArrayType.shape)
    if os.path.isdir(compiled):
        m = ct.models.CompiledMLModel(compiled, compute_units=ct.ComputeUnit[units])
    else:
        m = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit[units])
    x = {inp.name: np.random.rand(*shape).astype(np.float16)}
    for _ in range(5):
        m.predict(x)
    barrier.wait()
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        m.predict(x)
        n += 1
    t1 = time.time()
    q.put({"calls": n, "t0": t0, "t1": t1, "batch": shape[0]})


def run_procs(model, units, k, seconds):
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(k)
    q = ctx.Queue()
    ps = [ctx.Process(target=_worker, args=(model, units, seconds, barrier, q)) for _ in range(k)]
    for p in ps:
        p.start()
    rs = [q.get() for _ in range(k)]
    for p in ps:
        p.join()
    per = [r["calls"] * r["batch"] / (r["t1"] - r["t0"]) for r in rs]
    return {"per_img_s": per, "total_img_s": sum(per),
            "t0": max(r["t0"] for r in rs), "t1": min(r["t1"] for r in rs)}


def run_swift(binary, modelc, k, seconds, units="ne", shared=False):
    cmd = [binary, modelc, str(k), str(seconds), units] + (["--shared"] if shared else [])
    # Timing window: the binary prints only at the end, so bracket by its own load time.
    t_start = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    t_end = time.time()
    r = json.loads(out.strip().splitlines()[-1])
    r["t1"] = t_end
    r["t0"] = t_end - seconds
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="CPU_AND_NE model (.mlpackage)")
    ap.add_argument("--modelc", required=True, help="the same model compiled (.mlmodelc), for Swift")
    ap.add_argument("--swift-bin", required=True)
    ap.add_argument("--naive", help="optional second model to repeat procs-1/2 with")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gap", type=float, default=3.0, help="idle seconds between phases")
    ap.add_argument("--phases", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    phases = [
        ("procs", "CPU_AND_NE", 1), ("procs", "CPU_AND_NE", 2), ("procs", "CPU_AND_NE", 4),
        ("inproc", "ne", 1), ("inproc", "ne", 2), ("inproc", "ne", 4),
        ("shared", "ne", 2),
        ("procs", "CPU_AND_GPU", 1),
    ]
    if a.naive:
        phases += [("procs-naive", "CPU_AND_NE", 1), ("procs-naive", "CPU_AND_NE", 2)]
    if a.phases != "all":
        keep = set(a.phases.split(","))
        phases = [p for p in phases if f"{p[0]}-{p[2]}" in keep or p[0] in keep]

    pw = Power()
    time.sleep(2)
    idle = pw.window(time.time() - 1.8, time.time())
    results = {"idle_power": idle, "phases": []}
    try:
        for kind, units, k in phases:
            runs = []
            for rep in range(a.repeats):
                time.sleep(a.gap)
                if kind == "procs":
                    r = run_procs(a.model, units, k, a.seconds)
                elif kind == "procs-naive":
                    r = run_procs(a.naive, units, k, a.seconds)
                else:
                    r = run_swift(a.swift_bin, a.modelc, k, a.seconds, units, kind == "shared")
                r["power"] = pw.window(r["t0"], r["t1"])
                runs.append(r)
                p = r["power"] or {}
                print(f"  {kind:11s} {units:11s} k={k} rep{rep}: total {r['total_img_s']:7.1f} img/s "
                      f"per {[round(v, 1) for v in r['per_img_s']]}  "
                      f"ANE {p.get('ane_power', float('nan')):5.2f} W  "
                      f"all {p.get('all_power', float('nan')):5.2f} W", flush=True)
            tot = [r["total_img_s"] for r in runs]
            med = statistics.median(tot)
            pmed = {}
            for key in ("all_power", "ane_power", "cpu_power", "gpu_power", "sys_power"):
                vals = [r["power"][key] for r in runs if r["power"]]
                if vals:
                    pmed[key] = statistics.median(vals)
            rec = {"kind": kind, "units": units, "k": k, "runs": runs,
                   "median_total_img_s": med, "min": min(tot), "max": max(tot),
                   "spread_pct": (max(tot) - min(tot)) / med * 100, "power_median": pmed}
            if pmed.get("all_power"):
                rec["mJ_per_img_all"] = pmed["all_power"] / med * 1000
                rec["mJ_per_img_ane"] = pmed.get("ane_power", 0) / med * 1000
            results["phases"].append(rec)
            print(f"{kind:11s} {units:11s} k={k}: median {med:7.1f} img/s spread {rec['spread_pct']:4.1f}%  "
                  f"{rec.get('mJ_per_img_all', float('nan')):6.2f} mJ/img (CPU+GPU+ANE)", flush=True)
    finally:
        pw.stop()

    solo = next((p["median_total_img_s"] for p in results["phases"]
                 if p["kind"] == "procs" and p["k"] == 1 and p["units"] == "CPU_AND_NE"), None)
    if solo:
        print("\nscaling vs procs-1 on CPU_AND_NE:")
        for p in results["phases"]:
            if p["units"] in ("CPU_AND_NE", "ne"):
                print(f"  {p['kind']:11s} k={p['k']}: {p['median_total_img_s'] / solo:.2f}x")
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
