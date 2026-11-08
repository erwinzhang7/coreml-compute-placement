#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Which of M6's two Neural Engines does work land on?

Throughput alone cannot tell "one instance already uses both engines" from "one engine
does the work and the other idles while a second caller cannot reach it". IOReport can:
per engine it exposes interrupt counts (ane / ane1), the engine's IOP power state and its
SoC device state, all readable without root through bin/ioreport_ane. (The memory
controller's per-ANE byte counters are listed but do not come back in a subscription, so
per-engine DRAM traffic is not available this way.)

For each configuration: run bin/multi_model for SECONDS while sampling, keep the samples
inside the timed window, and report per-engine rates and the fraction of time each
engine's IOP reports Running.
"""
import argparse
import collections
import json
import subprocess
import time


def summarise(samples):
    tot = collections.defaultdict(float)
    secs = 0.0
    for s in samples:
        secs += s["ms"] / 1000
        for k, v in s["ch"].items():
            g, sg, n = (x.strip() for x in k.split("|"))
            if g.startswith("Interrupt") and n == "First Level Interrupt Handler Count":
                eng = sg.split()[0]
                if eng in ("ane", "ane1", "dart-ane", "dart-ane1"):
                    tot[f"irq_per_s {eng}"] += v
            elif g in ("ANE", "ANE1") and sg == "IOP State":
                t = sum(v.values()) or 1
                tot[f"iop_running_frac {g}"] += v.get("Running", 0) / t * s["ms"] / 1000
            elif g == "SoC Stats" and sg == "Device States" and n in ("ANE0_SYS", "ANE1_SYS"):
                t = sum(v.values()) or 1
                off = v.get("PWROFF", 0) + v.get("CLKOFF", 0)
                tot[f"sys_on_frac {n}"] += (1 - off / t) * s["ms"] / 1000
    return {k: round(v / secs, 3) for k, v in sorted(tot.items())} if secs else {}


def run(cfg, a):
    # To a file, not a pipe: each sample line is ~14 KB and nothing reads until the end, so
    # a pipe fills after four lines and blocks the sampler for the rest of the run.
    log = open(a.out + ".samples", "w+")
    smp = subprocess.Popen([a.ioreport, "sample", str(a.ms), str(int((a.seconds + 120) * 1000 / a.ms))],
                           stdout=log, text=True)
    t_start = time.time()
    if cfg["kind"] == "idle":
        time.sleep(a.seconds)
        res = {"total_img_s": 0.0}
        t1 = time.time()
    else:
        cmd = [a.swift_bin, cfg["model"], str(cfg["k"]), str(a.seconds), cfg.get("units", "ne")]
        res = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True)
                         .stdout.strip().splitlines()[-1])
        t1 = time.time()
    smp.terminate()
    smp.wait()
    log.seek(0)
    out = log.read()
    log.close()
    rows = []
    for l in out.splitlines():
        try:  # the last line can be cut off by terminate()
            rows.append(json.loads(l))
        except ValueError:
            pass
    t0 = t1 - a.seconds
    # A delta stamped t covers (t - ms, t]; keep deltas wholly inside the timed window.
    win = [r for r in rows if r["t"] - r["ms"] / 1000 >= t0 and r["t"] <= t1]
    return {"config": cfg, "img_s": res.get("total_img_s"), "per_img_s": res.get("per_img_s"),
            "n_samples": len(win), "engines": summarise(win), "load_wall_s": round(t0 - t_start, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swift-bin", default="bin/multi_model")
    ap.add_argument("--ioreport", default="bin/ioreport_ane")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--ms", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfgs = [
        {"name": "idle", "kind": "idle"},
        {"name": "ane-b16 x1", "kind": "run", "model": "models/siglip-ane-b16.mlmodelc", "k": 1},
        {"name": "ane-b1 x1", "kind": "run", "model": "models/siglip-ane-b1.mlmodelc", "k": 1},
        {"name": "ane-b16 x2", "kind": "run", "model": "models/siglip-ane-b16.mlmodelc", "k": 2},
        {"name": "naive-b16 x1", "kind": "run", "model": "models/siglip-vision-b16.mlmodelc", "k": 1},
        {"name": "ane-b16 x1 ALL", "kind": "run", "model": "models/siglip-ane-b16.mlmodelc", "k": 1,
         "units": "all"},
    ]
    results = []
    for c in cfgs:
        time.sleep(3)
        r = run(c, a)
        results.append(r)
        print(f"{c['name']:16s} {r['img_s'] or 0:7.1f} img/s  n={r['n_samples']}  {r['engines']}", flush=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
