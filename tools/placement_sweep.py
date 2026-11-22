#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Where does `ALL` actually put each model? Every model, every setting, into a file.

THE GAP THIS CLOSES. The README carried four rows showing what `ComputeUnit.ALL`
resolves to, and the most important of them is that on an M4 Pro the naive SigLIP
graph is split 78.9% ANE / 21.1% GPU by cost, and that split runs slower than pure
ANE *and* slower than pure GPU. That is the direct evidence that cross-device
handoff, rather than a bad choice of unit, is what makes the default lose.

Those four rows had no file behind them. They were produced by hand and typed into
a table. `anecheck.py` has always been able to write JSON, but `run.sh` called it
once, on one model, under one setting, so re-running the standard pipeline did not
regenerate three of the four. Everything else a claim rests on in this repo has a
file (results/conversion-check.json exists for exactly this reason), and this did
not.

PAPER.md 3.2 is deliberately weaker than the README here. It says the default can
be slower than both explicit placements, then declines to claim the mechanism:
"we did not instrument the partition". That is correct for the five-architecture
set, because the partition had only ever been measured for siglip. Running this
over all five is what would let that hedge become a result, so the interesting
output is not the siglip row we already have but resnet50 and whisper, the other
two architectures where `ALL` is below both.

STATIC, NOT TIMED. This reads MLComputePlan, which is a plan rather than a trace.
It costs a model load and a compile per cell and does not measure throughput, so
it is safe to run on a box that is busy with something else. The throughput
columns are joined in from an existing sweep file rather than measured here.

    python tools/placement_sweep.py --models *.mlpackage --sweep results/sweep-m4pro.json
    python tools/placement_sweep.py --models siglip-vision-b16.mlpackage --out /tmp/p.json
"""
import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coremltools as ct                                          # noqa: E402
from anecheck import analyse, ensure_compiled, summarise          # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
UNITS = ["ALL", "CPU_AND_NE", "CPU_AND_GPU"]


def chip_tag():
    """Short, filename-safe chip name, so two boxes cannot overwrite each other.

    The soak tool hit exactly this: LABEL derived from the brand string alone, so
    both M4 Pro minis resolved to `m4-pro` and wrote the same path. Include the
    hardware model, which differs between machine classes.
    """
    def s(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    brand = s("sysctl", "-n", "machdep.cpu.brand_string") or "unknown"
    return brand.replace("Apple ", "").replace(" ", "-").lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--units", nargs="+", default=UNITS)
    ap.add_argument("--sweep", default="",
                    help="a results/sweep-*.json or zoo-*.json to join throughput "
                         "from; without it the table prints placement only")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tag = chip_tag()
    out = args.out or str(REPO / "results" / f"placement-{tag}.json")

    # Throughput, if offered. Keyed (model basename, units) -> median.
    thr = {}
    if args.sweep:
        for r in json.loads(pathlib.Path(args.sweep).read_text()):
            thr[(os.path.basename(r["model"]), r["units"])] = r["median"]

    rows = []
    for m in args.models:
        if not os.path.exists(m):
            print(f"SKIP {m}: not found", file=sys.stderr)
            continue
        for u in args.units:
            try:
                with tempfile.TemporaryDirectory() as wd:
                    compiled = ensure_compiled(m, wd)
                    ops = analyse(compiled, ct.ComputeUnit[u], False)
                s = summarise(ops)
            except SystemExit as exc:
                print(f"SKIP {m} {u}: {exc}", file=sys.stderr)
                continue
            rows.append({
                "model": os.path.basename(m),
                "units": u,
                "ops": s["ops"],
                "by_cost_fraction": s["by_cost_fraction"],
                "by_count_fraction": s["by_count_fraction"],
                "ane_fraction_by_cost": s["ane_fraction_by_cost"],
                "median_images_per_s": thr.get((os.path.basename(m), u)),
            })
            frac = "  ".join(f"{d} {f:.1%}" for d, f in
                             sorted(s["by_cost_fraction"].items(), key=lambda x: -x[1]))
            print(f"{os.path.basename(m):28s} {u:12s} {frac}")

    if not rows:
        sys.exit("no models analysed")

    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"chip": tag, "macos": platform.mac_ver()[0],
                   "coremltools": ct.__version__, "rows": rows}, fh, indent=2)
    print(f"\nwrote {out}")

    # The table the README carries, generated rather than typed.
    print()
    print("| model | `ALL` placement (by cost) | `ALL` img/s | best pure | |")
    print("| --- | --- | ---: | ---: | --- |")
    for m in dict.fromkeys(r["model"] for r in rows):
        cells = {r["units"]: r for r in rows if r["model"] == m}
        a = cells.get("ALL")
        if not a:
            continue
        split = " / ".join(f"{f:.1%} {d}" for d, f in
                           sorted(a["by_cost_fraction"].items(), key=lambda x: -x[1])
                           if f >= 0.005)
        pure = {u: cells[u]["median_images_per_s"] for u in ("CPU_AND_NE", "CPU_AND_GPU")
                if u in cells and cells[u]["median_images_per_s"]}
        if a["median_images_per_s"] and pure:
            best_u = max(pure, key=pure.get)
            verdict = ("**below both**" if a["median_images_per_s"] < min(pure.values())
                       else "matches best" if a["median_images_per_s"] >= max(pure.values())
                       else "between")
            print(f"| `{m}` | {split} | {a['median_images_per_s']:.1f} | "
                  f"{max(pure.values()):.1f} ({best_u}) | {verdict} |")
        else:
            print(f"| `{m}` | {split} | n/a | n/a | throughput not joined |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
