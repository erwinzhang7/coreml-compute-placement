#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does the SHAPE of an `ALL` split predict what the default costs?

PAPER.md §3.2 shows the default costing 1.18x to 5.18x on the M4 Pro and being
slower than *both* explicit placements on three of five architectures, then
explicitly declines to explain it: "This is consistent with `ALL` splitting a
graph across engines and paying transfer and synchronisation costs that exceed
the benefit, but we did not instrument the partition and do not claim the
mechanism."

The partition is now instrumented (`tools/placement_sweep.py`), and the two chips
behave completely differently:

    M5 Max   ALL resolves to 100% GPU on all five architectures. It never splits.
    M4 Pro   ALL splits on every architecture measured so far.

That alone explains the asymmetry in §3.2: the default is free on the M5 Max
because it is not doing anything, and expensive on the M4 Pro because it is.

THIS SCRIPT ASKS THE NEXT QUESTION. If the cost is handoff, then a more BALANCED
split should cost more, because a lopsided split moves less work across the
boundary. Define the minority share as 1 minus the largest device's cost
fraction. On the first three M4 Pro architectures it is monotone:

    siglip     minority 21.1%  ->  1.19x
    bert       minority 22.5%  ->  1.36x
    whisper    minority 35.9%  ->  5.18x

THREE POINTS IS NOT A RESULT. Monotone ordering of three points happens by chance
one time in three. This is a hypothesis with a pre-registered test, not a finding,
and it was recorded here before the data that decides it existed.

PREDICTION, written 2026-08-28 before resnet50 and mobilenet were measured on the
M4 Pro. Their default costs were already known from the throughput sweep:

    resnet50   costs 2.41x, between bert and whisper
               -> predicted minority share between 22.5% and 35.9%
    mobilenet  costs 1.18x, the cheapest of the five
               -> predicted minority share at or below 21.1%, or no split at all

RESULT: 1 held, 1 FAILED. mobilenet came back 100% ANE, no split at all, as
predicted. resnet50 came back 87.6% ANE / 12.4% GPU, a minority share of 12.4%,
which is the SMALLEST split of the five and yet the second most expensive default.

    mobilenet    0.0%  ->  1.18x
    resnet50    12.4%  ->  2.41x
    siglip      21.1%  ->  1.19x
    bert        22.5%  ->  1.36x
    whisper     35.9%  ->  5.18x

So the handoff-share hypothesis is dead. How balanced a split is does not predict
what it costs, and resnet50 is the counterexample: 12.4% of the work moved to the
other engine costs 58% of the throughput.

WHAT REPLACED IT, and it is better because it separates two things the single
number was mixing. The default can fail in two independent ways:

    handoff       majority-engine throughput / ALL throughput
                  what the partition costs, measured against the engine that got
                  most of the work, so it is not confounded with unit choice

    wrong engine  best-engine throughput / majority-engine throughput
                  what choosing the wrong majority engine costs, with the
                  partition held fixed

Their product is the cost the paper quotes. That multiplication is an ALGEBRAIC
IDENTITY, (major/ALL) * (best/major) = best/ALL, so the fact that it comes out
exact is arithmetic and not evidence of anything. What is empirical is WHICH
factor is 1.0, and the five architectures fall into three groups:

    siglip, resnet50, bert    wrong engine 1.00x, all the cost is handoff
    mobilenet                 handoff 0.99x, all the cost is unit choice, because
                              it does not split at all and still picks the ANE
                              when the GPU is 1.19x faster
    whisper                   both, 2.01x handoff and 2.58x wrong engine

That is the useful statement: the default has two failure modes, they occur
separately, and only one of them has anything to do with splitting.

    python tools/split_cost.py
    python tools/split_cost.py --placement results/placement-m5-max.json \
                              --sweep results/zoo-m5max-v2.json
"""
import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# Written before the deciding data existed. Do not edit to fit an outcome; if a
# prediction fails, say so and change the hypothesis.
PREDICTIONS = {
    "resnet50":  (0.225, 0.359, "2.41x, between bert and whisper"),
    "mobilenet": (0.000, 0.211, "1.18x, the cheapest of the five"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", default=str(REPO / "results" / "placement-m4-pro.json"))
    ap.add_argument("--sweep", nargs="+",
                    default=["zoo-m4pro-dtypefix.json", "sweep-m4pro-rerun.json",
                             "sweep-m4pro.json"])
    args = ap.parse_args()

    pp = pathlib.Path(args.placement)
    if not pp.exists():
        sys.exit(f"no placement file: {pp}")
    pm = json.loads(pp.read_text())

    thr = {}
    for f in args.sweep:
        p = REPO / "results" / f
        if p.exists():
            for r in json.loads(p.read_text()):
                thr.setdefault((r["model"], r["units"]), r["median"])

    rows = []
    for r in pm["rows"]:
        if r["units"] != "ALL":
            continue
        m = r["model"]
        frac = r["by_cost_fraction"]
        if not frac:
            continue
        minority = 1 - max(frac.values())
        a, ane, gpu = (thr.get((m, u)) for u in ("ALL", "CPU_AND_NE", "CPU_AND_GPU"))
        if not (a and ane and gpu):
            continue
        # The engine that got most of the work, and what it does on its own.
        # Comparing ALL against THAT rather than against the best engine is what
        # separates the partition cost from the unit-choice cost.
        major = max(frac, key=frac.get)
        maj_thr = {"ANE": ane, "GPU": gpu}.get(major) or max(ane, gpu)
        rows.append({
            "model": m.replace("-b16.mlpackage", "").replace("siglip-vision", "siglip"),
            "minority": minority,
            "cost": max(ane, gpu) / a,
            "handoff": maj_thr / a,
            "wrong_engine": max(ane, gpu) / maj_thr,
            "split": " / ".join("%.1f%% %s" % (f * 100, d) for d, f in
                                sorted(frac.items(), key=lambda x: -x[1]) if f >= 0.005),
        })
    if not rows:
        sys.exit("no ALL rows with matching throughput; check --sweep")

    print("chip %s\n" % pm.get("chip", "?"))
    print("%-10s %-36s %9s %8s" % ("model", "ALL placement by cost", "minority", "cost"))
    for r in sorted(rows, key=lambda x: x["minority"]):
        print("%-10s %-36s %8.1f%% %7.2fx" % (r["model"], r["split"],
                                              r["minority"] * 100, r["cost"]))

    # The two failure modes, separated. The product is an identity and proves
    # nothing; which factor is 1.00 is the finding.
    print("\n%-10s %9s %13s %9s   %s" % ("model", "handoff", "wrong engine",
                                         "product", "dominant failure"))
    for r in sorted(rows, key=lambda x: -x["cost"]):
        split = r["minority"] >= 0.005
        h, w = r["handoff"] >= 1.02, r["wrong_engine"] >= 1.02
        if not split:
            # No partition happened, so whatever the handoff factor is, it is not
            # the cost of one. On the M5 Max `ALL` resolves to 100% GPU and still
            # comes in up to 1.09x under the pure GPU run; that is `ALL`-path
            # overhead or run-to-run spread, and calling it partition cost would
            # invent a mechanism the placement data rules out.
            mode = ("unit choice only" if w else
                    "no split; residual is ALL-path overhead or noise" if h else
                    "neither, default is fine")
        elif h and w:
            mode = "both"
        elif w:
            mode = "unit choice only"
        elif h:
            mode = "partition only"
        else:
            mode = "neither, default is fine"
        print("%-10s %8.2fx %12.2fx %8.2fx   %s"
              % (r["model"], r["handoff"], r["wrong_engine"],
                 r["handoff"] * r["wrong_engine"], mode))

    s = sorted(rows, key=lambda x: x["minority"])
    spread = s[-1]["minority"] - s[0]["minority"]
    if spread < 0.005:
        # Every model resolves to one device, so there is no minority share to
        # order by and monotonicity is undefined rather than false. This is the
        # M5 Max case and it is the CONTROL the hypothesis needs: no split, and
        # the default costs nothing.
        print("\nno split anywhere on this chip (minority share %.1f%% for every "
              "architecture)." % (s[0]["minority"] * 100))
        print("Monotonicity is undefined with no variation in x. The default costs "
              "%.2fx to %.2fx here,\nwhich is the control: where `ALL` does not split, "
              "it does not cost."
              % (min(r["cost"] for r in rows), max(r["cost"] for r in rows)))
    else:
        mono = all(s[i]["cost"] <= s[i + 1]["cost"] for i in range(len(s) - 1))
        print("\nmonotone in minority share: %s  (n=%d)" % ("YES" if mono else "NO", len(s)))
        if len(s) < 5:
            print("n<5, so this is still a hypothesis. Ordering three points monotonically\n"
                  "happens by chance one time in three.")

    # The predictions were registered for the M4 Pro, where `ALL` splits. Firing
    # them against another chip's file compares against intervals that were never
    # about it.
    chip = str(pm.get("chip", ""))
    hit = miss = 0
    for r in rows:
        if r["model"] not in PREDICTIONS or "m4" not in chip:
            continue
        lo, hi, why = PREDICTIONS[r["model"]]
        ok = lo <= r["minority"] <= hi
        hit, miss = hit + ok, miss + (not ok)
        print("\nPREDICTION %s (%s)" % (r["model"], why))
        print("  expected minority in [%.1f%%, %.1f%%], measured %.1f%%  -> %s"
              % (lo * 100, hi * 100, r["minority"] * 100, "HELD" if ok else "FAILED"))
    if hit or miss:
        print("\npre-registered predictions: %d held, %d failed" % (hit, miss))
        if miss:
            print("The handoff-share hypothesis is wrong as stated. Say so in PAPER.md\n"
                  "rather than widening the interval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
