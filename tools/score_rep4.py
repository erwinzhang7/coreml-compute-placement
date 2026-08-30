#!/usr/bin/env python
"""Score series 4 against a prediction fixed BEFORE it landed.

Series 4 started 16:00:22 (inference1) and 16:00:24 (inference2) on 2026-08-29,
12 soaks each, and finishes ~16:42. This file was written and committed at ~16:10
while the runs were in flight, so the prediction below cannot have been chosen
from the answer.

WHAT THE PRIOR TWO FLOOR SERIES SHOW, computed here from the local files rather
than quoted from memory (best_images_per_s, inference1 minus inference2):

    rep2floor   box diff +0.0250 img/s   within-series drift -0.0003 relative
    rep3floor   box diff +0.0158 img/s   within-series drift +0.0003 relative

Both boxes sit at ~177.6 img/s, so those differences are ~0.01% -- the floor
series exists precisely because the quantity of interest is this small.

PREDICTION, recorded before series 4 is readable:
  1. BOX DIFFERENCE lands in [0.000, 0.040] img/s, bracketing the two priors
     with room either side. Direction not predicted: two observations cannot
     establish a sign, and #177 already found the drift is not a box property.
  2. WITHIN-SERIES DRIFT stays under 0.001 in relative terms on both boxes,
     matching rep2 and rep3 which came in at 0.0003 in opposite directions.

FALSIFIER, and the more useful outcome: a box difference outside that interval,
or a drift an order of magnitude larger, would mean the floor series is not
stable across sessions and #173's "the floor is the harness" claim needs
revisiting rather than extending.

A UNITS TRAP, recorded because it nearly produced a false alarm. PAPER.md
records session-matched box differences of 0.0173 and 0.0203 for "#3 vs #4" at
fixed clock windows. Those are a DIFFERENT quantity from a DIFFERENT series than
the repNfloor runs, and comparing them to these numbers looked like a 100x
discrepancy until I found the source line. Two numbers in the same units and the
same rough magnitude are not necessarily the same measurement.
"""
import argparse
import glob
import json
import re
import statistics as st

PRIORS = {"rep2floor": (0.0250, -0.0003), "rep3floor": (0.0158, 0.0003)}
BOX_LO, BOX_HI, DRIFT_MAX = 0.000, 0.040, 0.001


def load(series, root):
    out = {}
    for f in glob.glob("%s/%s-*.json" % (root, series)):
        m = re.search(r"%s-([a-z0-9]+)-siglip-CPU_AND_GPU-(\d+)\.json" % series, f)
        if not m:
            continue
        out.setdefault(m.group(1), {})[int(m.group(2))] = \
            json.load(open(f))["summary"]["best_images_per_s"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/soak")
    ap.add_argument("--series", default="rep4floor")
    a = ap.parse_args()

    d = load(a.series, a.root)
    if not d:
        print("  %s not present yet" % a.series)
        raise SystemExit(0)
    boxes = sorted(d)
    print()
    for b in boxes:
        v = [d[b][i] for i in sorted(d[b])]
        f3, l3 = st.mean(v[:3]), st.mean(v[-3:])
        print("  %-12s n=%2d  mean %7.2f  drift %+.4f rel  %s"
              % (b, len(v), st.mean(v), (l3 - f3) / f3,
                 "OK" if len(v) == 12 else "INCOMPLETE -- do not score"))
    if len(boxes) != 2 or any(len(d[b]) != 12 for b in boxes):
        print("  REFUSING to score: need 12 runs on each of 2 boxes.")
        raise SystemExit(0)

    A = [d[boxes[0]][i] for i in sorted(d[boxes[0]])]
    B = [d[boxes[1]][i] for i in sorted(d[boxes[1]])]
    diff = st.mean(A) - st.mean(B)
    drifts = []
    for b in boxes:
        v = [d[b][i] for i in sorted(d[b])]
        drifts.append((st.mean(v[-3:]) - st.mean(v[:3])) / st.mean(v[:3]))
    print()
    print("  priors: rep2 %+.4f, rep3 %+.4f img/s" % (PRIORS["rep2floor"][0], PRIORS["rep3floor"][0]))
    print("  MEASURED box diff %+.4f img/s (%s minus %s)" % (diff, boxes[0], boxes[1]))
    print("  MEASURED drift    %+.4f / %+.4f relative" % tuple(drifts))
    print()
    ok_box = BOX_LO <= abs(diff) <= BOX_HI
    ok_drift = all(abs(x) < DRIFT_MAX for x in drifts)
    print("  PREDICTION 1 box diff in [%.3f, %.3f]:  %s" % (BOX_LO, BOX_HI, "HELD" if ok_box else "FALSIFIED"))
    print("  PREDICTION 2 drift under %.3f rel:      %s" % (DRIFT_MAX, "HELD" if ok_drift else "FALSIFIED"))
    if not (ok_box and ok_drift):
        print()
        print("  The floor series is not stable across sessions. #173's 'the floor")
        print("  is the harness' needs revisiting rather than extending.")
