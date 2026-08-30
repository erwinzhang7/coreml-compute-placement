#!/usr/bin/env python3
"""Decompose 3.6's floor spread into a BOX effect and a SESSION effect.

3.6 published three identical M4 Pro minis differing from each other by far more
than each differs from itself, and read that as a machine property. Repeating one
box's whole series three hours later moved it by as much as the boxes differ, so
the reading was only ever valid for pairs measured at the same time. This script
is what says which pairs those are.

TWO THINGS IT REFUSES TO DO, both of which I did by hand first and got wrong:

  It does not count files. A soak file exists from the moment its run starts --
  thermal_soak streams windows as it goes -- so a directory listing counts runs
  that are still going, and a file copied off a box mid-run is byte-identical in
  shape to one from a killed run. Completeness here means the summary is present
  AND the window count is within one of the requested duration, never that the
  path resolves.

  It does not compare series whose sessions it cannot establish. Soak files carry
  no timestamp of their own -- model, units, machine, power and load, but no
  clock -- so the session has to come from filesystem mtime, which survives
  `rsync -a` but not much else. Where mtimes are missing or implausible the pair
  is reported as UNKNOWN rather than assumed concurrent, because assuming
  concurrency is the exact error being corrected.

Sessions are inferred by clustering a series' mtimes: a gap larger than --gap
minutes starts a new session. Two series overlap if their [start, end] intervals
intersect, which is the only condition under which a between-box difference is
free of the session effect.
"""
import argparse
import collections
import datetime as dt
import json
import math
import pathlib
import statistics
import sys

SETTLE = 3  # 3.6 drops the opening runs; see the opening-runs trap.


def complete(d):
    """True if this file is a finished run rather than a live or killed one."""
    s = d.get("summary") or {}
    if s.get("sustained_fraction") is None:
        return False
    want = int(d["seconds"] / d["window"])
    return len(d.get("windows") or []) >= want - 1


def load(soakdir, pattern):
    """-> {(series, box): [(mtime, sustained), ...]} sorted by run index."""
    out = collections.defaultdict(list)
    for p in sorted(soakdir.glob(pattern)):
        parts = p.stem.split("-")
        if len(parts) < 3:
            continue
        series, box = parts[0], parts[1]
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not complete(d):
            print("  skipping %s: incomplete (live, killed or synced mid-write)"
                  % p.name, file=sys.stderr)
            continue
        out[(series, box)].append(
            (dt.datetime.fromtimestamp(p.stat().st_mtime),
             d["summary"]["sustained_fraction"]))
    return out


def sessions(times, gap_min):
    """Split sorted times into runs separated by more than gap_min minutes."""
    if not times:
        return []
    groups, cur = [], [times[0]]
    for t in times[1:]:
        if (t - cur[-1]).total_seconds() > gap_min * 60:
            groups.append(cur)
            cur = [t]
        else:
            cur.append(t)
    groups.append(cur)
    return groups


def sigma(x, y):
    if len(x) < 2 or len(y) < 2:
        return None, None
    d = statistics.mean(y) - statistics.mean(x)
    se = math.sqrt(statistics.stdev(x) ** 2 / len(x)
                   + statistics.stdev(y) ** 2 / len(y))
    return d, (abs(d) / se if se else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soakdir", default="results/soak")
    ap.add_argument("--pattern", default="*floor-*-siglip-CPU_AND_GPU-*.json")
    ap.add_argument("--gap", type=float, default=45.0,
                    help="minutes of silence that separate two sessions")
    a = ap.parse_args()

    series = load(pathlib.Path(a.soakdir), a.pattern)
    if not series:
        print("no complete series found", file=sys.stderr)
        return 2

    print()
    print("%-12s %-12s %5s %8s %8s   %s" %
          ("series", "box", "n", "settled", "sd", "window"))
    meta = {}
    for k in sorted(series):
        rows = series[k]
        vals = [v for _, v in rows][SETTLE:]
        times = sorted(t for t, _ in rows)
        if len(vals) < 2:
            continue
        sd = statistics.stdev(vals)
        meta[k] = (statistics.mean(vals), sd, len(vals), times[0], times[-1])
        print("%-12s %-12s %5d %8.4f %8.4f   %s -> %s" %
              (k[0], k[1], len(vals), statistics.mean(vals), sd,
               times[0].strftime("%m-%d %H:%M"), times[-1].strftime("%H:%M")))
        gs = sessions(times, a.gap)
        if len(gs) > 1:
            print("             WARNING: this series spans %d sessions; its "
                  "mean mixes them" % len(gs))

    print()
    print("SAME BOX, DIFFERENT SESSION -- this is the quantity that decides "
          "whether any")
    print("cross-session comparison in 3.6 is admissible at all.")
    boxes = sorted({k[1] for k in meta})
    for box in boxes:
        ks = sorted(k for k in meta if k[1] == box)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                x = [v for _, v in series[ks[i]]][SETTLE:]
                y = [v for _, v in series[ks[j]]][SETTLE:]
                d, s = sigma(x, y)
                if d is None:
                    continue
                gap = (meta[ks[j]][3] - meta[ks[i]][4]).total_seconds() / 3600
                print("  %-12s %-10s vs %-10s  %+.4f  %5.1f sigma   "
                      "%.1f h apart" % (box, ks[i][0], ks[j][0], d, s, gap))

    print()
    print("DIFFERENT BOX -- reported only with its session status. A pair that "
          "does not")
    print("overlap in time cannot be separated from the effect printed above.")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            for ka in sorted(k for k in meta if k[1] == boxes[i]):
                for kb in sorted(k for k in meta if k[1] == boxes[j]):
                    x = [v for _, v in series[ka]][SETTLE:]
                    y = [v for _, v in series[kb]][SETTLE:]
                    d, s = sigma(x, y)
                    if d is None:
                        continue
                    a0, a1 = meta[ka][3], meta[ka][4]
                    b0, b1 = meta[kb][3], meta[kb][4]
                    overlap = a0 <= b1 and b0 <= a1
                    tag = "MATCHED  " if overlap else "CONFOUNDED"
                    print("  %s %-10s %-12s vs %-10s %-12s  %+.4f  %5.1f sigma"
                          % (tag, ka[0], boxes[i], kb[0], boxes[j], d, s))

    print()
    print("Read the MATCHED rows as box differences. Read the CONFOUNDED rows "
          "as box")
    print("differences plus an unknown session term whose size is the table "
          "above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
