#!/usr/bin/env python3
"""
Turn soak JSONs into the sustained table this repo reports, ready to paste.

    python3 tools/summarise_soak.py results/soak/m4pro-*.json

Reads what thermal_soak.py wrote and prints one row per compute unit: peak,
sustained, and the fraction held. Pure stdlib, so it runs without the virtualenv
and on any Python 3.8+, exactly like summarise.py.

SUSTAINED FRACTION IS last window / best window. 1.00 means the unit held its
peak for the whole soak. It is not comparable across soaks of different length:
the denominator is the best window and the numerator is the last one, so a
longer soak scores lower on the same hardware. Compare only equal durations, and
this prints the duration so you can check.

The numbers are images/s. Higher is better.
"""

import argparse
import glob
import json
import sys
from collections import OrderedDict

UNIT_LABEL = OrderedDict([
    ("CPU_AND_NE", "ANE"),
    ("CPU_AND_GPU", "GPU"),
    ("ALL", "default (`ALL`)"),
])

THERMAL = {0: "nominal", 1: "moderate", 2: "heavy", 3: "trapping", 4: "sleeping"}


def load(paths):
    out = []
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        if "summary" not in d:
            print(f"{p}: no summary -- soak was interrupted before it finished, "
                  f"skipping", file=sys.stderr)
            continue
        out.append(d)
    if not out:
        raise SystemExit("no completed soaks among: " + ", ".join(paths))
    return out


def host_line(d):
    m = d.get("machine", {})
    bits = [m.get("cpu", "unknown chip")]
    if m.get("model"):
        bits.append(f"({m['model']})")
    if m.get("macos"):
        bits.append(f"macOS {m['macos']}")
    p = d.get("power_start", {})
    if p.get("source"):
        bits.append(p["source"])
    return " · ".join(bits)


def render(soaks):
    soaks = sorted(soaks, key=lambda d: list(UNIT_LABEL).index(d["units"])
                   if d["units"] in UNIT_LABEL else 99)
    L = [f"**{host_line(soaks[0])}**", ""]
    L.append("| compute unit | peak img/s | last img/s | sustained |")
    L.append("| --- | ---: | ---: | ---: |")
    best_sustained = max((d["summary"]["sustained_fraction"] or 0) for d in soaks)
    for d in soaks:
        s = d["summary"]
        label = UNIT_LABEL.get(d["units"], f"`{d['units']}`")
        frac = s["sustained_fraction"]
        cell = f"**{frac:.3f}**" if frac == best_sustained else f"{frac:.3f}"
        L.append(f"| {label} | {s['best_images_per_s']:.1f} | "
                 f"{s['last_images_per_s']:.1f} | {cell} |")
    d0 = soaks[0]
    L.append("")
    L.append(f"sustained = last window / best window · {d0['seconds']:.0f} s soak · "
             f"{d0['window']:.0f} s windows · batch {d0['batch']}")

    # A soak that was interrupted, ran on battery, or changed power source
    # mid-run is not a result. Say so in the pasted output rather than leaving it
    # in the JSON where nobody looks.
    warn = []
    for d in soaks:
        u = d["units"]
        if d["summary"].get("interrupted"):
            warn.append(f"`{u}` was INTERRUPTED and is incomplete")
        start = d.get("power_start", {}).get("source")
        end = d.get("power_end", {}).get("source")
        if start and end and start != end:
            warn.append(f"`{u}` changed power source mid-run ({start} to {end}) "
                        f"-- discard it")
        elif start == "battery":
            warn.append(f"`{u}` ran on BATTERY, which throttles independently of heat")
    maxt = max((d["summary"].get("max_thermal") or 0) for d in soaks)
    if maxt:
        warn.append(f"thermal pressure reached {THERMAL.get(maxt, maxt)} -- the "
                    f"machine was shedding work")
    else:
        warn.append("thermal pressure stayed nominal, which does NOT mean the parts "
                    "were cool: it is a shed-load signal, not a temperature. Read it "
                    "one way only -- above nominal proves trouble, nominal proves "
                    "nothing.")
    if warn:
        L.append("")
        for w in warn:
            L.append(f"> {w}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("soaks", nargs="+", help="JSON files written by thermal_soak.py")
    a = ap.parse_args()
    # Shells that do not expand globs (or a quoted pattern) would otherwise pass
    # the literal string through and fail with a confusing file-not-found.
    paths = []
    for p in a.soaks:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    print(render(load(paths)))


if __name__ == "__main__":
    sys.exit(main())
