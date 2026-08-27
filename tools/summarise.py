#!/usr/bin/env python3
"""
Turn a sweep JSON into the table this repo reports, ready to paste.

    python3 tools/summarise.py results/sweep-m5max.json

Reads what sweep.py wrote and prints one row per model: throughput under each
compute unit, which unit won, and by how much. Pure stdlib, so it runs without
the virtualenv and on any Python 3.8+.

The numbers are images/s. Higher is better. Reading them as latency inverts
every conclusion in the README, so the header says so explicitly.
"""

import argparse
import json
import sys
from collections import OrderedDict

UNIT_LABEL = OrderedDict([
    ("CPU_AND_NE", "ANE"),
    ("CPU_AND_GPU", "GPU"),
    ("ALL", "default (`ALL`)"),
])


def load(path):
    with open(path) as fh:
        rows = json.load(fh)
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path}: expected a non-empty list of sweep records")
    return rows


def host_line(rows):
    h = rows[0].get("host", {})
    bits = [h.get("cpu", "unknown chip")]
    if h.get("model"):
        bits.append(f"({h['model']})")
    if h.get("gpu_cores"):
        bits.append(f"{h['gpu_cores']} GPU cores")
    if h.get("macos"):
        bits.append(f"macOS {h['macos']}")
    return " · ".join(bits)


def summarise(rows):
    models = OrderedDict()
    for r in rows:
        models.setdefault(r["model"], {})[r["units"]] = r

    out = []
    for model, by_unit in models.items():
        cells, best, best_v, worst_v = {}, None, None, None
        for unit in UNIT_LABEL:
            rec = by_unit.get(unit)
            if not rec:
                continue
            v = rec["median"]
            cells[unit] = v
            if best_v is None or v > best_v:
                best_v, best = v, unit
            if worst_v is None or v < worst_v:
                worst_v = v
        out.append({
            "model": model, "cells": cells, "best": best,
            "spread": (best_v / worst_v) if worst_v else None,
            "batch": by_unit[next(iter(by_unit))].get("batch"),
            "repeats": by_unit[next(iter(by_unit))].get("repeats"),
        })
    return out


def render(rows, summary):
    L = []
    L.append(f"**{host_line(rows)}**")
    L.append("")
    L.append("| model | " + " | ".join(UNIT_LABEL.values()) + " | fastest | spread |")
    L.append("| --- | " + " | ".join("---:" for _ in UNIT_LABEL) + " | :---: | ---: |")
    for s in summary:
        cells = []
        for unit in UNIT_LABEL:
            v = s["cells"].get(unit)
            if v is None:
                cells.append("–")
            elif unit == s["best"]:
                cells.append(f"**{v:.1f}**")
            else:
                cells.append(f"{v:.1f}")
        spread = f"{s['spread']:.2f}x" if s["spread"] else "–"
        L.append(f"| `{s['model']}` | " + " | ".join(cells)
                 + f" | {UNIT_LABEL[s['best']]} | {spread} |")
    b = summary[0]
    L.append("")
    L.append(f"images/s, higher is better · batch {b['batch']} · median of {b['repeats']} runs")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", help="a JSON file written by tools/sweep.py")
    a = ap.parse_args()

    rows = load(a.sweep)
    summary = summarise(rows)
    print(render(rows, summary))

    # The interesting question is not the absolute rate, it is whether the
    # ranking matches what the README found on the two chips measured there.
    for s in summary:
        if s["best"] == "ALL":
            print(f"\n> `{s['model']}`: the default was fastest here.")
        elif s["cells"].get("ALL") is not None and s["cells"]["ALL"] < min(
                v for u, v in s["cells"].items() if u != "ALL"):
            print(f"\n> `{s['model']}`: the default was **slower than either pure "
                  f"placement**, the M4 Pro behaviour described in the README.")


if __name__ == "__main__":
    sys.exit(main())
