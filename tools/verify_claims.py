#!/usr/bin/env python3
"""Recompute every headline number in PAPER.md from results/ and require the
paper to contain it.

WHY THIS EXISTS. The M5 Max ANE figures were measured on a contended machine and
came out roughly 2x low. They were retracted, and the medians table and the
per-model table in 3.1 were corrected. Three prose sites were not:

    abstract      "the two that do not invert still move by 4.5x and 6.6x"
    3.1 prose     "still move by factors of 4.5 and 6.6"
    3.1 prose     "1.19x on the M4 Pro and 5.40x on the M5 Max"

The correct values are 2.4, 3.6 and 2.89. Halving the M5 Max ANE reproduces 4.9,
7.1 and 5.78, which is where those numbers came from. So the paper contradicted
its own table two lines above, for a full day, in the abstract.

A retraction is not done when the table is fixed. Every derived number has to be
recomputed, and the only way to know they all were is to recompute them from the
data every time.

DIRECTION OF THE CHECK. This does not parse the prose and compare loosely. It
recomputes each quantity, formats it exactly as the paper writes it, and requires
that string to appear in PAPER.md. A number that drifts in either the data or the
prose breaks the check. Text that says something the data does not support fails
even when the arithmetic elsewhere is right.

PROVENANCE IS PINNED, NOT SEARCHED. results/ holds several generations of sweep,
including pre-dtypefix and pre-retraction files whose numbers are wrong. Picking
files by glob would silently mix them. The two sources below are the ones that
reproduce the paper's 3.1 medians table exactly, and the script asserts that
before checking anything derived from them.

    python tools/verify_claims.py
    python tools/verify_claims.py --inject   # corrupt one median, require failure
"""
import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
PAPER = REPO / "PAPER.md"
# The README summarises the paper's headline numbers, so it can go stale on its
# own. It did: it still read "21 of 22 soaks ... in 41" long after the paper said
# 44 of 45 and 62, because this script only ever read PAPER.md. A checker that
# guards one copy of a claim and not the other is the "claims live in many
# places" failure with a tool attached.
README = REPO / "README.md"

# Pinned sources. M4 Pro's four added models come from the dtype-corrected zoo
# run; siglip predates the zoo and comes from its own sweep. M5 Max is one file.
M4_SOURCES = ["zoo-m4pro-dtypefix.json", "sweep-m4pro-rerun.json", "sweep-m4pro.json"]
M5_SOURCES = ["zoo-m5max-v2.json"]

# The 3.1 medians table, as printed. Recomputation must reproduce these before
# any derived claim is trusted.
PAPER_TABLE = {
    "siglip":    dict(m4_ane=204.4,  m4_gpu=178.8,   m5_ane=232.9,  m5_gpu=1077.7),
    "resnet50":  dict(m4_ane=929.6,  m4_gpu=659.5,   m5_ane=1094.6, m5_gpu=3277.5),
    "mobilenet": dict(m4_ane=3044.7, m4_gpu=3610.9,  m5_ane=3602.2, m5_gpu=10418.4),
    "bert":      dict(m4_ane=757.3,  m4_gpu=557.6,   m5_ane=822.5,  m5_gpu=3517.1),
    "whisper":   dict(m4_ane=56.3,   m4_gpu=145.1,   m5_ane=65.9,   m5_gpu=605.4),
}
ARCHITECTURES = list(PAPER_TABLE)


def load(names):
    """Merge the pinned files, first file wins so the corrected run takes
    precedence over the older one it supersedes."""
    by = {}
    for n in names:
        p = REPO / "results" / n
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            m = r["model"].replace("-b16.mlpackage", "").replace(".mlpackage", "")
            m = m.replace("siglip-vision", "siglip")
            if m == "siglip-ane":
                continue  # a second export of siglip, not one of the five
            by.setdefault(m, {}).setdefault(r["units"], r["median"])
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-set", action="store_true",
                    help="rewrite results/soak/PAPER-SET.txt from what is on disk. "
                         "Use when deliberately extending the dataset, then update "
                         "the prose in the same commit.")
    ap.add_argument("--inject", action="store_true",
                    help="halve one M5 Max ANE median, reproducing the retracted "
                         "contended measurement, and require this to fail")
    args = ap.parse_args()

    refresh = args.refresh_set
    m4, m5 = load(M4_SOURCES), load(M5_SOURCES)
    if args.inject:
        m5["mobilenet"]["CPU_AND_NE"] /= 2
        print("INJECTED: M5 Max mobilenet ANE halved, as the retracted runs had it\n")

    text = PAPER.read_text()
    readme = README.read_text() if README.exists() else ""
    fails, checks = [], []

    def require(label, s):
        checks.append((label, s, s in text))

    def require_readme(label, s):
        checks.append((label, s, s in readme))

    # The medians table itself. Everything below is derived from these.
    for m in ARCHITECTURES:
        want = PAPER_TABLE[m]
        got = dict(m4_ane=m4.get(m, {}).get("CPU_AND_NE"),
                   m4_gpu=m4.get(m, {}).get("CPU_AND_GPU"),
                   m5_ane=m5.get(m, {}).get("CPU_AND_NE"),
                   m5_gpu=m5.get(m, {}).get("CPU_AND_GPU"))
        for k, v in want.items():
            g = got[k]
            if g is None or abs(g - v) > 0.15:
                fails.append("3.1 medians %s %s: paper %.1f, results %s"
                             % (m, k, v, "MISSING" if g is None else "%.1f" % g))

    r4 = {m: m4[m]["CPU_AND_GPU"] / m4[m]["CPU_AND_NE"] for m in ARCHITECTURES if m in m4}
    r5 = {m: m5[m]["CPU_AND_GPU"] / m5[m]["CPU_AND_NE"] for m in ARCHITECTURES if m in m5}
    inverts = [m for m in ARCHITECTURES if (r4[m] > 1) != (r5[m] > 1)]
    same = [m for m in ARCHITECTURES if m not in inverts]

    require("inversion count", "three of five architectures")
    if len(inverts) != 3:
        fails.append("inversion count: paper says three of five, results say %d (%s)"
                     % (len(inverts), ", ".join(inverts)))

    # The two that keep their winner. Order follows the paper's sentence.
    moves = sorted(max(r4[m], r5[m]) / min(r4[m], r5[m]) for m in same)
    require("non-inverting ratio move, abstract",
            "move by %.1fx and %.1fx" % (moves[0], moves[1]))
    require("non-inverting ratio move, 3.1",
            "factors of %.1f and %.1f" % (moves[0], moves[1]))

    # mobilenet, the ANE-folklore result.
    require("mobilenet GPU advantage",
            "by %.2fx on the\nM4 Pro and %.2fx on the M5 Max"
            % (r4["mobilenet"], r5["mobilenet"]))

    # Chip sensitivity of each unit.
    ane = sorted(m5[m]["CPU_AND_NE"] / m4[m]["CPU_AND_NE"] for m in ARCHITECTURES)
    gpu = sorted(m5[m]["CPU_AND_GPU"] / m4[m]["CPU_AND_GPU"] for m in ARCHITECTURES)
    require("ANE chip sensitivity", "**%.2fx to %.2fx**" % (ane[0], ane[-1]))
    require("GPU chip sensitivity", "**%.2fx to %.2fx**" % (gpu[0], gpu[-1]))

    # The default. Cost against the better explicit placement, M4 Pro.
    cost4 = {m: max(m4[m]["CPU_AND_GPU"], m4[m]["CPU_AND_NE"]) / m4[m]["ALL"]
             for m in ARCHITECTURES if "ALL" in m4.get(m, {})}
    if cost4:
        lo, hi = min(cost4.values()), max(cost4.values())
        require("default cost range", "between %.2fx and %.2fx" % (lo, hi))
    worse_both = [m for m in ARCHITECTURES
                  if "ALL" in m4.get(m, {})
                  and m4[m]["ALL"] < min(m4[m]["CPU_AND_GPU"], m4[m]["CPU_AND_NE"])]
    if len(worse_both) != 3:
        fails.append("ALL slower than both: paper says three of five, results say "
                     "%d (%s)" % (len(worse_both), ", ".join(worse_both)))

    # 3.x sustained soaks. Reported to FOUR decimals on purpose. At three, every
    # ANE soak prints as 1.000, and an earlier draft read "the ANE returned
    # exactly 1.000 in every one of ten soaks". Only 7 of 16 M4 Pro soaks are
    # exactly 1.0000. Rounding is not a result, so the check works at the
    # precision that distinguishes them.
    # PINNED TO A STATED DATASET, NOT THE LIVE DIRECTORY. The fleet keeps
    # producing soaks, and globbing meant every sync "broke" the paper: the
    # counts moved four times in one session while the claim never did. That is
    # churn masquerading as a check, and worse, it trains you to edit numbers to
    # silence a red result.
    #
    # results/soak/PAPER-SET.txt lists the files 3.3 is computed from. Growing the
    # directory is now informational; only a disagreement between the paper and
    # ITS OWN set is a failure. Extending the dataset becomes a deliberate act:
    # --refresh-set, then update the prose, then commit both together.
    setf = REPO / "results" / "soak" / "PAPER-SET.txt"
    soakdir = REPO / "results" / "soak"
    on_disk = sorted(p.name for p in soakdir.glob("*.json"))
    if refresh:
        setf.write_text("\n".join(on_disk) + "\n")
        print("refreshed PAPER-SET.txt to %d files; now update the prose to match\n"
              % len(on_disk))
    names = ([l.strip() for l in setf.read_text().splitlines() if l.strip()]
             if setf.exists() else on_disk)
    missing = [n for n in names if not (soakdir / n).exists()]
    if missing:
        fails.append("PAPER-SET.txt lists %d file(s) that no longer exist, starting "
                     "with %s. The paper's numbers cannot be recomputed."
                     % (len(missing), missing[0]))
    extra = len(on_disk) - len(names)

    ane, gpu = [], []
    for n in names:
        p = soakdir / n
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        s = d.get("sustained_fraction") or (d.get("summary") or {}).get("sustained_fraction")
        u = d.get("units") or (d.get("summary") or {}).get("units")
        if s is None or u is None:
            continue
        (ane if u == "CPU_AND_NE" else gpu if u == "CPU_AND_GPU" else []).append(s)
    # The claim is no longer "the ranges are disjoint". Adding 21 recovered soaks
    # produced one ANE run at 0.9748 that sits below the best GPU run, so the
    # paper now says "21 of 22" and names the exception. Check the COUNT, which is
    # what the text asserts, not the disjointness it no longer claims.
    if ane and gpu:
        above = [x for x in ane if x > max(gpu)]
        require("ANE give-back, abstract",
                "gives back at most %.2f%% of its peak in %d of %d"
                % ((1 - min(above)) * 100 if above else 0, len(above), len(ane)))
        require("ANE beats every GPU soak, abstract",
                "sustain better than the best of the %d GPU soaks" % len(gpu))
        # Same numbers, the README's one-line summary of 3.3.
        require_readme("README 3.3 summary",
                       "in %d of %d soaks; the GPU gives back 0.8 to 16.3%% in %d"
                       % (len(above), len(ane), len(gpu)))
        if len(ane) - len(above) > 1:
            fails.append("more than one ANE soak now falls below the best GPU soak "
                         "(%d of %d). The paper names exactly one exception; "
                         "rewrite the claim rather than widening the exception."
                         % (len(ane) - len(above), len(ane)))

    for label, s, ok in checks:
        print("%-38s %s  %r" % (label, "ok  " if ok else "FAIL", s))
        if not ok:
            where = "README.md" if label.startswith("README") else "PAPER.md"
            fails.append("%s: %s does not contain %r" % (label, where, s))

    if extra:
        print("\nnote: %d soak file(s) on disk are not in PAPER-SET.txt. That is not a\n"
              "failure. Run --refresh-set and update the prose when you want them in."
              % extra)

    print()
    if fails:
        print("PAPER.md DISAGREES WITH results/ ON %d POINT(S):" % len(fails))
        for f in fails:
            print("  - %s" % f)
        return 1
    print("every headline number in PAPER.md recomputes from results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
