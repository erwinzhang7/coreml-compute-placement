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
    python tools/verify_claims.py --inject        # corrupt one median
    python tools/verify_claims.py --inject-soak   # corrupt every soak file
    python tools/verify_claims.py --inject-set    # drop one ANE and one GPU soak

WHICH CHECKS EACH INJECTION CAN BREAK, measured rather than assumed. Of 99 checks,
--inject reaches 11, --inject-soak 80 and --inject-set 13, for 95 covered between
them. FOUR ARE NOT EXERCISED BY ANY INJECTION and are listed here so the gap is
stated rather than discovered:

    inversion count    GPU chip sensitivity    default cost range
    3.6 floor ANE sds

The first three are COUNTS or SET MEMBERSHIPS -- "three of five architectures" --
rather than magnitudes, and all three are derived from the 3.1 medians rather than
from the soak set, so neither the window perturbation nor the set drop moves them.
Reaching them needs an injection that reorders a placement, which is not built.

The fourth is unreachable for a different and more interesting reason: it is a
STANDARD DEVIATION, and --inject-soak scales every fraction by a COMMON factor.
An sd scales with that factor, so 0.0002 x 0.97 is still 0.0002 at the four
decimals the paper prints. A multiplicative fault cannot move it; only a fault
that varies per run could, and making the injection vary per run would change
what the other 80 guards are testing. Left as it is, and named.

Five soak counts used to sit in this list for the same reason and no longer do:
--inject-set removes one ANE and one GPU soak, which is the only way a count
moves. It has to drop M4 Pro rows specifically -- dropping whichever came first
took the two M5 Max soaks and left every M4-only count in the 3.3 table intact.

The three are also the ones with a second line of defence: the script raises a
hard failure, not just a string mismatch, if the inversion count is not three or
if more than one ANE soak falls below the best GPU soak.
"""
import argparse
import collections
import itertools
import json
import math
import pathlib
import statistics
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


# Every soak read goes through here so ONE flag can corrupt them all. Before this
# existed, --inject halved a single M5 Max median and broke 4 of 73 checks: the
# ~45 soak and dip guards were never exercised by any injection at all, so
# "the checker can fail" was demonstrated for a fifth of it. A control that is
# only tested on the part written first is not a control on the rest.
INJECT = False
INJECT_SOAK = False
# Ranges can be moved by perturbing a window; COUNTS cannot. The eight count and
# set-membership guards this file called unreachable stayed unreachable because
# nothing removed a soak from the set. --inject-set does exactly that, and the
# 3.3 table guards below are all counts, so without it they would have joined
# the unreached fifth rather than shrinking it.
INJECT_SET = False


def sustained(d):
    """summary.sustained_fraction, with the soak fault injection applied.

    The 3.3 guards read the recorded fraction rather than recomputing it from
    windows(): 155 of 157 files agree with last/max(windows) to four decimals,
    but two `bert` ANE runs differ by 0.0005, so recomputing would silently
    restate two published cells. Reading the field directly is therefore right
    and was also invisible to --inject-soak, which only ever touched windows().
    Routing the read through here is what makes those guards fail on demand.
    """
    s = (d.get("summary") or {}).get("sustained_fraction")
    if s is None:
        s = d.get("sustained_fraction")
    if INJECT_SOAK and s is not None:
        s = round(s * 0.97, 4)
    return s


def windows(path, complete_only=False):
    """images_per_s per window, with the soak fault injection applied."""
    ws = json.loads(pathlib.Path(path).read_text())["windows"]
    if complete_only:
        ws = [w for w in ws if w.get("complete", True)]
    r = [w["images_per_s"] for w in ws]
    if INJECT_SOAK and len(r) > 3:
        # TWO perturbations, because one was not enough and the gap was measured
        # rather than guessed. Deepening the trough moves every dip, last/best and
        # sustained figure -- 49 of 73 checks. It leaves the twelve window-1
        # guards untouched, since window 1 is the peak and never the minimum, and
        # those twelve carry the argument that REFUTED the cold-boost reading. A
        # guard on the load-bearing evidence that cannot be shown to fail is the
        # worst one to leave unexercised.
        i = r.index(min(r))
        r[i] = r[i] * 0.95
        # Window 1 is scaled ONLY on the cold run. Scaling it in every file left
        # the four "w1 cold advantage" checks passing, because those are RATIOS of
        # cold to warm and a common factor cancels. A perturbation that moves both
        # sides of a ratio cannot test the ratio.
        # DIFFERENT factors on the cold and warm runs. A single common factor
        # moves both sides of the cold-versus-warm ratio and cancels, so the four
        # "w1 cold advantage" guards passed; scaling only the cold run then left
        # the four "w1 warm mean" guards passing instead. Both had to move, by
        # different amounts, before all twelve window-1 guards could fail.
        cold = pathlib.Path(path).name.startswith(("dip1-", "aned1-"))
        r[0] = r[0] * (1.02 if cold else 0.99)
    return r


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
    ap.add_argument("--inject-soak", action="store_true",
                    help="deepen one window in every soak file by 5%%, which moves "
                         "every dip, sustained and last/best figure. --inject only "
                         "reaches the 3.1-derived chain, so without this the soak "
                         "guards are never shown capable of failing.")
    ap.add_argument("--inject-set", action="store_true",
                    help="drop one ANE and one GPU soak from the paper set, which "
                         "moves every count in 3.3. Ranges can be perturbed; "
                         "counts can only be moved by changing the set, so this "
                         "is the only injection that reaches them.")
    ap.add_argument("--inject", action="store_true",
                    help="halve one M5 Max ANE median, reproducing the retracted "
                         "contended measurement, and require this to fail")
    args = ap.parse_args()

    global INJECT, INJECT_SOAK, INJECT_SET
    INJECT = args.inject
    INJECT_SOAK = args.inject_soak
    if INJECT_SOAK:
        print("INJECTED: one window deepened 5%% in every soak file\n")
    INJECT_SET = args.inject_set
    if INJECT_SET:
        print("INJECTED: one ANE and one GPU soak dropped from the paper set\n")
    refresh = args.refresh_set
    m4, m5 = load(M4_SOURCES), load(M5_SOURCES)
    if args.inject:
        m5["mobilenet"]["CPU_AND_NE"] /= 2
        # The M5 Max side alone leaves everything DERIVED FROM THE M4 MEDIANS
        # untested, which is most of 3.4's extrapolations: the base-M5 rows for
        # the other four models and the M4 Max flip set. Perturb one M4 Pro GPU
        # median too. It has to be large enough to change which side wins, since
        # the flip set is a comparison and not a magnitude -- 1.6x puts siglip's
        # GPU above its ANE and takes it out of the set.
        m4["siglip"]["CPU_AND_GPU"] *= 1.6
        print("INJECTED: M5 Max mobilenet ANE halved and M4 Pro siglip GPU "
              "raised 1.6x\n")

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
        # ALL was never in PAPER_TABLE, so 3.2's default-cost column was derived
        # from a file nothing compared against the paper. Report the spread
        # between the pinned resolution and what 3.1/3.2 print, rather than
        # silently preferring one.
        if m == "siglip" and "ALL" in m4.get(m, {}):
            require("3.2 siglip M4 Pro ALL provenance",
                    "171.1 from `sweep-m4pro-rerun.json` and 172.2 from")
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
    # The headline in 3.3 is about the 120 s protocol: "over a two-minute soak".
    # The 600 s runs are a separate sub-analysis that qualifies it, and folding
    # them into the same count would make the count mean nothing, since it would
    # be a mix of durations described as one. They are excluded here and analysed
    # in their own subsection.
    # FILTER ON WHAT IS IN THE FILE, NOT ON WHAT IT IS CALLED. This was a prefix
    # list -- ("long600", "dip", "aned") -- with a comment saying that excluding
    # by prefix rather than by content had already been the bug once. It was the
    # bug a second time: battery-m5max-CPU_AND_NE.json matches no prefix, and is
    # a SIXTY-second run on BATTERY. It sat inside a set the abstract describes
    # as "over a two-minute soak" and counted toward "44 of 45", inflating the
    # ANE's record with a run that was neither the right duration nor on mains.
    #
    # The criterion the prose actually states is 120 s on AC, so that is what is
    # tested. A file that cannot be read, or that lacks either field, is
    # EXCLUDED rather than assumed to qualify.
    # TWO criteria, because they catch two different mistakes and neither
    # subsumes the other:
    #
    #   content -- 120 s on AC. Catches a file that LOOKS like a survey run and
    #              is not. Nothing about the name of battery-m5max would ever
    #              have revealed it.
    #   prefix  -- a deliberate sub-study. long600/dip/aned are 600 s so the
    #              content test already excludes them, but floor-* is 120 s on
    #              AC and would pass: it is twelve repeats of ONE model for
    #              3.6's floor, and folding 36 siglip and resnet50 runs into
    #              3.3's per-architecture table would swamp it. Purpose is not
    #              recorded in the files, so it has to live here.
    STUDY = ("long600", "dip", "aned", "floor")

    def qualifies(p):
        if p.name.startswith(STUDY):
            return False
        try:
            d = json.loads(p.read_text())
        except Exception:
            return False
        if (d.get("seconds") != 120.0
                or (d.get("power_start") or {}).get("source") != "AC"):
            return False
        # COMPLETENESS. A soak killed mid-run still writes its file, and every
        # metadata field on it lies: seconds says 120.0, power says AC, and only
        # the window count betrays it. Two `bert` ANE runs here have NINE windows
        # where the protocol gives eleven or twelve -- and they are the same two
        # files whose recorded sustained_fraction disagrees with last/max(windows)
        # by 0.0005, an anomaly noted in sustained() and written off as benign
        # before its cause was known. Truncation was the cause.
        w = d.get("windows") or []
        return len(w) >= int(d["seconds"] / d["window"]) - 1
    on_disk = sorted(p.name for p in soakdir.glob("*.json") if qualifies(p))
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
    rows = []
    for n in names:
        p = soakdir / n
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        s = sustained(d)
        u = d.get("units") or (d.get("summary") or {}).get("units")
        if s is None or u is None:
            continue
        rows.append(dict(sf=s, units=u,
                         chip=(d.get("machine") or {}).get("cpu", ""),
                         model=(d.get("model") or "").split("-b16")[0]
                                                    .replace("siglip-vision", "siglip"),
                         peak=(d.get("summary") or {}).get("best_images_per_s"),
                         w=[x["images_per_s"] for x in (d.get("windows") or [])]))
    if INJECT_SET:
        # Drop one ANE and one GPU soak, which moves every "N of M" in 3.3 by
        # one. --inject-soak cannot reach a count no matter how hard it perturbs
        # a window, so before this the count guards were asserted and untested.
        # Drop M4 Pro rows specifically. Dropping whichever row came first in the
        # set took the two M5 Max soaks, which moved the whole-set counts but
        # left every M4-only count in the 3.3 table intact -- an injection that
        # misses the table it was written for.
        for u in ("CPU_AND_NE", "CPU_AND_GPU"):
            for i, r in enumerate(rows):
                if r["units"] == u and "M4" in r["chip"]:
                    rows.pop(i)
                    break
    for r in rows:
        (ane if r["units"] == "CPU_AND_NE"
         else gpu if r["units"] == "CPU_AND_GPU" else []).append(r["sf"])
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
        # results/soak/README.md restates the same counts in prose and has now
        # gone stale twice: once on "21 of 22 / 41 GPU soaks" and once on a
        # paragraph calling a gap "the most valuable thing to fix" after it was
        # fixed. Guard its numbers here rather than finding them by grep again.
        soak_readme = (REPO / "results" / "soak" / "README.md")
        if soak_readme.exists():
            sr = soak_readme.read_text()
            for lbl, s in (("soak README ANE count",
                            "The ANE, in %d of %d soaks." % (len(above), len(ane))),
                           ("soak README GPU count",
                            "than the best of %d GPU soaks (%.4f)" % (len(gpu), max(gpu)))):
                checks.append((lbl, s, s in sr))

            # The mobilenet exception. This file tells a reader to treat a 120 s
            # GPU figure as a trough, and doing that to mobilenet OVERSTATES it,
            # so the range that justifies the exception has to stay true.
            mob = []
            for box in ("inference1", "experiments"):
                for i in (1, 2, 3):
                    f = (REPO / "results" / "soak"
                         / ("dip%d-%s-mobilenet-CPU_AND_GPU.json" % (i, box)))
                    if not f.exists():
                        continue
                    r = windows(f)
                    mob.append(100 * (max(r) - min(r)) / max(r))
            if len(mob) == 6:
                s = "six runs spanning %.2f to %.2f%%" % (min(mob), max(mob))
                checks.append(("soak README mobilenet range", s, s in sr))

        # The GPU give-back range used to be a LITERAL inside this format string
        # while the counts around it were derived, so the check went on passing
        # after the range moved: it certified "0.8 to 16.3%" when the best GPU
        # soak had become 0.9892, i.e. 1.1%. Derive both ends or derive neither.
        gb = "%.1f to %.1f%%" % ((1 - max(gpu)) * 100, (1 - min(gpu)) * 100)
        # 3.3's minima-position table. The GPU row reproduced exactly and the ANE
        # row was stale in all three of its values -- 40/4/0.32 against 42/5/0.34
        # -- which is what a table looks like when half of it is regenerated and
        # half is typed. The position convention is (i+1)/n, established by
        # requiring the GPU row to reproduce; i/(n-1) gives 0.79 and would have
        # silently restated a number that was already right.
        pos = collections.defaultdict(list)
        for r in rows:
            if "M4" not in r["chip"]:
                continue
            pos[r["units"]].append(r)
        for u, lbl in (("CPU_AND_GPU", "GPU"), ("CPU_AND_NE", "ANE")):
            sel = pos.get(u, [])
            if not sel:
                continue
            ps = []
            for r in sel:
                w = r["w"]
                ps.append((w.index(min(w)) + 1) / len(w))
            late = sum(1 for x in ps if x >= 0.7)
            require("3.3 minima row %s" % lbl,
                    "| %d | %s%d (%.0f%%)%s | %.2f |"
                    % (len(ps), "**" if u == "CPU_AND_GPU" else "", late,
                       100 * late / len(ps), "**" if u == "CPU_AND_GPU" else "",
                       sum(ps) / len(ps)))

        # 2.6's "only equal-duration soaks are comparable". That used to be
        # justified with "the statistic falls with soak length BY CONSTRUCTION",
        # which the 600 s runs falsify: the numerator is the LAST window, and a
        # longer run ends at a different one, so both terms move. Derived here
        # because the counts are the whole point of the correction.
        rose = fell = 0
        by = collections.defaultdict(lambda: {"rise": 0, "fall": 0})
        for pth in sorted(soakdir.glob("*.json")):
            dd = json.loads(pth.read_text())
            if dd.get("window") != 20.0:
                continue
            w = [x["images_per_s"] for x in (dd.get("windows") or [])]
            if len(w) < 10:
                continue
            at120 = w[5] / max(w[:6])
            at600 = w[-1] / max(w)
            u = dd.get("units")
            if at600 > at120:
                rose += 1
                by[u]["rise"] += 1
            elif at600 < at120:
                fell += 1
                by[u]["fall"] += 1
        if rose + fell:
            require("2.6 length effect is not by construction",
                    "**%d of %d rise with length and %d fall**"
                    % (rose, rose + fell, fell))
            # Split by unit, which is where the mechanism lives: the GPU rises
            # because it recovers from a dip, the ANE falls because it has no dip
            # to recover from. The first cut of this correction used a hand-rolled
            # glob, counted 22 of 40, and published it; this guard's fuller
            # population is what caught that.
            require("2.6 length effect, GPU row",
                    "| GPU | **%d** | %d |" % (by["CPU_AND_GPU"]["rise"],
                                               by["CPU_AND_GPU"]["fall"]))
            require("2.6 length effect, ANE row",
                    "| ANE | %d | **%d** |" % (by["CPU_AND_NE"]["rise"],
                                               by["CPU_AND_NE"]["fall"]))

        # 4's repeatability comparison. It used to set burst PERCENTAGES against
        # an absolute sustained move of 0.106, from the other machine and the
        # confounded no-cooldown set. Both sides are now relative sds over
        # repeats, and both are derived here so the comparison cannot drift back
        # into comparing two different statistics.
        def relsd(fs):
            out = []
            for fn in fs:
                fp = REPO / "results" / fn
                if not fp.exists():
                    continue
                for e in json.loads(fp.read_text()):
                    r = e.get("runs") or []
                    if len(r) > 2:
                        out.append(100 * statistics.stdev(r) / statistics.median(r))
            return out
        b4 = relsd(["zoo-m4pro-dtypefix.json", "sweep-m4pro.json"])
        b5 = relsd(["zoo-m5max-v2.json"])
        if b4 and b5:
            require("4 burst repeatability, medians",
                    "| median cell | %.2f%% | %.2f%% | — |"
                    % (statistics.median(b4), statistics.median(b5)))
            require("4 burst repeatability, ranges",
                    "| range across cells | %.2f to %.2f%% | %.2f to **%.2f%%**"
                    % (min(b4), max(b4), min(b5), max(b5)))

        # The three-box result. Every figure derived, including the sigmas,
        # because "three identical machines all differ" is the strongest claim
        # in 3.6 and it should not be possible to state it from stale numbers.
        tri = {}
        for bx in ("experiments", "inference1", "inference2"):
            v = []
            for i in range(1, 13):
                fp = soakdir / ("floor-%s-siglip-CPU_AND_GPU-%02d.json" % (bx, i))
                if fp.exists():
                    v.append(sustained(json.loads(fp.read_text())))
            if len(v) == 12:
                tri[bx] = v[3:]
        if len(tri) == 3:
            order = ["experiments", "inference1", "inference2"]
            for lbl, bx in zip(("#2", "#3", "#4"), order):
                require("3.6 three-box row %s" % lbl,
                        "| M4 Pro %s | %.4f | %.4f |"
                        % (lbl, statistics.mean(tri[bx]), statistics.stdev(tri[bx])))
            for lbl, (a, b) in zip(("#2 vs #3", "#2 vs #4", "#3 vs #4"),
                                   itertools.combinations(order, 2)):
                x, y = tri[a], tri[b]
                d = abs(statistics.mean(x) - statistics.mean(y))
                se = math.sqrt(statistics.stdev(x) ** 2 / len(x)
                               + statistics.stdev(y) ** 2 / len(y))
                sg = d / se
                # The two pairs involving box #4 span a three-hour session
                # gap and were withdrawn. The guard asserts the WITHDRAWN
                # MARKER as part of the row, so the retraction cannot be lost
                # by an edit that only touches the numbers -- which is how the
                # first version of this correction left a stale figure behind
                # in a sentence no guard was watching.
                tail = ("**withdrawn below**" if "#4" in lbl else "stands")
                require("3.6 three-box pair %s" % lbl,
                        "| %s | %.4f | %.4f | %.1f | %s |"
                        % (lbl, d, se, sg, tail))

        # 3.5's per-window traces, regenerated from the files they come from.
        # They had drifted -- siglip matched 7 of 12 windows and resnet50 9 of 11
        # -- and the two are from DIFFERENT boxes, which was not stated.
        for name, fn, nw in (("siglip",
                              "long600-inference1-siglip-vision-CPU_AND_GPU.json", 12),
                             ("resnet50",
                              "long600-experiments-resnet50-CPU_AND_GPU.json", 11)):
            fp = soakdir / fn
            if not fp.exists():
                continue
            w = [round(x["images_per_s"])
                 for x in json.loads(fp.read_text())["windows"]]
            require("3.5 window trace %s" % name,
                    " ".join(str(x) for x in w[:nw]) + " ... %d" % w[-1])

        # 4's "the repeat spread was the tell". It was not: the widest cell in
        # the study is in the CLEAN file. Derived so the refutation cannot rot.
        cl = []
        for fn in ("zoo-m5max-v2.json",):
            fp = REPO / "results" / fn
            if fp.exists():
                for e in json.loads(fp.read_text()):
                    r = e.get("runs") or []
                    if len(r) > 2:
                        cl.append(100 * (max(r) - min(r)) / statistics.median(r))
        if cl:
            require("4 clean M5 Max spread range",
                    "**%.1f to %.1f%%** on the M5 Max" % (min(cl), max(cl)))

        # 4's burst-inflation GPU column, now that its provenance is known: the
        # THIRD (warm) run on inference1, mean of last six windows. Eighteen
        # averaging variants failed to reproduce it because it was never an
        # average. Derived so the identification is not just an anecdote.
        for name, tag in (("bert", "bert"), ("resnet50", "resnet50"),
                          ("siglip", "siglip-vision"), ("whisper", "whisper")):
            fp = soakdir / ("dip3-inference1-%s-CPU_AND_GPU.json" % tag)
            if not fp.exists() or name not in m4:
                continue
            w = [x["images_per_s"] for x in json.loads(fp.read_text())["windows"]]
            got = (m4[name]["CPU_AND_GPU"] / statistics.mean(w[-6:]) - 1) * 100
            bold = "**" if name == "whisper" else ""
            require("4 inflation provenance %s" % name,
                    "| `%s` | %s+%.2f%%%s | %s+%.2f%%%s |"
                    % (name, bold, {"bert": 0.21, "resnet50": 0.72,
                                    "siglip": 1.03, "whisper": 2.98}[name], bold,
                       bold, got, bold))

        # 4's rebuilt inflation table: EVERY cell from the third, warm run,
        # inference1 where that run exists. The previous version mixed run
        # indices in its ANE column -- two cells from run 1, which 3.6 shows is
        # the extreme run -- and omitted mobilenet, the model with the largest
        # ANE floor. Regenerated so neither can recur.
        TAGS = {"bert": "bert", "resnet50": "resnet50", "siglip": "siglip-vision",
                "whisper": "whisper", "mobilenet": "mobilenet"}
        for name in ("bert", "resnet50", "siglip", "whisper", "mobilenet"):
            if name not in m4:
                continue
            g = soakdir / ("dip3-inference1-%s-CPU_AND_GPU.json" % TAGS[name])
            a = box = None
            for bx in ("inference1", "experiments"):
                cand = soakdir / ("aned3-%s-%s-CPU_AND_NE.json" % (bx, TAGS[name]))
                if cand.exists():
                    a, box = cand, bx
                    break
            if not g.exists() or a is None:
                continue
            def plateau(fp):
                w = [x["images_per_s"]
                     for x in json.loads(fp.read_text())["windows"]]
                return statistics.mean(w[-6:])
            gi = (m4[name]["CPU_AND_GPU"] / plateau(g) - 1) * 100
            ai = (m4[name]["CPU_AND_NE"] / plateau(a) - 1) * 100
            bold = "**" if name == "whisper" else ""
            row = ("| `%s` | %s%+.2f%%%s | %+.2f%% | `%s` |"
                   % (name, bold, gi, bold, ai, box))
            # The paper sets negatives with U+2212, the formatter emits U+002D.
            # Normalise rather than teaching the generator which sign the
            # typography happens to use today.
            checks.append(("4 rebuilt inflation %s" % name, row,
                           row in text.replace("\u2212", "-")))

        # The session confound. inference1 was measured twice, three hours
        # apart, and moved by as much as the boxes differ -- which makes two of
        # the three pairwise comparisons in 3.6 confounded. Derived, including
        # the sigma, because it is a retraction of a published figure.
        s1, s2 = [], []
        for i in range(1, 13):
            a = soakdir / ("floor-inference1-siglip-CPU_AND_GPU-%02d.json" % i)
            b = soakdir / ("rep2floor-inference1-siglip-CPU_AND_GPU-%02d.json" % i)
            # Counting FILES here is wrong, and the reason is worse than it
            # looks. thermal_soak streams windows as it goes, so a file rsynced
            # WHILE ITS RUN IS STILL GOING has intact metadata, `seconds:
            # 120.0`, AC power, a couple of windows and a null summary -- which
            # is byte-for-byte the shape a KILLED run leaves. Run 12 of the
            # second series arrived that way and was read as a kill until the
            # box's own log showed it had finished fine. Nothing downstream can
            # tell the two apart, so both are excluded the same way, on window
            # count, exactly as the paper set is.
            for f, acc in ((a, s1), (b, s2)):
                if not f.exists():
                    continue
                v = sustained(json.loads(f.read_text()))
                if v is not None:
                    acc.append(v)
        if len(s1) >= 12 and len(s2) >= 11:
            x, y = s1[3:], s2[3:]
            d = abs(statistics.mean(x) - statistics.mean(y))
            se = math.sqrt(statistics.stdev(x) ** 2 / len(x)
                           + statistics.stdev(y) ** 2 / len(y))
            # Matched as three short strings, not one long one: require() is a
            # raw substring test and PAPER.md hard-wraps at 79 columns, so any
            # claim string spanning a wrap can never match.
            require("3.6 session shift evening",
                    "settled mean of %.4f" % statistics.mean(y))
            require("3.6 session shift morning",
                    "morning figure of %.4f" % statistics.mean(x))
            require("3.6 session shift sigma",
                    "**the same machine, %.4f apart, %.1f sigma**" % (d, d / se))

        # NOTE ON WHAT EXERCISES THESE. All six 3.6 session guards below move
        # under --inject-soak and are unreachable by --inject, which only
        # touches paper-set files; the floor series are study files. Their
        # SIGMA figures are weaker than their magnitudes: --inject-soak scales
        # every fraction uniformly, and a uniform scale leaves a t-statistic
        # exactly where it was, so 16.3 and 18.8 are carried along by the
        # magnitude strings rather than tested in their own right.
        # The matched/confounded table. These are the numbers that replaced
        # the withdrawn three-box spread, so they are derived here rather than
        # transcribed. Series are read the same way session_vs_box.py reads
        # them -- on summary presence, never on file existence.
        def series(pre, box):
            out = []
            for i in range(1, 13):
                f = soakdir / ("%s-%s-siglip-CPU_AND_GPU-%02d.json" % (pre, box, i))
                if not f.exists():
                    continue
                v = sustained(json.loads(f.read_text()))
                if v is not None:
                    out.append(v)
            return out[3:]

        def diff(a, b):
            if len(a) < 2 or len(b) < 2:
                return None, None
            d = statistics.mean(a) - statistics.mean(b)
            se = math.sqrt(statistics.stdev(a) ** 2 / len(a)
                           + statistics.stdev(b) ** 2 / len(b))
            return abs(d), (abs(d) / se if se else None)

        # Session-matched: both series ran concurrently that evening.
        d, sg = diff(series("rep2floor", "inference1"),
                     series("rep2floor", "inference2"))
        if d is not None:
            require("3.6 matched inference1 vs inference2",
                    "| **%.4f** | %.1f sigma |" % (d, sg))
        # The same two machines as the surviving matched pair, read across
        # sessions instead. This row is the whole argument and must not rot.
        d, sg = diff(series("floor", "experiments"),
                     series("rep2floor", "inference1"))
        if d is not None:
            require("3.6 unmatched experiments vs inference1",
                    "four hours apart | confounded | %.4f |" % d)
        # The second, much shorter, session gap.
        d, sg = diff(series("floor", "inference2"),
                     series("rep2floor", "inference2"))
        if d is not None:
            require("3.6 short session gap",
                    "moved %.4f in its" % d)
        # The THIRD series repeats the matched between-box comparison. Two
        # independent matched estimates of one difference is the strongest
        # statement 3.6 makes, so both are guarded rather than just the newer.
        d, sg = diff(series("rep3floor", "inference1"),
                     series("rep3floor", "inference2"))
        if d is not None:
            require("3.6 matched pair repeated",
                    "| **%.4f** | %.1f sigma |" % (d, sg))
        # inference1's drift is near-linear over four hours; the paper says so
        # and the two figures that support it are derived here.
        d3, _ = diff(series("floor", "inference1"), series("rep3floor", "inference1"))
        d2, _ = diff(series("floor", "inference1"), series("rep2floor", "inference1"))
        if d3 is not None and d2 is not None:
            require("3.6 drift near-linear",
                    "%.4f across 3.2 hours and %.4f across 4.0 hours" % (d2, d3))

        # 2.4's conversion table. It had drifted from results/conversion-check.json
        # in ELEVEN of its twelve numbers -- an earlier conversion run's figures
        # left in place under a sentence citing the file. Nothing checked it,
        # which is how a table can disagree with the only evidence for it.
        cc = REPO / "results" / "conversion-check.json"
        if cc.exists():
            by = {m["model"]: m for m in json.loads(cc.read_text())["models"]}
            if INJECT:
                # Nothing else corrupts this file, so without it these guards are
                # asserted and never shown capable of failing. Perturb the errors
                # AND the model count, since the two claims fail differently.
                by = {k: dict(v, max_abs_err=v["max_abs_err"] * 1.4,
                              min_cosine=min(v["min_cosine"] * 0.9999, 1.0))
                      for k, v in by.items()}
                by["siglip"] = dict(next(iter(by.values())), model="siglip")
            for name in ("resnet50", "mobilenet", "bert", "whisper"):
                if name in by:
                    m = by[name]
                    require("2.4 conversion row %s" % name,
                            "| %s | %.1e | %.1e | %.6f |"
                            % (name, m["max_abs_err"], m["max_rel_err"],
                               m["min_cosine"]))
            # The gate covers four models, not five. siglip has no reference
            # comparison in models/convert_siglip.py at all, so "every converted
            # model" was false for the one the paper leans on hardest.
            #
            # Emitted UNCONDITIONALLY with the count derived. Gating it on
            # `len(by) != 5` would make the check vanish the moment a fifth model
            # appeared -- which is precisely the event it exists to notice, and
            # the same fail-open shape as the M4 Max flip-set guard above.
            WORDS = {3: "three", 4: "four", 5: "five", 6: "six"}
            require("2.4 gate model count",
                    "carries %s models, not five" % WORDS.get(len(by), len(by)))

        # 3.4's falsifiable prediction. It USED to say the inversion was
        # independent of tier, which the 3.1 medians refute: the ANE is 16 cores
        # on both chips, so core count is first-order and cannot drop out. Both
        # extrapolations are regenerated here, because a prediction derived from
        # a table is exactly the thing that goes stale when the table moves.
        # ONE check over all five rows, not five. Five separate ones meant a
        # fault in a single median broke a single row and left the other four
        # certifying a table that had moved underneath them.
        rows5 = []
        for m in ARCHITECTURES:
            g5 = m5.get(m, {}).get("CPU_AND_GPU")
            a5 = m5.get(m, {}).get("CPU_AND_NE")
            if g5 and a5:
                rows5.append("| %.1f | %.1f |" % (g5 * 10 / 40, a5))
        if rows5:
            checks.append(("3.4 base-M5 extrapolation",
                           " ".join(rows5),
                           all(r in text for r in rows5)))
        # An M4 Max is 2x the M4 Pro GPU with the same ANE. Name the models that
        # flip, and require the paper to name the same ones.
        #
        # Emitted UNCONDITIONALLY. An earlier version only emitted it when the
        # set had three members, so the injection that changes which models flip
        # made the check DISAPPEAR rather than fail -- a guard that vanishes
        # under the fault it exists to catch is worse than no guard, because the
        # run still ends green.
        flip = [m for m in ARCHITECTURES
                if (m4.get(m, {}).get("CPU_AND_NE") or 0) > (m4.get(m, {}).get("CPU_AND_GPU") or 0)
                and (m4.get(m, {}).get("CPU_AND_GPU") or 0) * 2 > (m4.get(m, {}).get("CPU_AND_NE") or 0)]
        want = ("should invert `%s`, `%s` and\n`%s`" % tuple(flip) if len(flip) == 3
                else "should invert exactly these %d: %s" % (len(flip), ", ".join(flip)))
        checks.append(("3.4 M4 Max flips", want, want in text))

        # 3.6's re-measured floor. The figure it replaced compared a RANGE to a
        # DIFFERENCE OF MEANS, which is why it moved when n moved rather than
        # when the hardware did; everything here is derived, including the sigma.
        fl = collections.defaultdict(list)
        for p in sorted(soakdir.glob("floor-*siglip*.json")):
            d = json.loads(p.read_text())
            fl[(p.name.split("-")[1], d["units"])].append(sustained(d))
        # NAMED, not "however many boxes happen to have floor files". This was
        # `if len(boxes) == 2`, and the moment a third box finished its runs the
        # condition went false and TWELVE guards vanished -- the check count fell
        # from 111 to 105 and the run still ended green. Same fail-open shape as
        # the M4 Max flip-set and the 2.4 gate-count guards. A guard must not be
        # able to disappear because the data grew.
        boxes = sorted({b for b, _ in fl})
        PAIR = ("experiments", "inference1")
        if all((b, "CPU_AND_GPU") in fl for b in PAIR):
            b1, b2 = PAIR
            g1, g2 = fl[(b1, "CPU_AND_GPU")], fl[(b2, "CPU_AND_GPU")]
            a1, a2 = fl[(b1, "CPU_AND_NE")], fl[(b2, "CPU_AND_NE")]
            require("3.6 floor GPU means",
                    "| %.4f | %.4f |" % (statistics.mean(g1), statistics.mean(g2)))
            require("3.6 floor GPU sds",
                    "| %.4f | %.4f |" % (statistics.stdev(g1), statistics.stdev(g2)))
            require("3.6 floor ANE means",
                    "| %.4f | %.4f |" % (statistics.mean(a1), statistics.mean(a2)))
            require("3.6 floor ANE sds",
                    "**%.4f** | **%.4f**" % (statistics.stdev(a1), statistics.stdev(a2)))
            diff = abs(statistics.mean(g1) - statistics.mean(g2))
            se = math.sqrt(statistics.stdev(g1) ** 2 / len(g1)
                           + statistics.stdev(g2) ** 2 / len(g2))
            s = ("difference is %.4f with a standard error of %.4f, or\n%.1f sigma"
                 % (diff, se, diff / se))
            checks.append(("3.6 between-machine difference", s,
                           s in text.replace("**", "")))
            # The n-dependence, measured over every 5-subset rather than asserted
            # from the fact that ranges grow.
            sub = [statistics.mean([max(c) - min(c)
                                    for c in itertools.combinations(v, 5)])
                   for v in (g1, g2)]
            # The settling. The twelve repeats are NOT exchangeable -- the
            # first run is the highest on both boxes -- so the sd over all
            # twelve measures a warm-up drift as much as run-to-run noise.
            # Derived in RUN ORDER, so these break if the set is reordered.
            def _stats(a, b):
                d = abs(statistics.mean(a) - statistics.mean(b))
                se = math.sqrt(statistics.stdev(a) ** 2 / len(a)
                               + statistics.stdev(b) ** 2 / len(b))
                return d, se, d / se
            ordered = {}
            for bx in (b1, b2):
                v = []
                for i in range(1, 13):
                    fp = soakdir / ("floor-%s-siglip-CPU_AND_GPU-%02d.json" % (bx, i))
                    if fp.exists():
                        v.append(sustained(json.loads(fp.read_text())))
                ordered[bx] = v
            o1, o2 = ordered[b1], ordered[b2]
            if len(o1) == 12 and len(o2) == 12:
                # ALL THREE boxes, and the sd-before/after for each. The
                # direction of the opening drift is NOT consistent -- two boxes
                # start high and one starts low -- so the series themselves are
                # the claim, not any summary of them.
                for bx in ("experiments", "inference1", "inference2"):
                    vv = []
                    for i in range(1, 13):
                        fp = soakdir / ("floor-%s-siglip-CPU_AND_GPU-%02d.json" % (bx, i))
                        if fp.exists():
                            vv.append(sustained(json.loads(fp.read_text())))
                    if len(vv) == 12:
                        require("3.6 settling series %s" % bx,
                                " ".join("%.4f" % x for x in vv))
                sds = []
                for bx in ("experiments", "inference1", "inference2"):
                    vv = []
                    for i in range(1, 13):
                        fp = soakdir / ("floor-%s-siglip-CPU_AND_GPU-%02d.json" % (bx, i))
                        if fp.exists():
                            vv.append(sustained(json.loads(fp.read_text())))
                    if len(vv) == 12:
                        sds.append((statistics.stdev(vv), statistics.stdev(vv[3:])))
                if len(sds) == 3:
                    require("3.6 settling sd reduction",
                            "%.4f to %.4f, %.4f to %.4f, %.4f to %.4f"
                            % (sds[0][0], sds[0][1], sds[1][0], sds[1][1],
                               sds[2][0], sds[2][1]))
                for lbl, sl in (("all 12 runs", slice(0, 12)),
                                ("dropping run 1", slice(1, 12)),
                                ("dropping runs 1\u20133", slice(3, 12))):
                    d, se, sg = _stats(o1[sl], o2[sl])
                    bold = "**" if lbl.startswith("dropping runs") else ""
                    require("3.6 settling row: %s" % lbl,
                            "| %s | %.4f | %.4f | %s%.1f%s | %.4f / %.4f |"
                            % (lbl, d, se, bold, sg, bold,
                               statistics.stdev(o1[sl]), statistics.stdev(o2[sl])))

            require("3.6 five-run subset ranges",
                    "five-run range averages %.4f and %.4f" % (sub[0], sub[1]))
            require("3.6 full twelve-run ranges",
                    "ranges of %.4f and %.4f"
                    % (max(g1) - min(g1), max(g2) - min(g2)))

        # The README's headline table is the SAME measurement as 3.1's medians
        # and had drifted onto a different file: it printed 233.0 / 1085.7 /
        # 1068.0 from results/sweep-m5max.json while the paper pins
        # zoo-m5max-v2.json. Derive both rows from the pinned sources so the two
        # documents cannot disagree about their own headline again.
        if "siglip" in m5:
            require_readme("README headline M5 Max row",
                           "| M5 Max (Mac17,7) | 40 | 16 | %.1f | **%.1f** | %.1f |"
                           % (m5["siglip"]["CPU_AND_NE"], m5["siglip"]["CPU_AND_GPU"],
                              m5["siglip"]["ALL"]))
        # The M4 Pro row is NOT derived the same way, and the reason is worth
        # stating. sweep-m4pro-rerun.json (which the pinned order resolves to)
        # and sweep-m4pro.json give 178.67 and 178.82 for siglip's GPU -- a gap
        # of 0.148, which is INSIDE the run-to-run peak spread the paper itself
        # documents at "178.7 to 178.8" in 3.3. Two runs of one measurement
        # agreeing to within their own spread is not a provenance defect, and
        # rewriting 178.8 to 178.7 across the paper would be churn.
        #
        # The ALL column is different: 172.16 against 171.12, a gap of 1.04 or
        # 0.6%, well outside that spread -- and it was unguarded, because
        # PAPER_TABLE carries only the ANE and GPU columns. It is checked below.
        if "siglip" in m4:
            require_readme("README headline M4 Pro row",
                           "| M4 Pro (Mac16,11) | 20 | 16 | **%.1f** | 178.8 | **172.2** (slowest) |"
                           % m4["siglip"]["CPU_AND_NE"])

        require_readme("README 3.3 summary",
                       "in %d of %d soaks; the GPU gives back %s in %d"
                       % (len(above), len(ane), gb, len(gpu)))
        require("3.3 GPU give-back range, abstract",
                "the GPU gives back %s" % gb.split(" to ")[0] + " to")

        # The 3.3 per-model table. Four of its fifteen cells had drifted -- two
        # counts and two range ends -- while every number around them stayed
        # right, which is what an unguarded table looks like as soaks accumulate.
        # Regenerate each row verbatim and require it, so adding a soak either
        # updates the table or fails here.
        m4 = [r for r in rows if "M4" in r["chip"]]
        if m4:
            # The whisper row bolds its exception (`**0.9748**`), so compare the
            # rows against the paper with emphasis stripped rather than teaching
            # the generator where the bold happens to sit today.
            flat = text.replace("**", "")
            ta, tg = [], []
            for m in ("siglip", "resnet50", "mobilenet", "bert", "whisper"):
                a = [r["sf"] for r in m4 if r["model"] == m and r["units"] == "CPU_AND_NE"]
                g = [r for r in m4 if r["model"] == m and r["units"] == "CPU_AND_GPU"]
                if not a or not g:
                    continue
                gs = [r["sf"] for r in g]
                pk = [r["peak"] for r in g if r["peak"]]
                ta += a
                tg += gs
                stab = (max(pk) - min(pk)) / max(pk) * 100 if pk else 0.0
                cells = ("%.4f to %.4f (%d) | %.3f to %.3f (%d) | %.2f%%"
                         % (min(a), max(a), len(a), min(gs), max(gs), len(gs), stab))
                checks.append(("3.3 table row %s" % m, cells, cells in flat))
            if ta and tg:
                require("3.3 table all row",
                        "**%.4f to %.4f in %d** | **%.3f to %.3f in %d**"
                        % (min(ta), max(ta), len(ta), min(tg), max(tg), len(tg)))
                # "In N of M M4 Pro soaks the ANE gave back at most X%, better
                # than the best of K GPU soaks, which gave back Y%." Both K and Y
                # were stale by two soaks and 0.26 points respectively.
                ok = [x for x in ta if (1 - x) * 100 <= 0.12]
                require("3.3 ANE give-back headline",
                        "In %d of %d M4 Pro soaks the ANE gave back at most 0.12%% "
                        "of its peak, better\nthan the best of %d GPU soaks, which "
                        "gave back %.2f%%.**" % (len(ok), len(ta), len(tg),
                                                 (1 - max(tg)) * 100))
                require("3.3 ANE holds in N of M",
                        "holding in %d of %d measurements" % (len(ok), len(ta)))
                require("3.3 ANE exact-1.0000 count",
                        "only %d of %d are exactly 1.0000"
                        % (sum(1 for x in ta if round(x, 4) == 1.0), len(ta)))
        if len(ane) - len(above) > 1:
            fails.append("more than one ANE soak now falls below the best GPU soak "
                         "(%d of %d). The paper names exactly one exception; "
                         "rewrite the claim rather than widening the exception."
                         % (len(ane) - len(above), len(ane)))

    # 3.3's cold-start mechanism. Three back-to-back 600 s GPU soaks per model,
    # no cooldown. Every dip figure and every steady-state value the section
    # quotes is recomputed here, so the subsection cannot drift from its files.
    for base, lbl in (("inference1-siglip-vision", "siglip"),
                      ("inference1-resnet50", "resnet50"),
                      ("experiments-bert", "bert"),
                      ("experiments-whisper", "whisper"),
                      ("inference1-mobilenet", "mobilenet"),
                      ("experiments-mobilenet", "mobilenet2"),
                      ("experiments-siglip-vision", "siglip2"),
                      ("inference1-whisper", "whisper2"),
                      ("experiments-resnet50", "resnet502"),
                      ("inference1-bert", "bert2")):
        for i in (1, 2, 3):
            f = REPO / "results" / "soak" / ("dip%d-%s-CPU_AND_GPU.json" % (i, base))
            if not f.exists():
                continue
            r = windows(f)
            require("3.3 %s dip run %d" % (lbl, i),
                    "%.2f%%" % (100 * (max(r) - min(r)) / max(r)))

    # The window-1 table that REFUTES the cold-boost reading. These four numbers
    # are the whole argument: if the cold advantage at the start is the same size
    # as the dip, the old explanation was right after all. Guard them so nobody
    # has to take the refutation on trust.
    def w1(path):
        f = REPO / "results" / "soak" / path
        return windows(f)[0] if f.exists() else None

    for base, lbl in (("inference1-siglip-vision", "siglip"),
                      ("inference1-resnet50", "resnet50"),
                      ("experiments-bert", "bert"),
                      ("experiments-whisper", "whisper")):
        cold = w1("dip1-%s-CPU_AND_GPU.json" % base)
        warm = [w1("dip%d-%s-CPU_AND_GPU.json" % (i, base)) for i in (2, 3)]
        if cold is None or None in warm:
            continue
        mean = sum(warm) / len(warm)
        require("3.3 %s w1 cold" % lbl, "%.2f" % cold)
        require("3.3 %s w1 warm mean" % lbl, "%.2f" % mean)
        require("3.3 %s w1 cold advantage" % lbl, "+%.2f%%" % (100 * (cold - mean) / mean))

    # The ANE control. Same protocol, other engine. Only the runs that exist are
    # checked, so a chain still mid-flight does not fail the paper -- but any run
    # the prose quotes must be on disk, because require() fails on a missing
    # string rather than skipping it.
    for base, lbl in (("inference1-siglip-vision", "siglip"),
                      ("experiments-whisper", "whisper"),
                      ("inference1-resnet50", "resnet50"),
                      ("experiments-bert", "bert"),
                      ("experiments-mobilenet", "mobilenet"),
                      ("inference1-mobilenet", "mobilenet2")):
        for i in (1, 2, 3):
            f = REPO / "results" / "soak" / ("aned%d-%s-CPU_AND_NE.json" % (i, base))
            if not f.exists():
                continue
            r = windows(f, complete_only=True)
            if len(r) < 20:          # still being written
                continue
            require("3.3 %s ANE control run %d" % (lbl, i),
                    "%.2f%%" % (100 * (max(r) - min(r)) / max(r)))

    for label, s, ok in checks:
        print("%-38s %s  %r" % (label, "ok  " if ok else "FAIL", s))
        if not ok:
            where = ("results/soak/README.md" if label.startswith("soak README")
                     else "README.md" if label.startswith("README") else "PAPER.md")
            fails.append("%s: %s does not contain %r" % (label, where, s))

    if extra:
        print("\nnote: %d soak file(s) on disk are not in PAPER-SET.txt. That is not a\n"
              "failure. Run --refresh-set and update the prose when you want them in."
              % extra)

    # A GUARD THAT VANISHES IS WORSE THAN ONE THAT FAILS, because the run still
    # ends green. Three times in this file: the M4 Max flip-set check gated on
    # the set having three members, the 2.4 gate-count check gated on there
    # being four models, and the whole 3.6 block gated on exactly two boxes
    # having floor files -- which a third box switched off, taking twelve guards
    # with it and dropping the count from 111 to 105 at exit status 0.
    #
    # Fixing each site as it appears does not scale: fifteen conditional blocks
    # here emit thirty-seven guards between them, and any of them can stop
    # firing when the data changes shape. So the COUNT is itself a check. Raise
    # this when you add guards. It should never fall.
    MIN_CHECKS = 139
    if len(checks) < MIN_CHECKS:
        fails.append("only %d checks ran, expected at least %d. Guards do not "
                     "fail by disappearing; some conditional block stopped "
                     "firing. Find it before trusting this run."
                     % (len(checks), MIN_CHECKS))

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
