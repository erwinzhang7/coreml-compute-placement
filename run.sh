#!/usr/bin/env bash
#
# Reproduce the measurement on your own machine and print a paste-ready table.
#
#   ./run.sh                      burst throughput (~2 min after setup)
#   ./run.sh --soak               burst, then a 120 s sustained soak per unit
#   ./run.sh --batch 8 --quick    faster, noisier
#   ./run.sh --soak-seconds 300   a longer soak; implies --soak
#
# Builds both model variants, sweeps the three compute-unit settings, and
# summarises. The venv and the built .mlpackages are gitignored; the sweep
# JSON is written into results/, which is tracked, so you can commit yours.
#
# --soak answers a different question from the sweep. The sweep measures the
# rate a unit can REACH; the soak measures the rate it HOLDS. On an M5 Max the
# GPU keeps only 0.837 of its peak over two minutes while the ANE keeps 0.999,
# so a burst-only number can rank the units differently from a long-running job.
#
# If your chip disagrees with the two in the README, that is the interesting
# case: please open an issue with the tables this prints.

set -euo pipefail

BATCH=16
ITERS=30
REPEATS=5
LABEL=""
SOAK=0
SOAK_SECONDS=120

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)   BATCH="$2"; shift 2 ;;
        --label)   LABEL="$2"; shift 2 ;;
        --quick)   ITERS=10; REPEATS=3; shift ;;
        --soak)    SOAK=1; shift ;;
        --soak-seconds) SOAK=1; SOAK_SECONDS="$2"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ── environment guards, because each of these fails confusingly later ──────
[[ "$(uname -s)" == "Darwin" ]] || { echo "This needs macOS on Apple silicon." >&2; exit 1; }
[[ "$(uname -m)" == "arm64"  ]] || { echo "This needs Apple silicon, not Intel." >&2; exit 1; }

# coremltools ships no native extension for 3.13+, and without it MLComputePlan
# and MLModel do not exist, so the failure surfaces as a confusing AttributeError.
PY="$(command -v python3.12 || true)"
[[ -n "$PY" ]] || {
    echo "Python 3.12 not found. coremltools has no native extension for 3.13+," >&2
    echo "so 3.12 specifically is required (brew install python@3.12)." >&2
    exit 1
}

CHIP="$(sysctl -n machdep.cpu.brand_string)"
MODEL_ID="$(sysctl -n hw.model)"
OS="$(sw_vers -productVersion)"
# LABEL derives from the chip brand string, so two physically different machines
# with the same chip produce the same label and overwrite each other's results.
# This study runs TWO identical M4 Pro Mac minis specifically to measure
# machine-to-machine variation, so a collision would silently destroy the control
# it depends on. An audit caught it; the soaks had been hand-named and got lucky.
# LocalHostName disambiguates without leaking anything a public result should not
# carry, and is stable across reboots.
[[ -n "$LABEL" ]] || {
    LABEL="$(echo "$CHIP" | tr '[:upper:] ' '[:lower:]-' | sed 's/^apple-//')"
    HOSTTAG="$(scutil --get LocalHostName 2>/dev/null | tr '[:upper:]' '[:lower:]' \
               | sed 's/[^a-z0-9]//g' | tail -c 9)"
    [[ -n "$HOSTTAG" ]] && LABEL="${LABEL}-${HOSTTAG}"
}

echo "── $CHIP ($MODEL_ID), macOS $OS, batch $BATCH ──"

if [[ ! -x .venv/bin/python ]]; then
    echo "── creating .venv (this pulls in torch, so it is not quick) ──"
    "$PY" -m venv .venv
    ./.venv/bin/pip install --quiet --upgrade pip
    ./.venv/bin/pip install --quiet -r requirements.txt
fi
VPY=./.venv/bin/python

NAIVE="siglip-vision-b${BATCH}.mlpackage"
ANE="siglip-ane-b${BATCH}.mlpackage"

# Converting downloads the HuggingFace weights the first time.
[[ -d "$NAIVE" ]] || { echo "── building $NAIVE ──"; "$VPY" models/convert_siglip.py --batch "$BATCH" --out "$NAIVE"; }
[[ -d "$ANE"   ]] || { echo "── building $ANE ──";   "$VPY" models/ane_siglip.py    --batch "$BATCH" --out "$ANE"; }

echo "── where the ops land ──"
"$VPY" tools/anecheck.py "$NAIVE" --compute-units ALL || true

OUT="results/sweep-${LABEL}.json"
mkdir -p results
echo "── sweeping (this is the slow part) ──"
"$VPY" tools/sweep.py --models "$NAIVE" "$ANE" \
    --batch "$BATCH" --iters "$ITERS" --repeats "$REPEATS" --out "$OUT"

echo
echo "══════════ paste this into an issue ══════════"
python3 tools/summarise.py "$OUT"
echo "═════════════════════════════════════════════"
echo
echo "Raw JSON: $OUT"

if [[ "$SOAK" == "1" ]]; then
    # Three units at SOAK_SECONDS each, plus a 20 s gap so the next unit does not
    # start inside the previous one's thermal tail.
    TOTAL=$(( (SOAK_SECONDS + 20) * 3 / 60 ))
    echo
    echo "── sustained soak: 3 units x ${SOAK_SECONDS}s, about ${TOTAL} min ──"
    echo "   Leave the machine otherwise idle. A soak on a busy box measures the"
    echo "   other job. On a laptop, plug it in: on battery the GPU throttles for"
    echo "   reasons that have nothing to do with heat."
    mkdir -p "results/soak"
    for U in CPU_AND_NE CPU_AND_GPU ALL; do
        "$VPY" tools/thermal_soak.py "$NAIVE" --units "$U" \
            --seconds "$SOAK_SECONDS" --window 10 \
            --out "results/soak/${LABEL}-${U}.json"
        sleep 20
    done
    echo
    echo "══════════ sustained, paste this too ══════════"
    python3 tools/summarise_soak.py "results/soak/${LABEL}-"*.json
    echo "══════════════════════════════════════════════"
    echo
    echo "Raw JSON: results/soak/${LABEL}-*.json"
fi
