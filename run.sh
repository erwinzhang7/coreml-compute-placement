#!/usr/bin/env bash
#
# Reproduce the measurement on your own machine and print a paste-ready table.
#
#   ./run.sh
#   ./run.sh --batch 8 --quick
#
# Builds both model variants, sweeps the three compute-unit settings, and
# summarises. The venv and the built .mlpackages are gitignored; the sweep
# JSON is written into results/, which is tracked, so you can commit yours.
#
# If your chip disagrees with the two in the README, that is the interesting
# case: please open an issue with the table this prints.

set -euo pipefail

BATCH=16
ITERS=30
REPEATS=5
LABEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)   BATCH="$2"; shift 2 ;;
        --label)   LABEL="$2"; shift 2 ;;
        --quick)   ITERS=10; REPEATS=3; shift ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
[[ -n "$LABEL" ]] || LABEL="$(echo "$CHIP" | tr '[:upper:] ' '[:lower:]-' | sed 's/^apple-//')"

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
