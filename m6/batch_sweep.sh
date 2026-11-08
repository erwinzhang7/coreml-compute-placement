#!/usr/bin/env bash
# One instance on CPU_AND_NE across batch sizes: if a single large request were split
# across both engines, img/s would step up at larger batches instead of flattening.
set -euo pipefail
cd "$(dirname "$0")/.."
for B in 1 8 16 32; do
    IT=$(( 480 / B )); (( IT < 30 )) && IT=30
    .venv/bin/python tools/sweep.py --models models/siglip-ane-b${B}.mlpackage \
        --units CPU_AND_NE --batch $B --iters $IT --repeats 5 --out out/batch-b${B}.json
done
