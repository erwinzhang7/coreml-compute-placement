#!/usr/bin/env bash
# Memory bandwidth: GPU STREAM (m6/stream_bw.py), the repo's original gpu_bw.py for continuity,
# and CPU streaming read (tools/membw.c) over a thread sweep.
#   m6/bw_all.sh <published GB/s> <GiB per array> "<thread counts>"
set -euo pipefail
cd "$(dirname "$0")/.."
PUB=$1; GIB=$2; THREADS=$3
V=.venv-llm/bin/python
[[ -x bin/membw ]] || { mkdir -p bin && cc -O3 -o bin/membw tools/membw.c; }
$V -u m6/stream_bw.py --gib "$GIB" --published "$PUB" --out out/stream-bw.json
echo "== tools/gpu_bw.py (MLX ops, best of several)"
$V -u tools/gpu_bw.py --gb "$GIB" --secs 3
echo "== CPU streaming read, tools/membw.c, median of 3 per thread count"
for T in $THREADS; do
    R=$(for r in 1 2 3; do bin/membw --threads "$T" --gb "$GIB" --secs 2 --quiet; done | awk "{print \$3}" | sort -n | sed -n 2p)
    echo "threads $T  $R GB/s"
done | tee out/cpu-membw.txt
