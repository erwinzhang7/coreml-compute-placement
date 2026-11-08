#!/usr/bin/env bash
# M6 setup, run once from the staging dir: venv pinned to the versions the
# M4 Pro / M5 Max results used, the ANE SigLIP at batch 1/8/32 (b16 is copied from the
# published run, not rebuilt), compiled .mlmodelc for the Swift bench, the device probes,
# and MLComputePlan residency for every model.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
OUT=out; mkdir -p "$OUT"
PY=/opt/homebrew/bin/python3.12

if [[ ! -x .venv/bin/python ]]; then
    "$PY" -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet coremltools==9.0 numpy==2.5.2 torch==2.8.0 transformers==5.16.1
fi
V=.venv/bin/python
$V -c "import coremltools,numpy,torch;print('coremltools',coremltools.__version__,'numpy',numpy.__version__,'torch',torch.__version__)"

for B in 1 8 32; do
    M=models/siglip-ane-b${B}.mlpackage
    # NNPACK off: at batch 32 torch.jit.trace's forward pass took NNPACK's FFT path for the
    # 1x1 convolutions and ran >10 min on the CPU (b8: 15 s). The trace is the same either way.
    [[ -d $M ]] || $V -c "import sys, runpy, torch; torch.backends.nnpack.set_flags(False); \
sys.argv = ['ane_siglip.py', '--batch', '$B', '--out', '$M']; \
runpy.run_path('models/ane_siglip.py', run_name='__main__')"
done

mkdir -p bin
swiftc -O m6/compile_model.swift -o bin/compile_model
swiftc -O m6/multi_model.swift -o bin/multi_model
swiftc -O m6/probe_devices.swift -o bin/probe_devices
swiftc -O m6/probe_ane_private.swift -o bin/probe_ane_private

# Compiled copies for Swift (MLModel(contentsOf:) wants .mlmodelc).
for M in models/siglip-ane-b16.mlpackage models/siglip-vision-b16.mlpackage; do
    [[ -d ${M%.mlpackage}.mlmodelc ]] || bin/compile_model "$M"
done

{
    echo "# $(date '+%Y-%m-%d %H:%M %Z')  $(sysctl -n machdep.cpu.brand_string) $(sysctl -n hw.model) macOS $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "## MLComputeDevice.allComputeDevices"
    bin/probe_devices
    echo "## private AppleNeuralEngine.framework (read-only)"
    bin/probe_ane_private
    echo "## IOKit ANE services"
    ioreg -r -c H11ANEIn -d 1 -w0 | grep -E '^\+-o|"(ane-type|ane-subtype|IONameMatched|device-type|ane-num|num-|cores)' || true
    ioreg -r -c H1xANELoadBalancer -d 1 -w0 | head -40 || true
    ioreg -l -w0 | grep -E '"#ane-num-clusters"|"ane-power-' | sort -u | head || true
} > "$OUT/devices.txt" 2>&1
cat "$OUT/devices.txt"

mkdir -p "$OUT/residency"
for M in models/siglip-ane-b{1,8,16,32}.mlpackage models/siglip-vision-b16.mlpackage; do
    for U in CPU_AND_NE ALL; do
        N=$(basename "$M" .mlpackage)
        $V tools/anecheck.py "$M" --compute-units $U --json "$OUT/residency/$N-$U.json" \
            | tail -8 || echo "anecheck failed: $M $U"
    done
done
echo SETUP_DONE
