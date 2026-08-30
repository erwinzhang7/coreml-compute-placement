# Where does your Core ML model actually run?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22136103.svg)](https://doi.org/10.5281/zenodo.22136103)

On Apple silicon a model can run on the Neural Engine, the GPU, or the CPU. Core ML picks
for you, reports success either way, and never tells you what it chose. This repo measures
what it chooses and what that costs.

**The right compute unit does not transfer between configurations, and Core ML's default is
sometimes the slowest of the available options.**

| chip | GPU cores | NPU cores | ANE | GPU | default (`ALL`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| M4 Pro (Mac16,11) | 20 | 16 | **204.4** | 178.8 | **172.2** (slowest) |
| M5 Max (Mac17,7) | 40 | 16 | 232.9 | **1077.7** | 1053.2 |

images/s, SigLIP-base-224 vision tower, batch 16, fp16, median of 5, from the same pinned
sources as PAPER.md §3.1 (`results/zoo-m5max-v2.json`). The ANE-rewrite figures further down
quote 1085.7 for the M5 Max GPU instead: those come from `results/sweep-m5max.json`, where the
naive and rewritten models were measured in one run, and a before/after has to use two numbers
from the same run rather than a headline from one file and a variant from another. On the M4 Pro the
Neural Engine is the fastest unit and the default is the slowest option. On the M5 Max the
GPU is 4.7x faster than the ANE. Neither result generalises to the other chip, and nothing
in the API tells you which situation you are in.

**Not the same macOS build**: the M4 Pro ran 26.5.1 and the M5 Max ran 26.6. Core ML's
placement heuristic ships inside the OS, so chip and OS cannot be separated with these two
machines. [PAPER.md §3.5](PAPER.md) sets out what that does and does not undermine.

## Run it on your own chip

```sh
./run.sh              # sweep the three compute-unit settings, print a table
./run.sh --soak       # add the sustained pass: what a unit HOLDS, not what it reaches
```

Two chips is a thin basis for a claim about chip families. If yours ranks them differently,
that is the result I most want to see. There is an
[issue template](.github/ISSUE_TEMPLATE/chip-report.yml), and a
`results/sweep-<chip>.json` in a pull request is better still.

Run on an **idle** machine, and on a laptop **plug it in**: on battery the GPU throttles for
reasons unrelated to heat, and the summariser flags the run if you forget.

## The paper

[**PAPER.md**](PAPER.md) is the full treatment across **five** architectures rather than the
one above, with method, mechanism, threats to validity and citations. In short:

- **§3.1** The faster unit inverts between chips on three of five architectures. The two that
  do not invert still move their margin by 2.4x and 3.6x.
- **§3.2** The default costs 1.18x to 5.18x on the M4 Pro, and on three of five is slower
  than *both* explicit placements. On the M5 Max it is free.
- **§3.3** Peak does not predict sustained. Over two minutes the ANE gives back at most 0.14%
  in 41 of 42 soaks; the GPU gives back 1.1 to 16.3% in 60.
- **§3.4** Mechanism: the M5's per-GPU-core neural accelerators, with a falsifiable prediction.
- **§4** Threats to validity. Read these before citing any number.

`python3 tools/verify_claims.py` recomputes every headline number in PAPER.md from
`results/` and fails if the prose and the data disagree.

---

## Further results, not in the paper

These are measured and reproducible but sit outside the paper's scope, which is
compute-unit placement for a single model in a single process.

### Concurrency is nearly free on one chip and expensive on the other

Both units driven simultaneously from separate processes, same model:

| chip | ANE alone | GPU alone | combined | % of ideal sum | uplift vs best single |
| --- | ---: | ---: | ---: | ---: | ---: |
| M4 Pro | 204.4 | 178.9 | 381.3 | **99%** | 1.87x |
| M5 Max | 231.3 | 901.5 | 970.3 | **86%** | 1.08x |

**The GPU pays the contention, not the ANE.** On the M5 Max the GPU drops to 84% of its solo
rate while the ANE holds 93%. A GPU already sustaining ~900 img/s is near the
memory-bandwidth limit, so the ANE's traffic displaces it; the M4 Pro's GPU at 179 img/s is
far from that limit and both units run unimpeded. So concurrency is worth most exactly where
it is cheapest.

### The CPU's share of its own memory bus nearly halves between chips

Whether the CPU is a candidate at all, and it changes by chip the same way. Measured
streaming-read bandwidth per engine (`tools/membw.c`, `tools/gpu_bw.py`), GB/s:

| chip | CPU cores | GPU cores | bus peak | 1 core | CPU all cores | GPU | CPU share | GPU/CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M4 Pro | 14 | 20 | 273 | 89.8 | 249.5 | 253.5 | **91%** | **1.02x** |
| M5 Max | 18 | 40 | 614 | 87.8 | 303.8 | 566.7 | **49%** | **1.87x** |

GPU bandwidth grows 2.24x against 2.0x the cores and a 2.25x bus, tracking the hardware
almost exactly. CPU bandwidth grows 1.22x. The bus was sized for the GPU and the CPU was not
scaled with it.

**It is not core count.** Both chips saturate their CPU-side path long before running out of
cores:

| threads | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 18 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M4 Pro (of 14) | 89.8 | 160.6 | 198.6 | 233.5 | 239.3 | 239.9 | 249.4 | n/a |
| M5 Max (of 18) | 87.8 | 161.8 | 233.1 | 287.3 | 299.0 | **303.8** | 272.2 | 296.3 |

The M5 Max is done at 8 threads; its remaining ten cores add nothing. Single-core bandwidth
is essentially identical across the two, 89.8 against 87.8, despite a 2.25x difference in bus
width. Whatever limits one core is not the DRAM.

Running both engines at once adds **4-5%** over the GPU alone on either chip. Over-committing
takes it below GPU-only **on the M5 Max only**: at all 18 threads it reads 0.99x (-7.1 GB/s),
while the M4 Pro at all 14 threads is still 1.03x (+7.5 GB/s). Per-repeat numbers, with the contended rate solved
for rather than assumed, are in [`results/membw-m4pro.txt`](results/membw-m4pro.txt) and
[`results/membw-m5max.txt`](results/membw-m5max.txt) (`tools/contention.py`). This bears on CPU/GPU-offload
designs ported from discrete-GPU systems, where the premise is a slow link between two memory
pools. Neither Apple part has that link, and on neither does recruiting the CPU add more than
about 5%.

Three methodology points, each of which produced a confident wrong number first: align the
windows and solve for the contended rate rather than letting each process time itself, which
once reported an aggregate above the hardware peak; warm up before the first timed case,
since GPU clock ramp costs 50-70% and dwarfs the effect; interleave configurations and keep
the machine idle, since a background job was observed changing the same binary's result by 3x.

### The ANE rewrite is worth 10-13% here, and makes the GPU slower

Applying the full `ml-ane-transformers` treatment (`(B,C,1,S)` layout, `nn.Linear` to 1x1
`nn.Conv2d`, per-head chunked attention, channel-dim LayerNorm) gives 204.4 to 231.3 img/s on
M4 Pro and 233.0 to 256.7 on M5 Max. It is not more because **the naive conversion was already
100% ANE-resident**: a ViT starts ANE-shaped, since its patch embedding is a Conv2d emitting
`(B,C,H,W)`. Apple's larger reported gains were on models that fell off the ANE entirely, and
there was no such cliff to recover here. Verified against the HuggingFace model at max abs
diff 3.8e-06, min cosine 0.99999988.

The rewrite is **unit-specific and costs the GPU**: M4 Pro GPU 178.8 to 149.5, M5 Max GPU
1085.7 to 720.1. If you run both units, ship a different variant to each; doing that on the
M4 Pro measured 408.3 img/s combined against 381.3 sharing one model.

### Energy

M5 Max, `powermetrics`, sustained load, batch 16:

| units | img/s | ANE | GPU | CPU | total | energy/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `CPU_AND_NE` | 233.0 | 4.8 W | 0.07 W | 4.0 W | 8.9 W | 38.2 mJ |
| `CPU_AND_GPU` | 962.1 | 0 W | 39.4 W | 4.7 W | 44.1 W | 45.8 mJ |

The ANE is ~1.2x better per image end to end at 5x lower power. But **45% of the ANE
configuration's power is CPU**, Python `predict()` driving overhead, against 11% for the GPU
configuration. On silicon alone the ANE is ~20.6 mJ/image against the GPU's ~41.0 mJ, about 2.0x better;
the 2.2x an earlier version gave here compared the ANE's silicon-only figure with the GPU's
end-to-end one, which is not a like-for-like ratio. Any serious ANE
deployment must eliminate that driving overhead before its efficiency can be judged.
`ANE Power: 0 mW` under `CPU_AND_GPU` independently confirms the compute plan: placement
analysis and the wattmeter agree.

### What `ALL` actually resolves to

`MLComputePlan`, cost-weighted, across all five architectures. This is now generated by
`tools/placement_sweep.py` into `results/placement-<chip>.json` rather than measured by
hand, and it answers the asymmetry in PAPER.md §3.2:

| model | M4 Pro, `ALL` resolves to | M5 Max, `ALL` resolves to |
| --- | --- | --- |
| `siglip` | **78.9% ANE / 21.1% GPU** | 100% GPU |
| `bert` | 77.5% ANE / 17.3% GPU / 5.1% CPU | 100% GPU |
| `whisper` | 64.1% ANE / 35.9% GPU | 100% GPU |
| `resnet50` | **87.6% ANE / 12.4% GPU** | 100% GPU |
| `mobilenet` | **100% ANE** | 100% GPU |

**On the M5 Max `ALL` never splits.** One engine every time, always the GPU, which is the
right answer on that chip, and the default costs 1.01x to 1.09x. **On the M4 Pro it splits
on four of five architectures**, and there it costs 1.18x to 5.18x. `mobilenet` is the
exception: the runtime puts it entirely on the ANE and it still costs 1.18x. So the default is free
on the M5 Max because it is not doing anything.

`siglip` on the M4 Pro is the clearest case: the 79/21 split runs slower than pure ANE
(204.4) *and* slower than pure GPU (178.8). Handoff costs more than the parallelism gains.

### The compute plan goes blank in exactly the configuration that fails to compile

Measuring the ANE-rewrite variant to back the last unsourced rows turned up something
better than a confirmation. On the M4 Pro under `ALL`:

| units | ops | placement **by cost** | placement **by count** |
| --- | ---: | --- | --- |
| `ALL` | 1658 | **{}, nothing returned** | 98.5% ANE, 1.5% unknown |
| `CPU_AND_NE` | 1658 | 100% ANE | 100% ANE |
| `CPU_AND_GPU` | 1658 | 100% GPU | 100% GPU |

`MLComputePlan` returns **no per-operation cost estimates at all** for that one cell,
while the naive model on the same box under the same setting returns
78.9% ANE / 21.1% GPU quite happily. Cost weighting is the thing `anecheck.py` exists
to provide, so this is the case where the tool has least to say.

It is the same cell that fails to compile. `results/anecc-probe.md` establishes by
exhaustive sweep that `ANECCompile() FAILED` needs all four of **ANE-rewritten model,
`ALL`, cold compile, M4 Pro**, and the sweep above reproduced that error five times
while measuring throughput. So Core ML's introspection goes silent in precisely the
corner where its compilation breaks. We report the co-occurrence; four conditions
matching exactly is not proof of causation.

**The README previously read "~98.5% ANE" for this cell.** That figure is real but it
is the *by-count* number, sitting in a table whose every other entry is cost-weighted.
Corrected above rather than deleted.

| chip | model | `ALL` img/s | best pure | |
| --- | --- | ---: | ---: | --- |
| M4 Pro | ANE-rewritten | 240.8 | 231.2 (ANE) | slightly better |
| M5 Max | ANE-rewritten | 720.4 | 720.1 (GPU) | matches best, **still hand-measured** |

The M4 Pro row is now backed by `results/sweep-variant-m4pro.json` and
`results/placement-variant-m4pro.json`, and it reproduces the hand-typed 240.9 / 231.3
to within 0.1. The M5 Max row has no file behind it; weight it accordingly.

### A compile failure the framework never surfaces

On the M4 Pro a **cold** compile of the ANE-rewritten model under `ALL` emits
`_ANECompiler : ANECCompile() FAILED`, falls back silently, and runs anyway as the *fastest*
configuration for that model, with bit-identical outputs. The message never reaches the
calling process's stderr, and it disappears once the compiled artifact is cached, which is why
it reads as intermittent rather than deterministic. Four conditions are all required to see
it, pinned down in [`results/anecc-probe.md`](results/anecc-probe.md).

---

## Reproducing

Requires macOS 15+ on Apple silicon and **Python 3.12**: coremltools ships no native
extension for 3.13+, and without it `MLComputePlan` and `MLModel` do not exist.

`./run.sh` does everything below in one step. The individual steps, to vary something:

```sh
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# build the two model variants (ane_siglip.py refuses to emit a model that does
# not match the HuggingFace reference numerically)
./.venv/bin/python models/convert_siglip.py --batch 16 --out siglip-vision-b16.mlpackage
./.venv/bin/python models/ane_siglip.py    --batch 16 --out siglip-ane-b16.mlpackage
./.venv/bin/python models/convert_zoo.py                    # the other four architectures

# where do the ops go?
./.venv/bin/python tools/anecheck.py siglip-vision-b16.mlpackage --compute-units ALL

# throughput, repeated, with spread
./.venv/bin/python tools/sweep.py --models siglip-vision-b16.mlpackage siglip-ane-b16.mlpackage

# both units at once, optionally a different variant per unit
./.venv/bin/python tools/bench_concurrent.py siglip-vision-b16.mlpackage \
    --model-ane siglip-ane-b16.mlpackage --model-gpu siglip-vision-b16.mlpackage

# what a unit HOLDS rather than what it reaches
./.venv/bin/python tools/thermal_soak.py siglip-vision-b16.mlpackage \
    --units CPU_AND_GPU --seconds 120 --window 10 --out results/soak/mychip-gpu.json
python3 tools/summarise_soak.py "results/soak/mychip-*.json"

# memory bandwidth
cc -O3 -o tools/membw tools/membw.c
./tools/membw --threads 10 --gb 16 --secs 3        # --qos bg steers to the E cores
python tools/gpu_bw.py --gb 4 --secs 3
python tools/contention.py --threads 10 --secs 15 --reps 4
```

`contention.py` waits for the load average to settle and prints `UNRELIABLE` rather than a
verdict if run-to-run spread exceeds 20%.

### Tools

| tool | what it does |
| --- | --- |
| `tools/anecheck.py` | per-op compute-device placement from `MLComputePlan`, **cost-weighted**, with `--assert-ane-fraction` as a CI gate |
| `tools/sweep.py` | repeated throughput runs, reports per-run values and spread |
| `tools/bench_concurrent.py` | both units simultaneously, separate processes, optional per-unit model variant |
| `tools/bench_power.py` | sustained-load driver to pair with `powermetrics` |
| `tools/bench_coreml.py` | single-configuration throughput |
| `tools/probe_gpu.py` | matched matmul-bound and bandwidth-bound models, to separate matmul-specific hardware from general GPU uplift |
| `tools/membw.c` | CPU streaming-read bandwidth, per core tier via QoS, emitting its timed window as `CLOCK_MONOTONIC` timestamps |
| `tools/gpu_bw.py` | GPU streaming bandwidth via MLX, several op shapes with explicit traffic accounting |
| `tools/contention.py` | both engines in aligned windows, interleaved repeats, solves for the contended rate, refuses a verdict above 20% spread |
| `tools/thermal_soak.py` | sustained load bucketed into windows, with thermal pressure, power source and **concurrent load** per run; writes each window as it completes so a kill costs one window |
| `tools/verify_conversions.py` | numerical equivalence of every converted model against its source |
| `tools/verify_claims.py` | recomputes PAPER.md's headline numbers from `results/` and fails on disagreement |
| `tools/summarise.py` | renders a sweep JSON into the reporting table, stdlib only |
| `tools/summarise_soak.py` | renders soak JSONs into the sustained table, flagging battery, power-source changes and interruptions |
| `tools/fleet_busy.sh` | which machines are running work, distinguishing "between chain steps" from idle |

## Core AI

Apple shipped Core AI at WWDC 2026, a pipeline layer above Core ML. Placement did not go away
with it: `MLComputeUnits` becomes `SpecializationOptions(preferredComputeUnitKind:)`, an
allow-list becomes a single *preferred* unit, and it moves to Swift. Apple's own reference
recipes choose that unit from **model structure with no chip term**, which is precisely what
§3.1 measures to be wrong.

**None of it can be measured here.** `CoreAI.framework` is absent from macOS 26.6 and the
Xcode 26.6 SDK, and Apple's package declares `platforms: [.macOS("27.0")]`.
[**CORE-AI.md**](CORE-AI.md) sets out the tension and what it would take to settle.

## Prior art

`anecheck.py` is not the first tool of its kind. Existing compute-plan tooling:

- [`john-rocky/CoreML-LLM`](https://github.com/john-rocky/CoreML-LLM),
  `conversion/audit_ane_residency.py`, a mature ANE residency auditor. Its nested-block
  traversal is more correct than this repo's first version was, and was the reason that bug
  got fixed here.
- [`pytorch/executorch`](https://github.com/pytorch/executorch),
  `examples/apple/coreml/scripts/coreml_compute_plan.py`
- [`Anemll/Anemll`](https://github.com/Anemll/Anemll), `anemll/utils/ane_profiler.py`
- [`freedomtan/coreml_modelc_profling`](https://github.com/freedomtan/coreml_modelc_profling)

What this one adds is **cost weighting**, residency weighted by `MLComputePlan`'s per-op
estimated cost rather than by op count, so one stranded matmul is not hidden by fifty cheap
ANE ops, plus a pass/fail threshold for CI.

The ANE model rewrite follows [`apple/ml-ane-transformers`](https://github.com/apple/ml-ane-transformers)
and the Apple ML research article *Deploying Transformers on the Apple Neural Engine*.

## Licence

MIT. SigLIP weights are Google's, under their own terms.
