# Where does your Core ML model actually run?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22136103.svg)](https://doi.org/10.5281/zenodo.22136103)

On Apple silicon a model can run on the Neural Engine, the GPU, or the CPU. Core ML picks
for you, reports success either way, and never tells you what it chose. This repo measures
what it chooses and what that costs.

The short version: **the right compute unit depends on the chip, and Core ML's default is
sometimes the slowest of the available options.**

Same model, same batch, same macOS build, two chips:

| chip | GPU cores | NPU cores | ANE | GPU | default (`ALL`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| M4 Pro (Mac16,11) | 20 | 16 | **204.4** | 178.8 | **172.2**  ← slowest |
| M5 Max (Mac17,7) | 40 | 16 | 233.0 | **1085.7** | 1068.0 |

images/s, SigLIP-base-224 vision tower, batch 16, fp16. Median of 5 runs; spread below.

On the M4 Pro the Neural Engine is the **fastest** unit and the default is the **slowest**
option. On the M5 Max the GPU is 4.7x faster than the ANE. Neither result generalises to
the other chip, and nothing in the API tells you which situation you are in.

The same non-generality shows up on an axis that has nothing to do with the ANE. Measuring
memory bandwidth per engine, **the CPU complex reaches 91% of the bus on the M4 Pro and 49%
on the M5 Max** — so on one chip the CPU is a peer of the GPU for bandwidth-bound work and
on the other it is half as good. Finding 4.

There is also a failure the framework never tells you about. On the M4 Pro, a cold compile
of the ANE-rewritten model under `ALL` emits `_ANECompiler : ANECCompile() FAILED`, falls
back silently, and runs anyway — as the *fastest* configuration for that model, with
bit-identical outputs. The message never reaches the calling process's stderr, and it
disappears once the compiled artifact is cached, which is why it reads as intermittent
rather than deterministic. Four conditions are all required to see it, and they are pinned
down in [`results/anecc-probe.md`](results/anecc-probe.md).

**Run it on your own chip:**

```sh
./run.sh
```

That builds both model variants, sweeps the three compute-unit settings, and
prints a table ready to paste. Two chips is a thin basis for a claim about chip
families — if yours ranks them differently, that is the result I most want to
see. There is an [issue template](.github/ISSUE_TEMPLATE/chip-report.yml) for it.

---

## The four findings

### 1. The ANE/GPU ratio inverts within one chip family

| chip | ANE img/s | GPU img/s | ANE/GPU |
| --- | ---: | ---: | ---: |
| M4 Pro | 204.4 | 178.8 | **1.14x** |
| M5 Max | 233.0 | 1085.7 | **0.21x** |

**ANE throughput is nearly constant across the two chips** (204 vs 233, and both have the
same 16-core NPU). GPU throughput moves **6x** for **2x** the core count.

That gap is **matmul-specific**. Two synthetic Core ML models of comparable runtime, one
arithmetic-bound and one bandwidth-bound (`tools/probe_gpu.py`):

| probe | M5 Max | M4 Pro | ratio |
| --- | ---: | ---: | ---: |
| matmul, 103.1 GFLOP/call | 2.63 ms (**39.2 TFLOPS**) | 16.21 ms (**6.4 TFLOPS**) | **6.17x** |
| elementwise, 1610 MB/call | 13.43 ms (120 GB/s) | 17.04 ms (95 GB/s) | **1.27x** |

The two separate by a factor of five. On bandwidth-bound work the M5 Max is only 1.27x
faster, *less* than its 2x core-count advantage; on matmul it is 6.17x. Three checks:

- The matmul ratio (6.17x) matches the SigLIP GPU gap (6.07x) almost exactly, so the
  real-world difference is fully accounted for by matmul throughput.
- The ANE is a control at **1.14x** - same 16-core NPU on both, barely moved. This is not
  "M5 is a newer chip".
- Dividing 6.17x by the 2x core count leaves **~3.08x more matmul throughput per GPU core**,
  which is not a clock or process change.

That signature is what M5's per-core GPU neural accelerators would produce, and Apple
announced exactly that feature for M5. Note the limit of the claim: this measures the
*effect* as matmul-specific, it does not identify the mechanism in silicon.

The practical consequence: advice of the form "use the ANE for inference" or "the ANE is
too slow, use the GPU" is chip-specific, and both are currently stated as though they were
general.

### 2. Concurrency is nearly free on one chip and expensive on the other

Both units driven simultaneously from separate processes, same model:

| chip | ANE alone | GPU alone | combined | % of ideal sum | uplift vs best single |
| --- | ---: | ---: | ---: | ---: | ---: |
| M4 Pro | 204.4 | 178.9 | 381.3 | **99%** | 1.87x |
| M5 Max | 231.3 | 901.5 | 970.3 | **86%** | 1.08x |

**The GPU pays the contention, not the ANE.** On the M5 Max the GPU drops to 84% of its
solo rate while the ANE holds 93%. The plausible reading is that a GPU already sustaining
~900 img/s is near the memory-bandwidth limit, so the ANE's traffic displaces it; the M4
Pro's GPU at 179 img/s is far from that limit and both units run unimpeded.

So concurrency is worth most exactly where it is cheapest, which is a convenient shape but
not one you would guess.

### 3. The default `ComputeUnit.ALL` is unpredictable, and sometimes worst

`ALL` placement is chip- **and** model-dependent:

| chip | model | `ALL` placement (by cost) | `ALL` img/s | best pure | |
| --- | --- | --- | ---: | ---: | --- |
| M5 Max | naive | 100% GPU | 1068.0 | 1085.7 (GPU) | matches best |
| M5 Max | ANE-rewritten | 100% GPU | 720.4 | 720.1 (GPU) | matches best |
| M4 Pro | naive | **78.9% ANE / 21.1% GPU** | **172.2** | 204.4 (ANE) | **16% worse than either pure option** |
| M4 Pro | ANE-rewritten | ~98.5% ANE | 240.9 | 231.3 (ANE) | slightly better |

The M4 Pro naive row is the interesting one. `ALL` splits the graph 79/21 across ANE and
GPU, and that split runs **slower than pure ANE (204.4) and slower than pure GPU (178.8)**.
Cross-device handoff costs more than the parallelism gains.

So the default is best on one chip, worst on another, and better than pure placement in a
third case. **There is no safe default. Measure and pin.**

Related anomaly, characterised in `results/anecc-probe.md`: on the M4 Pro, a **cold**
compile of the ANE-rewritten model under `ALL` emits

```
E5RT encountered an STL exception. msg = MILCompilerForANE error: failed to compile
ANE model using ANEF. Error=_ANECompiler : ANECCompile() FAILED.
```

and then runs anyway. It is narrow and deterministic: only that model, only `ALL`, only on
a cold compile, only on M4 Pro. `CPU_AND_NE` and `CPU_AND_GPU` are clean on the same model,
the naive model is clean under every unit, and a warm (already compiled) run is always
clean. **Outputs are unaffected** - `ALL` and `CPU_AND_NE` agree to the last bit - and `ALL`
is in fact the fastest configuration for that model at 240.9 img/s, so the failed ANE
compile falls back silently and costs nothing measurable.

Two practical notes. The message does **not** go to the calling process's stderr, so
redirecting or capturing at the Python level will not see it; it surfaces from a helper
process at teardown, which makes it easy to miss and hard to attribute. And because it only
fires on a cold compile, it disappears as soon as the compiled artifact is cached, which is
why it looks intermittent. Related in spirit to
[coremltools#2758](https://github.com/apple/coremltools/issues/2758), where an
`ANECCompile` failure was likewise not surfaced at load. This is Core ML framework
behaviour rather than a conversion bug, so Feedback Assistant is the right destination.

---

### 4. The CPU's share of its own memory bus nearly doubles between chips

The findings above are about which compute unit to pick. This one is about whether the CPU
is a candidate at all, and it changes by chip the same way.

Start with what you are buying:

| chip | CPU cores | GPU cores | CPU:GPU | bus peak |
| --- | ---: | ---: | ---: | ---: |
| M4 Pro | 14 (10P + 4E) | 20 | **0.70** | 273 GB/s |
| M5 Max | 18 (6 + 12) | 40 | **0.45** | 614 GB/s |

Between the two, CPU cores grow **1.29x**, GPU cores **2.0x**, and the bus **2.25x**. Measured
streaming-read bandwidth per engine (`tools/membw.c`, `tools/gpu_bw.py`):

| chip | 1 core | CPU, all cores | GPU | CPU share of bus | CPU/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| M4 Pro | 89.8 | 249.5 | 253.5 | **91%** | **1.02x** |
| M5 Max | 87.8 | 303.8 | 566.7 | **49%** | **1.87x** |

GPU bandwidth grows **2.24x** against 2.0x the cores and a 2.25x bus — it tracks the hardware
almost exactly. CPU bandwidth grows **1.22x**. The bus was sized for the GPU, and the CPU was
not scaled with it. So on the M4 Pro the CPU is a peer of the GPU for bandwidth-bound work,
and on the M5 Max it is roughly half as good.

**It is not simply core count.** Both chips saturate their CPU-side path long before running
out of cores:

| threads | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 18 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M4 Pro (of 14) | 89.8 | 160.6 | 198.6 | 233.5 | 239.3 | 239.9 | 249.4 | — |
| M5 Max (of 18) | 87.8 | 161.8 | 233.1 | 287.3 | 299.0 | **303.8** | 272.2 | 296.3 |

The M5 Max is done at 8 threads; its remaining ten cores add nothing. Single-core bandwidth
is also essentially identical across the two chips, 89.8 against 87.8, despite a 2.25x
difference in bus width — whatever limits one core is not the DRAM. The CPU's path to memory
tops out somewhere around 250-300 GB/s on both parts, and that ceiling barely moved while the
bus more than doubled.

Two framings, both true: commercially you are buying GPU cores, which is why nobody widened
the CPU's path; mechanically that path has a ceiling that is now less than half the bus on
the larger chip.

#### Running both at once does not add much on either chip

Aligned windows, with the contended rate solved for rather than assumed (`tools/contention.py`):

| chip | threads | CPU alone | GPU alone | together | vs GPU alone |
| --- | ---: | ---: | ---: | ---: | ---: |
| M4 Pro | 10 | 240.5 | 253.7 | 90.4 + 176.6 = 266.9 | **1.05x** |
| M4 Pro | 14 | 250.2 | 253.5 | 147.8 + 113.2 = 261.0 | 1.03x |
| M5 Max | 6 | 300.1 | 554.6 | 130.7 + 444.0 = 574.8 | **1.04x** |
| M5 Max | 18 | 295.6 | 556.5 | 142.4 + 407.0 = 549.4 | 0.99x |

Despite the very different CPU shares, both chips land in the same place: about **4-5%**
above what the GPU achieves alone, and the exchange is close to one-for-one. Adding the CPU
to a GPU-saturated bus is not additive on either part. Over-committing makes it worse — at
full thread count the M4 Pro drops to 1.03x and the M5 Max to 0.99x, below GPU-only.

The practical consequence mirrors finding 1. "The CPU has bandwidth going spare, offload to
it" and "the CPU is useless for bandwidth-bound work" are both chip-specific claims stated
generally. On the M4 Pro the two engines are interchangeable for bandwidth-bound work, which
is scheduling freedom; on the M5 Max the CPU is the strictly worse engine. On neither does
running both add meaningful throughput.

This also bears on CPU/GPU-offload designs ported from discrete-GPU systems, where the
premise is a slow link between two separate memory pools. Neither Apple part has that link,
and on neither does recruiting the CPU add more than about 5%.

#### What the measurement needs to get right

Three methodology points, each of which produced a confident, wrong number first:

- **Align the windows and measure the overlap.** Two processes each timing their own window
  reported an aggregate *above* the hardware peak. `membw.c` emits its timed window as
  `CLOCK_MONOTONIC` timestamps so the contended rate is solved for.
- **Warm up before the first timed case.** The first configuration measured in a process runs
  50-70% slow on GPU clock ramp, larger than the effect being measured.
- **Interleave configurations, and keep the machine idle.** Batching all of A then all of B
  confounds drift with configuration. A background job was observed changing the same
  binary's result by 3x, and an earlier version of the M5 Max contention row in this table
  was measured on a busy machine and reported 1.00x instead of 1.04x. `contention.py` waits
  for the load average to settle and prints `UNRELIABLE` above 20% spread.

## Two smaller results

**The `ml-ane-transformers` rewrite is worth 10-13% here, not more.** Applying the full
treatment - `(B,C,1,S)` layout, `nn.Linear` to 1x1 `nn.Conv2d`, per-head chunked attention,
channel-dim LayerNorm - gives 204.4 -> 231.3 img/s on M4 Pro and 233.0 -> 256.7 on M5 Max.
The reason it is not more: the naive conversion was **already 100% ANE-resident**. A ViT
starts ANE-shaped because its patch embedding is a Conv2d emitting `(B,C,H,W)`. Apple's
larger reported gains were on models that fell off the ANE entirely; there was no such
cliff to recover here. Verified numerically equivalent to the HuggingFace model:
max abs diff 3.8e-06, min cosine 0.99999988.

**The rewrite is unit-specific and makes the GPU slower**: M4 Pro GPU 178.8 -> 149.5,
M5 Max GPU 1085.7 -> 720.1. If you run both units, ship a different variant to each. Doing
that on the M4 Pro measured 408.3 img/s combined, against 381.3 sharing one model.

**Energy** (M5 Max, `powermetrics`, sustained load, batch 16):

| units | img/s | ANE | GPU | CPU | total | energy/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `CPU_AND_NE` | 233.0 | 4.8 W | 0.07 W | 4.0 W | 8.9 W | 38.2 mJ |
| `CPU_AND_GPU` | 962.1 | 0 W | 39.4 W | 4.7 W | 44.1 W | 45.8 mJ |

The ANE is ~1.2x better per image end to end, at 5x lower power. Note that **45% of the
ANE configuration's power is CPU** - Python `predict()` driving overhead - against 11% for
the GPU configuration. On silicon alone the ANE is ~20.6 mJ/image, about 2.2x better. Any
serious ANE deployment needs to eliminate that driving overhead before its efficiency can
be judged. `ANE Power: 0 mW` under `CPU_AND_GPU` independently confirms the compute plan:
placement analysis and the wattmeter agree.

---

## Results from other chips

Measured here so far:

| chip | macOS | fastest on the naive model | default (`ALL`) |
| --- | --- | :---: | --- |
| M4 Pro (Mac16,11), 20 GPU cores | 26.5.1 | ANE | slowest of the three |
| M5 Max (Mac17,7), 40 GPU cores | 26.6 | GPU, by 4.7x | within 2% of the GPU |

Two chips, one generation apart, disagreeing. That is enough to show the ranking
is not a property of the framework, and not enough to say what it *is* a property
of. M1/M2/M3, the base and Max variants of each, and anything with a different
NPU core count would all narrow it down.

`python3 tools/summarise.py results/sweep-<chip>.json` renders any sweep into the
table above. Reports welcome as issues; a `results/sweep-<chip>.json` in a pull
request is even better.

---

## Limitations

Please read these before citing any number above.

- **Two chips.** M4 Pro and M5 Max, one machine each. Nothing here says what M1/M2/M3, base M4, Max/Ultra
  variants, or A-series parts do. The central claim is precisely that results do not
  generalise across chips, which applies to these results too.
- **One model family.** A vision transformer, which is an unusually good ANE fit. CNNs and
  decoder-only LLMs will behave differently, and decoders are known to be much harder.
- **One framework version.** coremltools 9.0, torch 2.8.0, macOS 26.5.1 (M4 Pro) and 26.6
  (M5 Max). Placement heuristics are Core ML internals and can change with any OS update.
- **Python driving overhead is in every throughput number.** It penalises faster
  configurations more, which flatters the ANE at small batch. Batch-16 rows are the more
  trustworthy ones.
- **Power measured on one chip only**, and confounded by the driving overhead as noted.
- **`ALL` mixed-placement conclusions rest on `MLComputePlan`**, a static analysis. It has
  been cross-checked against wall-clock and against ANE wattage, but it is a plan, not a
  trace.
- Single-model, single-process. No multi-model residency, no sustained thermal soak.
- **The bandwidth numbers in finding 4 are streaming reads**, which model weight streaming
  during decode. They are not a claim about mixed read/write, random access, or what any
  real kernel achieves. The GPU figure is the best of four MLX op shapes, so it is a floor
  on what the hardware can do, not a ceiling.
- **Finding 4's CPU figures depend on QoS steering**, which macOS honours as a hint, not a
  guarantee. Core placement was not verified with `powermetrics` on every run.

Spreads across 5 repeats were 0.0-3.5% (median 0.3%), so the differences reported here are
far larger than run-to-run noise. Raw per-run values are in `results/`.

---

## Reproducing

Requires macOS 15+ on Apple silicon and **Python 3.12** - coremltools ships no native
extension for 3.13+, and without it `MLComputePlan` and `MLModel` do not exist.

`./run.sh` does everything below in one step and prints the summary table. The
individual steps, if you want to vary something:

```sh
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# build the two model variants
./.venv/bin/python models/convert_siglip.py --batch 16 --out siglip-vision-b16.mlpackage
./.venv/bin/python models/ane_siglip.py    --batch 16 --out siglip-ane-b16.mlpackage

# where do the ops go?
./.venv/bin/python tools/anecheck.py siglip-vision-b16.mlpackage --compute-units ALL

# throughput, repeated, with spread
./.venv/bin/python tools/sweep.py --models siglip-vision-b16.mlpackage siglip-ane-b16.mlpackage

# both units at once, optionally with a different variant per unit
./.venv/bin/python tools/bench_concurrent.py siglip-vision-b16.mlpackage \
    --model-ane siglip-ane-b16.mlpackage --model-gpu siglip-vision-b16.mlpackage
```

`models/ane_siglip.py` verifies numerical equivalence against the HuggingFace model on
every build and refuses to emit a model that does not match.

Memory bandwidth (finding 4):

```sh
cc -O3 -o tools/membw tools/membw.c

# per-tier CPU sweep; --qos bg steers to the efficiency tier
./tools/membw --threads 10 --gb 16 --secs 3
./tools/membw --threads 10 --gb 16 --secs 3 --qos bg

# GPU, several op shapes
python tools/gpu_bw.py --gb 4 --secs 3

# both at once, aligned windows, interleaved repeats
python tools/contention.py --threads 10 --secs 15 --reps 4
```

Run these on an otherwise idle machine. `contention.py` waits for the load average to settle
and prints `UNRELIABLE` rather than a verdict if run-to-run spread exceeds 20%.

### Tools

| tool | what it does |
| --- | --- |
| `tools/anecheck.py` | per-op compute-device placement from `MLComputePlan`, **cost-weighted**, with `--assert-ane-fraction` as a CI gate |
| `tools/sweep.py` | repeated throughput runs, reports per-run values and spread |
| `tools/bench_concurrent.py` | both units simultaneously, separate processes, optional per-unit model variant |
| `tools/bench_power.py` | sustained-load driver to pair with `powermetrics` |
| `tools/bench_coreml.py` | single-configuration throughput |
| `tools/probe_gpu.py` | builds matched matmul-bound and bandwidth-bound models, to separate matmul-specific hardware from general GPU uplift |
| `tools/membw.c` | CPU streaming-read bandwidth, per core tier via QoS, emitting its timed window as `CLOCK_MONOTONIC` timestamps |
| `tools/gpu_bw.py` | GPU streaming bandwidth via MLX, several op shapes with explicit traffic accounting |
| `tools/contention.py` | both engines in aligned windows, interleaved repeats, solves for the contended rate and refuses to render a verdict above 20% spread |
| `tools/summarise.py` | renders a sweep JSON into the reporting table, stdlib only so it runs outside the virtualenv |

---

## Prior art

`anecheck.py` is not the first tool of its kind and does not claim to be. Existing
compute-plan tooling worth knowing about:

- [`john-rocky/CoreML-LLM`](https://github.com/john-rocky/CoreML-LLM) -
  `conversion/audit_ane_residency.py`, a mature ANE residency auditor. Its nested-block
  traversal is more correct than this repo's first version was, and was the reason that bug
  got fixed here.
- [`pytorch/executorch`](https://github.com/pytorch/executorch) -
  `examples/apple/coreml/scripts/coreml_compute_plan.py`
- [`Anemll/Anemll`](https://github.com/Anemll/Anemll) - `anemll/utils/ane_profiler.py`
- [`freedomtan/coreml_modelc_profling`](https://github.com/freedomtan/coreml_modelc_profling)

What this one adds is **cost weighting** - residency weighted by `MLComputePlan`'s per-op
estimated cost rather than by op count, so one stranded matmul is not hidden by fifty cheap
ANE ops - and a pass/fail threshold for CI.

The ANE model rewrite follows the principles in
[`apple/ml-ane-transformers`](https://github.com/apple/ml-ane-transformers) and the Apple
ML research article *Deploying Transformers on the Apple Neural Engine*.

## Licence

MIT. SigLIP weights are Google's, under their own terms.
