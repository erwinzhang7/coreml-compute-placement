# Compute-unit placement in Core ML is chip-dependent, and the default is not a safe choice

**Erwin Zhang**

---

## Abstract

On Apple silicon a Core ML model can execute on the Neural Engine (ANE), the GPU,
or the CPU. The framework selects for the developer, reports success either way,
and exposes no interface reporting what it chose. We measure five model
architectures across two chip tiers (M4 Pro, M5 Max) and three compute-unit
settings, in both burst and sustained regimes.

Three results. First, **which unit is faster inverts between chips on three of
five architectures** — a vision transformer, a dense-convolution CNN, and a text
encoder — while the two that do not invert still move by 4.5x and 6.6x. The
choice is therefore not a property of the model, and it does not transfer across
a chip generation. Second, **the default `ComputeUnit.ALL` is not a safe
default**: on the M4 Pro it costs between 1.19x and 5.16x against the better
explicit placement, and on three of five architectures it is slower than *both*
explicit placements. On the M5 Max it is free at peak, which we report as
prominently. Third, **peak throughput does not predict sustained throughput**:
over a two-minute soak the ANE holds 0.999–1.000 of its peak on both chips while
the GPU holds 0.837 and the default 0.721 on the M5 Max. A benchmark of a few
seconds — the standard form — cannot see this.

We quantify measurement reproducibility explicitly, using two physically
identical M4 Pro machines and five repeats, and report a within-machine range of
0.011 and a between-machine difference of 0.019 on the noisiest metric. The
cross-chip effects we report exceed both by an order of magnitude.

All measurement code, model conversion code, and raw per-run JSON are in this
repository, and a single command reproduces the tables on any Apple silicon Mac.

---

## 1. Introduction

Core ML's `MLComputeUnits` is an allow-list, not an instruction. `.all` permits
the runtime to use any engine; `.cpuAndNeuralEngine` and `.cpuAndGPU` restrict
it. In all cases the runtime decides per-operation, and the resulting placement
is not reported through any public API. `MLComputePlan` exposes a *planned*
device per operation, which is a static analysis rather than a trace.

The practical question a developer faces is therefore: *leave it on the default,
or pick?* The folklore answer is that the default is fine, and that the ANE is
the right target for the kinds of models it was built for. This paper measures
both claims and finds the first false on one of two chips tested, and the second
false on the architecture most often cited as the ANE's best case.

The question has become more pointed, not less, with Core AI (WWDC 2026), which
retains placement control as `SpecializationOptions(preferredComputeUnitKind:)`
— a single *preferred* unit rather than an allow-list — and whose published
reference recipes select that unit from **model structure alone, with no chip
term**. See [CORE-AI.md](CORE-AI.md).

### Contributions

1. A cross-chip, cross-architecture placement measurement (5 architectures x 2
   chips x 3 settings) showing the optimal unit inverts.
2. A quantification of what the default costs, including cases where it is worse
   than every explicit alternative.
3. A sustained-throughput measurement showing peak and sustained rank the units
   differently, with a thermal-attribution method that does not require root.
4. An explicit reproducibility floor from physically identical hardware, which
   most single-machine benchmark papers do not establish.
5. Open tooling: conversion, sweep, soak, and summarisation, reproducible with
   one command.

---

## 2. Method

### 2.1 Hardware

| label | chip | model id | GPU cores | NPU cores | chassis | power | macOS |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| M4 Pro #1 | Apple M4 Pro | Mac16,11 | 20 | 16 | Mac mini | mains | 26.5.1 |
| M4 Pro #2 | Apple M4 Pro | Mac16,11 | 20 | 16 | Mac mini | mains | 26.5.1 |
| M5 Max | Apple M5 Max | Mac17,7 | 40 | 16 | MacBook Pro | mains | 26.6 |

The two M4 Pro machines are physically distinct units of identical
configuration. They exist in this study specifically to measure how much a
result moves for reasons that are not the chip (§3.5).

The M5 Max runs a **non-stock fan curve**, with the die held near 97 °C under
sustained GPU load. This is recorded because it matters for §3.3 and for §4.

### 2.2 Software

coremltools 9.0, PyTorch 2.8.0, transformers 5.16.1, Python 3.12. Models are
converted to `mlprogram` with `compute_precision=FLOAT16` and
`minimum_deployment_target=macOS15`.

### 2.3 Models

Five architectures, selected for structural difference rather than popularity,
all obtained through `transformers` so that reproducing this adds no dependency
beyond what the harness already requires.

| name | architecture | source | input | shape (batch 16) |
| --- | --- | --- | --- | --- |
| `siglip` | vision transformer | `google/siglip-base-patch16-224` | image | (16, 3, 224, 224) |
| `resnet50` | dense convolution | `microsoft/resnet-50` | image | (16, 3, 224, 224) |
| `mobilenet` | depthwise-separable conv | `google/mobilenet_v2_1.0_224` | image | (16, 3, 224, 224) |
| `bert` | text encoder | `distilbert-base-uncased` | int32 tokens | (16, 128) |
| `whisper` | audio encoder | `openai/whisper-tiny` | log-mel | (16, 80, 3000) |

Each is wrapped to emit one embedding vector per item, so all five are
benchmarked as feature extractors with comparable output structure.

### 2.4 Conversion correctness

A model that converts incorrectly still produces a throughput number. Every
converted model is therefore checked against a PyTorch fp32 reference **on the
Core ML model actually benchmarked**, not against a second PyTorch module, and
conversion is refused on mismatch.

Cosine similarity is the acceptance gate rather than absolute or relative error.
The outputs of these five models differ in scale by orders of magnitude, so no
single `atol` is meaningful across the set; and relative error is degenerate for
post-ReLU activations, where reference values approach zero. ResNet-50 converts
correctly with a maximum *relative* error of 4.6 x 10², which is why relative
error is reported but not gated on.

| model | max abs err | max rel err | **min cosine** |
| --- | ---: | ---: | ---: |
| resnet50 | 3.4e-02 | 4.6e+02 | 0.999945 |
| mobilenet | 4.7e-02 | 8.9e+02 | 0.999979 |
| bert | 6.2e-03 | 1.3e+01 | 0.999990 |
| whisper | 1.1e-02 | 1.4e+01 | 0.999998 |

### 2.5 Burst protocol

`tools/sweep.py`. For each (model, compute unit): warm-up calls, then a timed
block of *iters* inference calls, repeated *repeats* times. We report the median
across repeats and the spread (max−min)/median. Batch 16 throughout.
`repeats=5, iters=30` for `siglip`; `repeats=3, iters=20` for the four added
architectures.

Python call overhead is included in every figure. It is a fixed per-call cost, so
it penalises faster configurations proportionally more, which **flatters the
slower unit**. All differences reported below are therefore conservative with
respect to the faster unit.

### 2.6 Sustained protocol

`tools/thermal_soak.py`. One model, one compute unit, driven continuously for
120 s, with throughput bucketed into 10 s windows. The reported statistic is

    sustained fraction = (last window) / (best window)

1.000 means the unit held its peak for the whole soak. This statistic **falls
with soak length by construction**, since the denominator is a maximum over more
windows and the numerator is fixed at the end; only equal-duration soaks are
comparable, and duration is reported with every figure.

Thermal pressure is sampled per window through the `notify(3)` name
`com.apple.system.thermalpressurelevel`, which requires no elevated privilege.
`powermetrics` would give die temperature but requires root, which would make the
measurement unreproducible for a reader; we accept the weaker instrument in
exchange for reproducibility and state its limits in §4.

Power source and Low Power Mode are recorded at start and end of every soak. A
soak whose power source changes mid-run is discarded; one such run occurred
during this study and was discarded rather than reported.

---

## 3. Results

### 3.1 The faster unit inverts between chips

ANE throughput divided by GPU throughput, batch 16. Values above 1 mean the ANE
is faster.

| model | M4 Pro | M5 Max | outcome |
| --- | ---: | ---: | --- |
| `siglip` | **1.143** | 0.215 | **inverts** |
| `resnet50` | **1.409** | 0.168 | **inverts** |
| `bert` | **1.346** | 0.132 | **inverts** |
| `mobilenet` | 0.842 | 0.185 | same winner, ratio moves 4.5x |
| `whisper` | 0.388 | 0.059 | same winner, ratio moves 6.6x |

**Three of five architectures flip which unit is faster between an M4 Pro and an
M5 Max.** The two that do not flip still move by factors of 4.5 and 6.6, so even
where the ranking survives, the margin does not.

This is not a property of vision transformers. A dense-convolution CNN
(`resnet50`) and a text encoder (`bert`) both invert. Nor is it a property of
image models specifically: `bert` takes integer token input and inverts;
`whisper` takes log-mel audio and does not.

`mobilenet` is the result that most constrains ANE folklore.
Depthwise-separable convolution is the workload the Neural Engine is most often
described as ideal for, and **the GPU is faster on both chips** — by 1.19x on the
M4 Pro and 5.40x on the M5 Max.

Underlying medians (img/s):

| model | M4 Pro ANE | M4 Pro GPU | M5 Max ANE | M5 Max GPU |
| --- | ---: | ---: | ---: | ---: |
| `siglip` | 204.4 | 178.8 | 233.0 | 1085.7 |
| `resnet50` | 928.6 | 658.8 | 540.5 | 3225.8 |
| `mobilenet` | 3064.4 | 3641.3 | 1897.7 | 10249.4 |
| `bert` | 755.9 | 561.4 | 454.6 | 3439.5 |
| `whisper` | 56.3 | 145.1 | 34.2 | 577.9 |

**ANE throughput is far less chip-sensitive than GPU throughput.** Between M4 Pro
and M5 Max the ANE moves by 0.58x–1.14x depending on model — and in **four of
five cases it gets *slower* on the newer, larger chip** — while the GPU moves by
2.8x–6.1x in every case. Both chips have 16 NPU cores; the GPU count doubles from
20 to 40.

That the ANE regresses on four of five architectures across a chip generation is
itself notable, and we do not have an explanation for it. Identical NPU core
counts would predict parity, not a 0.58x–0.62x regression on the four
non-`siglip` models. Possible causes include differences in the compiler shipped
with the two OS versions (26.5.1 vs 26.6), memory-subsystem contention, or
per-generation ANE microarchitecture changes not reflected in core count. None of
these was tested, and the two machines differ in OS version as well as chip, so
this observation is confounded and is reported as an observation only.

### 3.2 The default is not a safe default

Cost of `ComputeUnit.ALL` relative to the better of the two explicit placements:

| model | M4 Pro | M5 Max |
| --- | ---: | ---: |
| `siglip` | 1.19x | 1.02x |
| `mobilenet` | 1.19x | 1.00x |
| `bert` | 1.32x | 1.00x |
| `resnet50` | **2.41x** | 1.00x |
| `whisper` | **5.16x** | 1.01x |

On the M5 Max the default is effectively free: it tracks the GPU to within 2% on
every architecture, because the GPU is the right answer on that chip and the
runtime selects it. **We report this as prominently as the negative result.**

On the M4 Pro the default costs 1.19x to 5.16x. More seriously, on three of five
architectures it is slower than **both** explicit placements, not merely slower
than the better one:

| model | ANE | GPU | `ALL` | |
| --- | ---: | ---: | ---: | --- |
| `siglip` | 204.4 | 178.8 | **172.2** | below both |
| `resnet50` | 928.6 | 658.8 | **384.6** | below both |
| `whisper` | 56.3 | 145.1 | **28.1** | below both |

The `whisper` case is the strongest form: the default returns 28.1 img/s where
the ANE alone returns 56.3 and the GPU alone returns 145.1. **The default is
half the throughput of the slower of the two engines it is choosing between.**
This is consistent with `ALL` splitting a graph across engines and paying
transfer and synchronisation costs that exceed the benefit, but we did not
instrument the partition and do not claim the mechanism.

The correct statement is therefore not "the default is sometimes worst" but:
**the default is chip-dependent in the same way placement is** — free on one
chip, up to 5.16x on the other.

### 3.3 Peak does not predict sustained

Sustained fraction over 120 s, `siglip`, batch 16, all machines on mains:

| compute unit | M4 Pro #1 | M4 Pro #2 | M5 Max |
| --- | ---: | ---: | ---: |
| ANE | 1.000 | 1.000 | 0.999 |
| GPU | 0.973 | 0.949 | **0.837** |
| `ALL` | 0.998 | 1.000 | **0.721** |

**The ANE holds its rate on both chips. On the M5 Max the GPU gives up 16% and
the default gives up 28%.** The default is the *worst* sustainer on the chip
where §3.2 found it free at peak — so a burst benchmark and a sustained
benchmark disagree about it, and only the sustained one reflects a long-running
job.

The ANE result is not specific to `siglip`. **All five architectures**, soaked on
M4 Pro #2 under identical conditions, 120 s each:

| model | ANE | GPU | `ALL` |
| --- | ---: | ---: | ---: |
| `siglip` | 1.000 | 0.949 | 1.000 |
| `resnet50` | 0.999 | 0.964 | 0.994 |
| `mobilenet` | 1.000 | 0.968 | 1.000 |
| `bert` | 1.000 | 0.954 | 0.998 |
| `whisper` | 1.000 | 0.937 | 0.977 |
| **range** | **0.999–1.000** | **0.937–0.968** | 0.977–1.000 |

**The separation is complete: every ANE figure is at or above 0.999, and every
GPU figure is at or below 0.968.** The two ranges do not overlap, across five
architectures spanning vision transformer, dense CNN, depthwise CNN, text encoder
and audio encoder. Adding the M5 Max, where the ANE holds 0.999 and the GPU 0.837,
the ANE is at 0.999–1.000 in all six measured (chip, architecture) cells and the
GPU has never once matched it.

The GPU loses something in every single case — 3.2%–6.3% on the M4 Pro, 16.3% on
the M5 Max. The default's loss ranges from nothing to 28% depending on chip and
architecture.

This has a practical consequence the burst tables alone cannot support: **for a
continuously loaded service the ANE's advertised rate is the rate you get, and no
other unit's is.** A capacity plan built from burst numbers will over-provision
the GPU's contribution by 3–16% and the default's by up to 28%.

We did not measure beyond 120 s, so nothing here says where the GPU's decline
stops. On the M4 Pro it had flattened by the end of the soak; on the M5 Max the
`ALL` curve was still descending in the final window (§3.3, first table), so 0.721
is an upper bound on what a longer run would report, not a floor.

The GPU's advantage therefore shrinks with the duration of the measurement:

| | GPU / ANE, M5 Max |
| --- | ---: |
| 30-iteration burst (§3.1) | 4.66x |
| first 10 s window of the soak | 4.38x |
| last 10 s window of the soak | **3.67x** |

The burst figure exceeds even the first soak window because the decline begins
within the first seconds. On the M4 Pro, where there is little decline, burst
and soak agree to within 0.1% (204.4 vs 204.5 ANE; 178.8 vs 178.8 GPU).

**Thermal pressure remained `nominal` in every window of every soak reported
here, and this does not establish that heat was absent.** Thermal pressure is the
operating system's signal that it is about to shed user work, not a die
temperature; the M5 Max was held near 97 °C by an aggressive fan curve and still
reported nominal throughout. The reading is informative in one direction only:
above nominal is evidence of thermal trouble, nominal is not evidence of its
absence.

Two observations argue the decline is a per-engine power limit rather than
enclosure cooling. The ANE holds 0.999 **on the same machine, in the same
session, at the same die temperature** as the GPU run that lost 16%; a cooling
limit that affects one engine and not the other is not a cooling limit. And the
M5 Max GPU is measured at 39.2 TFLOPS against the M4 Pro's 6.4 (§3.4), so an
engine performing 4.7x the work per second draws proportionally more power and
approaches a ceiling the smaller GPU never reaches. We regard the direction as
established and the exact magnitude as chassis-sensitive (§4).

### 3.4 Mechanism: Apple moved matrix acceleration into the GPU

The inversion in §3.1 is not an accident of these particular models. It is the
predictable consequence of a documented architectural change between the two chip
generations, and the measurements below are the signature that change should
produce.

**What the vendor documents.** The M5 generation places a matrix-multiply
accelerator inside every GPU core. Apple describes the M5 GPU as having "a
dedicated Neural Accelerator in each core, enabling GPU-based AI workloads to run
dramatically faster" and claims "over 4x peak GPU compute compared to M4"
[[1]](#ref1). The M5 Max carries "Neural Accelerators built into each core"
across an "up-to-40-core GPU" and is claimed "3.9x faster than M4 Max" on AI
workloads [[2]](#ref2). The M4 generation has no equivalent: its added machine
learning hardware is described as "enhanced machine learning (ML) accelerators in
the CPUs" [[3]](#ref3), not in the GPU cores.

**Meanwhile the Neural Engine did not change in width.** Both generations ship a
16-core Neural Engine [[1]](#ref1)[[3]](#ref3). The M5's is described as
"improved", but it is not widened, and no multiple is claimed for it in the way
one is claimed for the GPU.

| | M4 Pro | M5 Max | ratio |
| --- | ---: | ---: | ---: |
| GPU cores | 20 [[3]](#ref3) | 40 [[2]](#ref2) | 2.0x |
| per-GPU-core matrix accelerator | none | yes [[1]](#ref1)[[2]](#ref2) | — |
| Neural Engine cores | 16 | 16 | 1.0x |
| unified memory bandwidth | 273 GB/s [[3]](#ref3) | 614 GB/s [[2]](#ref2) | 2.25x |

**So the two engines received very different investments across this generation:
the GPU gained both 2x the cores and a new per-core matrix unit, while the Neural
Engine kept its width.** A model whose cost is dominated by matrix multiplication
should therefore move much further on the GPU than on the ANE — which is exactly
what §3.1 measures, and it is why the ANE can be the faster unit on an M4 Pro and
the slower unit on an M5 Max for the same model.

**The measurement matches that signature.** Two synthetic Core ML models of
comparable runtime, one arithmetic-bound and one bandwidth-bound
(`tools/probe_gpu.py`):

Two synthetic Core ML models of comparable runtime, one arithmetic-bound and one
bandwidth-bound (`tools/probe_gpu.py`):

| probe | M5 Max | M4 Pro | ratio |
| --- | ---: | ---: | ---: |
| matmul, 103.1 GFLOP/call | 2.63 ms (**39.2 TFLOPS**) | 16.21 ms (**6.4 TFLOPS**) | **6.17x** |
| elementwise, 1610 MB/call | 13.43 ms (120 GB/s) | 17.04 ms (95 GB/s) | 1.27x |

**GPU arithmetic throughput moves 6.17x for a 2.0x core-count increase.** Core
count alone cannot produce that; the surplus is the per-core matrix accelerator.
Bandwidth-bound work moves only 1.27x on the same pair of chips, so the gain is
specific to arithmetic and not a general uplift.

Two independent cross-checks on that 6.17x. Apple claims "over 4x peak GPU
compute" for M5 against M4 at equal core count [[1]](#ref1); doubling the cores
on top of that predicts something approaching 8x, and 6.17x sits below it, which
is the expected direction once clocks and scaling losses are included. Apple
separately claims the M5 Max is 3.9x faster than the M4 **Max** on AI
[[2]](#ref2); since the M4 Max has roughly twice the GPU of the M4 **Pro** used
here, a figure well above 3.9x against an M4 Pro is what that implies.

Every cross-chip GPU gain in §3.1 falls **strictly between the two probe bounds**
— 2.81x for `mobilenet`, 6.13x for `bert`, against 1.27x bandwidth-bound and
6.17x arithmetic-bound. That is the expected ordering if each model sits somewhere
on the spectrum between bandwidth- and compute-bound: the most arithmetic-dense
model approaches the matmul ceiling and none exceeds it.

**Two honest limits on this argument.** First, the elementwise probe reaches only
95 GB/s of the M4 Pro's 273 GB/s and 120 GB/s of the M5 Max's 614 GB/s, so it is
not bandwidth-saturating; its 1.27x is a floor on the achievable-bandwidth ratio,
not the hardware's 2.25x. The conclusion survives — even the generous 2.25x
hardware bandwidth ratio is far below the 6.17x arithmetic ratio — but the probe
should not be read as a bandwidth benchmark. Second, we did not perform a
per-model roofline analysis, so we do not claim to know where each of the five
architectures sits on that spectrum; we claim only that all five land inside the
bounds, which is what the mechanism predicts.

**A falsifiable prediction.** If the mechanism is the per-GPU-core matrix
accelerator, then the inversion should appear on **any** M5-family part and on no
M4-or-earlier part, independent of tier. A base M5, an M5 Pro or an M5 Ultra
should rank the units as the M5 Max does here; an M4 Max — which has twice the
M4 Pro's GPU but still no per-core matrix unit — should rank them as the M4 Pro
does, and should *not* invert merely because it has more GPU cores. We do not have
those machines. This is the cheapest experiment that could falsify the
explanation, and we invite it.

The ANE's much smaller movement is consistent with both chips carrying the same
16-core Neural Engine, though it does not explain the regression noted in §3.1.

### 3.5 Measurement reproducibility

Most single-machine benchmark results do not establish how much they move for
reasons other than the variable under study. Two physically identical M4 Pro
machines and five repeats of the noisiest metric give that floor directly.

Sustained fraction, GPU, `siglip`, 120 s:

| quantity | value |
| --- | ---: |
| M4 Pro #2, five repeats | 0.949, 0.950, 0.951, 0.957, 0.960 (mean 0.954, sd 0.0045) |
| within-machine range | **0.011** |
| between-machine difference (#1 at 0.973) | **0.019** (1.7x the within-machine range) |
| M4 Pro mean 0.963 → M5 Max 0.837 | **0.126** (11x within-machine, 7x between-machine) |

Two conclusions. Identical hardware is **not** identical: the between-machine
difference is 1.7x the run-to-run range, so a single sustained GPU figure should
not be quoted to three decimals. And the cross-chip effect exceeds both sources
of variation by roughly an order of magnitude, so it is not machine variation.

Burst measurements are far more repeatable than sustained ones. Spreads across
repeats were 0.0–1.2% on the M4 Pro for every model and unit. On the M5 Max they
were 0.3–3.7% except for two ANE cells — `mobilenet` at 17.2% and `bert` at
12.3% — which are the least reliable numbers in this paper and are flagged as
such in §4.

---

## 4. Threats to validity

**Two chips.** M4 Pro and M5 Max. Nothing here establishes behaviour on M1/M2/M3,
base or Ultra variants, or A-series parts. The central claim is precisely that
results do not generalise across chips, which applies to these results too.

**Chassis is confounded with chip for the sustained results.** The M4 Pro
machines are desktops with stock cooling; the M5 Max is a laptop with a
non-stock fan curve. §3.3 gives two arguments that the sustained differences are
per-engine power rather than enclosure, and we consider the direction
established — but the specific value 0.837 belongs to this machine with this fan
curve. Separating them requires an M5-family desktop or an M4 Pro laptop,
neither of which was available. This does **not** affect §3.1 or §3.2, which are
within-machine ratios at burst.

**Small repeat counts on the added architectures.** `n=3` for the four models
added in §3.1, against `n=5` for `siglip`. This is adequate for effects of
4x–17x and thin for the two high-variance M5 Max ANE cells noted in §3.5, where
the reported medians (`mobilenet` 1897.7, `bert` 454.6) should be treated as
approximate. No conclusion in this paper depends on those two cells: both are
cases where the GPU wins by more than 5x.

**One batch size.** Batch 16 throughout. Small-batch behaviour is known to be
dominated by per-call overhead and is not measured here.

**Python driving overhead is in every throughput figure.** It penalises faster
configurations proportionally more, so it is conservative with respect to every
gap reported.

**Thermal pressure is a weak instrument.** It never left nominal, which as
discussed rules nothing out. Die temperature was not sampled during the soaks.
The *cause* of the M5 Max GPU and `ALL` decline is therefore not established:
power limiting, clock ramp-down after a boost window, and thermal limiting at
97 °C are all consistent with what was measured and are not separated. What is
established is the operational fact that the throughput declines and the ANE's
does not.

**No decoder-only LLM.** The five architectures cover vision transformer, dense
CNN, depthwise CNN, text encoder and audio encoder, but not autoregressive
decoding with a KV cache — which is both the hardest case and the structure
Apple's own Core AI recipes route to the ANE. This is the most significant
remaining gap.

**`ALL` mechanism not instrumented.** §3.2 shows the default can be slower than
both explicit placements but does not demonstrate *why*. Graph partitioning with
inter-engine transfer is the natural hypothesis and is untested here.

**One framework version.** coremltools 9.0, macOS 26.5.1 and 26.6. Placement
heuristics are runtime internals and may change with any OS update.

---

## 5. Related work

**Placement inspection tooling.** Several tools read `MLComputePlan` to report
where operations are planned to run: `john-rocky/CoreML-LLM`
(`conversion/audit_ane_residency.py`), `pytorch/executorch`
(`examples/apple/coreml/scripts/coreml_compute_plan.py`), `Anemll/Anemll`
(`anemll/utils/ane_profiler.py`), and `freedomtan/coreml_modelc_profling`. These
answer *where is this planned to run*. They do not measure what that placement
costs, and a plan is not a trace. The tooling accompanying this paper adds
cost-weighted residency — residency weighted by `MLComputePlan`'s per-operation
estimated cost rather than by operation count, so one stranded matmul is not
concealed by fifty cheap ANE operations — and a pass/fail threshold usable in CI.

**ANE-oriented model rewriting.** `apple/ml-ane-transformers` and the associated
Apple ML research article *Deploying Transformers on the Apple Neural Engine*
[[4]](#ref4) establish the (B, C, 1, S) layout, `Linear`→`Conv2d` substitution and
chunked attention used by the ANE-rewritten variant in this repository. That line of work
optimises a model *for* the ANE and reports the resulting speedup. It does not
address whether the ANE is the correct target on a given chip, which is the
question here — and §3.1 finds that on an M5 Max it frequently is not. We note in
passing that on the M5 Max the ANE rewrite is actively harmful to GPU throughput
(720.1 img/s against the unmodified model's 1085.7), so a rewrite undertaken for
the ANE forecloses the faster option on that chip.

**Vendor guidance.** Apple's Core AI reference recipes select a compute unit from
model structure, discussed in §6. To our knowledge no published work measures
whether the ANE/GPU ranking is stable across chip generations for a fixed model,
which is the gap this paper addresses.

**What is not comparable.** Published MLPerf-style inference results on Apple
silicon generally report a single configuration and do not vary the compute-unit
setting, so they cannot be used to check §3.1 or §3.2.

---

## 6. Relationship to Core AI

Core AI (WWDC 2026) [[5]](#ref5) sits above Core ML and does not remove placement
control: it becomes `SpecializationOptions(preferredComputeUnitKind:)`, taking a single
*preferred* unit (`.cpu`, `.gpu`, `.neuralEngine`) rather than an allow-list, and
moves to Swift — the Python conversion package has no placement surface at all.
As with `ALL`, it is a preference, and nothing reports what was honoured.

Apple's published reference recipes (`apple/coreai-models`,
`swift/Sources/CoreAIShared/Runtime/ModelStructure.swift`) select the preferred
unit from **model structure with no chip term**: static-shape and segmenter
models are routed to `.neuralEngine`, dynamic-shape models to `.gpu`. §3.1 finds
the optimal unit inverting between chips for a fixed model. These are in tension,
and resolving it is future work: we have not measured whether the Core AI runtime
honours the preference, nor whether the inversion holds for the static-shape
decoder structure those recipes route to the ANE. Details in
[CORE-AI.md](CORE-AI.md).

---

## 7. Reproducibility

```sh
./run.sh --soak
```

builds the models, sweeps the three compute-unit settings, runs the sustained
soaks, and prints both tables ready to paste. Raw per-run JSON for every figure
in this paper is under `results/`; the sustained runs and their conditions are in
`results/soak/`.

The summariser refuses to let an invalid soak read as a result: it flags runs
that were on battery, that changed power source mid-run, or that were
interrupted. `models/convert_zoo.py` refuses to save a model that fails its
numerical check.

If your chip ranks the units differently from either of ours, that is the result
we most want to see.

---

## Data availability

Every number in this paper is derived from JSON committed in this repository.
`results/sweep-*.json` and `results/zoo-*.json` hold burst runs with per-repeat
values; `results/soak/*.json` hold per-window sustained runs with the power and
thermal state of each.

Vendor figures in §3.4 are quoted from Apple press material and are **claims, not
measurements**: they are used to establish what changed architecturally between
the two generations, not as evidence of performance. Every performance number
attributed to this study was measured on the hardware described in §2.1.

---

## References

<a id="ref1"></a>[1] Apple. *Apple unleashes M5, the next big leap in AI
performance for Apple silicon.* Apple Newsroom, 15 October 2025.
<https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/>
— "The 10-core GPU features a dedicated Neural Accelerator in each core, enabling
GPU-based AI workloads to run dramatically faster"; "over 4x peak GPU compute
compared to M4"; "an improved 16-core Neural Engine".

<a id="ref2"></a>[2] Apple. *Apple introduces new Mac Studio with M5 Max and M5
Ultra.* Apple Newsroom, 25 August 2026.
<https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/>
— "up-to-40-core GPU"; "Neural Accelerators built into each core"; "up to 614GB/s
of unified memory bandwidth"; "3.9x faster than M4 Max".

<a id="ref3"></a>[3] Apple. *Apple introduces M4 Pro and M4 Max.* Apple Newsroom,
30 October 2024.
<https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/>
— M4 Pro: up to 20-core GPU, 273GB/s unified memory bandwidth; "enhanced machine
learning (ML) accelerators in the CPUs".

<a id="ref4"></a>[4] Apple Machine Learning Research. *Deploying Transformers on
the Apple Neural Engine.* 2022.
<https://machinelearning.apple.com/research/neural-engine-transformers> — the
(B, C, 1, S) layout, `Linear`→`Conv2d` substitution and chunked attention used by
the ANE-rewritten variant in this repository, and the reference implementation
`apple/ml-ane-transformers`.

<a id="ref5"></a>[5] Apple. *Core AI.* Apple Developer Documentation, 2026.
<https://developer.apple.com/documentation/coreai> — and the reference recipes at
<https://github.com/apple/coreai-models>, whose
`swift/Sources/CoreAIShared/Runtime/ModelStructure.swift` selects
`preferredComputeUnitKind` from model structure.
