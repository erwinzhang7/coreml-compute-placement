# Compute-unit placement in Core ML does not transfer between configurations, and the default is not a safe choice

**Erwin Zhang**

---

## Abstract

On Apple silicon a Core ML model can execute on the Neural Engine (ANE), the GPU,
or the CPU. The framework selects for the developer, reports success either way,
and exposes no interface reporting what it chose. We measure five model
architectures across two chip tiers (M4 Pro, M5 Max) and three compute-unit
settings, in both burst and sustained regimes.

Three results. First, **which unit is faster inverts between chips on three of
five architectures**: a vision transformer, a dense-convolution CNN, and a text
encoder. The two that do not invert still move by 2.4x and 3.6x. The
choice is therefore not a property of the model, and it does not transfer across
a chip generation. Second, **the default `ComputeUnit.ALL` is not a safe
default**: on the M4 Pro it costs between 1.18x and 5.18x against the better
explicit placement, and on three of five architectures it is slower than *both*
explicit placements. On the M5 Max it is close to free at peak, costing 1.01x to
1.09x, which we report as prominently. Third, **peak throughput does not predict sustained throughput**:
over a two-minute soak the ANE gives back at most 0.14% of its peak in 41 of 42
soaks on both chips and four separate machines, while the GPU gives back 1.1 to
13.2% on an M4 Pro and 4.8 to 16.3% on an M5 Max across 60. Those 41 ANE soaks all
sustain better than the best of the 60 GPU soaks. The forty-second is a
`whisper` run that gave back 2.52% after reaching the same peak as seven runs that
did not, one of them a repeat on the same machine, which we report and do not
explain in §3.3. A benchmark of a few seconds, which is the standard form, cannot
see any of this.

We also report that the sustained *fraction* is confounded by the thermal state a
run starts in on a throttling chip, which makes the ANE's insensitivity to that
confound the most robust result in this section.

We quantify measurement reproducibility explicitly, using two physically
identical M4 Pro machines and five repeats, and report a within-machine range of
0.011 and a between-machine difference of 0.019 on the noisiest metric. **The
cross-chip sustained fraction does not reliably clear that floor**: repeated, the
gap against the M4 Pro mean runs from 0.129 down to 0.023, and the low end is
comparable to the between-machine difference itself (§3.6). What does clear it by
a wide margin is the absolute rate — 172 to 177 img/s for the M4 Pro GPU against
762 to 888 for the M5 Max — and the burst ratios of §3.1 and §3.2.

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
retains placement control as `SpecializationOptions(preferredComputeUnitKind:)`,
a single *preferred* unit rather than an allow-list, and whose published
reference recipes select that unit from **model structure alone, with no chip
term**. See [CORE-AI.md](CORE-AI.md).

### Contributions

1. A cross-chip, cross-architecture placement measurement (5 architectures x 2
   chips x 3 settings) showing the optimal unit inverts.
2. A quantification of what the default costs, including cases where it is worse
   than every explicit alternative.
3. A sustained-throughput measurement showing the peak/sustained gap is large
   enough to change a capacity plan, with a thermal-attribution method that does
   not require root. An earlier version of this list claimed peak and sustained
   *rank* the units differently; §3.3 withdraws that, because with the tool fixed
   the ordering reverses depending on which pair of runs is compared.
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
| M4 Pro #3 | Apple M4 Pro | Mac16,11 | 20 | 16 | Mac mini | mains | 26.5.1 |
| M5 Max | Apple M5 Max | Mac17,7 | 40 | 16 | MacBook Pro | mains | 26.6 |

**Four machines.** The three M4 Pro machines are physically distinct units of
identical configuration; they appear as `experiments`, `inference1` and
`inference2` in the raw filenames and as box 1, box 2 and box 3 in §3.3. Their
identity is measured rather than assumed: each reports `Apple M4 Pro`,
`Mac16,11`, 14 CPU cores, 20 GPU cores and macOS 26.5.1.

The five-repeat reproducibility floor in §3.6 uses two of the three; the soak
corpus in §3.3 and the cold/warm study use all three. They exist in this study specifically to measure how much a
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

A model that converts incorrectly still produces a throughput number. The four
models built by `models/convert_zoo.py` are therefore checked against a PyTorch
fp32 reference **on the Core ML model actually benchmarked**, not against a
second PyTorch module, and conversion is refused on mismatch.

**`siglip` is not gated, and it is the model most of this paper rests on.**
`models/convert_siglip.py` traces, converts and saves; it performs no reference
comparison and can refuse nothing. `results/conversion-check.json` accordingly
carries four models, not five. We report the gate as covering what it covers.

Cosine similarity is the acceptance gate rather than absolute or relative error.
The outputs of these five models differ in scale by orders of magnitude, so no
single `atol` is meaningful across the set; and relative error is degenerate for
post-ReLU activations, where reference values approach zero. ResNet-50 converts
correctly with a maximum *relative* error of 390, which is why relative
error is reported but not gated on.

| model | max abs err | max rel err | **min cosine** |
| --- | ---: | ---: | ---: |
| resnet50 | 2.9e-02 | 3.9e+02 | 0.999953 |
| mobilenet | 4.7e-02 | 1.0e+03 | 0.999977 |
| bert | 4.1e-03 | 3.6e+00 | 0.999999 |
| whisper | 2.7e-03 | 2.3e+00 | 1.000000 |

### 2.5 Burst protocol

`tools/sweep.py`. For each (model, compute unit): warm-up calls, then a timed
block of *iters* inference calls, repeated *repeats* times. We report the median
across repeats and the spread (max - min) / median. Batch 16 throughout.
The two chips were not run at the same settings, and we state it rather than
leave it implicit: on the **M4 Pro**, `repeats=5, iters=30` for `siglip` and
`repeats=3, iters=20` for the four added architectures; on the **M5 Max**,
`repeats=5, iters=30` for all five. The M5 Max side is therefore the better
sampled of the two, which matters for §3.6's spread comparisons and not for the
medians themselves.

Python call overhead is included in every figure. It is a fixed per-call cost, so
it penalises faster configurations proportionally more, which **flatters the
slower unit**. All differences reported below are therefore conservative with
respect to the faster unit.

### 2.6 Sustained protocol

`tools/thermal_soak.py`. One model, one compute unit, driven continuously, with
throughput bucketed into fixed windows. **Two protocols are used and the results
are not pooled:** 120 s in 10 s windows for the survey of §3.3, and 600 s in 20 s
windows for the duration, cold-start and ANE-control sub-studies. The 120 s set
is pinned in `results/soak/PAPER-SET.txt`; the 600 s runs are excluded from it, so
no count in §3.3 mixes the two. The reported statistic is

    sustained fraction = (last window) / (best window)

1.000 means the unit held its peak for the whole soak. **Only equal-duration
soaks are comparable**, and duration is reported with every figure.

An earlier version of this paragraph justified that with "the statistic falls
with soak length by construction, since the denominator is a maximum over more
windows and the numerator is fixed at the end". The first half is right and the
second is not: the numerator is the *last* window, and a longer run ends at a
different window. Taking every 600 s soak in `results/soak` and reading it both
ways — the fraction it would have reported at 120 s against the one it reports at
600 s — **23 of 63 rise with length and 40 fall**, and the split is by *unit*:

| unit | rises with length | falls |
| --- | ---: | ---: |
| GPU | **20** | 16 |
| ANE | 3 | **24** |

The GPU rises more often than not, by as much as +0.031, because it dips early
and recovers (§3.5), so a longer run ends further into the recovery and the
numerator climbs faster than the maximum does. The ANE has no dip to recover
from, so its small decline simply accumulates and the fraction falls. Length
biases the denominator upward and nothing more; the direction of the net effect
belongs to the unit, not to the arithmetic.

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
| `siglip` | **1.143** | 0.216 | **inverts** |
| `resnet50` | **1.410** | 0.334 | **inverts** |
| `bert` | **1.358** | 0.234 | **inverts** |
| `mobilenet` | 0.843 | 0.346 | same winner, ratio moves 2.4x |
| `whisper` | 0.388 | 0.109 | same winner, ratio moves 3.6x |

**Three of five architectures flip which unit is faster between an M4 Pro and an
M5 Max.** The two that do not flip still move by factors of 2.4 and 3.6, so even
where the ranking survives, the margin does not.

This is not a property of vision transformers. A dense-convolution CNN
(`resnet50`) and a text encoder (`bert`) both invert. Nor is it a property of
image models specifically: `bert` takes integer token input and inverts;
`whisper` takes log-mel audio and does not.

`mobilenet` is the result that most constrains ANE folklore.
Depthwise-separable convolution is the workload the Neural Engine is most often
described as ideal for, and **the GPU is faster on both chips**, by 1.19x on the
M4 Pro and 2.89x on the M5 Max.

Underlying medians (img/s):

| model | M4 Pro ANE | M4 Pro GPU | M5 Max ANE | M5 Max GPU |
| --- | ---: | ---: | ---: | ---: |
| `siglip` | 204.4 | 178.8 | 232.9 | 1077.7 |
| `resnet50` | 929.6 | 659.5 | 1094.6 | 3277.5 |
| `mobilenet` | 3044.7 | 3610.9 | 3602.2 | 10418.4 |
| `bert` | 757.3 | 557.6 | 822.5 | 3517.1 |
| `whisper` | 56.3 | 145.1 | 65.9 | 605.4 |

**ANE throughput is far less chip-sensitive than GPU throughput.** Between M4 Pro
and M5 Max the ANE moves by **1.09x to 1.18x** across the five architectures,
while the GPU moves by **2.89x to 6.31x**. Both chips have a 16-core Neural
Engine; the GPU count doubles from 20 to 40 and gains a per-core matrix unit
(§3.4).

That asymmetry is the whole of the inversion. The ANE improves modestly and
uniformly, consistent with an engine of the same width described by the vendor as
"improved" [[1]](#ref1). The GPU improves by 2.89x to 6.31x. Any
model whose better unit was the ANE by less than the GPU's gain therefore changes
hands.

### 3.2 The default is not a safe default

Cost of `ComputeUnit.ALL` relative to the better of the two explicit placements:

| model | M4 Pro | M5 Max |
| --- | ---: | ---: |
| `mobilenet` | 1.18x | 1.04x |
| `siglip` | 1.19x | 1.02x |
| `bert` | 1.36x | 1.01x |
| `resnet50` | **2.41x** | 1.01x |
| `whisper` | **5.18x** | 1.09x |

On the M5 Max the default is close to free: it costs 1.01x to 1.09x, tracking the
GPU to within 2% on `resnet50` and `bert` and falling 2.3%, 3.6% and 8.7% below it
on `siglip`, `mobilenet` and `whisper`. The direction is right — the GPU is the
better answer on that chip and the runtime selects it — but "within 2% on every
architecture" was an overstatement of a table that already said 1.09x two lines
above. **We report this as prominently as the negative result.**

On the M4 Pro the default costs 1.18x to 5.18x. More seriously, on three of five
architectures it is slower than **both** explicit placements, not merely slower
than the better one:

| model | ANE | GPU | `ALL` | |
| --- | ---: | ---: | ---: | --- |
| `siglip` | 204.4 | 178.8 | **172.2** | below both |
| `resnet50` | 929.6 | 659.5 | **386.1** | below both |
| `whisper` | 56.3 | 145.1 | **28.0** | below both |

The `whisper` case is the strongest form: the default returns 28.0 img/s where
the ANE alone returns 56.3 and the GPU alone returns 145.1. **The default is
half the throughput of the slower of the two engines it is choosing between.**

**A provenance note on `siglip`'s M4 Pro row.** Two runs of that sweep exist, and
they do not agree equally well on every column. The ANE and GPU medians differ by
0.012 and 0.148 img/s between them, which is inside the run-to-run peak spread
this paper measures at 178.7 to 178.8 (§3.3) — two runs of one measurement
agreeing to within their own noise. The `ALL` column is not like that: it reads
171.1 from `sweep-m4pro-rerun.json` and 172.2 from `sweep-m4pro.json`, a gap of
1.04 img/s or 0.6%, several times the spread of the other two columns. We print
172.2 and note that the default's cost for `siglip` is 1.19x or 1.20x depending
on which run is used. Nothing in this paper turns on that third decimal, but the
`ALL` column was the one quantity in the medians table with no check against the
pinned sources at all, and an unchecked column is where this kind of thing
lives.

#### The partition, and why the two chips differ

Reading `MLComputePlan` for each model under each setting (`tools/placement_sweep.py`)
answers the asymmetry directly. Placement is cost-weighted, so a fraction is a
share of estimated work rather than a share of operations:

| model | M4 Pro, `ALL` resolves to | M5 Max, `ALL` resolves to |
| --- | --- | --- |
| `mobilenet` | 100% ANE | 100% GPU |
| `resnet50` | 87.6% ANE / 12.4% GPU | 100% GPU |
| `siglip` | 78.9% ANE / 21.1% GPU | 100% GPU |
| `bert` | 77.5% ANE / 17.3% GPU / 5.1% CPU | 100% GPU |
| `whisper` | 64.1% ANE / 35.9% GPU | 100% GPU |

**On the M5 Max `ALL` never splits.** It resolves to a single engine on all five
architectures, that engine is the GPU, and the GPU is the right answer on that
chip, so the default costs 1.01x to 1.09x. **On the M4 Pro it splits on four of
five**, and it is there that it costs up to 5.18x.

So the asymmetry is not that one runtime chooses better than the other. One of
them declines to partition and the other does not.

#### Two failure modes, and they occur separately

The single cost figure mixes two things. Comparing `ALL` against the engine that
received *most* of the work separates them:

- **partition** = majority-engine throughput / `ALL` throughput. What splitting
  costs, with the choice of majority engine held fixed.
- **unit choice** = best-engine throughput / majority-engine throughput. What
  picking the wrong majority engine costs, with the partition held fixed.

Their product is the cost above. That multiplication is an algebraic identity and
is not evidence of anything; what is empirical is which factor equals one.

| model | partition | unit choice | total | failing on |
| --- | ---: | ---: | ---: | --- |
| `whisper` | 2.01x | 2.58x | 5.18x | both |
| `resnet50` | 2.41x | 1.00x | 2.41x | partition only |
| `bert` | 1.36x | 1.00x | 1.36x | partition only |
| `siglip` | 1.19x | 1.00x | 1.19x | partition only |
| `mobilenet` | 0.99x | 1.19x | 1.18x | unit choice only |

`mobilenet` is the case that separates them cleanly. `ALL` puts 100% of it on the
ANE, so there is no partition and no partition cost, 0.99x. Its entire 1.18x is
the runtime choosing the ANE when the GPU is 1.19x faster. **The default can be
wrong without splitting at all.**

#### The size of the partition does not predict its cost

The obvious hypothesis is that a more balanced split costs more, since it moves
more work across the boundary. We registered that as a prediction in
`tools/split_cost.py` before `resnet50` and `mobilenet` were measured on this
chip, and **it failed**:

| model | minority share | cost |
| --- | ---: | ---: |
| `mobilenet` | 0.0% | 1.18x |
| `resnet50` | **12.4%** | **2.41x** |
| `siglip` | 21.1% | 1.19x |
| `bert` | 22.5% | 1.36x |
| `whisper` | 35.9% | 5.18x |

`resnet50` has the *smallest* split of the four that split and the second most
expensive default. Moving 12.4% of the estimated work to the other engine costs
58% of the throughput. So the partition cost is real and can be severe, but it is
not a function of how much was moved, and a developer cannot bound it by reading
the compute plan.

The correct statement is not "the default is sometimes worst" but: **the cost of
the default is not portable.** It is free on one of our two configurations and up
to 5.18x on the other. We deliberately do not say "chip-dependent" here. Which
engine `ALL` resolves to is a runtime-heuristic output, and our two configurations
differ in macOS build as well as silicon (§3.5).

### 3.3 Peak does not predict sustained

All figures below are from the corrected tool. An earlier version of this section
used a soak that inserted `pmset` calls between windows and clamped its final
window to the deadline; both are fixed, and §4 records what the fix moved.

**M4 Pro, three physically distinct machines, `siglip`, 120 s, mains:**

| compute unit | box 1 | box 2 | box 3 | last-window img/s |
| --- | ---: | ---: | ---: | --- |
| ANE | 1.000 | 1.000 | 1.000 | 204.4, 204.5, 204.4 |
| GPU | 0.987 | 0.979 | 0.960 | 176.6, 174.9, 171.8 |
| `ALL` | 0.996 | 0.994 | 0.993 | 172.4, 171.9, 171.5 |

**The ANE holds its peak to within 0.01%, on three separate machines.** The GPU
gives back 2% to 4%. The default gives back under 1%, and on this chip it
sustains slightly *better* than the pure GPU placement rather than worse.

The absolute last-window rates agree to 2.7% across three boxes, which is the
reproducibility this measurement has when the chip does not throttle.

**M5 Max, same model, same protocol, and the number does not hold still:**

| run | unit | peak | last img/s | sustained |
| --- | --- | ---: | ---: | ---: |
| 1 | ANE | 235.3 | 235.2 | 1.000 |
| 1 | GPU | 1049.7 | 887.8 | 0.846 |
| 2 | GPU | 1041.8 | 882.4 | 0.847 |
| 3 | GPU | 799.9 | 761.7 | **0.952** |
| 1 | `ALL` | 1018.6 | 875.8 | 0.860 |
| 2 | `ALL` | 1030.3 | 800.4 | 0.777 |
| 3 | `ALL` | 784.7 | 773.6 | **0.986** |

**On this chip the sustained fraction is not a stable quantity, and the reason is
mechanical.** Sorted by starting peak it is monotone in both units: GPU goes
0.846, 0.847, 0.952 as the peak falls 1050, 1042, 800; `ALL` goes 0.777, 0.860,
0.986 as the peak falls 1030, 1019, 785. A machine that starts cool reaches a high
peak and gives a lot back. A machine already warm starts low, gives little back,
and therefore **scores better on a statistic that is supposed to mean "holds its
rate"**. Run 3 of each pair followed five earlier soaks with 15 to 20 s between
them; its peak is a quarter lower because the box had not recovered.

So on a throttling chip, `sustained fraction` measures the compute unit *and the
thermal state it started in*, and this protocol does not separate them. Compare
absolute last-window rates instead: GPU 887.8, 882.4, 761.7 and `ALL` 875.8,
800.4, 773.6, which still vary but at least expose the instability rather than
normalising it away.

**A control confirms this belongs to the chip, not to the statistic.** The same
no-cooldown protocol that produced the effect on the M5 Max was run four times
back to back on an M4 Pro:

| | M4 Pro | M5 Max |
| --- | ---: | ---: |
| peak spread across repeats | **0.06%** (178.7 to 178.8) | **25.9%** (799.9 to 1049.7) |
| sustained spread | 0.022 | 0.106 |
| peak correlated with sustained? | no | yes, monotone |

**The M4 Pro peak does not move at all**, 463x less than the M5 Max's, so there is
no starting state for the fraction to track and its 0.022 spread is ordinary
run-to-run noise. The confound therefore appears only on hardware that actually
throttles under this load, which is why the M4 Pro figures in this section are
usable and the M5 Max magnitudes are not.

**What survives all of this is the ANE.** It gave back at most 0.14% of its peak
in every run, on every box, on both chips, regardless of starting state — with a
single unexplained exception at 2.52%, named and discussed below. That
insensitivity is exactly
what an engine that does not degrade should look like, and it is why the ANE
result is the robust one and the M5 Max magnitudes are not.

**Retracted.** An earlier version said the default was the *worst* sustainer on
the M5 Max, at 0.721 against the GPU's 0.837, and drew the conclusion that a burst
benchmark and a sustained benchmark disagree about it. With the tool fixed and the
run repeated, `ALL` and the GPU are indistinguishable, and the ordering reverses
depending on which pair of runs is compared. The claim is withdrawn. What remains
is weaker and still worth saying: on the M5 Max both the GPU and the default give
back roughly a sixth of their peak over two minutes, and the ANE gives back
nothing.

The GPU's advantage over the ANE therefore still shrinks with the length of the
measurement, but the size of the shrink is not a fixed number on this machine.

The ANE result is not specific to `siglip`. **98 soaks across five architectures
and three physically distinct M4 Pro machines**, corrected tool, 90 s cooldown
between runs:

| model | ANE (n) | GPU (n) | GPU peak stability |
| --- | --- | --- | ---: |
| `siglip` | 0.9996 to 1.0000 (10) | 0.949 to 0.987 (20) | 0.76% |
| `resnet50` | 0.9988 to 1.0000 (8) | 0.962 to 0.989 (10) | 0.51% |
| `mobilenet` | 0.9990 to 0.9998 (8) | 0.963 to 0.984 (10) | 0.84% |
| `bert` | 0.9997 to 1.0000 (6) | 0.952 to 0.984 (8) | 0.08% |
| `whisper` | **0.9748** to 1.0000 (8) | 0.868 to 0.967 (8) | 0.65% |
| **all** | **0.9748 to 1.0000 in 40** | **0.868 to 0.989 in 56** | |

**In 39 of 40 M4 Pro soaks the ANE gave back at most 0.12% of its peak, better
than the best of 56 GPU soaks, which gave back 1.08%.** That holds across a
vision transformer, a dense CNN, a depthwise CNN, a text encoder and an audio
encoder.

**One run is the exception and we do not explain it.** A `whisper` ANE soak on
one of the three M4 Pros gave back 2.52%: flat at 56.3 img/s for seven windows,
then a monotone fall to 54.9 that plateaued. It is not a contended peak, which is
the usual cause of a low reading here and the one that cost us the M5 Max figures
in §3.3. Contention lowers the peak, and this run reached 56.34 img/s against
56.25 to 56.35 for the seven other `whisper` ANE runs, none of which declined. So
the engine reached full rate and then lost it. The soak tool did not record what
else was running when that run was taken, so its own file cannot settle whether
something started mid-run.

**It did not reproduce.** A repeat on the same machine returned 0.9998, and that
repeat carries a concurrent-load record showing the box at 3.7% of its 14 cores,
so it was taken under conditions we can vouch for rather than assume. One repeat
is not proof that the first reading was spurious, and we leave it in the table
rather than dropping it, but nothing we have measured since supports treating it
as a property of the engine.

We report it rather than dropping it. With that run included the ANE and GPU
ranges overlap at a single point; with it excluded they are disjoint. A reader
should treat "the ANE does not degrade" as holding in 39 of 40 measurements, not
as a law.

We report the ANE column to four decimals deliberately. At three, every one of
these forty prints as `1.000`, and an earlier draft of this table read
"1.000 in 10 of 10" and claimed the ANE "returned exactly 1.000". That was a
rounding artefact: only 14 of 40 are exactly 1.0000. The distinction matters
because "exactly" invites a mechanism that does not exist, a hard clamp at peak,
where what the data supports is the weaker and sufficient claim that ANE
throughput does not measurably decay over two minutes.

The GPU peaks are stable to between 0.08% and 0.84% across repeats and across
boxes, so none of the M4 Pro variation is the thermal-state confound of §3.3;
these are genuine run-to-run differences of 0.02 to 0.05 in the fraction.

`whisper` declines most (0.868 to 0.967) and is the most arithmetic-dense model in
the set, which is the direction the power explanation in §3.4 predicts, though
with five models this is a consistent observation rather than a demonstrated
relationship.

The default sits with the GPU rather than below it. On this chip it sustains at
least as well as the pure GPU placement on four of five architectures, which is
the opposite of what an earlier version of this paper claimed (§4).

This has a practical consequence the burst tables alone cannot support: **for a
continuously loaded service the ANE's advertised rate is the rate you get, and no
other unit's is.** A capacity plan built from burst numbers will over-provision
the GPU's contribution by 1 to 16% and the default's by up to 28%.

The 120 s protocol says nothing about where the GPU's decline stops, and the
600 s runs later in this section say it does not keep going: four of five models
recover most of the dip and settle. What is unmeasured is beyond **600 s**. On the M4 Pro it had flattened by the end of the soak. On the M5 Max it had
not, which is part of why the fraction there depends on the starting state (§3.3):
a run that begins warm has already spent the decline before the clock starts.

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
engine performing 6.17x the work per second draws proportionally more power and
approaches a ceiling the smaller GPU never reaches. We regard the direction as
established and the exact magnitude as chassis-sensitive (§4).

#### Two minutes measures the GPU's trough, not its steady state

Every figure above comes from a 120-second soak. Extending to 600 seconds on four
architectures shows that duration is not a neutral choice, and that the
two-minute protocol systematically understates the GPU. **Everything in this
subsection is M4 Pro.** The M5 Max behaves differently and is treated at the end.

**The GPU dips and recovers.** It does not decline monotonically. Reading the
per-window trace rather than only the endpoints — `siglip` from
`long600-inference1-…` and `resnet50` from `long600-experiments-…`, which are
different boxes and were previously printed together unlabelled:

```
siglip    GPU   178 178 173 171 171 171 172 172 173 174 176 176 ... 177
resnet50  GPU   656 656 656 649 647 648 648 650 652 653 655 ... 655
```

Peak, a minimum around 80 to 100 seconds, then monotone recovery to a plateau
just below peak. **A 120-second soak ends inside that trough.**

The position of the minimum is the strongest evidence that this is real and
engine-specific. Across four 600 s runs of each engine, in windows of 20 s:

| | minimum falls in window |
| --- | --- |
| GPU | **4, 4, 5, 5** of 29 |
| ANE | 1, 6, 17, 26 of 29 |

All four GPU minima land in a two-window band. If position were uniform over 29
windows the chance of that is about 2e-5. The ANE's minima are scattered, which is
what noise around a flat line looks like rather than a trajectory.

**The same signature is already present in the 120 s corpus**, which is a
prediction this explanation makes about data collected before it existed. If a
120 s run ends inside the trough, its slowest window should be near its end.
Across every M4 Pro 120 s soak:

| | n | minimum in the last 30% of the run | mean position |
| --- | ---: | ---: | ---: |
| GPU | 56 | **43 (77%)** | 0.81 |
| ANE | 40 | 4 (10%) | 0.32 |

Uniform position would put about 30% of minima in the last 30%. The GPU is at
77%, the ANE at 10%. So the dip is not an artefact of four long runs: it is
visible in fifty-six independent short ones, and the ANE shows the opposite
pattern, its slowest window falling early as a flat line with a slight warm-up
would.

**Both readouts, from the same runs.** Taking the window that ends at 120 s and
the mean of the last six windows out of the *same* 600 s soak removes every
between-run confound:

| model | unit | at 120 s | at steady state |
| --- | --- | ---: | ---: |
| `siglip` | ANE | 1.0000 | 0.9995 |
| `siglip` | GPU | 0.9635 | **0.9959** |
| `resnet50` | ANE | 0.9963 | 0.9995 |
| `resnet50` | GPU | 0.9876 | **0.9978** |
| `bert` | ANE | 0.9980 | 0.9994 |
| `bert` | GPU | 0.9835 | **0.9976** |
| `whisper` | ANE | 0.9996 | 0.9999 |
| `whisper` | GPU | 0.9496 | **0.9737** |

The ANE moves by at most 0.4 points between the two readouts. The GPU moves by up
to 3.2. **So the 120 s figure conflates two different things**: a transient dip,
which is universal across models and is an artefact of when measurement stops,
and a true steady-state decline, which is not universal at all. After recovery
`whisper` still gives back 2.6% on the GPU while `resnet50` gives back 0.2%, a
twelve-fold spread between architectures on one chip. From the four-decimal
values it is (1 − 0.9737) / (1 − 0.9978) = 11.95; "thirteen-fold" only appears
if the give-backs are rounded to one significant figure first, which is the
rounding artefact this section warns about two paragraphs earlier.

This also explains the width of the 120 s M4 Pro GPU range, 0.868 to 0.989, without
appealing to noise: those runs sample different points on a recovery curve, and
different architectures settle at different floors.

**What this does not change.** The ANE still sustains better than the GPU on every
architecture at every duration measured. What changes is the size and the shape of
the claim: at steady state the advantage runs from 4.0x on `bert` to 8.2x on `siglip`, and
263x on `whisper` whose ANE gives back 0.01% against the GPU's 2.63% — so it is
not a single multiple at all, rather than the order of magnitude the two-minute
numbers suggest, and the correct statement about
the GPU is not "it declines" but "it dips, recovers, and then settles at a
model-dependent floor".

**The M5 Max does not do this.** Its GPU drops roughly 15% within the first two
windows and then holds, rather than dipping a few percent and recovering:

```
M4 Pro GPU    178  178  173  171  171  171  172  173  174  176  177 ... 177
M5 Max GPU   1050  933  907  895  894  896  889  892  892  889  888
```

So the shape is not a property of "the GPU" but of a particular chip under a
particular load. On the M5 Max the loss is immediate and permanent within the
run, which is consistent with the *arithmetic-throughput* reading in §3.4: a GPU
sustaining 39.2 TFLOPS against the M4 Pro's 6.4 reaches a ceiling the smaller
part never approaches. We call it that rather than a "power-limit reading",
which is what this sentence said before, because **§3.4 contains no power
measurement** — `results/gpu-probe.json` records times and derived FLOPS and
nothing electrical. Power is the mechanism we infer, not one we measured. The ANE is flat on **both** chips, 233 and 235 img/s across every
window of the M5 Max runs.

This matters for the correction above. On the M4 Pro a 120 s soak understates the
GPU because it stops in a trough that later recovers. On the M5 Max a 120 s soak
does **not** understate anything, because there is no recovery to miss. Applying
one correction to both would be wrong in the second case.

#### The dip is a cold-start transient on four of five models, and on `mobilenet` it is neither

Die temperature would settle this directly, and `powermetrics` requires root,
which we declined to wire into the tool so that anyone can reproduce these runs.
A back-to-back design discriminates without it. Three 600 s GPU soaks per box with
**no cooldown between them**: run 1 starts idle, runs 2 and 3 start on a machine
that has been at full GPU load for the preceding ten and twenty minutes.

| model | run 1, idle | run 2, warm | run 3, warm | minimum at window |
| --- | ---: | ---: | ---: | --- |
| `siglip` | **3.05%** | 0.70% | 0.50% | 6, 19, 29 |
| `resnet50` | **2.71%** | 0.34% | 0.34% | 5, 15, 27 |
| `bert` | **2.13%** | 0.21% | 0.28% | 4, 14, 13 |
| `whisper` | **3.83%** | 1.14% | 1.40% | 4, 4, 5 |

**The steady state does not move.** siglip settles at 176.7, 176.6, 176.8 img/s and
`whisper` at 144.0, 144.1, 144.1 across the three runs. Warming the machine removes
the early loss and changes nothing else.

**Neither does the starting point, and that rules out the obvious explanation.**
An earlier draft of this section called the dip a cold-start *boost* being given
up. It is not, and the first window refutes it: if a cold box were boosting, it
would start measurably faster than a warm one, and it does not.

| model | window 1, cold | window 1, warm (mean) | difference | dip, cold |
| --- | ---: | ---: | ---: | ---: |
| `siglip` | 177.69 | 177.19 | +0.28% | 3.05% |
| `resnet50` | 656.88 | 656.38 | +0.08% | 2.71% |
| `whisper` | 145.63 | 145.35 | +0.19% | 3.83% |
| `bert` | 557.69 | 557.16 | +0.09% | 2.13% |

The cold advantage at the start is **0.08 to 0.28%**, an order of magnitude too
small to be the 2.13 to 3.83% that is subsequently lost. Cold and warm runs begin
at the same rate and end at the same rate; the only difference is in between.

What the series actually shows is an **excursion**. A cold run holds its opening
rate for two or three windows, drops in a single step — siglip goes 177.3 to 173.2
between windows 3 and 4 — bottoms around 80 to 120 s, and then climbs back to the
plateau over the following minutes. A warm run does not take the step at all.

So the finding is that **the excursion is conditional on starting cold**, and that
it is transient rather than degradation, since the endpoint is identical either
way. Naming its cause needs die temperature and clock residency, which need root,
and this paper does not claim one. A control loop converging once on a cold ramp
fits, and so do other things.

**`mobilenet` does neither of those things, and it is the reason to run all five.**
Added to the protocol after the other four, and run on **both** M4 Pro machines
rather than one, so box and engine are not aliased:

| box | run 1, idle | run 2, warm | run 3, warm | plateau, last 6 windows |
| --- | ---: | ---: | ---: | ---: |
| inference1 | 2.15% | 2.35% | 2.44% | 0.980, 0.977, 0.976 |
| experiments | 2.31% | 2.10% | 2.15% | 0.979, 0.980, 0.980 |

Two claims above fail on it. The dip does **not** shrink on a warm box — six runs
spanning 2.10 to 2.44% with no cold/warm ordering on either machine — so it is not
conditional on starting cold. And the shape is not an excursion. Windows 1 to 12
of the cold run, as a percentage of peak:

    mobilenet   100.0  99.6  99.2  98.4  98.2  98.3  98.1  97.8  98.0  98.0  98.0  98.0   ... 98.0
    siglip      100.0  99.9  99.8  97.5  97.0  96.9  97.1  97.3  97.7  98.1  98.7  99.0   ... 99.4

`mobilenet` settles at 98% by the eighth window and holds it for the remaining
twenty-one; the second machine settles at 97.7% by the ninth and holds that. Both
are a settling. `siglip` bottoms at 96.9% and climbs back to 99.4%, which is an
excursion, and only the second recovers.

So the M4 Pro GPU has **two** behaviours, not one, and which appears depends on
the model. That also means the "read the 120 s number as a trough" correction
above applies to the four that recover and not to `mobilenet`, whose 120 s number
is already its steady state.

**The two machines agree more closely here than §3.6's floor would suggest.** The
six GPU peaks span 3615.6 to 3632.0 img/s, 0.5%, and the six ANE peaks 3075.8 to
3077.2, 0.05% — against the 2.7% agreement on `siglip` last-window rates that
§3.6 reports. A between-machine floor measured on one model and one metric does
not transfer to another, and the honest reading of §3.6 is that 2.7% is what that
comparison gave, not a constant.

**On three of five the effect disappears rather than shrinking.** For `siglip`,
`resnet50` and `bert` the dip falls to 0.70% or less *and* the minimum stops
occupying the 4 to 6 band. Across all twelve warm runs of those three models the
minima land at 8, 10, 13, 14, 15, 17, 19, 21, 22, 22, 27 and 29, where the
position carries no information. Losing both the magnitude and the location is what a
signature going away looks like; losing only the magnitude would not be.

#### Every model, on both machines

The four findings above each rested on one model measured on one box. `mobilenet`
showed what that can hide, so the whole protocol was repeated for all five models
on both M4 Pro machines — thirty runs, ten model-box cells:

| model | box | cold | warm | warm | minima |
| --- | --- | ---: | ---: | ---: | --- |
| `siglip` | inference1 | 3.05% | 0.70% | 0.50% | 6, 19, 29 |
| `siglip` | experiments | 2.59% | 0.26% | 0.21% | 4, 21, 22 |
| `resnet50` | inference1 | 2.71% | 0.34% | 0.34% | 5, 15, 27 |
| `resnet50` | experiments | 1.49% | 0.21% | 0.34% | 5, 22, 8 |
| `bert` | experiments | 2.13% | 0.21% | 0.28% | 4, 14, 13 |
| `bert` | inference1 | 2.33% | 0.22% | 0.28% | 5, 10, 17 |
| `whisper` | experiments | 3.83% | 1.14% | 1.40% | 4, 4, 5 |
| `whisper` | inference1 | 4.01% | 2.63% | 2.50% | 6, 15, 28 |
| `mobilenet` | inference1 | 2.15% | 2.35% | 2.44% | 8, 9, 28 |
| `mobilenet` | experiments | 2.31% | 2.10% | 2.15% | 22, 9, 19 |

**The pattern is robust and the magnitude is not.** On four of five models, on both
machines, the cold run's dip is several times the warm runs' and the warm minimum
leaves the 4-to-6 band. `mobilenet` fails to be cold-conditional on *both*
machines, so it is a consistent exception rather than a one-box artefact. But the
size of the cold dip moves between physically identical boxes — `resnet50` reads
2.71% on one and 1.49% on the other, `siglip` 3.05% against 2.59% — so **the
direction transfers and the number does not**. A single-box cold dip should be
quoted as a pattern, not as a value.

`bert` is the closest thing to a clean replication in the set: 2.13 / 0.21 / 0.28
against 2.33 / 0.22 / 0.28.

**`whisper`'s warm residual is the one thing that does not survive the second
machine.** Its cold dip reproduces closely, 3.83 against 4.01. Its warm residual
does not: roughly twice as large on `inference1` **and its minimum leaves the
4-to-6 band**, landing at 15 and 28, which is the pattern this section calls a
signature going away. The two pieces of evidence that made `whisper` look
model-specific — a residual that survives warming, and a minimum that stays put —
point in opposite directions once a second machine is used.

So we weaken that claim rather than repeat it. `whisper` retains more of its dip
on a warm box than the other four do, on both machines. Whether that residual is a
property of the **model** is not established: its size and its position both move
between two identical machines, which is what a machine effect looks like. Model
load and cache warming remain candidates; so does something about the individual
box, and none of the three is measured.

**The ANE, run through the identical protocol, shows none of it.** Every run above
is `CPU_AND_GPU`, so the design cannot by itself distinguish a property of the GPU
from a property of the harness or of the first seconds of any measurement. Three
back-to-back 600 s `CPU_AND_NE` soaks on the same two boxes, same script, same
window length, no cooldown:

| model | box | run 1, idle | run 2, warm | run 3, warm | cold−warm spread | minimum at window |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `siglip` | inference1 | 0.06% | 0.05% | 0.06% | 0.01 pp | 21, 23, 10 |
| `whisper` | experiments | 0.05% | 0.04% | 0.05% | 0.02 pp | 4, 4, 25 |
| `resnet50` | inference1 | 0.30% | 0.31% | 0.32% | 0.02 pp | 8, 5, 6 |
| `bert` | experiments | 0.23% | 0.25% | 0.25% | 0.01 pp | 24, 13, 26 |
| `mobilenet` | experiments | 0.76% | 0.77% | 0.77% | 0.01 pp | 2, 2, 1 |
| `mobilenet` | inference1 | 0.77% | 0.69% | 0.73% | 0.08 pp | 1, 27, 10 |

**Starting cold changes nothing on any of the five.** The spread across the three
runs is 0.01 to 0.02 percentage points on four of them and 0.08 pp on
`mobilenet`/inference1, and on three of six rows the cold run is the *smallest*.
The GPU's cold run is larger than its warm runs by 1.22 to 2.56 percentage points
across the four models that dip, on the same two machines — and by **−0.24 and
+0.19 pp** on `mobilenet`, which does not. Peaks agree to within 0.06% across
runs on every model. Anything living in the tool, the model load, or the first
minute of a measurement would have moved this too, so the excursion is the GPU's.

**The ANE's floor is model-dependent, which we did not expect.** `siglip` and
`whisper` give back 0.04 to 0.06%; `resnet50` and `bert` give back 0.23 to 0.36%,
five times more; and `mobilenet` gives back **0.69 to 0.77%**, the largest of the
five and more than ten times `siglip`. So the spread across models is not five-fold
but closer to twenty-fold. Against the GPU's cold dip of roughly 2 to 4% that
still leaves the ANE 3 to 50 times better depending on the model, rather than the
uniform 7 to 16 times an earlier version claimed from the four models it had. It
is reproducible rather than noisy — three runs per model agree to 0.08 pp or
better — so it is a real property of those graphs on that engine and not a
measurement artefact. It is reported because it was predicted otherwise: a
pre-registered expectation of 0.04 to 0.08% for all four was written before these
ran, and it was wrong twice over, since `mobilenet` is further out again.

That also qualifies an argument made above. `resnet50`'s minimum sits at windows
8, 5 and 6 — inside the early band — in all three runs *including the warm ones*,
so "the position is uninformative on the ANE" is not universal. The stronger form of that
argument does not survive either: on `whisper`/experiments the minimum sits at
4, 4 and then 25, and on `mobilenet`/inference1 at 1, 27 and 10, so the position
*does* move between cold and warm on two of the six rows. What distinguishes the
ANE from the GPU is therefore not that its minimum stays put — sometimes it does
not — but that its **magnitude** stays put: 0.01 to 0.08 pp across cold and warm,
against the GPU's 1.2 to 2.6.

So the excursion is a property of **the GPU**, not of the tool, the model load, or
the first minute of a measurement. Any of those would have moved the ANE too.

What this does **not** identify is the physical variable. On a cold machine clock
headroom, power budget and fan state all move together, and this design separates
warm from cold without separating those. The 120 s tables above are retained
because the five-architecture comparison rests on them, and they are now labelled
for what they measure.

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
| per-GPU-core matrix accelerator | none | yes [[1]](#ref1)[[2]](#ref2) | n/a |
| Neural Engine cores | 16 | 16 | 1.0x |
| unified memory bandwidth | 273 GB/s [[3]](#ref3) | 614 GB/s [[2]](#ref2) | 2.25x |

**So the two engines received very different investments across this generation:
the GPU gained both 2x the cores and a new per-core matrix unit, while the Neural
Engine kept its width.** A model whose cost is dominated by matrix multiplication
should therefore move much further on the GPU than on the ANE. That is exactly
what §3.1 measures, and it is why the ANE can be the faster unit on an M4 Pro and
the slower unit on an M5 Max for the same model.

**The measurement matches that signature.** Two synthetic Core ML models of
comparable runtime, one arithmetic-bound and one bandwidth-bound
(`tools/probe_gpu.py`):

| probe | M5 Max | M4 Pro | ratio |
| --- | ---: | ---: | ---: |
| matmul, 103.1 GFLOP/call | 2.63 ms (**39.2 TFLOPS**) | 16.21 ms (**6.4 TFLOPS**) | **6.17x** |
| elementwise, 1610 MB/call | 13.43 ms (120 GB/s) | 17.04 ms (95 GB/s) | 1.27x |

**GPU arithmetic throughput moves 6.17x for a 2.0x core-count increase.** Core
count alone cannot produce that; the surplus is the per-core matrix accelerator.
Bandwidth-bound work moves only 1.27x on the same pair of chips, so the gain is
specific to arithmetic and not a general uplift.

Two vendor figures BOUND that 6.17x, and we call them bounds rather than
cross-checks because neither can fail on its own. Apple claims "over 4x peak GPU
compute" for M5 against M4 at equal core count [[1]](#ref1); doubling the cores
on top of that predicts something approaching 8x. Apple separately claims the M5
Max is 3.9x faster than the M4 **Max** on AI [[2]](#ref2); since the M4 Max has
roughly twice the GPU of the M4 **Pro** used here, that implies well above 3.9x
against an M4 Pro.

**Together they admit anything between about 3.9x and 8x, and 6.17x is simply
inside that window.** A GPU improvement produced entirely by clocks, process or
compiler would land in the same window, so agreement here is consistency, not
evidence for the matrix unit specifically. What would actually discriminate is
the *shape* of the gain across workloads — 6.17x arithmetic-bound against 1.27x
bandwidth-bound on the same chip pair — and that is the probe above, not these
two numbers.

Every cross-chip GPU gain in §3.1 sits **at or between the two probe bounds**:
2.89x for `mobilenet` up to 6.31x for `bert`, against 1.27x bandwidth-bound and
6.17x arithmetic-bound. `bert` exceeds the matmul probe, by 2.2%, and **we cannot
dismiss that as probe noise, because the probe has no repeat spread to appeal
to**: `results/gpu-probe.json` carries repeats on the M4 Pro side only (matmul
16.08 and 16.35 ms, 1.65% apart) and a single M5 Max timing, so the *ratio* was
measured once. Even taking the M4 Pro's slowest repeat, which is the direction
that flatters the ceiling, gives 16.35/2.63 = 6.22x, still short of `bert`'s
6.31x. The honest statement is that one of five models sits 2.2% above a ceiling
measured without repeats on the side that matters, which is a caveat on the probe
rather than a finding about `bert`. That is the expected ordering if each model sits somewhere
on the spectrum between bandwidth- and compute-bound: the most arithmetic-dense
model sits at the matmul ceiling and none clears it materially.

**Two honest limits on this argument.** First, the elementwise probe reaches only
95 GB/s of the M4 Pro's 273 GB/s and 120 GB/s of the M5 Max's 614 GB/s, so it is
not bandwidth-saturating; its 1.27x is a floor on the achievable-bandwidth ratio,
not the hardware's 2.25x. The conclusion survives, because even the generous 2.25x
hardware bandwidth ratio is far below the 6.17x arithmetic ratio, but the probe
should not be read as a bandwidth benchmark. Second, we did not perform a
per-model roofline analysis, so we do not claim to know where each of the five
architectures sits on that spectrum; we claim only that all five land inside the
bounds, which is what the mechanism predicts.

**A falsifiable prediction.** An earlier version of this section predicted that
the inversion would appear on **any** M5-family part and on no M4-or-earlier part,
*independent of tier* — that a base M5 would rank the units as the M5 Max does,
and that an M4 Max would not invert merely for having more GPU cores. That
prediction is wrong, and the medians in §3.1 are enough to show it. Tier cannot
drop out, because the ranking is a race between a GPU that scales with core count
and an ANE that is 16 cores on **both** chips: core count is a first-order term in
the mechanism, not a nuisance parameter.

Scaling the M5 Max GPU medians by 10/40 for a base M5 and leaving the ANE alone:

| model | base M5 GPU, scaled | M5 Max ANE | winner |
| --- | ---: | ---: | --- |
| `siglip` | 269.4 | 232.9 | GPU |
| `resnet50` | 819.4 | 1094.6 | **ANE** |
| `mobilenet` | 2604.6 | 3602.2 | **ANE** |
| `bert` | 879.3 | 822.5 | GPU |
| `whisper` | 151.4 | 65.9 | GPU |

So the mechanism predicts a base M5 inverts on **three of five**, not five of
five: `resnet50` and `mobilenet` should go back to the ANE. And an M4 Max, at
twice the M4 Pro's GPU and no matrix unit, should invert `siglip`, `resnet50` and
`bert` — every model the M4 Pro gives to the ANE — which it also does at only 71%
of linear core scaling. It should invert, on the paper's own arithmetic.

That makes for a sharper test than the original, because the two parts now differ
in *pattern* rather than in a yes-or-no: an M4 Max should invert everything, while
a base M5 — despite the per-core matrix unit — should still leave two models on
the ANE. Pure core-count scaling and the matrix-unit mechanism predict different
sets, so one machine of either kind discriminates between them.

Both extrapolations assume linear core scaling, which our own probe says is
generous — 6.17x arithmetic for a 2.0x core increase means the M5 gains more than
cores alone — and neither accounts for memory bandwidth scaling sub-linearly. We
do not have those machines. This remains the cheapest experiment that could
falsify the explanation, and we invite it.

The ANE's much smaller movement is consistent with both chips carrying the same
16-core Neural Engine. It moves 1.09x to 1.18x between chips, in the M5 Max's
favour on all five models; the regression an earlier version reported here was a
contended measurement and does not exist (§4).

### 3.5 What is chip and what is OS build

The two configurations differ in chip, GPU core count, chassis **and macOS build**
(26.5.1 against 26.6). These are fully collinear. Core ML's placement heuristic
ships inside the OS, so for any given result we have to ask which difference is
doing the work. The answer is not the same for every result.

**The inversion (§3.1) has a chip attribution and it does not depend on the ANE.**
§3.4 gives a vendor-documented hardware change, an independent within-machine
probe, and a falsifiable prediction. It also survives deleting the ANE side
entirely: hold the ANE at its measured M4 Pro value and combine with the measured
M5 Max GPU, and all three inversions still occur, because every GPU gain
(2.89x to 6.31x) exceeds the largest M4 Pro ANE/GPU ratio (1.410). The inversion is
carried by the GPU, which is the side with the documented mechanism.

**Retracted.** This paragraph read that `mobilenet` "moves 4.5x" and `whisper`
"moves 6.6x", and that both were inflated roughly 1.6x by the ANE regression. Both
halves are dead: §3.1 measures those moves at 2.44x and 3.56x from the clean
sources, and §4 finds the regression does not exist. Holding the ANE at its M4 Pro
value gives 2.89x and 4.17x — larger than the published figures, not smaller — so
the correction, had it been needed, pointed the other way.

**The default (§3.2) has no chip attribution at all, and we withdraw the claim
that it has one.** Which engine `ALL` resolves to is an output of a runtime
heuristic that ships in the OS. §3.4's hardware mechanism says nothing about it,
and we did not instrument the partition. The honest statement is that the cost of
the default **differs between these two configurations**, by up to 5.16x, and that
we cannot say whether the responsible difference is the silicon or the OS build.
That is still the practical warning a developer needs: the default is not
portable. It is not a claim about chips.

**Placement is identical on both configurations, which we measured.** Measured
with `tools/anecheck.py`, cost-weighted, `CPU_AND_NE`, on both machines:

| model | M4 Pro ANE by cost | M5 Max ANE by cost |
| --- | ---: | ---: |
| `siglip` | 100.0% | 100.0% |
| `resnet50` | 100.0% | 100.0% |
| `mobilenet` | 100.0% | 100.0% |
| `whisper` | 100.0% | 100.0% |
| `bert` | 95.2% | 96.3% |

Four models are fully ANE-resident on each machine, and `bert` puts an identical
95.3% of operations there by count on both. So the ANE comparison in §3.1 is
like-for-like: the same graph on the same engine, and the 1.09x to 1.18x
improvement is the engine getting faster rather than more work reaching it.

We report this because an earlier version of this paper contained an ANE
*regression* that did not survive re-measurement (§4, measurement hygiene), and
this residency check was run to explain it. The regression was an artefact; the
residency result stands on its own as a control.

**The experiment that would settle this is cheap and we have the hardware.**
Update M4 Pro #3 to 26.6, keep #1 and #2 on 26.5.1, and rerun both sweeps. That
gives a third cell and separates OS build from silicon for every result in the
paper. Upgrading the third rather than the second matters: it leaves a *pair* on
the old build, so the OS difference can be read against the between-machine floor
§3.6 measures rather than against a single machine. We have not done it.

### 3.6 Measurement reproducibility

Most single-machine benchmark results do not establish how much they move for
reasons other than the variable under study. **Two physically identical M4 Pro
Mac minis, twelve repeats each, corrected tool, same 120 s protocol and 90 s
cooldown on both**, give that floor directly. Sustained fraction, GPU, `siglip`:

| | M4 Pro #2 | M4 Pro #3 |
| --- | ---: | ---: |
| mean of 12 | 0.9726 | 0.9597 |
| within-machine sd | 0.0042 | 0.0047 |
| ANE control, mean of 4 | 0.9997 | 0.9995 |
| ANE control, sd | **0.0002** | **0.0001** |

**The between-machine difference is 0.0130 with a standard error of 0.0018, or
7.1 sigma. Identical hardware is not identical**, and a single sustained GPU
figure should not be quoted to three decimals.

The ANE arm is the control: same boxes, same session, same harness, holding to an
sd of 0.0002. Twenty times tighter than the GPU's, so the spread is a property of
the GPU under sustained load and not of the measurement.

**It is not one odd machine. A third M4 Pro mini, same protocol, differs from
both of the others** — though only one of the three pairs below turned out to be
a fair comparison, for reasons the rest of this section is about. Settled means
over the last nine of twelve repeats, and every pairwise difference as first
measured:

| box | settled mean | settled sd |
| --- | ---: | ---: |
| M4 Pro #2 | 0.9705 | 0.0012 |
| M4 Pro #3 | 0.9575 | 0.0015 |
| M4 Pro #4 | 0.9495 | 0.0012 |

| pair | difference | se | sigma | |
| --- | ---: | ---: | ---: | --- |
| #2 vs #3 | 0.0130 | 0.0006 | 20.7 | stands |
| #2 vs #4 | 0.0211 | 0.0006 | 37.4 | **withdrawn below** |
| #3 vs #4 | 0.0080 | 0.0006 | 12.8 | **withdrawn below** |

Read on its own, that table says three machines of the same model, same OS, same
tool, same protocol, all on mains, spread over 0.021 of sustained fraction while
each holds itself to 0.0012 to 0.0015. That is what we published, and two of its
three rows do not survive.

**Two of those three pairs are confounded with session, and we found it by
running a second series rather than by re-reading the first.** `experiments` and
`inference1` were measured concurrently, 16:09 to 16:47. `inference2` was
measured three hours later, 19:05 to 19:42. Repeating the whole twelve-run series
on `inference1` that evening gives a settled mean of 0.9703 against its own
morning figure of 0.9575 — **the same machine, 0.0128 apart, 16.3 sigma**, which
is the size of the entire between-box effect.

`inference1` and `inference2` then ran concurrent second series that evening,
which supplies a session-matched replacement for one of the two broken pairs.
Sorting every pairwise comparison by whether the two series actually overlapped
in time (`tools/session_vs_box.py`):

| pair | session | difference | |
| --- | --- | --- | --- |
| #2 vs #3, both 16:09–16:47 | **matched** | **0.0130** | 20.7 sigma |
| #3 vs #4, both 20:0x–20:4x | **matched** | **0.0173** | 18.8 sigma |
| #3 vs #4 again, both 20:4x–21:2x | **matched** | **0.0203** | 23.6 sigma |
| #2 vs #4, three hours apart | confounded | 0.0211 | withdrawn |
| #3 vs #4, three hours apart | confounded | 0.0080 | withdrawn |
| #2 vs #3, four hours apart | confounded | 0.0003 | withdrawn |

The first and last rows are **the same two machines**. Measured at the same time
they differ at 20.7 sigma; measured four hours apart they are indistinguishable.
Nothing about the machines changed between those two readings — only whether the
comparison was matched — and either one, taken alone, is publishable-looking.

The heading claim survives on the matched rows, and is if anything understated:
the session-matched `inference1` vs `inference2` difference is 0.0173, more than
twice the 0.0080 the confounded comparison reported. The session term was hiding
half of that box difference rather than manufacturing it. What does not survive
is the three-way spread of 0.021, which mixes a box effect with a session effect
of its own size.

The #3 vs #4 pair was then measured a second time, on concurrent third series,
and reproduces at 0.0203. Two independent session-matched estimates of the same
box difference, 0.0173 and 0.0203, differing by about one hour's drift.

The session effect itself scales with elapsed time on both machines. On
`inference1` the shift is 0.0128 across 3.2 hours and 0.0157 across 4.0 hours —
0.0040 and 0.0039 per hour, near-linear — while `inference2` moved 0.0035 in its
first 0.4 hours and then nothing. So this is a drift in time rather than
independent per-session draws, and the two machines are at different phases of
it. What it drifts *toward*, and whether it ever returns, is not established:
every series so far runs later than the last, so a monotone drift and a
time-of-day cycle fit equally well. Separating them needs a series started at
the same hour as the first. What distinguishes the
machines is still unknown: they differ in serial number and in nothing we
controlled. What is now known is that it can only be measured simultaneously.

**What this replaces.** An earlier version of this section led with five repeats
on one machine and reported a within-machine *range* of 0.011 against a
between-machine *difference* of 0.019, calling the ratio 1.7x. That comparison
was wrong in two ways. It set a *range* against a *difference of means*, which
are different statistics: a range grows with n, so the same machine that gives
0.011 over five runs gives more over twelve, and the comparison moves without the
hardware moving. And the five runs behind the 0.011 were pre-correction — twelve
windows with the final one clamped to 9.04 to 9.11 s, the signature the corrected
tool removed — so the floor and the figures compared against it did not come from
the same tool.

The replacement is stronger, not weaker: 7.1 sigma on a valid comparison against
a ratio of two incommensurable statistics. The n-dependence is also quantified
rather than assumed — over all C(12,5) subsets a
five-run range averages 0.0094 and 0.0101 on the two boxes against full twelve-run
ranges of 0.0142 and 0.0163, so any range quoted without its n is uninterpretable.

**Most of that within-machine spread is the opening runs, not noise, and a 90 s
cooldown does not remove it.** The twelve runs are not exchangeable:

    experiments  0.9831 0.9781 0.9751 0.9694 0.9696 0.9703 0.9700 0.9711 0.9689 0.9713 0.9725 0.9718
    inference1   0.9721 0.9642 0.9618 0.9598 0.9594 0.9571 0.9568 0.9589 0.9568 0.9562 0.9558 0.9569
    inference2   0.9385 0.9479 0.9469 0.9500 0.9481 0.9472 0.9495 0.9507 0.9497 0.9502 0.9491 0.9508

On every box the first run is the extreme one and the series settles after about
three. Dropping those three cuts the standard deviation on all three machines —
0.0042 to 0.0012, 0.0047 to 0.0015, 0.0033 to 0.0012 — so the opening runs are
reliably unrepresentative and the sd over all twelve is measuring them as much as
it measures run-to-run variability.

**They are not unrepresentative in a consistent direction, which kills the
obvious explanation.** An earlier version of this paragraph called it a warm-up
and said each soak starts a little warmer than the last. That fits `experiments`
and `inference1`, whose first three run 0.008 and 0.009 *above* their last six
with rank correlations of −0.28 and −0.89. It does not fit `inference2`, whose
first run is its **lowest** and whose first three run 0.006 *below* the last six,
rank correlation **+0.74**. A warming story predicts one direction; two boxes go
one way and the third goes the other. We report the effect, drop the opening runs
because doing so demonstrably tightens every box, and do not claim to know why.

Excluding the settling sharpens the result without moving it:

| | between-machine difference | se | sigma | within-machine sd |
| --- | ---: | ---: | ---: | ---: |
| all 12 runs | 0.0130 | 0.0018 | 7.1 | 0.0042 / 0.0047 |
| dropping run 1 | 0.0131 | 0.0011 | 11.4 | 0.0028 / 0.0026 |
| dropping runs 1–3 | 0.0130 | 0.0006 | **20.7** | 0.0012 / 0.0015 |

**The effect is invariant to the treatment; only the noise estimate moves.** The
difference between the machines stays at 0.0130 to 0.0131 in all three rows while
the within-machine sd falls by a factor of three and a half. We report 7.1 sigma
as the headline because it makes no choice about which runs to keep, but the
settled rows say the steady-state repeatability of this measurement is about
0.0013 rather than 0.0045.

This has a consequence beyond this section. **Any back-to-back soak series in
this paper carries the same drift**, so a protocol that compares its first run
against its later ones — which is what §3.3's cold-start study does by
construction — measures the drift along with whatever else it intends to. §3.3's
effect is 2 to 4 percentage points against this 0.8 to 0.9, so it survives, but
by a smaller margin than it appears to.

**The cross-chip sustained comparison does NOT clear that floor reliably, and an
earlier version of this paper claimed it did.** It said the cross-chip effect
exceeded both sources of variation by roughly an order of magnitude, computed from
a single M5 Max figure of 0.837. Repeated, that figure ranges 0.846 to 0.952 for
the same configuration (§3.3), so the gap against the M4 Pro mean ranges 0.129
down to **0.023**, and the low end is comparable to the between-machine difference
itself. The claim is withdrawn.

What survives is the comparison the confound cannot touch. The ANE gave back at
most 0.14% on four machines across two chips in 41 of 42 runs, with the single
`whisper` exception of §3.3, so its flat line is not a number that could have been
produced by a lucky starting state. And the absolute
throughputs are not close: M4 Pro GPU sustains 172 to 177 img/s while the M5 Max
sustains 762 to 888, which no amount of thermal-state variation reconciles.

**Which measurement is more repeatable depends on the chip**, and an earlier
version of this paragraph got that wrong in both the numbers and the conclusion.
It read "burst measurements are far more repeatable than sustained ones",
comparing burst *percentages* against 0.106 — an absolute fraction move, on the
other machine, from the no-cooldown set §3.3 shows is confounded. Three
mismatches in one sentence.

Compared like with like, as a relative standard deviation across repeats:

| | burst, M4 Pro | burst, M5 Max | 120 s soak, settled |
| --- | ---: | ---: | ---: |
| median cell | 0.05% | 0.97% | — |
| range across cells | 0.00 to 1.72% | 0.05 to **8.42%** | 0.12 to 0.19% |

On the M4 Pro the burst is the tighter measurement, by roughly a factor of three
at the median. **On the M5 Max it is not: its burst spreads reach 8.42% where a
settled soak on identical hardware holds 0.12 to 0.19%.** The worst M5 Max burst
cell spans 22.06% peak to trough. So the advice is not "prefer bursts"; it is
that a burst is cheap and tight on a part that does not throttle, and is the
*less* stable instrument on one that does — which is the same chip-dependence
every other result in this paper has.

The soak figures are the settled runs of §3.6, dropping the three warm-up runs.
Including them the relative sd is 0.42 to 0.51%, still below the M5 Max burst
and above the M4 Pro's. Reported both ways because which one is fair depends on
whether the protocol includes a warm-up, and ours now does not.

---

## 4. Threats to validity

**Burst figures read above the steady state, and asymmetrically.** A burst is over
in about two seconds, so it samples the opening rate, and §3.3 shows the opening
rate sits slightly above the plateau the same run settles to. Every burst number
in §3.1 and §3.2 is measured from a cold start, and the cold excursion documented
in §3.3 has not begun that early, so the burst misses it entirely — this is not
the 2 to 3% dip reappearing, it is a much smaller offset between the first seconds
and the plateau. Comparing each burst median against the steady state of the
**third, warm run** of the corresponding 600 s series — mean of its last six
windows — bounds it:

| model | GPU inflation | ANE inflation | ANE run from |
| --- | ---: | ---: | --- |
| `bert` | +0.20% | +0.38% | `experiments` |
| `resnet50` | +0.74% | −0.08% | `inference1` |
| `siglip` | +1.08% | +0.18% | `inference1` |
| `whisper` | **+2.94%** | +0.16% | `experiments` |
| `mobilenet` | +2.01% | −0.39% | `inference1` |

The ANE is within 0.4% either way, which is what an engine whose opening rate and
plateau are the same thing looks like. The GPU is inflated by up to 3%, and by
more than 2% on the two models — `whisper` and `mobilenet` — whose GPU dip §3.5
finds largest.

**This table was rebuilt, and the previous one is worth describing because of how
it failed.** It printed four models rather than five, and its ANE column mixed
run indices: `siglip` and `whisper` came from the third run of their series while
`bert` and `resnet50` came from the **first**. §3.6 shows the first run is the
extreme one on every machine measured, so two cells were anchored to the least
representative run available and two were not. Every cell now uses run three. The
GPU column is unchanged to within 0.05 pp because it was already consistent; the
ANE column moves, most visibly `resnet50` from −0.25% to −0.08% and `bert` from
+0.24% to +0.38%.

`mobilenet` was missing entirely, and it is not a neutral omission: §3.3 measures
its ANE floor at 0.69 to 0.77%, the largest of the five, so the four rows shown
excluded the model most likely to stress the claim. Its GPU inflation of +2.01%
is the second largest.

The ANE column still cannot be made to match the GPU column's *machine*: there
are no 600 s ANE runs for `bert` or `whisper` on `inference1`, so those two are
from `experiments` and the column names its box per row rather than implying one.

**How the previous version was traced, since the method is reusable.** Its
caption said the same thing this one does, but reading it literally reproduced
nothing: each burst median against the mean of the last six windows of *the*
corresponding soaks gave 1.77% for `whisper`'s GPU cell against a printed 2.98%,
and eighteen further combinations of plateau definition, box selection and soak
duration all left that cell at least 0.86 pp out. It was recorded as
unreproducible and the numbers were left alone rather than overwritten with a
reconstruction that could not be validated.

Inverting the question found it. Rather than guessing a method and comparing,
solve each printed cell for the plateau it *implies* and ask which committed
measurement equals that. **All four GPU cells turned out to be a single run: the
third, warm run on `inference1`**, mean of its last six windows.

| | printed | from `dip3-inference1` |
| --- | ---: | ---: |
| `bert` | +0.21% | +0.20% |
| `resnet50` | +0.72% | +0.74% |
| `siglip` | +1.03% | +1.08% |
| `whisper` | **+2.98%** | **+2.94%** |

That is why averaging failed: the table never averaged. The identification is
strong rather than coincidental. A 0.05% match is routine in general — across the
eight cells, 94 of 210 candidate statistics land inside that window — but
`whisper`'s implied plateau matches only **two of thirty**, and both are that one
file. Two cells discriminate and six do not, which is why the six alone would
have identified nothing.

**Its ANE column had not come from the same runs, which is what made the rebuild
necessary rather than merely tidy.** `siglip` and `whisper` matched the third run
as the GPU cells did, but `bert` and `resnet50` matched `aned1`, the **first**
run of their series. §3.6 shows the first run is the extreme one on every machine
we have measured, so two of the four ANE cells were anchored to the least
representative run available and two were not. That is why the ANE column moved
when it was rebuilt and the GPU column did not.

The `bert` identification is firm — two of nine candidates, both `aned1`.
`resnet50`'s is weaker: seven of eighteen candidates fall within the same window,
so "run 1" is suggestive there rather than established. Either way the column was
not built on one rule.

**This does not threaten §3.1 or §3.2, and it makes their M4 Pro results
conservative.** The smallest margin in §3.1 is `siglip` at 1.14x, which 3%
cannot invert, and the default costs in §3.2 run from 1.18x to 5.18x, where 3% on
one term moves 5.18x to about 5.03x. But the direction matters: the bias flatters
the GPU, so the M4 Pro cases where the *ANE* wins are understated, and the M5 Max
cases where the GPU wins are slightly overstated. This is a second asymmetry on top
of the Python call overhead noted in §2.5, and the two push in opposite directions.

**Measurement hygiene, and a retraction.** An earlier version of this paper
reported that the ANE gets *slower* on the M5 Max on four of five architectures,
called it unexplained, and built a residency investigation around it. That result
was an artefact. Those M5 Max ANE figures were taken while the machine was running
other work, and they were low by roughly a factor of two. The paper flagged those two cells as unreliable
and used them anyway, which was the error.

**We described the repeat spread as the tell, and it is not one.** The claim was
that 17.2% on `mobilenet` and 12.3% on `bert` stood out against "0.0% to 1.3% for
every clean measurement in this study". Recomputed, the clean files span 0.0 to
3.3% on the M4 Pro and **0.1 to 22.1%** on the M5 Max — and the widest cell in
the whole study, `mobilenet` under `ALL` at 22.1%, comes from the *clean* re-run.
A spread that large is therefore compatible with an uncontended machine on this
chip, so it could not have identified the contaminated cells and did not. What
identified them was re-running them idle.

Re-measured on an idle machine, all five architectures are **faster** on the M5
Max ANE, by 1.09x to 1.18x, with spreads of 0.4% to 0.8%, and the values reproduce
across independent runs. The regression does not exist. Every ANE figure in §3.1
is from the clean re-measurement (`results/zoo-m5max-v2.json`); the contaminated
file is retained as `results/zoo-m5max.json` so the correction is auditable.

Two further defects in the harness were found at the same time and fixed. The
sweep fed float32 to every model regardless of declared input dtype, so `siglip`
paid an fp32-to-fp16 conversion per call and `bert` received token ids of zero
(uniform [0,1) cast to int32). And the reported rate multiplied by a
command-line `--batch` that was never compared against the model's actual input
batch, which would have scaled every figure by declared/actual had they ever
disagreed. They did not, verified across all five models. `tools/sweep.py` now
feeds the declared dtype and refuses a mismatched batch.

None of this changed either headline. The inversion is unchanged at three of five
architectures, and the default's cost moved from 1.19x-5.16x to 1.18x-5.18x. It
changed a third claim from a finding into an artefact, which is why it is here
rather than in a footnote.

**Two chips.** M4 Pro and M5 Max. Nothing here establishes behaviour on M1/M2/M3,
base or Ultra variants, or A-series parts. The central claim is precisely that
results do not generalise across chips, which applies to these results too.

**Chassis is confounded with chip for the sustained results.** The M4 Pro
machines are desktops with stock cooling; the M5 Max is a laptop with a
non-stock fan curve. §3.3 gives two arguments that the sustained differences are
per-engine power rather than enclosure, and we consider the direction
established. But no specific value belongs to the M5 Max at all (§3.3); the ones
we measured range 0.846 to 0.952 for the same configuration, and they belong to
this machine with this fan
curve. Separating them requires an M5-family desktop or an M4 Pro laptop,
neither of which was available. This does **not** affect §3.1 or §3.2, which are
within-machine ratios at burst.

**Small repeat counts on the added architectures.** `n=3` for the four models
added in §3.1, against `n=5` for `siglip`. This is adequate for effects of
4x to 17x. It was previously called thin for "the two high-variance M5 Max ANE
cells", quoting medians of 1897.7 for `mobilenet` and 454.6 for `bert`. Those are
the retracted contended figures; the clean re-run gives 3602.2 and 822.5, and its
ANE repeat spreads are 0.76% and 0.62%, so neither cell is high-variance any more.
The GPU margins on the two are 2.89x and 4.28x, not "more than 5x". The caveat
that remains is the plain one: `n=3` is fewer repeats than `n=5`.

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
decoding with a KV cache, which is both the hardest case and the structure
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
cost-weighted residency, meaning residency weighted by `MLComputePlan`'s per-operation
estimated cost rather than by operation count, so one stranded matmul is not
concealed by fifty cheap ANE operations, and a pass/fail threshold usable in CI.

**ANE-oriented model rewriting.** `apple/ml-ane-transformers` and the associated
Apple ML research article *Deploying Transformers on the Apple Neural Engine*
[[4]](#ref4) establish the (B, C, 1, S) layout, Linear to Conv2d substitution and
chunked attention used by the ANE-rewritten variant in this repository. That line of work
optimises a model *for* the ANE and reports the resulting speedup. It does not
address whether the ANE is the correct target on a given chip, which is the
question here, and §3.1 finds that on an M5 Max it frequently is not. We note in
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
moves to Swift. The Python conversion package has no placement surface at all.
As with `ALL`, it is a preference, and nothing reports what was honoured.

**None of it can be measured on this hardware.** `canImport(CoreAI)` is false
under Swift 6.3.3, and the framework is absent from `/System/Library/Frameworks`,
from PrivateFrameworks, and from the Xcode 26.6 SDK, which does carry
`CoreML.framework` and `FoundationModels.framework`. Apple's own package declares
`platforms: [.macOS("27.0")]` and requires Xcode 27.0+. This fleet is on macOS
26.5.1 and 26.6. So every Core AI statement in this paper is a reading of Apple's
published source, and no throughput, residency or honoured-placement number for it
exists here or can be taken until the OS ships.

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

builds **the two `siglip` variants**, sweeps the three compute-unit settings on
them, runs the sustained soaks, and prints both tables ready to paste. It does
not invoke `models/convert_zoo.py`, so it does not reproduce the four added
architectures of §2.3 — those were built and swept separately, and their raw
JSON is committed rather than regenerated by this script. Raw per-run JSON for every figure
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

Every number in this paper is derived from JSON committed in this repository,
with one exception we name rather than leave for a reader to find: the **97 °C**
die temperature in §2.1, §3.3 and §4 is not in any committed file. The soak JSON
records the `notify(3)` thermal-pressure level and nothing else -- there is no
temperature field anywhere in `results/` -- because we declined to wire
`powermetrics` and its root requirement into the tool (§3.3). That figure was
read off the fan-control utility by hand and should be treated as context for
the fan curve, not as a measurement this repository backs.
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
Quoting: "The 10-core GPU features a dedicated Neural Accelerator in each core, enabling
GPU-based AI workloads to run dramatically faster"; "over 4x peak GPU compute
compared to M4"; "an improved 16-core Neural Engine".

<a id="ref2"></a>[2] Apple. *Apple introduces new Mac Studio with M5 Max and M5
Ultra.* Apple Newsroom, 25 August 2026.
<https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/>
Quoting: "up-to-40-core GPU"; "Neural Accelerators built into each core"; "up to 614GB/s
of unified memory bandwidth"; "3.9x faster than M4 Max".

<a id="ref3"></a>[3] Apple. *Apple introduces M4 Pro and M4 Max.* Apple Newsroom,
30 October 2024.
<https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/>
Quoting: M4 Pro up to 20-core GPU, 273GB/s unified memory bandwidth; "enhanced machine
learning (ML) accelerators in the CPUs".

<a id="ref4"></a>[4] Apple Machine Learning Research. *Deploying Transformers on
the Apple Neural Engine.* 2022.
<https://machinelearning.apple.com/research/neural-engine-transformers>
Quoting: the
(B, C, 1, S) layout, Linear to Conv2d substitution and chunked attention used by
the ANE-rewritten variant in this repository, and the reference implementation
`apple/ml-ane-transformers`.

<a id="ref5"></a>[5] Apple. *Core AI.* Apple Developer Documentation, 2026.
<https://developer.apple.com/documentation/coreai>
and the reference recipes at
<https://github.com/apple/coreai-models>, whose
`swift/Sources/CoreAIShared/Runtime/ModelStructure.swift` selects
`preferredComputeUnitKind` from model structure.
