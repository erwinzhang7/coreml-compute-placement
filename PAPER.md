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
explicit placements. On the M5 Max it is free at peak, which we report as
prominently. Third, **peak throughput does not predict sustained throughput**:
over a two-minute soak the ANE gives back at most 0.14% of its peak in 44 of 45
soaks on both chips and four separate machines, while the GPU gives back 0.8 to
13.2% on an M4 Pro and 4.8 to 16.3% on an M5 Max across 62. Those 44 ANE soaks all
sustain better than the best of the 62 GPU soaks. The forty-fifth is a
`whisper` run that gave back 2.52% after reaching the same peak as seven runs that
did not, one of them a repeat on the same machine, which we report and do not
explain in §3.3. A benchmark of a few seconds, which is the standard form, cannot
see any of this.

We also report that the sustained *fraction* is confounded by the thermal state a
run starts in on a throttling chip, which makes the ANE's insensitivity to that
confound the most robust result in this section.

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
retains placement control as `SpecializationOptions(preferredComputeUnitKind:)`,
a single *preferred* unit rather than an allow-list, and whose published
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
correctly with a maximum *relative* error of 460, which is why relative
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
across repeats and the spread (max - min) / median. Batch 16 throughout.
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
"improved" [[1]](#ref1). The GPU improves by between three and six times. Any
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

On the M5 Max the default is effectively free: it tracks the GPU to within 2% on
every architecture, because the GPU is the right answer on that chip and the
runtime selects it. **We report this as prominently as the negative result.**

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

**What survives all of this is the ANE.** It measured 1.000 in every run, on every
box, on both chips, regardless of starting state. That insensitivity is exactly
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

**Thermal pressure read `nominal` in every window of every soak reported here, and
that establishes nothing.** Thermal pressure is the operating system's signal that
it is about to shed user work, not a die temperature; the M5 Max was held near
97 °C by an aggressive fan curve and still reported nominal throughout. The
reading is informative in one direction only: above nominal is evidence of
trouble, nominal is not evidence of its absence.

Two observations argue the decline is a per-engine power limit rather than
enclosure cooling. The ANE holds 1.000 **on the same machine, in the same session,
at the same die temperature** as the GPU run that lost 15%; a cooling limit that
affects one engine and not the other is not a cooling limit. And the M5 Max GPU is
measured at 39.2 TFLOPS against the M4 Pro's 6.4 (§3.4), so an engine performing
several times the work per second draws proportionally more power and approaches a
ceiling the smaller GPU never reaches.

The ANE result is not specific to `siglip`. **28 soaks across five architectures
and three physically distinct M4 Pro machines**, corrected tool, 90 s cooldown
between runs:

| model | ANE (n) | GPU (n) | GPU peak stability |
| --- | --- | --- | ---: |
| `siglip` | 0.9996 to 1.0000 (10) | 0.949 to 0.992 (21) | 0.76% |
| `resnet50` | 0.9988 to 1.0000 (8) | 0.962 to 0.989 (10) | 0.51% |
| `mobilenet` | 0.9990 to 0.9998 (8) | 0.963 to 0.984 (10) | 0.84% |
| `bert` | 0.9997 to 1.0000 (8) | 0.952 to 0.984 (8) | 0.08% |
| `whisper` | **0.9748** to 1.0000 (8) | 0.868 to 0.967 (9) | 0.65% |
| **all** | **0.9748 to 1.0000 in 42** | **0.868 to 0.992 in 58** | |

**In 41 of 42 M4 Pro soaks the ANE gave back at most 0.12% of its peak, better
than the best of 58 GPU soaks, which gave back 0.82%.** That holds across a
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
should treat "the ANE does not degrade" as holding in 18 of 19 measurements, not
as a law.

We report the ANE column to four decimals deliberately. At three, every one of
these sixteen prints as `1.000`, and an earlier draft of this table read
"1.000 in 10 of 10" and claimed the ANE "returned exactly 1.000". That was a
rounding artefact: only 7 of 16 are exactly 1.0000. The distinction matters
because "exactly" invites a mechanism that does not exist, a hard clamp at peak,
where what the data supports is the weaker and sufficient claim that ANE
throughput does not measurably decay over two minutes.

The GPU peaks are stable to between 0.08% and 0.84% across repeats and across
boxes, so none of the M4 Pro variation is the thermal-state confound of §3.3;
these are genuine run-to-run differences of 0.02 to 0.05 in the fraction.

`whisper` declines most (0.938 to 0.955) and is the most arithmetic-dense model in
the set, which is the direction the power explanation in §3.4 predicts, though
with five models this is a consistent observation rather than a demonstrated
relationship.

The default sits with the GPU rather than below it. On this chip it sustains at
least as well as the pure GPU placement on four of five architectures, which is
the opposite of what an earlier version of this paper claimed (§4).

This has a practical consequence the burst tables alone cannot support: **for a
continuously loaded service the ANE's advertised rate is the rate you get, and no
other unit's is.** A capacity plan built from burst numbers will over-provision
the GPU's contribution by 3 to 16% and the default's by up to 28%.

We did not measure beyond 120 s, so nothing here says where the GPU's decline
stops. On the M4 Pro it had flattened by the end of the soak. On the M5 Max it had
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

Every cross-chip GPU gain in §3.1 sits **at or between the two probe bounds**:
2.89x for `mobilenet` up to 6.31x for `bert`, against 1.27x bandwidth-bound and
6.17x arithmetic-bound. `bert` slightly exceeds the matmul probe, by 2%, which is
inside the probe's own repeat spread and so is not evidence of anything beyond
the arithmetic ceiling. That is the expected ordering if each model sits somewhere
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

**A falsifiable prediction.** If the mechanism is the per-GPU-core matrix
accelerator, then the inversion should appear on **any** M5-family part and on no
M4-or-earlier part, independent of tier. A base M5, an M5 Pro or an M5 Ultra
should rank the units as the M5 Max does here. An M4 Max, which has twice the
M4 Pro's GPU but still no per-core matrix unit, should rank them as the M4 Pro
does, and should *not* invert merely because it has more GPU cores. We do not have
those machines. This is the cheapest experiment that could falsify the
explanation, and we invite it.

The ANE's much smaller movement is consistent with both chips carrying the same
16-core Neural Engine, though it does not explain the regression noted in §3.1.

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
(2.81x to 6.13x) exceeds the largest M4 Pro ANE/GPU ratio (1.410). The inversion is
carried by the GPU, which is the side with the documented mechanism.

**The magnitudes for the two non-inverting models are not clean.** `mobilenet`
"moves 4.5x" and `whisper` "moves 6.6x" between chips. With the ANE held flat
those become 2.81x and 3.98x. The published figures are inflated roughly 1.6x by
the ANE regression, which we cannot attribute (§3.1). Read the direction, not the
multiple.

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
Update M4 Pro #2 to 26.6, keep M4 Pro #1 on 26.5.1, and rerun both sweeps. That
gives a third cell and separates OS build from silicon for every result in the
paper. We have not done it.

### 3.6 Measurement reproducibility

Most single-machine benchmark results do not establish how much they move for
reasons other than the variable under study. Two physically identical M4 Pro
machines and five repeats of the noisiest metric give that floor directly.

Sustained fraction, GPU, `siglip`, 120 s:

| quantity | value |
| --- | ---: |
| M4 Pro #2, five repeats | 0.949, 0.950, 0.951, 0.957, 0.960 (mean 0.954, sd 0.0045) |
| within-machine range | **0.011** |
| between-machine difference (#1 at 0.973) | **0.019** (1.7x the within-machine range) |
| M4 Pro mean 0.975 vs M5 Max 0.846 to 0.952 | **0.023 to 0.129**, see below |

**Identical hardware is not identical.** The between-machine difference is 1.7x
the run-to-run range, so a single sustained GPU figure should not be quoted to
three decimals. Three separate M4 Pro boxes measured with the corrected tool give
0.987, 0.979 and 0.960, and their absolute last-window rates agree to 2.7%.

**The cross-chip sustained comparison does NOT clear that floor reliably, and an
earlier version of this paper claimed it did.** It said the cross-chip effect
exceeded both sources of variation by roughly an order of magnitude, computed from
a single M5 Max figure of 0.837. Repeated, that figure ranges 0.846 to 0.952 for
the same configuration (§3.3), so the gap against the M4 Pro mean ranges 0.129
down to **0.023**, and the low end is comparable to the between-machine difference
itself. The claim is withdrawn.

What survives is the comparison the confound cannot touch. The ANE measured 1.000
on four machines across two chips in every run, so its zero decline is not a
number that could have been produced by a lucky starting state. And the absolute
throughputs are not close: M4 Pro GPU sustains 172 to 177 img/s while the M5 Max
sustains 762 to 888, which no amount of thermal-state variation reconciles.

**Burst measurements are far more repeatable than sustained ones**, and this is
the practical lesson. M4 Pro burst spreads run 0.0 to 3.7% across every model and
unit; M5 Max burst spreads run 0.3 to 6.5%. Sustained fractions on the same
hardware move by 0.106 between runs. If a study can answer its question with a
burst measurement, it should.

---

## 4. Threats to validity

**Measurement hygiene, and a retraction.** An earlier version of this paper
reported that the ANE gets *slower* on the M5 Max on four of five architectures,
called it unexplained, and built a residency investigation around it. That result
was an artefact. Those M5 Max ANE figures were taken while the machine was running
other work, and they were low by roughly a factor of two. The repeat spreads at
the time were the tell: 17.2% on `mobilenet` and 12.3% on `bert` against 0.0% to
1.3% for every clean measurement in this study. The paper flagged those two cells
as unreliable and used them anyway, which was the error.

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
4x to 17x and thin for the two high-variance M5 Max ANE cells noted in §3.5, where
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
