# Sustained soaks

Burst benchmarks measure what a compute unit can *reach*. These measure what it
*holds*. They are the same model and batch as the committed sweeps, so the two
are directly comparable.

## Both chips, sustained fraction (last window / best window)

Three machines, including **two identical M4 Pro Mac minis**, which is what makes
the cross-chip numbers readable rather than suggestive.

| compute unit | M4 Pro #1 | M4 Pro #2 | M5 Max | machine floor | chip gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CPU_AND_NE` (ANE) | 1.000 | 1.000 | 0.999 | 0.000 | 0.001 |
| `CPU_AND_GPU` | 0.973 | 0.949 | **0.837** | 0.024 | **0.124 — 5x the floor** |
| `ALL` (default) | 0.998 | 1.000 | **0.721** | 0.002 | **0.278 — 132x the floor** |

The two M4 Pros are the same chip in the same chassis on mains, so the spread
between them is the reproducibility floor of this measurement — how much it moves
for reasons that are not the chip.

The GPU is the noisy one, so it was repeated **five times on one box**:

| | GPU sustained |
| --- | --- |
| M4 Pro #2, n=5 | 0.949, 0.950, 0.951, 0.957, 0.960 (mean 0.954, sd 0.0045) |
| within-machine range | **0.011** |
| between-machine difference (#1 0.973 vs #2 mean 0.954) | **0.019** — 1.7x the within-machine range |
| M4 Pro mean 0.963 → M5 Max 0.837 | **0.126** — **11x** within-machine, **7x** between-machine |

Two things fall out. The two minis genuinely differ a little — 0.019 against a
run-to-run range of 0.011 — so identical hardware is not identical, and a single
GPU soak should not be quoted to three decimals. And the cross-chip gap is an
order of magnitude larger than either source of variation, so it is not machine
noise.

The ANE's 0.001 does **not** clear the floor and should be read as "the ANE holds
its rate on both chips", not as a difference.

**The ANE holds. The GPU and the default do not, and only on the M5 Max.** That
extends [finding 1](../../README.md#1-the-anegpu-ratio-inverts-within-one-chip-family)
onto a second axis: the burst numbers say the *ranking* of the units differs
between chips, and these say the *durability* does too.

### Why this reads as the chip, not the cooling

The M5 Max here is a laptop with a custom fan curve and its die near 97 °C, so
cooling is the obvious alternative explanation. Two things argue against it being
the driver:

- **The ANE does not sag on that same machine.** 0.999 over two minutes, in the
  same enclosure, at the same die temperature, in the same session as the GPU
  run that lost 16%. A cooling limit that only touches one engine is not a
  cooling limit; it is a per-engine power limit.
- **The mechanism is downstream of the GPU being fast.** `tools/probe_gpu.py`
  measures the M5 Max GPU at 39.2 TFLOPS against the M4 Pro's 6.4 — 6.17x on
  matmul. An engine doing 4.7x the work per second draws far more power, so it
  reaches a ceiling the M4 Pro's GPU never approaches. The M4 Pro GPU has no
  headroom problem because it has no headroom to spend.

What the chassis *can* still affect is the exact number: 0.837 is this laptop
with this fan curve. A stock MacBook Pro, or an M5 Max in a desktop enclosure,
could land elsewhere. The **direction and rough size** survive, because they
clear the machine floor by 5x and 132x; the third decimal does not.

None of this touches finding 1's burst inversion, which is a within-machine ratio
on each chip and independently explained by the matmul probe.

### The M4 Pro soak reproduces the committed sweep

Run months later, on a freshly built venv, from a clean clone:

| unit | sweep (30-iteration burst) | soak, first 10 s window |
| --- | ---: | ---: |
| ANE | 204.4 | 204.5 |
| GPU | 178.8 | 178.8 |
| `ALL` | 172.2 | 172.9 |

On the M5 Max the same comparison does *not* line up — 1085.7 burst against
1020.7 in the first soak window — and that is the finding rather than a
discrepancy. The M5 Max GPU declines fast enough that even a 10 s window has
already lost some of the peak a 2 s burst captures. On the M4 Pro there is
almost nothing to lose, so the two agree.

### Conditions, and what is still unseparated

Both M4 Pros are Mac minis on mains with stock cooling and nothing else running
(each soak logged `0 avm procs` and no resident chat model at start). The M5 Max
is a MacBook Pro with a custom fan curve.

An earlier version of this file said the chip/chassis pair was unseparated and
left it there. That was too weak a reading: the two-mini floor plus the ANE's
flat 0.999 on the hot laptop are evidence, and they point the same way. What
remains genuinely unseparated is narrower — **how much of the 0.124 and 0.278 is
the enclosure rather than the silicon.** Settling that needs an M5 Max in a
desktop chassis, or an M4 Pro laptop, and neither exists here.

It does not need settling for the claim being made, which is that the units'
durability is ordered differently on the two chips.

## M5 Max, 120 s, batch 16, SigLIP-base-224 vision tower

| compute unit | peak img/s | last img/s | sustained | shape |
| --- | ---: | ---: | ---: | --- |
| `CPU_AND_NE` (ANE) | 233.1 | 232.8 | **0.999** | flat for 12 windows |
| `CPU_AND_GPU` | 1020.7 | 854.3 | **0.837** | falls ~16% in 20 s, then plateaus |
| `ALL` (default) | 1006.5 | 725.6 | **0.721** | still falling at 120 s |

**The ANE gives up nothing. The GPU gives up a sixth. The default gives up more
than a quarter and had not stabilised when the run ended.**

The GPU's advantage over the ANE therefore shrinks with the length of the
measurement:

| | GPU / ANE |
| --- | ---: |
| peak (this soak's first window) | 4.38x |
| sustained (last window) | **3.67x** |
| README headline, 30-iteration burst | 4.66x |

Nothing here contradicts the burst numbers. A 30-iteration burst is over in about
two seconds, and the decline has barely started by then — which is exactly why
the burst figure is higher than even the first 10 s window. The two measure
different things, and only one of them is what a long-running job gets.

## Conditions, because they change the answer

- **AC power.** Recorded per run in the JSON. An earlier GPU soak was discarded:
  the machine was plugged in mid-run and the guard in `thermal_soak.py` caught the
  battery→AC transition. A 60 s GPU soak on battery gave 0.879, but that is **not**
  comparable to the 0.837 above — the AC run is 120 s, and sustained fraction falls
  with soak length by construction, since the denominator is the best window and
  the numerator is the last one. Comparing AC against battery needs equal
  durations and has not been done.
- **Custom fan curve, die pinned at ~97 °C.** This is not a stock machine. A
  stock MacBook Pro will ramp its fans differently and may land somewhere else
  entirely. Sustained numbers are more sensitive to cooling than burst numbers
  are, so this caveat matters more here than anywhere else in the repo.
- **Thermal pressure read `nominal` in every window of every run, and that means
  nothing here.** macOS thermal pressure signals that the OS is about to shed
  user work; it is not a die temperature. A machine held at 97 °C by aggressive
  fans reports nominal throughout while the GPU clocks itself down anyway. Read
  the level in one direction only: above nominal is proof of trouble, nominal is
  not proof of its absence.
- Single process, single model, no other GPU client. Python driving overhead is
  in every figure, as it is in the burst numbers.

## What is not established

The **cause** of the GPU and `ALL` decline. Thermal pressure cannot settle it for
the reason above, and the die temperature was not sampled during these runs —
that needs `sudo powermetrics --samplers smc`, which is deliberately not wired
into `thermal_soak.py` so the tool stays runnable without root. Power limiting,
clock ramp-down after a boost window, and thermal limiting at 97 °C are all
consistent with what was measured, and they are not separated.

What *is* established is the operational fact: over two minutes, the ANE holds
its rate and the GPU and the default do not.

## Reproducing

```sh
python tools/thermal_soak.py MODEL --units CPU_AND_NE  --seconds 120 --window 10 --out out.json
python tools/thermal_soak.py MODEL --units CPU_AND_GPU --seconds 120 --window 10 --out out.json
python tools/thermal_soak.py MODEL --units ALL         --seconds 120 --window 10 --out out.json
```

Windows are written as they complete, so an interrupted soak keeps everything up
to the last one. If your chip or your cooling ranks these differently, that is
the result worth reporting.
