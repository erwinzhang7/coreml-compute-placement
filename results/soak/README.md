# Sustained soaks

Burst benchmarks measure what a compute unit can *reach*. These measure what it
*holds*. They are the same model and batch as the committed sweeps, so the two
are directly comparable.

## Read this before the tables

Two things below were wrong for a while and are now corrected in place.

1. Every soak taken before commit 54ac0a0 used a tool that called `pmset` between
   windows, clamped its final window to the deadline, and could publish an
   interrupted window as the final value. All figures here are from the corrected
   tool.
2. **On the M5 Max the sustained fraction depends on how warm the machine started**,
   so a single number for that chip is not meaningful. Details below.

## M4 Pro, three physically distinct machines, corrected tool

`siglip`, batch 16, 120 s, mains, each box otherwise idle.

| compute unit | box 1 | box 2 | box 3 | last-window img/s |
| --- | ---: | ---: | ---: | --- |
| `CPU_AND_NE` (ANE) | 1.000 | 1.000 | 1.000 | 204.4, 204.5, 204.4 |
| `CPU_AND_GPU` | 0.987 | 0.979 | 0.960 | 176.6, 174.9, 171.8 |
| `ALL` (default) | 0.996 | 0.994 | 0.993 | 172.4, 171.9, 171.5 |

**The ANE holds its peak to within 0.01% on all three.** The GPU gives back 2 to
4%. The default gives back under 1%, so on this chip it sustains slightly
*better* than the pure GPU placement.

The three ANE cells print as `1.000` at the precision of this table. They are not
exactly 1: at four decimals they are 1.0000, 1.0000 and 0.9999. An earlier draft
read "holds its peak exactly", which is a rounding artefact rather than a result.

Absolute last-window rates agree to 2.7% across three separate machines. That is
what this measurement's reproducibility looks like on a chip that does not
throttle.

## M5 Max, and why there is no single number

| run | unit | peak | last img/s | sustained |
| --- | --- | ---: | ---: | ---: |
| 1 | ANE | 235.3 | 235.2 | 1.000 |
| 1 | GPU | 1049.7 | 887.8 | 0.846 |
| 2 | GPU | 1041.8 | 882.4 | 0.847 |
| 3 | GPU | 799.9 | 761.7 | **0.952** |
| 1 | `ALL` | 1018.6 | 875.8 | 0.860 |
| 2 | `ALL` | 1030.3 | 800.4 | 0.777 |
| 3 | `ALL` | 784.7 | 773.6 | **0.986** |

Sorted by starting peak the fraction is monotone in both units. A machine that
starts cool reaches a high peak and gives a lot back; a machine already warm
starts low, gives little back, and **scores better on a statistic that is supposed
to mean "holds its rate"**. Run 3 of each pair followed five earlier soaks with 15
to 20 s between them, and its peak is a quarter lower because the box never
recovered.

So `sustained fraction` on this chip measures the compute unit *and* the state it
started in. Compare absolute last-window rates across runs instead, and give a
throttling machine a real cooldown before each soak.

### Two claims withdrawn

**"The default is the worst sustainer on the M5 Max."** It was 0.721 against the
GPU's 0.837 with the old tool. Corrected and repeated, `ALL` is 0.777, 0.860,
0.986 and the GPU is 0.846, 0.847, 0.952. Indistinguishable, and the ordering
flips depending on which runs are compared.

**"The cross-chip gap is 5x to 132x the machine floor."** That was computed from
single M5 Max figures. With the range, the GPU gap against the M4 Pro mean runs
from 0.129 down to 0.023, and the low end is comparable to the between-machine
difference itself.

### What survives

The ANE, in 21 of 22 soaks. Across four machines and two chips those 21 measured
0.9986 to 1.0000, giving back at most 0.14% of peak regardless of starting state,
which is what an engine that does not degrade should look like, and it is that
insensitivity that makes it immune to the confound above. All 21 sustain better
than the best of 41 GPU soaks (0.9918).

**The twenty-second is a `whisper` ANE soak on inference2 at 0.9748**, and it is
not dismissed. It ran flat at 56.3 img/s for seven windows, then fell monotonically
to 54.9 and plateaued. It is not the contended-peak artefact that cost us the M5
Max numbers: contention lowers the peak, and this run reached 56.34 against 56.34
and 56.35 for the two whisper ANE runs on other boxes that held their rate. The
engine reached full speed and then lost 2.5% of it.

We cannot say why, because **`thermal_soak.py` records the machine and the thermal
level but not what else was running**. That is the same blind spot that made the
M5 Max figures unexplainable until they were re-run on an idle box, and it is now
the most valuable thing to fix in the tool.

And the absolute rates, which are not close: M4 Pro GPU sustains 172 to 177 img/s,
M5 Max 762 to 888.

### The M4 Pro soak reproduces the committed sweep

Run months later, on a freshly built venv, from a clean clone:

| unit | sweep (30-iteration burst) | soak, first 10 s window |
| --- | ---: | ---: |
| ANE | 204.4 | 204.5 |
| GPU | 178.8 | 178.8 |
| `ALL` | 172.2 | 172.9 |

On the M5 Max the same comparison does *not* line up. 1085.7 burst against
1020.7 in the first soak window, and that is the finding rather than a
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
remains genuinely unseparated is narrower: **how much of the 0.124 and 0.278 is
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
two seconds, and the decline has barely started by then, which is exactly why
the burst figure is higher than even the first 10 s window. The two measure
different things, and only one of them is what a long-running job gets.

## Conditions, because they change the answer

- **AC power.** Recorded per run in the JSON. An earlier GPU soak was discarded:
  the machine was plugged in mid-run and the guard in `thermal_soak.py` caught the
  battery-to-AC transition. A 60 s GPU soak on battery gave 0.879, but that is **not**
  comparable to the 0.837 above. The AC run is 120 s, and sustained fraction falls
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
the reason above, and the die temperature was not sampled during these runs.
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
