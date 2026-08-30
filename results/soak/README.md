# Sustained soaks

Burst benchmarks measure what a compute unit can *reach*. These measure what it
*holds*. They are the same model and batch as the committed sweeps, so the two
are directly comparable.

## Read this before the tables

Four things below were wrong for a while and are now corrected in place.

-1. **A sixty-second run on battery was counted as a two-minute soak.**
   `battery-m5max-CPU_AND_NE.json` is `seconds=60.0`, six windows, `power_start`
   and `power_end` both `battery`. It sat inside `PAPER-SET.txt` and counted
   toward "44 of 45", so a claim about *two-minute soaks on mains* was being
   supported in part by a run that was neither. It read 1.0000, so it flattered
   the ANE; removing it moves the count to **43 of 44** and changes nothing else.

   The cause is worth more than the number. The set was filtered by FILENAME
   PREFIX — `("long600", "dip", "aned")` — under a comment noting that filtering
   by prefix rather than by content had already been the bug once. Nothing about
   the name `battery-m5max` would ever have caught it. The filter now reads
   `seconds` and `power_start.source` out of each file, and keeps a prefix list
   only for deliberate sub-studies, whose purpose is not recorded anywhere in
   the JSON.

0. **Only soaks carrying a `concurrent_load` field recorded what else was running.**
   The sampler was added to `thermal_soak.py` only for the `mx*` batch, so 32 of the
   157 soaks in `PAPER-SET.txt` carry it and **the other 125 are blind** to the one
   confound the field exists to catch. An earlier version of this note had the
   count inverted, naming the 32 that record as the ones that do not. The boxes were checked by hand and were quiet at the time
   (largest non-soak process 0.2 GB at 0.0% CPU), but that is a point-in-time
   check rather than a per-run record and is weaker evidence.

   The field also marks a throughput difference. Sampling costs **0.18%** of
   absolute rate, measured three runs each way with non-overlapping groups. It
   does not move `sustained_fraction`, which is a within-run ratio where a
   constant offset cancels. So compare absolute rates only within one group.

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

**Read that GPU number as a trough, not a steady state — for four of the five
models.** 600-second soaks (`long600-*.json`) show the M4 Pro GPU dipping to a
minimum around 80 to 100 seconds and then recovering to a plateau just under peak.
A 120-second run stops inside that dip. Measured from the *same* 600 s runs,
taking the window that ends at 120 s against the mean of the last six:

| model | unit | at 120 s | at steady state |
| --- | --- | ---: | ---: |
| `siglip` | GPU | 0.9635 | 0.9959 |
| `resnet50` | GPU | 0.9876 | 0.9978 |
| `whisper` | GPU | 0.9496 | 0.9737 |
| `siglip` | ANE | 1.0000 | 0.9995 |

The signature is in the short runs too: across 56 M4 Pro 120 s GPU soaks the
slowest window falls in the last 30% of the run 77% of the time, against 10% for
40 ANE soaks and ~30% for chance. **This is M4 Pro only.** The M5 Max GPU loses
about 15% inside two windows and holds, with no recovery to miss. PAPER.md §3.3
has the full treatment.

**`mobilenet` is the exception and it was added last.** It does not dip and
recover; it settles at about 98% of peak by the eighth window and holds that for
the remaining twenty-one, on both M4 Pro machines and whether the box starts cold
or warm — six runs spanning 2.10 to 2.44%. So there is no trough to correct for: its
120 s figure is already its steady state, and applying the "read it as a trough"
adjustment to `mobilenet` would overstate it.

The M4 Pro GPU has two behaviours, not one, and which appears depends on the
model. That is the reason to run all five rather than four.

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

The ANE, in 41 of 42 soaks. Across four machines and two chips those 41 measured
0.9986 to 1.0000, giving back at most 0.14% of peak regardless of starting state,
which is what an engine that does not degrade should look like, and it is that
insensitivity that makes it immune to the confound above. All 41 sustain better
than the best of 60 GPU soaks (0.9892). Counts are the 120 s protocol only; the
600 s runs are analysed separately and deliberately excluded (PAPER.md §3.3).

**The forty-second is a `whisper` ANE soak on inference2 at 0.9748**, and it is
not dismissed. It ran flat at 56.3 img/s for seven windows, then fell monotonically
to 54.9 and plateaued. It is not the contended-peak artefact that cost us the M5
Max numbers: contention lowers the peak, and this run reached 56.34 against 56.34
and 56.35 for the seven other `whisper` ANE runs in the paper set, all of which
held their rate. The
engine reached full speed and then lost 2.5% of it.

We could not say why at the time, because `thermal_soak.py` recorded the machine and
the thermal level but not what else was running, the same blind spot that made the
M5 Max figures unexplainable until they were re-run on an idle box. **That is now
fixed**: every soak records peak non-self CPU, when it peaked, and the busiest
processes, in a `concurrent_load` field. The original run predates it and remains
unexplainable from its own file.

It did not reproduce. A repeat on the same machine returned 0.9998 with the box
recorded at 3.7% of its 14 cores, and a 600 s `whisper` ANE run is flat at 0.9998
over 29 windows. One reading, never seen again at its own duration or at five
times it.

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

All **three** M4 Pros are Mac minis on mains with stock cooling. The M5 Max is a
MacBook Pro with a custom fan curve.

An earlier version of this paragraph said "both" M4 Pros, and claimed each soak
"logged `0 avm procs` and no resident chat model at start". **No soak file
records anything of the kind.** The only keys any of them carry are model, units,
batch, seconds, window, machine, power_start, power_end, thermal_readable,
summary, windows and — in the `mx*` batch and later — concurrent_load. What the
boxes were doing was checked by hand at the time and not written down, which is
weaker evidence than a per-run record and should not have been described as one.
The `concurrent_load` field is that record, and it exists for 130 of the runs;
see item −1 above for what it can and cannot see.

An earlier version of this file said the chip/chassis pair was unseparated and
left it there. That was too weak a reading: the two-mini floor plus the ANE's
flat 0.999 on the hot laptop are evidence, and they point the same way. What
remains genuinely unseparated is narrower: **how much of the 0.129 and 0.278 is
the enclosure rather than the silicon.** The second reproduces as the extreme M4
Pro `ALL` against the extreme M5 Max `ALL`, 1.0000 − 0.7209 = 0.279. The first
was printed as 0.124 and does not reproduce from any comparison of these files;
PAPER.md §3.6 derives the corresponding GPU gap against the M4 Pro mean as 0.129,
which is the figure used here now. Settling that needs an M5 Max in a
desktop chassis, or an M4 Pro laptop, and neither exists here.

It does not need settling for the claim being made, which is that the units'
durability is ordered differently on the two chips.

## M5 Max, 120 s, batch 16, SigLIP-base-224 vision tower

| compute unit | peak img/s | last img/s | sustained | shape |
| --- | ---: | ---: | ---: | --- |
| `CPU_AND_NE` (ANE) | 233.1 | 232.8 | **0.999** | flat for 12 windows |
| `CPU_AND_GPU` | 1020.7 | 854.3 | **0.837** | falls ~16% in 20 s, then plateaus |
| `ALL` (default) | 1006.5 | 725.6 | **0.721** | still falling at 120 s |

**These three runs are from the OLD tool** — `ac-m5max-*.json`, twelve windows with
the final one clamped to 9.31-9.42 s rather than a full 10 s, which is the exact
signature the correction removed. They predate it by two hours. Item 1 above says
every figure here is from the corrected tool; that is true of the rest of this
file and not of this table, which is kept because it is the run the retraction is
*about*.

**The ANE gives up nothing. The GPU gives up a sixth.** The third line does not
survive: a repeat with the corrected tool put `ALL` and the GPU within noise of
each other, and the ordering reverses depending on which pair of runs is compared,
so "the default gives up more than a quarter" is withdrawn (PAPER.md §3.3). What
holds on this chip is that both the GPU and the default give back roughly a sixth
over two minutes, and the ANE gives back nothing.

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
  comparable to the 0.837 above, because the durations differ. Comparing AC against
  battery needs equal durations and has not been done.

  An earlier version of this note argued that sustained fraction falls with soak
  length **by construction**, since the denominator is the best window and the
  numerator is the last one. **That is wrong, and the 600 s runs falsify it.** The
  denominator is fixed once the peak has passed, usually within two windows, while
  the numerator can *rise* if the engine recovers. On the M4 Pro it does: `siglip`
  on the GPU reads 0.9635 at 120 s and 0.9959 at 600 s, in the same run. The
  argument only holds where throughput decreases monotonically, which is the M5 Max
  and not the M4 Pro.
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
