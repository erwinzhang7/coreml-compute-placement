# M6 (dual Neural Engine): results

Measured 2026-11-07 on a 14-inch MacBook Pro loaner: Apple M6 (Mac18,1), 12-core CPU, 12-core GPU,
32 GB, macOS 27.2 (26B7124), on AC, thermal pressure nominal at every sweep start. Same models, harness
and protocol as the M4 Pro and M5 Max results: SigLIP-base-224 vision tower, batch 16, `tools/sweep.py`
(30 timed calls after 10 warm-up, median of 5 fresh loads), coremltools 9.0, numpy 2.5.2, torch 2.8.0.

The b16 models are the published ones: `siglip-vision-b16` is the repo-root build; `siglip-ane-b16`
was built by the same `models/ane_siglip.py`. Both have byte-identical `weight.bin` to the copies the
earlier chips ran (the `model.mlmodel` files differ only in build metadata). b1, b8 and b32 were built
on the M6 with the same script, each passing its numerical check (min cosine 0.99999988).

Raw files are in `results/m6/`. The OS differs between chips (M4 Pro 26.5.1, M5 Max 26.6, M6 27.2),
and the M6 was also in interactive use during the runs, which the spreads below include.

## 1. The published protocol, third chip

img/s, batch 16, median of 5:

| chip | GPU cores | naive ANE | naive GPU | naive `ALL` | ANE-rewritten ANE | ANE-rewritten GPU | ANE/GPU (naive) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M4 Pro | 20 | 204.4 | 178.8 | 172.2 | 231.3 | 149.5 | 1.14x |
| M5 Max | 40 | 233.0 | 1085.7 | 1068.0 | 256.7 | 720.1 | 0.21x |
| **M6** | 12 | **426.4** | **437.1** | 424.3 | **530.0** | 157.0 | **0.98x** |

Spreads on the M6 are 0.1-1.8%. The M6 ANE is **2.09x the M4 Pro's and 1.83x the M5 Max's** on the
naive model, and 2.29x / 2.06x on the ANE rewrite. Throughout this repo ANE throughput had stayed
flat across chips, so this is the first time it has moved.

Residency (`tools/anecheck.py`, MLComputePlan, cost-weighted): the ANE rewrite is 100% ANE at b1, b8,
b16 and b32 under both `CPU_AND_NE` and `ALL`. The **naive model is also 100% ANE under `ALL`** on M6
(M4 Pro split it 79/21 ANE/GPU; M5 Max put it 100% on the GPU). So `ALL` on M6 is the ANE, and
`ALL` ≈ ANE throughput in the table agrees. The 424.3 vs 426.4 gap is within spread; the summariser's
"default slower than either pure placement" flag is noise on this chip.

No `ANECCompile() FAILED` appeared in the unified log during any run. Unlike the M4 Pro, the ANE
rewrite under `ALL` compiled cleanly here.

### Metal GPU through MLX

The kill criterion names the Metal GPU, and MLX is how people actually run on it. `m6/mlx_siglip.py`
implements the same tower (patch embed, 12 pre-LN layers, post-LN, mean pool) in MLX 0.32.4, fp16,
verified against the HuggingFace fp32 reference (min cosine 0.999999), `mx.compile`d and evaluated
synchronously per call:

| path | batch 16 | batch 32 |
| --- | ---: | ---: |
| MLX fp16, GPU | 405.1 | 408.8 |
| Core ML `CPU_AND_GPU`, naive | 437.1 | |

On this GPU Core ML's own GPU path is the faster of the two, so the best Metal number is 437.1 img/s.

## 2. Dual ANE: one instance already gets both engines

What the system reports:

- Core ML `MLComputeDevice.allComputeDevices`: **one** `neuralEngine`, `totalCoreCount=32`
  (M5 Max: one, 16).
- Private `_ANEDeviceInfo` (read-only class getters): `numANEs=2`, `numANECores=32`, architecture `h18g`.
- IOKit: two `H11ANEIn` (`H11ANE`, `H11ANE1`) under one `H1xANELoadBalancer`
  (`ANEDevicePropertyNumANEs=2`); the PMGR node has `#ane-num-clusters = 2`.

Concurrency, ANE rewrite b16 on `CPU_AND_NE`, 10 s closed loop, median of 3 (`m6/dual_ane.py`):

| configuration | total img/s | vs one instance |
| --- | ---: | ---: |
| 1 process, 1 model | 512.4 | 1.00x |
| 2 processes | 505.2 | 0.99x |
| 4 processes | 495.5 | 0.97x |
| 1 process, 2 MLModels, 2 threads (Swift) | 497.4 | 0.97x |
| 1 process, 4 MLModels, 4 threads (Swift) | 492.1 | 0.96x |
| 1 process, 1 MLModel, 2 threads (Swift) | 489.0 | 0.95x |
| naive model, 1 process | 405.0 | |
| naive model, 2 processes | 406.0 | 1.00x of its solo |

A second caller adds nothing, in any form. The in-process rows use Swift so the GIL cannot be the
reason.

Batch sweep, one instance, ANE rewrite on `CPU_AND_NE` (median of 5):

| batch | 1 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: |
| img/s | 313.9 | 512.6 | 519.2 | 521.9 |

Throughput is saturated by batch 8. Even batch 1 (314 img/s, 3.2 ms per image) beats every
earlier chip's best ANE figure at batch 16.

Per-engine activity from IOReport, no root (`m6/ioreport_ane.c`, `m6/per_engine.py`; 500 ms deltas
inside a 10 s timed window, n = 15-17):

| configuration | img/s | IOP Running, ANE / ANE1 | interrupts/s, ane / ane1 |
| --- | ---: | --- | --- |
| idle | 0 | 0 / 0 | 0 / 0 |
| rewrite b16, 1 instance | 549.9 | 1.0 / 1.0 | 127 / 128 |
| rewrite b1, 1 instance | 322.0 | 1.0 / 1.0 | 1069 / 1061 |
| rewrite b16, 2 instances | 526.2 | 1.0 / 1.0 | 140 / 132 |
| naive b16, 1 instance | 424.8 | 1.0 / 1.0 | 103 / 102 |
| rewrite b16, 1 instance, `ALL` | 505.3 | 1.0 / 1.0 | 117 / 116 |

**Answer: about 2x, and a single instance already gets it.** One closed-loop caller, at batch 1 or
16, drives both engines at equal interrupt rates. Running a second instance changes neither the
total nor the split. Three things together suggest that each request is split across both engines,
rather than requests alternating between them: a sequential caller cannot overlap alternating
requests, yet one caller gets 1.8-2.3x the single-engine chips; Core ML presents one 32-core device; and
power doubles at constant energy per image (section 3). That is the reading, not something measured:
the counters show both engines working, not how one request is divided between them. (The memory
controller's per-ANE byte counters, `AMC ANE0/ANE1`, are listed by IOReport but do not come back in
a subscription.)

The Swift harness (`m6/multi_model.swift`) reads a few percent higher than the Python one (550 vs
512-530): it has less driving overhead.

Targeting one engine: nothing found. Neither the public API nor the private client classes take an
engine or device index (`_ANEClient`, `_ANEModel`, `_ANERequest`, `_ANEProgramForEvaluation` expose
procedure and QoS-queue indices only). That was checked read-only; SIP stayed on and nothing ran as
root.

## 3. Energy

macmon 0.11 (IOReport energy counters, no root), mean over each timed window. The M5 Max row is the
README's `powermetrics` figure. The two meters differ, so compare silicon-only ANE figures loosely.

| chip / path | img/s | ANE W | GPU W | ANE or GPU mJ/img |
| --- | ---: | ---: | ---: | ---: |
| M5 Max, naive, ANE (powermetrics) | 233.0 | 4.8 | | 20.6 |
| M6, rewrite, ANE, 1 instance | 512.4 | 10.85 | 0 | 21.2 |
| M6, naive, ANE, 1 instance | 421.4 | 10.75 | 0 | 25.5 (22.5 in the dual run) |
| M6, naive, Core ML GPU | 389.5 | 0 | 19.31 | 49.6 |
| M6, MLX fp16 GPU | 317.8 | 0 | 12.66 | 39.8 |

These are medians over three consecutive 12-15 s windows, and the GPU throughput was falling across
them (section 4), so the GPU rows are not burst figures.

On the rewrite the M6 ANE draws 2.3x the M5 Max's ANE power for 2.2x the throughput, at about the same
energy per image (21.2 vs 20.6 mJ). On the naive model it is 1.8x the throughput for 2.2x the power
(22.5-25.5 mJ). Either way the second engine is doing proportional work, which is not what a single,
more efficient engine would look like. Against the GPU, the ANE is 1.6-2.4x better per image on
silicon alone. macmon's CPU figure (0.1-0.4 W for the
Python driver) is far below the 4.0 W `powermetrics` gave on the M5 Max for the same loop, so the
driving-overhead share is not compared across meters here.

## 4. Sustained

Over back-to-back 15 s windows (naive b16, `m6/dual_ane.py --phases procs-1`), Core ML GPU ran
434 → 390 → 352 img/s as GPU power fell 21.3 → 17.5 W, while the ANE ran 426 → 421 → 411. The
published soak protocol (`tools/thermal_soak.py`, 120 s, 10 s windows) gives:

Run in this order, 20 s apart: naive on ANE, naive on GPU, rewrite on ANE. The tool's load
sampler marked all three "quiet"; the busiest other process was a browser at ~40% of one core.

| unit, model | peak img/s | img/s at 120 s | sustained | thermal pressure |
| --- | ---: | ---: | ---: | --- |
| ANE, naive | 424.1 | 369.9 | 0.872 | nominal throughout |
| GPU, naive | 436.6 | 340.0 | 0.779 | moderate from 20 s, heavy from 70 s |
| ANE, rewrite | 512.9 | 437.0 | 0.852 | moderate (started warm, after the GPU soak) |
| *M5 Max ANE, naive* | *233.1* | *232.8* | *0.999* | |
| *M5 Max GPU, naive* | *1020.7* | *854.3* | *0.837* | |

The two units lose throughput differently. The GPU drops ~21% within 20 s and then holds flat at ~341.
The ANE falls steadily, about 1% per 10 s window, and had nearly flattened by 120 s (rewrite:
441.7, 438.6, 437.0 in the last three windows). On the naive model that happened at **nominal**
thermal pressure. The M5 Max's single ANE did not decline at all (0.999), so this is new to M6. That
fits a dual-engine block meeting a power or thermal limit that one 16-core engine never reached, but
the counters here cannot tell those apart. At 120 s the ANE leads the GPU by 1.09x (naive) and
1.29x (rewrite), against 0.98x and 1.21x at burst. Whether the ANE keeps falling over 5-10 minutes
was not measured.

## 5. Kill criterion

Erwin's spec: stop if ANE-optimised Core ML runs below ~60% of Metal GPU throughput on the same encoder.

| chip | ANE-rewritten on ANE | best Metal GPU | ratio | verdict |
| --- | ---: | ---: | ---: | --- |
| M4 Pro | 231.3 | 178.8 (Core ML) | 1.29 | passes |
| M5 Max | 256.7 | 1085.7 (Core ML) | 0.24 | **fires** |
| M6 | 530.0 | 437.1 (Core ML; MLX 405.1) | **1.21** | passes |

On M6 the criterion does not fire, and over a 120 s soak the gap widens (1.29x, section 4). The paper's thesis
holds with a third data point: the right unit does not transfer between chips. On M6 the reason is
that the ANE doubled while a base-tier GPU stayed small.

What it means for the ANE inference-server direction:

- **The concurrency argument (§6 of the spec) is dead on M6.** One instance saturates both engines,
  so a server gains nothing from running several ANE instances; extra callers only add queueing.
  The ANE's value here is raw throughput and ~2x energy per image, not parallel slots.
- **The ANE + GPU split (both units at once) is now the interesting configuration on M6**, because
  the two are roughly equal (530 vs 437 burst). If they add the way they did on the M4 Pro (99% of
  ideal), that is ~900 img/s. On the M5 Max they reached 86%. Not measured here
  (`tools/bench_concurrent.py` is the harness).
- **The verdict is per chip, and the fleet's chips are M4 Pro (passes) and M5 Max (fires).** M6 is a
  loaner. A base M6 is a laptop part, and an M6 Pro/Max, with a much bigger GPU, could flip the
  verdict back the way the M5 Max did.
- Model load costs a full ANE compile every time through `ct.models.MLModel(<.mlpackage>)`: 30-70 s
  per load for SigLIP on macOS 27.2 (from `ANECompilerService`'s log), because each load compiles into
  a fresh temp path. Loading a stable `.mlmodelc` (`ct.models.CompiledMLModel`) avoids it. Any server
  must hold models resident and load compiled artifacts.

## Reproduce

`m6/setup.sh` (venv, b1/b8/b32 builds, Swift tools, probes, residency), then `tools/sweep.py` for §1,
`m6/dual_ane.py` for §2-3, `m6/batch_sweep.sh`, `m6/per_engine.py`, `m6/mlx_siglip.py`, and
`tools/thermal_soak.py` for §4. Each was run as a single short job on an otherwise idle GPU.
`ct.models.MLModel` on an `.mlpackage` recompiles for the ANE on every load (see above), so
`tools/sweep.py` on M6 takes minutes per model, not seconds.

> **Invalid as of 2026-11-07 19:00:** the M6 fan was stuck off (0 RPM, SoC ~104 C) during every run on this branch, by a fan-control bug since fixed. Rerun before using any number here.
