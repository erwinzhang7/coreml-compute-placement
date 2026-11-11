# M6 (dual Neural Engine): results

Measured 2026-11-07, 19:42-20:49 EST, on a 14-inch MacBook Pro loaner: Apple M6 (Mac18,1), 12-core CPU,
12-core GPU, 32 GB, macOS 27.2 (26B7124), on AC. Same models, harness and protocol as the M4 Pro and
M5 Max results: SigLIP-base-224 vision tower, batch 16, `tools/sweep.py` (30 timed calls after 10
warm-up, median of 5 fresh loads), coremltools 9.0, numpy 2.5.2, torch 2.8.0; MLX 0.32.4.

**This replaces a first run from the same day that is invalid.** During it the fan was stuck at 0 RPM
in manual mode (SoC around 104 C) because of a bug in the fan-control app. Those files and their
write-up are kept, unchanged, in `results/m6/invalid-2026-11-07-fan-off/`. Section 9 compares the two.

The b16 models are the published ones: `siglip-vision-b16` is the repo-root build, `siglip-ane-b16` was
built by the same `models/ane_siglip.py`. Both have byte-identical `weight.bin` to the copies the earlier
chips ran; the `model.mlmodel` files differ only in build metadata. b1, b8 and b32 were built on the M6
with the same script and each passed its numerical check (min cosine 0.99999988). The OS differs
between chips (M4 Pro 26.5.1, M5 Max 26.6, M6 27.2).

## How every run was gated

Each timed run went through `m6/guarded.sh`, one short job at a time, with nothing else on the GPU:

- **Before:** a fan and thermal snapshot, at least 60 s idle, then a wait until the hottest CPU/GPU die
  sensor was under 60 C. Every run started at 26-35 C, thermal pressure nominal.
- **During:** the fan and die sampled every 5 s into `results/m6/thermal/<run>.during.jsonl`.
- **After:** another snapshot.
- **Fault rule:** the run fails if a fan sits at 0 RPM while a die sensor is above 85 C for two
  consecutive samples (10 s or more). That is the stuck-fan signature, and no run showed it.

This machine's fan-control app does not leave the fan in macOS's automatic mode. When the die is cool
it parks the fan in "auto" at 0 RPM, which is normal for this chassis. Once the die warms it switches
to "manual" with its own targets, up to 8250 RPM. So "manual" alone is not a fault. Every run log
shows both modes and the fan reaching 8000-8278 RPM under load. The hottest any die got was 88.5 C,
briefly, in the sweep. Thermal pressure stayed nominal in every run.

The first version of the guard required auto mode throughout, which made the sweep job
exit 4 after completing all its measurements. Its log passes the corrected rule (0 fault samples), so
its numbers are used. Every later job ran under the corrected rule.

## 1. The published protocol, third chip

img/s, batch 16, median of 5, spread 0.2-1.1%:

| chip | GPU cores | naive ANE | naive GPU | naive `ALL` | ANE-rewritten ANE | ANE-rewritten GPU | ANE/GPU (naive) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M4 Pro | 20 | 204.4 | 178.8 | 172.2 | 231.3 | 149.5 | 1.14x |
| M5 Max | 40 | 233.0 | 1085.7 | 1068.0 | 256.7 | 720.1 | 0.21x |
| **M6** | 12 | **429.4** | **436.8** | 429.5 | **559.6** | 156.8 | **0.98x** |

On the naive model the M6 ANE runs 2.10x the M4 Pro's and 1.84x the M5 Max's; on the ANE rewrite,
2.42x and 2.18x. Throughout this repo ANE throughput had stayed flat across chips, so this is the
first time it has moved.

**Residency** (`tools/anecheck.py`, MLComputePlan, cost-weighted, `results/m6/residency/`):
- The ANE rewrite is 100% ANE at b1, b8, b16 and b32, under both `CPU_AND_NE` and `ALL`.
- The naive model is also 100% ANE under `ALL` on M6. The M4 Pro split it 79/21 ANE/GPU, and the
  M5 Max put it entirely on the GPU.
- So on M6, `ALL` means the ANE, and `ALL` ≈ ANE throughput in the table.
- No `ANECCompile() FAILED` appeared in the unified log.

**Metal GPU through MLX** (`m6/mlx_siglip.py`): the same tower in MLX fp16, checked against the
HuggingFace fp32 reference (min cosine 0.999999) and `mx.compile`d. It runs at 405.0 img/s at b16
and 409.5 at b32, below Core ML's own GPU path (436.8). So the best Metal number on this chip is 436.8.

## 2. Dual ANE: one instance already gets both engines

What the system reports (`results/m6/devices.txt`):
- Core ML lists one `neuralEngine` with `totalCoreCount=32`. The M5 Max shows one, with 16.
- The private `_ANEDeviceInfo` class getters, read-only, give `numANEs=2`, `numANECores=32` and
  architecture `h18g`.
- IOKit has two `H11ANEIn` (`H11ANE`, `H11ANE1`) under one `H1xANELoadBalancer`
  (`ANEDevicePropertyNumANEs=2`).

Concurrency, ANE rewrite b16 on `CPU_AND_NE`, 10 s closed loop, median of 3, spread 0.0-0.1%
(`m6/dual_ane.py`):

| configuration | total img/s | vs one instance |
| --- | ---: | ---: |
| 1 process, 1 model | 558.3 | 1.00x |
| 2 processes | 574.8 | 1.03x |
| 4 processes | 576.3 | 1.03x |
| 1 process, 2 MLModels, 2 threads (Swift) | 570.1 | 1.02x |
| 1 process, 4 MLModels, 4 threads (Swift) | 571.5 | 1.02x |
| 1 process, 1 MLModel, 2 threads (Swift) | 570.0 | 1.02x |
| naive model, 1 process / 2 processes | 429.3 / 439.5 | 1.02x |

A second caller adds 2-3% at most, the cost of keeping the queue full between calls. It does not
add a second engine's worth. The in-process rows use Swift, so the GIL is not the reason.

Batch sweep, one instance, ANE rewrite on `CPU_AND_NE`, median of 5, spread 0.3-0.5%:

| batch | 1 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: |
| img/s | 316.0 | 554.4 | 559.4 | 565.2 |

Per-engine activity from IOReport, read without root (`m6/ioreport_ane.c`, `m6/per_engine.py`;
500 ms deltas inside a 10 s window, n = 15-17):

| configuration | img/s | IOP Running, ANE / ANE1 | interrupts/s, ane / ane1 |
| --- | ---: | --- | --- |
| idle | 0 | 0 / 0 | 0 / 0 |
| rewrite b16, 1 instance | 568.6 | 1.0 / 1.0 | 142 / 140 |
| rewrite b1, 1 instance | 321.1 | 1.0 / 1.0 | 1073 / 1070 |
| rewrite b16, 2 instances | 570.4 | 1.0 / 1.0 | 146 / 144 |
| naive b16, 1 instance | 437.3 | 1.0 / 1.0 | 104 / 101 |
| rewrite b16, 1 instance, `ALL` | 569.1 | 1.0 / 1.0 | 133 / 132 |

**Answer: about 2x, and a single instance already gets it.** One closed-loop caller drives both
engines at equal interrupt rates, at batch 1 and at batch 16. A second instance changes neither the
total nor the split.

That each request is split across both engines, rather than requests alternating between them, is an
inference, not a measurement. A sequential caller cannot overlap alternating requests, yet one caller
gets 1.8-2.4x the single-engine chips, and Core ML presents a single 32-core device. The counters show
both engines working; they do not show how one request is divided. (IOReport lists the memory
controller's per-ANE byte counters, `AMC ANE0/ANE1`, but they do not come back in a subscription.)

**Targeting one engine: nothing found.** Neither the public API nor the private client classes take
an engine or device index; `_ANEClient`, `_ANEModel`, `_ANERequest` and `_ANEProgramForEvaluation`
expose procedure and QoS-queue indices only. This was checked read-only, with SIP on and nothing run
as root.

## 3. ANE and GPU at the same time

`tools/bench_concurrent.py`, separate processes, 15 s, started together:

| chip | models | ANE alone | GPU alone | together | of ideal | vs best single unit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M4 Pro | naive / naive | 204.4 | 178.9 | 381.3 | 99% | 1.87x |
| M5 Max | naive / naive | 231.3 | 901.5 | 970.3 | 86% | 1.08x |
| M6 | naive / naive | 431.1 | 438.2 | 541.1 | 62% | 1.23x |
| M6 | rewrite on ANE / naive on GPU | 559.2 | 438.1 | 774.8 | 78% | 1.39x |

On the naive pair, the M6's GPU keeps 52% of its solo rate and the ANE 72%. With the rewrite on the ANE,
each keeps 78%. The two units now contend harder than on either earlier chip. That fits two 16-core
engines and a 12-core GPU sharing 170 GB/s (section 7), but the shared bus is a reading, not something
isolated here. The best M6 configuration is still 775 img/s: rewrite on the ANE plus naive on the GPU.

## 4. Energy

macmon 0.11 (IOReport energy counters, no root), mean over each timed window, medians of 3.

| path | img/s | unit W | unit mJ/img | CPU+GPU+ANE mJ/img |
| --- | ---: | ---: | ---: | ---: |
| ANE, rewrite | 558.3 | 15.76 | 28.2 | 28.9 |
| ANE, naive | 430.4 | 12.85 | 29.9 | 30.6 |
| GPU (Core ML), naive | 437.5 | 20.36 | 46.5 | 48.5 |
| GPU (MLX fp16) | 405.3 | 20.78 | 51.3 | 52.8 |
| *M5 Max ANE, naive (powermetrics)* | *233.0* | *4.8* | *20.6* | |

On silicon alone the ANE is 1.6-1.8x more efficient per image than the GPU on the same chip.
macmon and powermetrics are different meters, so read the M5 Max row as approximate.

In the invalid fan-off run the ANE drew less: 10.8 W for 512 img/s, 21 mJ/img. With cooling it
draws 45% more power for 9% more throughput. That fits a hot chip held at a lower, more efficient
clock and voltage point, and a cool one running at its top point. The ANE's clock state is not
visible here, so this is an inference.

## 5. Sustained

`tools/thermal_soak.py`, 120 s, 10 s windows, run back to back with the gated cool-down between them:

| unit, model | peak img/s | img/s at 120 s | sustained | max die | thermal pressure |
| --- | ---: | ---: | ---: | ---: | --- |
| ANE, naive | 429.3 | 428.8 | 0.999 | 73 C | nominal |
| GPU, naive | 436.9 | 436.7 | 1.000 | 81 C | nominal |
| ANE, rewrite | 557.7 | 553.9 | 0.993 | 78 C | nominal |
| *M5 Max ANE, naive* | *233.1* | *232.8* | *0.999* | | |
| *M5 Max GPU, naive* | *1020.7* | *854.3* | *0.837* | | |

With working cooling, nothing on the M6 throttles in two minutes. The decline reported from the
invalid run (ANE 0.85-0.87, GPU 0.78) came entirely from the stuck fan. Longer soaks were not run.

## 6. Kill criterion

Erwin's spec: stop if ANE-optimised Core ML runs below ~60% of Metal GPU throughput on the same encoder.

| chip | ANE-rewritten on ANE | best Metal GPU | ratio | verdict |
| --- | ---: | ---: | ---: | --- |
| M4 Pro | 231.3 | 178.8 (Core ML) | 1.29 | passes |
| M5 Max | 256.7 | 1085.7 (Core ML) | 0.24 | **fires** |
| M6 | 559.6 | 436.8 (Core ML; MLX 405.0) | **1.28** | passes |

The ratio is the same at 120 s (553.9 / 436.7 = 1.27). On M6 the criterion does not fire.

What this means for serving:

- **Running several ANE instances does not help on M6.** One instance already saturates both
  engines, so running several ANE instances buys nothing.
- **The ANE's value on M6** is throughput (1.28x the GPU) and 1.6-1.8x lower energy per image. Driving
  ANE and GPU together adds 1.39x over the ANE alone, less than the M4 Pro's near-perfect overlap.
- **The verdict is per chip.** Of the two earlier chips, the M4 Pro passes and the M5 Max fires. This
  M6 is a base-tier loaner, and an M6 Pro or Max, with a much larger GPU, could fire it again.
- **Model loading.** `ct.models.MLModel(<.mlpackage>)` pays a full ANE compile on every load on
  macOS 27.2 (30-70 s for SigLIP), because each load compiles into a fresh temp path. Loading a stable
  `.mlmodelc` with `ct.models.CompiledMLModel` avoids it. A server must keep models resident and load
  compiled artifacts.

## 7. Memory bandwidth

`m6/stream_bw.py`: STREAM copy/scale/add/triad as MLX custom Metal kernels (float4, 1 GiB per array,
7 repeats of 20 calls, median; STREAM byte counting, each kernel checked for correctness).
`tools/gpu_bw.py` is the repo's earlier MLX-op probe. CPU figures are streaming reads from
`tools/membw.c`, median of 3 per thread count.

| chip | published | GPU STREAM (copy / scale / add / triad) | best of the four | GPU read-only | CPU read, best |
| --- | ---: | --- | ---: | ---: | ---: |
| M6 | 170 GB/s | 150.5 / 150.1 / 150.1 / 150.6 | 150.6 (88.6%) | 157.1 (92.4%) | 166.2 at 12 threads (97.8%) |
| M5 Max | 614 GB/s | 521.9 / 523.9 / 536.8 / 536.8 | 536.8 (87.4%) | 518.6 (84.5%) | 297.9 at 18 threads (48.5%) |

Spreads were 0.2-3.8%. The M5 Max here is the 14-inch MacBook Pro, Mac17,7, with a 40-core GPU and
128 GB, on macOS 27.2 (26B123), measured 2026-11-08 01:54-01:58 EST. It is not necessarily the same unit as the M5 Max in the earlier results.
The earlier repo figure of 566.7 GB/s came from `tools/gpu_bw.py` at 4 GiB per operand on macOS 26; this
run used 1 GiB, so the two are not directly comparable.

Both GPUs reach about 87-89% of their published peak on STREAM. The CPU side differs: all 12 M6 cores
reading together pull slightly more than its GPU does (166 vs 157 GB/s), while the M5 Max's CPU reaches
under half of the bus.

The M5 Max runs were done twice. The first time, another user job (two CPU threads, niced) started on
the machine the same minute. It cut the CPU sweep to 105.6 GB/s and dropped one STREAM copy repeat to
375 GB/s. Those runs are kept, unused, in `results/m5max-14/shared-with-concurrent-job/`. The repeat
reserved all 16 schedulable cores so nothing else could start alongside. The M5 Max runs were gated
the same way as the M6's (`m6/guarded.sh`, `results/m5max-14/thermal/`). Its fans run under the same
control app: auto at 0 RPM when cool, manual up to about 7850 RPM under load. The peak die temperature
was 88 C in the bandwidth run and 69 C in the LLM run, with thermal pressure nominal and no fault samples.

## 8. LLM decode and prefill

`m6/llm_decode.py`: `mlx-community/Qwen2.5-7B-Instruct-4bit` (snapshot c26a38f, the same SHA-256 on
both machines), MLX 0.32.4 / mlx-lm 0.32.2, a fixed 542-token chat prompt, greedy, exactly 256 generated
tokens (EOS ignored), one warm-up run, then median of 5.

Decode reads every weight once per token, except the embedding table, from which it reads one row.
That is 3.978 GB per token (4.284 GB of weights minus 0.307 GB of `embed_tokens`; `lm_head` is
separate and read in full), plus about 46 MB of KV cache. So the ceiling is bandwidth / 3.98 GB.
(`llm_decode.py` divides by the full 4.28 GB, so its JSON ceilings are about 7% low.)

| chip | prefill tok/s | decode tok/s | effective decode GB/s | ceiling at published BW | of published | of best measured GPU BW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M6 | 1117 (TTFT 485 ms) | 36.09 (spread 0.4%) | 143.6 | 42.7 tok/s | 84% | 91% (of 157.1) |
| M5 Max | 3317 (TTFT 163 ms) | 119.38 (spread 3.1%) | 474.9 | 154.3 tok/s | 77% | 88% (of 536.8) |
| M5 Max / M6 | 2.97x | 3.31x | | | | |

Decode scales with bandwidth: the M5 Max has 3.61x the published bandwidth and 3.56x the measured,
and decodes 3.31x faster. Each chip decodes at about 90% of its own measured GPU bandwidth, so the
remaining gap to the published figures is mostly bandwidth the GPU cannot reach at all. Prefill is
compute-bound and scales 2.97x, close to the GPU core ratio (40 vs 12, 3.33x).

On the M5 Max the last two of five runs were 2-3% slower in both decode and prefill, on both
attempts. On the second the machine was to itself and the die stayed under 69 C, so this points to the
chip's own clock behaviour over consecutive runs, not to contention. The median is used. Peak memory was 4.7-4.9 GB on both machines.

## 9. Before and after the fan fix (M6)

| measurement | fan stuck off (invalid) | fan working | change |
| --- | ---: | ---: | ---: |
| naive, ANE, img/s | 426.4 | 429.4 | +0.7% |
| naive, GPU | 437.1 | 436.8 | -0.1% |
| naive, `ALL` | 424.3 | 429.5 | +1.2% |
| rewrite, ANE | 530.0 | 559.6 | +5.6% |
| rewrite, `ALL` | 523.9 | 558.7 | +6.6% |
| rewrite, GPU | 157.0 | 156.8 | -0.1% |
| MLX fp16, b16 | 405.1 | 405.0 | 0.0% |
| dual: 1 / 2 / 4 processes | 512.4 / 505.2 / 495.5 | 558.3 / 574.8 / 576.3 | +9% / +14% / +16% |
| dual: 2 vs 1 instance | 0.99x | 1.03x | |
| batch 1 / 8 / 16 / 32 | 314 / 513 / 519 / 522 | 316 / 554 / 559 / 565 | +1% / +8% / +8% / +8% |
| ANE W, rewrite | 10.85 | 15.76 | +45% |
| ANE mJ/img, rewrite | 21.2 | 28.2 | +33% |
| soak sustained, ANE naive / GPU / ANE rewrite | 0.872 / 0.779 / 0.852 | 0.999 / 1.000 / 0.993 | |
| kill-criterion ratio (burst) | 1.21 | 1.28 | |

The burst figures barely moved. The 30 s sweep runs finish before a cold die heats up, even with the fan
off. Everything that runs long moved: the ANE rewrite under sustained load (+6-16%), the soaks, and power.
None of the qualitative findings changed: one instance uses both engines, the ANE beats the GPU on M6,
and the kill criterion does not fire. Two findings from the invalid run are withdrawn: that the M6 ANE
throttles within two minutes, and that it matches the M5 Max ANE's ~21 mJ per image.

## Reproduce

`m6/setup.sh` (venv, b1/b8/b32 builds, Swift tools, probes, residency). Then, each wrapped in
`m6/guarded.sh <name> ...`:

| section | script |
| --- | --- |
| §1 | `tools/sweep.py` |
| §2 | `m6/dual_ane.py`, `m6/batch_sweep.sh`, `m6/per_engine.py` |
| §3 | `tools/bench_concurrent.py` |
| §4 | `m6/dual_ane.py --phases procs-1`, `m6/mlx_siglip.py --power` |
| §5 | `tools/thermal_soak.py` |
| §7 | `m6/llm_prep.sh`, then `m6/bw_all.sh <published GB/s> 1 "<thread counts>"` |
| §8 | `m6/llm_decode.py` |

`ct.models.MLModel` on an `.mlpackage` recompiles for the ANE on every load (see §6), so
`tools/sweep.py` on M6 takes minutes per model, not seconds.
