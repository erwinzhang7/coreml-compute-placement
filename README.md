# Where does your Core ML model actually run?

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

---

## The three findings

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

Spreads across 5 repeats were 0.0-3.5% (median 0.3%), so the differences reported here are
far larger than run-to-run noise. Raw per-run values are in `results/`.

---

## Reproducing

Requires macOS 15+ on Apple silicon and **Python 3.12** - coremltools ships no native
extension for 3.13+, and without it `MLComputePlan` and `MLModel` do not exist.

```sh
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# build the two model variants
./.venv/bin/python models/convert_siglip.py --batch 16 --out siglip-naive-b16.mlpackage
./.venv/bin/python models/ane_siglip.py    --batch 16 --out siglip-ane-b16.mlpackage

# where do the ops go?
./.venv/bin/python tools/anecheck.py siglip-naive-b16.mlpackage --compute-units ALL

# throughput, repeated, with spread
./.venv/bin/python tools/sweep.py --models siglip-naive-b16.mlpackage siglip-ane-b16.mlpackage

# both units at once, optionally with a different variant per unit
./.venv/bin/python tools/bench_concurrent.py siglip-naive-b16.mlpackage \
    --model-ane siglip-ane-b16.mlpackage --model-gpu siglip-naive-b16.mlpackage
```

`models/ane_siglip.py` verifies numerical equivalence against the HuggingFace model on
every build and refuses to emit a model that does not match.

### Tools

| tool | what it does |
| --- | --- |
| `tools/anecheck.py` | per-op compute-device placement from `MLComputePlan`, **cost-weighted**, with `--assert-ane-fraction` as a CI gate |
| `tools/sweep.py` | repeated throughput runs, reports per-run values and spread |
| `tools/bench_concurrent.py` | both units simultaneously, separate processes, optional per-unit model variant |
| `tools/bench_power.py` | sustained-load driver to pair with `powermetrics` |
| `tools/bench_coreml.py` | single-configuration throughput |
| `tools/probe_gpu.py` | builds matched matmul-bound and bandwidth-bound models, to separate matmul-specific hardware from general GPU uplift |

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
