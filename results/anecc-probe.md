# ANECCompile() failure under ComputeUnit.ALL: investigation log

Chased 2026-08-07 on `mini-experiments` (Mac16,11, M4 Pro, 20 GPU / 16 NPU, macOS 26.5.1,
coremltools 9.0). Recorded including the wrong turns, because two of them are traps that
would mislead anyone else reproducing this.

## The message

```
E5RT encountered an STL exception. msg = MILCompilerForANE error: failed to compile
ANE model using ANEF. Error=_ANECompiler : ANECCompile() FAILED.
```

## Conditions, established by exhaustive sweep

| model | compute units | cold compile | result |
| --- | --- | --- | --- |
| `siglip-ane-b16` (ANE rewrite) | **ALL** | **yes** | **FAILS** |
| `siglip-ane-b16` | ALL | no (warm) | clean |
| `siglip-ane-b16` | CPU_AND_NE | yes | clean |
| `siglip-ane-b16` | CPU_AND_GPU | yes | clean |
| `siglip-vision-b16` (naive) | ALL / CPU_AND_NE / CPU_AND_GPU | yes | clean |
| `probe-attn1` (1 attention layer) | ALL / CPU_AND_NE / CPU_AND_GPU | yes | clean |
| any of the above on **M5 Max** | any | yes | clean |

So: **ANE-rewritten model + `ALL` + cold compile + M4 Pro.** All four conditions required.

## Severity: low

- **Outputs are unaffected.** `ALL` and `CPU_AND_NE` agree exactly (max deviation from a
  CPU_ONLY reference is 1.5234 for both, min cosine 0.99586 for both). The ~11% max
  relative deviation from CPU_ONLY is fp16-versus-fp32 and appears identically on
  `CPU_AND_GPU`, where no ANE is involved, and on the naive model. It is not related.
- **`ALL` is the fastest configuration for this model anyway** (240.9 img/s, against 231.3
  for CPU_AND_NE and 149.5 for CPU_AND_GPU). Whatever the fallback does, it costs nothing
  measurable.

So this is a noisy diagnostic for an optimisation that silently and correctly degrades.
Worth reporting, not worth blocking on.

## Two traps

**1. The message does not go to the calling process's stderr.** Redirecting fd 2 around the
load and predict calls does not capture it; neither does `2> file` on the Python process.
It surfaced *between* two separate `python` invocations in a shell loop, which is what gave
it away: it comes from a helper process at teardown. Any attempt to attribute it to a
specific call by capturing Python-level stderr will produce false negatives. Three of the
sweeps here reported "clean" for that reason before the capture method was fixed.

**2. It only fires on a cold compile,** so it vanishes once the compiled artifact is cached.
Re-running the exact sweep that first produced it came back clean. Reproducing reliably
requires forcing a fresh compile:

```python
m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)
shutil.copytree(m.get_compiled_model_path(), fresh_dir)   # cold compile happens here
```

Together these two make it look intermittent and unattributable when it is neither.

## Hypotheses tested and rejected

| hypothesis | test | result |
| --- | --- | --- |
| needs many predictions | 30 predicts, single unit | clean |
| needs multiple compute units in one process | CPU_AND_NE,CPU_AND_GPU,ALL sequentially | clean |
| needs ANE contention | two concurrent ANE processes | clean |
| a single attention layer suffices | `probe-attn1`, 4.6 MB | clean |

## Not yet minimal

The smallest known reproducer is still the full 164 MB rewritten SigLIP tower. A single
ANE-style attention layer does not trigger it, so the next step is a depth bisect
(`probe-attn12`, `probe-mlp12` are built by `tools/probe_gpu.py`'s sibling in the ane-bench
scratch tree) to find the threshold. A report is much more useful with a small artifact.

## Destination

This is Core ML framework behaviour, not a conversion bug: the same `.mlpackage` compiles
cleanly under two of three compute-unit settings. Per the coremltools maintainer's guidance
on [#2758](https://github.com/apple/coremltools/issues/2758), "If it works for one compute
type, but not for another compute, that is an issue with the Core ML Framework, not the
conversion", Feedback Assistant is the right destination, not the coremltools repo.
