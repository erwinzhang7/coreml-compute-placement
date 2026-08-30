# Core AI, and why this repo's question got sharper

Apple shipped **Core AI** at WWDC 2026: a layer above Core ML covering the whole
pipeline rather than just the runtime.

| piece | what it is |
| --- | --- |
| Core AI framework | Swift runtime API, OS-provided |
| [`apple/coreai-torch`](https://github.com/apple/coreai-torch) | PyTorch → Core AI IR, `pip install coreai-torch` |
| [`apple/coreai-optimization`](https://github.com/apple/coreai-optimization) | quantization and palettization |
| [`apple/coreai-models`](https://github.com/apple/coreai-models) | 26 export recipes, Python primitives, Swift runtime utilities |
| Core AI Debugger | Xcode integration, graph inspection, traces back to Python source |

Everything below was checked against the repositories themselves, at
`coreai-torch` v0.4.2 and `coreai-models` as of 2026-08-27.

## Placement did not go away. It was renamed and narrowed.

`coreai-torch` contains **no** reference to compute units, the Neural Engine, or
device placement anywhere in its 40 submodules — its exports are `TorchConverter`,
`TorchMetalKernel`, `composite_ops`, `externalize`, `debugging`. There is no
`coreai` runtime package on PyPI either. The Python surface is conversion only.

The knob is on the Swift side:

```swift
SpecializationOptions(preferredComputeUnitKind: .neuralEngine)  // or .gpu, .cpu
```

`SpecializationOptions` is not defined in `coreai-models`, so it is framework API.

| | Core ML | Core AI |
| --- | --- | --- |
| type | `MLComputeUnits` | `SpecializationOptions.preferredComputeUnitKind` |
| shape | allow-list | a single **preferred** unit |
| values | `.all`, `.cpuOnly`, `.cpuAndNE`, `.cpuAndGPU` | `.cpu`, `.gpu`, `.neuralEngine` |

Note *preferred*. It is a request, and nothing in the API reports what was
actually honoured — the same gap that made [finding 3](README.md#3-the-default-computeunitall-is-unpredictable-and-sometimes-worst)
necessary for `ComputeUnit.ALL`.

## Apple's own placement policy has no chip term

`swift/Sources/CoreAIShared/Runtime/ModelStructure.swift` derives the unit from
the **structure of the model** and nothing else:

| model structure | preferred unit |
| --- | --- |
| `chunkedStatic` (fixed batch, static shapes) | `.neuralEngine` |
| `multiFunctionSegmenter` | `.neuralEngine` |
| `dynamic` (single `main`, dynamic shapes) | `.gpu`, plus `expectFrequentReshapes` |

Two callers skip the heuristic and pin a unit outright: the diffusion pipeline
takes `.gpu`, the speech recognizer takes `.cpu`.

There is no chip anywhere in that decision.

**This repo's central measured result is that the optimal unit inverts between
chips for the same model** — ANE/GPU is 1.14x on an M4 Pro and 0.21x on an M5
Max. If a structure-only policy is right, that inversion should not matter. If
this repo's measurement is right, then on an M5 Max a `chunkedStatic` model is
routed to `.neuralEngine`: the unit measured here losing 4.7x to the GPU on that
chip.

Those two things cannot both be fully right, which makes this a claim about
shipping code rather than a general observation about an API.

## What is not established

Three things, and none of them should be skipped before the tension above is
presented as a result.

- **Whether the framework's own default agrees with these recipes.**
  `ModelStructure.swift` is Apple's published reference library. That is strong
  evidence about recommended practice and is *not* proof of what the Core AI
  runtime does when you say nothing.
- **Whether `preferredComputeUnitKind` is honoured.** It is a preference. It
  needs exactly the treatment finding 3 gave `ALL`: measure where the work landed
  rather than where it was asked to land.
- **Whether the inversion holds for a `chunkedStatic` decoder-LLM.** That is the
  structure the heuristic sends to the ANE, and everything measured in this repo
  so far is a vision transformer ([one model family](README.md#limitations) is
  already the first stated limitation). Until that is measured, the tension is an
  inference.

## The harness cannot measure Core AI yet

Every tool here drives Core ML from Python through `coremltools`. Core AI
inference is Swift, with no Python runtime package, so none of it applies. A
Swift benchmark runner is the one piece of work between this repo and covering
the pipeline; `coreai-models/swift/Sources` has 212 files to model it on,
including executable targets under `Tools/`.

Until that exists, everything in this file is a reading of Apple's source, not a
measurement — which is the opposite of how the rest of this repo is sourced, and
is the reason it lives in its own document instead of in the findings.
