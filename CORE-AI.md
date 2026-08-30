# Core AI, and why this repo's question got sharper

Apple shipped **Core AI** at WWDC 2026: a layer above Core ML covering the whole
pipeline rather than just the runtime.

| piece | what it is |
| --- | --- |
| Core AI framework | Swift runtime API, OS-provided |
| [`apple/coreai-torch`](https://github.com/apple/coreai-torch) | PyTorch to Core AI IR, `pip install coreai-torch` |
| [`apple/coreai-optimization`](https://github.com/apple/coreai-optimization) | quantization and palettization |
| [`apple/coreai-models`](https://github.com/apple/coreai-models) | 26 export recipes, Python primitives, Swift runtime utilities |
| Core AI Debugger | Xcode integration, graph inspection, traces back to Python source |

Everything below was checked against the repositories themselves, at
`coreai-torch` v0.4.2 and `coreai-models` as of 2026-08-27.

## Placement did not go away. It was renamed and narrowed.

`coreai-torch` contains **no** reference to compute units, the Neural Engine, or
device placement anywhere in its 40 submodules. Its exports are `TorchConverter`,
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
actually honoured. That is the same gap that made [finding 3](README.md#3-the-default-computeunitall-is-unpredictable-and-sometimes-worst)
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
chips for the same model**. ANE/GPU is 1.14x on an M4 Pro and 0.21x on an M5
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
  structure the heuristic sends to the ANE, and nothing measured in this repo is
  a decoder-LLM: the five architectures of PAPER.md §2.3 are a vision
  transformer, two CNNs, a text encoder and an audio encoder. An earlier version
  of this line said everything measured here was a vision transformer, which
  stopped being true when the zoo was added, and §3.1 makes a point of the
  inversion not being a ViT property. Until that is measured, the tension is an
  inference.

## Core AI cannot be measured on this hardware at all

Not because the harness drives Core ML from Python, which was the earlier reading.
The runtime framework does not exist on any machine here.

    canImport(CoreAI)                       false, Swift 6.3.3, target macosx26.0
    /System/Library/Frameworks/CoreAI.*     absent
    /System/Library/PrivateFrameworks/      absent
    Xcode 26.6 SDK                          CoreML.framework yes,
                                            FoundationModels.framework yes,
                                            CoreAI.framework NO

Apple's own package says why. `coreai-models/Package.swift` declares

```swift
platforms: [.macOS("27.0"), .iOS("27.0")]
```

and its README requires **Xcode 27.0+**. This fleet runs macOS 26.5.1 and 26.6
with Xcode 26.6, so the framework is a major OS version away.

What that means for this repo:

- The conversion side is usable today. `coreai-torch` is a pure-Python wheel, it
  installs on 3.11+, and everything in this document about placement APIs and the
  structure-based heuristic was read from source that runs anywhere.
- The **execution** side is unreachable. No throughput number, no residency
  measurement, and no check of whether `preferredComputeUnitKind` is honoured can
  be taken until macOS 27 is on at least one box.
- So the tension this document sets out, between a structure-only placement policy
  and a measured chip-dependent optimum, stays a reading of Apple's source rather
  than a measurement. That is the honest status and it is not fixable by writing
  more code.

A Swift benchmark runner is still the right shape when the OS arrives:
`coreai-models/swift/Sources` has 212 files including executable targets under
`Tools/` to model it on. It is not worth writing before then, because it could not
be compiled, let alone run.
