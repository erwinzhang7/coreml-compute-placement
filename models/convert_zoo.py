#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert structurally different models to Core ML, so the findings stop resting on one ViT.

The README's first stated limitation is "one model family. A vision transformer,
which is an unusually good ANE fit. CNNs and decoder-only LLMs will behave
differently." Every number in this repo -- the ANE/GPU inversion, the
ALL-is-slowest result, the concurrency finding, the sustained soak -- comes from
SigLIP's vision tower. If the inversion is a property of ViTs rather than of the
chips, the central claim is much weaker than it reads.

WHAT IS HERE, chosen for structural difference rather than popularity:

    resnet50    dense convolutions, no attention at all
    mobilenet   depthwise-separable convolutions -- the ANE's best case, so a
                hostile test of "the ANE is not always the right answer"
    bert        text encoder: INTEGER token inputs, attention over a sequence
                rather than over image patches
    whisper     audio encoder: conv frontend into a transformer, and a much
                longer sequence (1500 positions) than any of the above

Together with the existing SigLIP ViT that is five shapes across four families.

EVERYTHING COMES FROM `transformers`, WHICH IS ALREADY A DEPENDENCY. Adding
torchvision or timm for two CNNs would grow the install for everyone who only
wants to reproduce the headline, and these are the same architectures.

CORRECTNESS IS CHECKED AGAINST THE CONVERTED MODEL, not against another torch
module. models/ane_siglip.py verifies its rewrite torch-against-torch, which is
the right check for a rewrite. Here the thing that can go wrong is the
CONVERSION, so the reference is torch fp32 and the candidate is the Core ML fp16
model actually being benchmarked. A model that converts to garbage would
otherwise still produce a throughput number, and a fast wrong answer is the one
failure mode a benchmark cannot afford.

COSINE IS THE GATE, not absolute error. These outputs have wildly different
scales -- ResNet pooled features, BERT hidden states and Whisper encoder states
are not comparable in magnitude -- so a single atol would be meaningless across
the zoo. Cosine similarity is scale-free. Absolute and relative error are printed
anyway, because they are what tells you *how* it failed.

    python models/convert_zoo.py --list
    python models/convert_zoo.py resnet50 --batch 16 --out resnet50-b16.mlpackage
    python models/convert_zoo.py --all --batch 16
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

import coremltools as ct
import numpy as np
import torch


class Pooled(torch.nn.Module):
    """Reduce a model's output to one vector per item.

    Every shape here is benchmarked as a feature extractor, so they all need to
    emit something comparable. Sequence models mean-pool; CNNs already pool.
    """

    def __init__(self, model, kind):
        super().__init__()
        self.model = model
        self.kind = kind

    def forward(self, x):
        if self.kind == "image_pooled":
            # ResNet/MobileNet expose pooler_output as (B, C, 1, 1).
            out = self.model(pixel_values=x).pooler_output
            return out.flatten(1)
        if self.kind == "image_seq":
            return self.model(pixel_values=x).last_hidden_state.mean(dim=1)
        if self.kind == "tokens":
            return self.model(input_ids=x).last_hidden_state.mean(dim=1)
        if self.kind == "audio":
            return self.model(input_features=x).last_hidden_state.mean(dim=1)
        raise ValueError(self.kind)


def _resnet50(batch):
    from transformers import ResNetModel
    m = ResNetModel.from_pretrained("microsoft/resnet-50", dtype=torch.float32).eval()
    return m, "image_pooled", "pixel_values", (batch, 3, 224, 224), torch.float32


def _mobilenet(batch):
    from transformers import MobileNetV2Model
    m = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224",
                                         dtype=torch.float32).eval()
    return m, "image_pooled", "pixel_values", (batch, 3, 224, 224), torch.float32


def _bert(batch):
    from transformers import DistilBertModel
    m = DistilBertModel.from_pretrained("distilbert-base-uncased",
                                        attn_implementation="eager",
                                        dtype=torch.float32).eval()
    # 128 tokens: long enough that attention is not a rounding error, short
    # enough to stay a realistic encoder workload.
    return m, "tokens", "input_ids", (batch, 128), torch.int32


def _whisper(batch):
    from transformers import WhisperModel
    m = WhisperModel.from_pretrained("openai/whisper-tiny",
                                     attn_implementation="eager",
                                     dtype=torch.float32).eval()
    enc = m.encoder.eval()
    n_mels = m.config.num_mel_bins
    frames = m.config.max_source_positions * 2  # encoder halves this via conv stride
    return enc, "audio", "input_features", (batch, n_mels, frames), torch.float32


SPECS = {
    "resnet50":  (_resnet50,  "dense convolutions, no attention"),
    "mobilenet": (_mobilenet, "depthwise-separable convolutions, the ANE's best case"),
    "bert":      (_bert,      "text encoder, integer token inputs"),
    "whisper":   (_whisper,   "audio encoder, conv frontend into a transformer"),
}


def example_input(shape, dtype):
    torch.manual_seed(0)
    if dtype == torch.int32:
        # Token ids. 1000 keeps well inside every vocab here and avoids special
        # tokens, whose embeddings are not representative of the average row.
        return torch.randint(1, 1000, shape, dtype=torch.int32)
    return torch.randn(*shape)


def verify(wrapper, mlmodel, input_name, example, tol_cos=0.999):
    """torch fp32 reference against the Core ML fp16 model actually benchmarked."""
    with torch.no_grad():
        ref = wrapper(example).float().numpy()
    got = mlmodel.predict({input_name: example.numpy()})
    got = np.asarray(next(iter(got.values())), dtype=np.float32)

    if ref.shape != got.shape:
        raise SystemExit(f"FAIL shape: torch {ref.shape} vs Core ML {got.shape}")

    diff = np.abs(ref - got)
    rel = (diff / np.maximum(np.abs(ref), 1e-6)).max()
    a = ref.reshape(ref.shape[0], -1)
    b = got.reshape(got.shape[0], -1)
    cos = float((
        (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    ).min())
    print(f"  max abs {diff.max():.3e}   max rel {rel:.3e}   min cosine {cos:.6f}")
    if not np.isfinite(cos) or cos < tol_cos:
        raise SystemExit(
            f"FAIL: converted model is not equivalent (min cosine {cos:.6f} < {tol_cos}). "
            f"Refusing to save -- a fast wrong answer is worse than no number.")
    print("  numerical equivalence: PASS")


def build(name, batch, out, tol_cos):
    fn, blurb = SPECS[name]
    print(f"loading {name}: {blurb}")
    model, kind, input_name, shape, dtype = fn(batch)
    wrapper = Pooled(model, kind).eval()
    example = example_input(shape, dtype)
    print(f"  input {input_name} {tuple(shape)} {str(dtype).replace('torch.','')}")

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)

    print("converting to Core ML (fp16, mlprogram, macOS15)")
    ct_dtype = np.int32 if dtype == torch.int32 else np.float32
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=input_name, shape=example.shape, dtype=ct_dtype)],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    print("verifying against the torch fp32 reference")
    verify(wrapper, mlmodel, input_name, example, tol_cos)
    mlmodel.save(out)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", choices=sorted(SPECS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="")
    ap.add_argument("--tol-cosine", type=float, default=0.999)
    args = ap.parse_args()

    if args.list:
        for n, (_, blurb) in sorted(SPECS.items()):
            print(f"  {n:<10} {blurb}")
        return
    if args.all:
        failed = []
        for n in sorted(SPECS):
            print(f"\n===== {n}")
            try:
                build(n, args.batch, f"{n}-b{args.batch}.mlpackage", args.tol_cosine)
            except SystemExit as e:
                # One model failing to convert must not silently take the others
                # with it: the point of the zoo is coverage, and a partial zoo is
                # still worth benchmarking as long as the gap is stated.
                print(f"  {n} FAILED: {e}")
                failed.append(n)
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"  {n} FAILED: {type(e).__name__}: {e}")
                failed.append(n)
        if failed:
            raise SystemExit("failed to build: " + ", ".join(failed))
        return
    if not args.model:
        raise SystemExit("give a model name, --all, or --list")
    build(args.model, args.batch,
          args.out or f"{args.model}-b{args.batch}.mlpackage", args.tol_cosine)


if __name__ == "__main__":
    main()
