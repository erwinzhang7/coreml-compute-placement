#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert a SigLIP vision tower to Core ML, naively - no ANE rewrite.

This is the baseline arm of milestone 1 in the ANE spec: what does an off-the-shelf
`ct.convert()` of a real image encoder actually place on the Neural Engine? The answer
is the thing every "runs on ANE" claim skips.

Deliberately does NOT apply the apple/ml-ane-transformers treatment ((B,C,1,S) layout,
Linear->Conv2d, chunked attention). That is the comparison arm, built later.
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

import coremltools as ct
import torch
from transformers import SiglipVisionModel


class VisionTower(torch.nn.Module):
    """Encoder stack plus a mean pool, yielding one embedding vector per image.

    SigLIP's own `pooler_output` comes from SiglipMultiheadAttentionPoolingHead, which
    wraps `nn.MultiheadAttention`. That does not survive the Core ML torch frontend: it
    traces `head_dim = embed_dim // num_heads` as operations on Long *tensors*, producing
    `aten::Int` nodes whose input is not 0-dimensional, and the converter dies with
      TypeError: only 0-dimensional arrays can be converted to Python scalars
    All 9 such nodes in the traced graph are inside that head; the 12-layer encoder body
    is clean.

    Mean pooling is substituted so the model still emits an embedding. For a residency and
    throughput measurement this is the right trade: the head is a single cross-attention
    with one query token against 196 keys, next to 12 full transformer layers, so it is a
    rounding error in the compute profile. It is NOT equivalent for retrieval quality - a
    traceable reimplementation of the real head is needed before these embeddings mean
    anything semantically.
    """

    def __init__(self, model, pool="mean"):
        super().__init__()
        self.model = model
        self.pool = pool

    def forward(self, pixel_values):
        hidden = self.model(pixel_values=pixel_values).last_hidden_state
        if self.pool == "mean":
            return hidden.mean(dim=1)
        return hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/siglip-base-patch16-224")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default="siglip-vision.mlpackage")
    ap.add_argument("--pool", default="mean", choices=["mean", "none"])
    args = ap.parse_args()

    print(f"loading {args.model_id}")
    hf = SiglipVisionModel.from_pretrained(
        args.model_id, attn_implementation="eager", dtype=torch.float32
    ).eval()

    size = hf.config.image_size
    print(f"  image_size={size} hidden={hf.config.hidden_size} "
          f"layers={hf.config.num_hidden_layers} heads={hf.config.num_attention_heads}")

    wrapper = VisionTower(hf, pool=args.pool).eval()
    example = torch.randn(args.batch, 3, size, size)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)

    print("converting to Core ML (fp16, mlprogram, macOS15)")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=example.shape)],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    mlmodel.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
