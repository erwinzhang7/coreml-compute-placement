#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SigLIP vision tower rewritten in ANE-friendly form, weights loaded from the HF checkpoint.

Follows the three principles from Apple's "Deploying Transformers on the Apple Neural
Engine" (apple/ml-ane-transformers):

  1. **(B, C, 1, S) layout**, not (B, S, C). The ANE moves data in an NCHW-shaped way;
     putting the sequence last is what lets it stream.
  2. **nn.Linear -> nn.Conv2d with a 1x1 kernel.** Mathematically identical - the weight is
     just (out, in) reshaped to (out, in, 1, 1) - but it maps to an ANE-native op.
  3. **Per-head chunked attention** instead of one big batched matmul, so each head's
     working set fits.

LayerNorm also has to be rewritten: nn.LayerNorm normalises the *last* dimension, which in
this layout is the sequence. We need it over the channel dim.

Correctness is not assumed. `verify()` checks this against the HuggingFace model on real
inputs and fails loudly on mismatch.
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn


class LayerNormANE(nn.Module):
    """LayerNorm over the channel dim of a (B, C, 1, S) tensor."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def linear_to_conv2d(linear: nn.Linear) -> nn.Conv2d:
    conv = nn.Conv2d(linear.in_features, linear.out_features, kernel_size=1)
    conv.weight.data = linear.weight.data[:, :, None, None].contiguous()
    conv.bias.data = linear.bias.data.contiguous()
    return conv


class ANEAttention(nn.Module):
    """Per-head chunked self-attention on (B, C, 1, S)."""

    def __init__(self, hf_attn, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q = linear_to_conv2d(hf_attn.q_proj)
        self.k = linear_to_conv2d(hf_attn.k_proj)
        self.v = linear_to_conv2d(hf_attn.v_proj)
        self.out = linear_to_conv2d(hf_attn.out_proj)
        self.head_dim = hf_attn.q_proj.out_features // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        q = self.q(x) * self.scale
        k = self.k(x)
        v = self.v(x)

        # split channels into per-head chunks; each is (B, head_dim, 1, S)
        qs = q.split(self.head_dim, dim=1)
        ks = k.split(self.head_dim, dim=1)
        vs = v.split(self.head_dim, dim=1)

        outs = []
        for qi, ki, vi in zip(qs, ks, vs):
            # (B, head_dim, 1, S_q) x (B, head_dim, 1, S_k) -> (B, S_k, 1, S_q)
            w = torch.einsum("bchq,bchk->bkhq", qi, ki)
            w = w.softmax(dim=1)
            # (B, head_dim, 1, S_k) x (B, S_k, 1, S_q) -> (B, head_dim, 1, S_q)
            outs.append(torch.einsum("bchk,bkhq->bchq", vi, w))

        return self.out(torch.cat(outs, dim=1))


class ANEMLP(nn.Module):
    def __init__(self, hf_mlp):
        super().__init__()
        self.fc1 = linear_to_conv2d(hf_mlp.fc1)
        self.fc2 = linear_to_conv2d(hf_mlp.fc2)

    def forward(self, x):
        # SigLIP uses gelu_pytorch_tanh
        return self.fc2(nn.functional.gelu(self.fc1(x), approximate="tanh"))


class ANEEncoderLayer(nn.Module):
    def __init__(self, hf_layer, num_heads):
        super().__init__()
        self.ln1 = _copy_ln(hf_layer.layer_norm1)
        self.attn = ANEAttention(hf_layer.self_attn, num_heads)
        self.ln2 = _copy_ln(hf_layer.layer_norm2)
        self.mlp = ANEMLP(hf_layer.mlp)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def _copy_ln(hf_ln) -> LayerNormANE:
    ln = LayerNormANE(hf_ln.weight.shape[0], eps=hf_ln.eps)
    ln.weight.data = hf_ln.weight.data.clone()
    ln.bias.data = hf_ln.bias.data.clone()
    return ln


class ANESiglipVision(nn.Module):
    """Encoder stack + mean pool, matching convert_siglip.py's output for comparability."""

    def __init__(self, hf_vision):
        super().__init__()
        cfg = hf_vision.config
        # transformers 5.x exposes embeddings/encoder/post_layernorm directly on
        # SiglipVisionModel; 4.x nests them under .vision_model.
        vm = getattr(hf_vision, "vision_model", hf_vision)
        emb = vm.embeddings
        # patch_embedding is already a Conv2d producing (B, C, H, W) - exactly the layout
        # we want, which is why vision transformers are a good ANE fit to begin with.
        self.patch_embedding = emb.patch_embedding
        pos = emb.position_embedding.weight.data  # (S, C)
        self.register_buffer("pos", pos.t()[None, :, None, :].contiguous())  # (1, C, 1, S)

        self.layers = nn.ModuleList(
            [ANEEncoderLayer(l, cfg.num_attention_heads) for l in vm.encoder.layers]
        )
        self.post_ln = _copy_ln(vm.post_layernorm)

    def forward(self, pixel_values):
        x = self.patch_embedding(pixel_values)          # (B, C, H, W)
        # flatten(2).unsqueeze(2) rather than reshape(b, c, 1, h * w): deriving the
        # sequence length from x.shape emits an aten::Int over a length-1 tensor, which
        # the coremltools 9.0 torch frontend cannot fold - the exact failure in
        # apple/coremltools#2755, fixed in main but not in the released wheel.
        x = x.flatten(2).unsqueeze(2)                   # (B, C, 1, S)
        x = x + self.pos
        for layer in self.layers:
            x = layer(x)
        x = self.post_ln(x)
        return x.mean(dim=3).squeeze(2)                 # mean over sequence -> (B, C)


def build(model_id="google/siglip-base-patch16-224"):
    from transformers import SiglipVisionModel

    hf = SiglipVisionModel.from_pretrained(
        model_id, attn_implementation="eager", dtype=torch.float32
    ).eval()
    return hf, ANESiglipVision(hf).eval()


def reference(hf, pixel_values):
    """What convert_siglip.py's baseline computes: encoder stack + mean pool."""
    return hf(pixel_values=pixel_values).last_hidden_state.mean(dim=1)


def verify(hf, ane, batch=2, tol=2e-3):
    size = hf.config.image_size
    torch.manual_seed(0)
    x = torch.randn(batch, 3, size, size)
    with torch.no_grad():
        ref = reference(hf, x)
        got = ane(x)
    if ref.shape != got.shape:
        raise SystemExit(f"shape mismatch: reference {tuple(ref.shape)} vs ANE {tuple(got.shape)}")
    diff = (ref - got).abs()
    rel = (diff / ref.abs().clamp(min=1e-6)).max().item()
    cos = nn.functional.cosine_similarity(ref, got, dim=-1).min().item()
    print(f"  max abs diff {diff.max().item():.3e}   max rel {rel:.3e}   min cosine {cos:.8f}")
    if diff.max().item() > tol:
        raise SystemExit(f"FAIL: rewrite is not numerically equivalent (tol {tol})")
    print("  numerical equivalence: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/siglip-base-patch16-224")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="siglip-ane-b16.mlpackage")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    print(f"building ANE rewrite of {args.model_id}")
    hf, ane = build(args.model_id)
    print("verifying against HuggingFace reference")
    verify(hf, ane)
    if args.verify_only:
        return

    import coremltools as ct

    size = hf.config.image_size
    example = torch.randn(args.batch, 3, size, size)
    with torch.no_grad():
        traced = torch.jit.trace(ane, example)

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
