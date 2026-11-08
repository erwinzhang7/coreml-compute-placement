#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The same SigLIP-base-224 vision tower on the Metal GPU through MLX: the comparison the
kill criterion names ("ANE-optimised Core ML below ~60% of Metal GPU throughput"). The
Core ML CPU_AND_GPU number is one Metal path; MLX is the one people actually run.

Same computation as the Core ML models: patch embed, 12 pre-LN encoder layers,
post-LayerNorm, mean pool over the 196 tokens (the attention-pool head is excluded from
both, see models/convert_siglip.py). fp16 weights and activations, batch 16 by default.
Verified against the HuggingFace fp32 reference by cosine similarity before timing.
"""
import argparse
import json
import statistics
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class Layer(nn.Module):
    def __init__(self, d=768, h=12, ff=3072, eps=1e-6):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d, eps=eps)
        self.ln2 = nn.LayerNorm(d, eps=eps)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)

    def __call__(self, x):
        B, L, D = x.shape
        y = self.ln1(x)
        q, k, v = (t.reshape(B, L, self.h, D // self.h).transpose(0, 2, 1, 3)
                   for t in (self.q(y), self.k(y), self.v(y)))
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=(D // self.h) ** -0.5)
        x = x + self.o(a.transpose(0, 2, 1, 3).reshape(B, L, D))
        return x + self.fc2(nn.gelu_approx(self.fc1(self.ln2(x))))


class Vision(nn.Module):
    def __init__(self, n=12, d=768, patch=16, size=224, eps=1e-6):
        super().__init__()
        self.patch = nn.Conv2d(3, d, patch, stride=patch)
        self.pos = mx.zeros(((size // patch) ** 2, d))
        self.layers = [Layer(d) for _ in range(n)]
        self.post = nn.LayerNorm(d, eps=eps)

    def __call__(self, x):  # x: NHWC
        x = self.patch(x)
        B = x.shape[0]
        x = x.reshape(B, -1, x.shape[-1]) + self.pos
        for l in self.layers:
            x = l(x)
        return self.post(x).mean(axis=1)


def load(model_id, dtype):
    import torch
    from transformers import SiglipVisionModel
    hf = SiglipVisionModel.from_pretrained(model_id, attn_implementation="eager",
                                           dtype=torch.float32).eval()
    sd = {k: v.numpy() for k, v in hf.state_dict().items()}
    # The prefix moved between transformers versions ("vision_model." or none); find it.
    p = next(k for k in sd if k.endswith("embeddings.patch_embedding.weight"))[
        : -len("embeddings.patch_embedding.weight")]
    m = Vision()
    w = {"patch.weight": sd[p + "embeddings.patch_embedding.weight"].transpose(0, 2, 3, 1),
         "patch.bias": sd[p + "embeddings.patch_embedding.bias"],
         "pos": sd[p + "embeddings.position_embedding.weight"],
         "post.weight": sd[p + "post_layernorm.weight"], "post.bias": sd[p + "post_layernorm.bias"]}
    names = {"ln1": "layer_norm1", "ln2": "layer_norm2", "q": "self_attn.q_proj",
             "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.out_proj",
             "fc1": "mlp.fc1", "fc2": "mlp.fc2"}
    for i in range(12):
        for ours, theirs in names.items():
            for t in ("weight", "bias"):
                w[f"layers.{i}.{ours}.{t}"] = sd[f"{p}encoder.layers.{i}.{theirs}.{t}"]
    m.load_weights([(k, mx.array(v).astype(dtype)) for k, v in w.items()])
    mx.eval(m.parameters())
    return hf, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/siglip-base-patch16-224")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--power", action="store_true", help="sample macmon per repeat")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dtype = getattr(mx, a.dtype)

    import torch
    hf, m = load(a.model_id, dtype)

    torch.manual_seed(0)
    xt = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        ref = hf(pixel_values=xt).last_hidden_state.mean(dim=1).numpy()
    got = np.array(m(mx.array(xt.numpy().transpose(0, 2, 3, 1)).astype(dtype)).astype(mx.float32))
    cos = float(min((r @ g) / np.linalg.norm(r) / np.linalg.norm(g) for r, g in zip(ref, got)))
    print(f"verify vs HF fp32: min cosine {cos:.6f}  max abs diff {np.abs(ref - got).max():.3e}")
    if cos < 0.999:
        sys.exit("FAIL: MLX tower does not match the reference")

    f = mx.compile(m)
    x = mx.random.uniform(shape=(a.batch, 224, 224, 3)).astype(dtype)
    for _ in range(a.warmup):
        mx.eval(f(x))
    pw = None
    if a.power:
        sys.path.insert(0, __file__.rsplit("/", 1)[0])
        from dual_ane import Power
        pw = Power()
        time.sleep(1)
    runs, power = [], []
    for _ in range(a.repeats):
        w0 = time.time()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            mx.eval(f(x))  # synchronous per call, like Core ML predict()
        runs.append(a.batch * a.iters / (time.perf_counter() - t0))
        if pw:
            power.append(pw.window(w0, time.time()))
    if pw:
        pw.stop()
    med = statistics.median(runs)
    if power and all(power):
        gpu_w = statistics.median(p["gpu_power"] for p in power)
        all_w = statistics.median(p["all_power"] for p in power)
        print(f"  power: GPU {gpu_w:.2f} W, CPU+GPU+ANE {all_w:.2f} W -> "
              f"{all_w / med * 1000:.1f} mJ/img (GPU only {gpu_w / med * 1000:.1f})")
    print(f"MLX {a.dtype} batch {a.batch}: median {med:.1f} img/s  "
          f"spread {(max(runs) - min(runs)) / med * 100:.1f}%  runs {[round(r, 1) for r in runs]}")
    json.dump({"backend": "mlx", "mlx": mx.__version__, "dtype": a.dtype, "batch": a.batch,
               "iters": a.iters, "repeats": a.repeats, "runs": runs, "median": med,
               "verify_min_cosine": cos, "power": power}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
