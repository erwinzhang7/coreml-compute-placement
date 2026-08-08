#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Is the M5 Max GPU advantage matmul-specific, or general?

M5 Max is ~6x the M4 Pro on a SigLIP vision tower for 2x the GPU core count. Two
explanations fit that: M5's per-core GPU neural accelerators (matmul-specific), or a
general uplift in clocks/bandwidth/cores (workload-agnostic).

These separate. Build two Core ML models of comparable runtime:

  matmul  - a stack of square GEMMs, arithmetic-bound, the thing tensor-style units target
  elemwise- a stack of fused multiply-adds on a large tensor, bandwidth-bound, no GEMM

Then compare the M5 Max / M4 Pro GPU ratio for each.

  matmul ratio >> elemwise ratio  -> matmul-specific hardware, i.e. neural accelerators
  ratios roughly equal           -> general GPU improvement, hypothesis rejected

The ANE is a useful built-in control: both chips have the same 16-core NPU, and ANE
throughput differed by only 1.14x, so whatever changed did not touch the NPU.
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

import coremltools as ct
import torch
import torch.nn as nn


class Matmul(nn.Module):
    """Arithmetic-bound: repeated square GEMMs, tiny activation memory."""

    def __init__(self, dim=2048, depth=24):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(depth)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class Elemwise(nn.Module):
    """Bandwidth-bound: fused multiply-adds over a large tensor, no matmul at all."""

    def __init__(self, dim=2048, depth=24):
        super().__init__()
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.randn(dim) * 0.01 + 1.0) for _ in range(depth)]
        )
        self.offsets = nn.ParameterList(
            [nn.Parameter(torch.randn(dim) * 0.01) for _ in range(depth)]
        )

    def forward(self, x):
        for s, o in zip(self.scales, self.offsets):
            x = x * s + o
        return x


def build(kind, batch, dim, depth, out):
    model = (Matmul(dim, depth) if kind == "matmul" else Elemwise(dim, depth)).eval()
    ex = torch.randn(batch, dim)
    with torch.no_grad():
        traced = torch.jit.trace(model, ex)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=ex.shape)],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    ml.save(out)
    # FLOPs / bytes for a rough intensity figure
    if kind == "matmul":
        work = 2 * batch * dim * dim * depth
        print(f"  {out}: {work/1e9:.1f} GFLOP per call")
    else:
        work = batch * dim * depth * 2 * 2  # read+write, fp16
        print(f"  {out}: {work/1e6:.1f} MB of traffic per call")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--elem-batch", type=int, default=8192,
                    help="elemwise needs a much larger tensor to leave the "
                         "per-call overhead floor and actually hit bandwidth")
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=24)
    args = ap.parse_args()
    build("matmul", args.batch, args.dim, args.depth, "probe-matmul.mlpackage")
    build("elemwise", args.elem_batch, args.dim, args.depth, "probe-elemwise.mlpackage")


if __name__ == "__main__":
    main()
