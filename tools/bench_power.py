#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sustained-load driver for power measurement. Run under an external powermetrics.

Prints a start marker, hammers the model for `--seconds`, prints a stop marker and the
image count. Pair with `powermetrics --samplers ane_power,gpu_power` to get joules per
image, which is the number the ANE was actually designed to win on.
"""
import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import coremltools as ct
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--units", required=True, choices=[u.name for u in ct.ComputeUnit])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    model = ct.models.MLModel(args.model, compute_units=ct.ComputeUnit[args.units])
    spec = model.get_spec()
    name = spec.description.input[0].name
    shape = tuple(spec.description.input[0].type.multiArrayType.shape)
    x = {name: np.random.rand(*shape).astype(np.float32)}

    for _ in range(5):
        model.predict(x)

    print(f"LOAD_START {args.units}", flush=True)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < args.seconds:
        model.predict(x)
        n += 1
    dt = time.perf_counter() - t0
    print(f"LOAD_STOP calls={n} images={n * args.batch} seconds={dt:.2f} "
          f"img_per_s={n * args.batch / dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
