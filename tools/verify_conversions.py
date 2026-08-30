#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Re-verify every converted model against a PyTorch fp32 reference, and COMMIT the result.

models/convert_zoo.py checks each conversion at build time and refuses to save a
model that fails. That check printed to a console and nowhere else, so the
conversion-correctness table in PAPER.md had no backing file: the numbers existed
only in a terminal I happened to be looking at. An audit caught it. This makes the
table reproducible on demand and writes it where the paper can cite it.

    python tools/verify_conversions.py --out results/conversion-check.json
"""
import argparse, json, platform, subprocess, sys, warnings
warnings.filterwarnings("ignore")
import coremltools as ct
import numpy as np
import torch
sys.path.insert(0, "models")
from convert_zoo import SPECS, Pooled, example_input  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results/conversion-check.json")
    a = ap.parse_args()

    host = {
        "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip(),
        "model": subprocess.run(["sysctl", "-n", "hw.model"],
                                capture_output=True, text=True).stdout.strip(),
        "macos": platform.mac_ver()[0],
        "coremltools": ct.__version__,
        "torch": torch.__version__,
        "python": ".".join(map(str, sys.version_info[:3])),
    }
    rows = []
    for name in sorted(SPECS):
        path = f"{name}-b{a.batch}.mlpackage"
        try:
            fn, _ = SPECS[name]
            model, kind, input_name, shape, dtype = fn(a.batch)
            wrapper = Pooled(model, kind).eval()
            ex = example_input(shape, dtype)
            with torch.no_grad():
                ref = wrapper(ex).float().numpy()
            ml = ct.models.MLModel(path)
            got = np.asarray(next(iter(ml.predict({input_name: ex.numpy()}).values())),
                             dtype=np.float32)
            diff = np.abs(ref - got)
            A = ref.reshape(ref.shape[0], -1); B = got.reshape(got.shape[0], -1)
            cos = float(((A * B).sum(1) /
                         (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)).min())
            rows.append({"model": name, "package": path, "batch": a.batch,
                         "input": input_name, "shape": list(shape),
                         "max_abs_err": float(diff.max()),
                         "max_rel_err": float((diff / np.maximum(np.abs(ref), 1e-6)).max()),
                         "min_cosine": cos, "status": "PASS" if cos >= 0.999 else "FAIL"})
            print(f"  {name:<10} max abs {diff.max():.3e}  min cosine {cos:.6f}  "
                  f"{rows[-1]['status']}")
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": name, "package": path, "status": "ERROR",
                         "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {name:<10} ERROR {type(exc).__name__}: {exc}")
    with open(a.out, "w") as fh:
        json.dump({"host": host, "gate": {"metric": "min_cosine", "threshold": 0.999},
                   "models": rows}, fh, indent=2)
    print(f"wrote {a.out}")
    return 1 if any(r["status"] != "PASS" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
