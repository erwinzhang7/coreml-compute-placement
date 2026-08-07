#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""anecheck - report where a Core ML model's ops actually run, and fail if not the ANE.

Core ML places unsupported ops on GPU/CPU and reports success. You get correct outputs,
no warning, and none of the speedup. Most "running X on the ANE" claims were never
checked. This turns the question into a number you can assert on in CI.

    anecheck.py model.mlpackage
    anecheck.py model.mlpackage --assert-ane-fraction 0.95
    anecheck.py model.mlpackage --json report.json --compute-units CPU_AND_NE

Residency is reported two ways, and the difference matters:

  * **by cost**   - weighted by MLComputePlan's per-op estimated cost. This is the number
                    to care about. One heavy matmul on the CPU outweighs fifty cheap ANE ops.
  * **by op count** - unweighted. Useful for spotting a single stubborn op type.

`const` ops are excluded by default: they are weights being materialised, not compute, and
counting them inflates or deflates residency depending only on how many weights a model has.
Pass --include-const to see them.

Note on `preferred` vs `supported`: an op can be ANE-*capable* and still be placed on the
CPU, and that is the common silent-fallback shape. Both are reported.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

_DEVICE_SHORT = {
    "MLCPUComputeDevice": "CPU",
    "MLGPUComputeDevice": "GPU",
    "MLNeuralEngineComputeDevice": "ANE",
}


def _short(device) -> str:
    if device is None:
        return "?"
    return _DEVICE_SHORT.get(type(device).__name__, type(device).__name__)


def ensure_compiled(path: str, workdir: str) -> str:
    """Return a path to a compiled .mlmodelc, compiling an .mlpackage if needed.

    This is not a nicety. Handing MLComputePlan.load_from_path an uncompiled .mlpackage
    aborts the whole process with SIGABRT from a CoreML dispatch queue - uncatchable from
    Python. See apple/coremltools#2757. Compiling up front sidesteps it entirely.
    """
    if os.path.isfile(os.path.join(path, "coremldata.bin")):
        return path

    if not path.rstrip("/").endswith((".mlpackage", ".mlmodel")):
        raise SystemExit(f"not a model path: {path}")

    # Hold the MLModel reference: get_compiled_model_path() hands back a tempdir owned by
    # the model object, deleted as soon as it is collected.
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = os.path.join(workdir, "model.mlmodelc")
    shutil.copytree(model.get_compiled_model_path(), compiled)
    del model
    return compiled


def analyse(compiled_path: str, compute_units: ct.ComputeUnit, include_const: bool):
    plan = MLComputePlan.load_from_path(path=compiled_path, compute_units=compute_units)
    structure = plan.model_structure

    if structure.program is None:
        raise SystemExit(
            "Only mlprogram models are supported. This looks like a neuralnetwork or "
            "pipeline model; convert with convert_to='mlprogram'."
        )

    def walk(block, func_name):
        """Recurse into nested blocks - ops inside cond/while bodies are real ops.

        Only walking `function.block.operations` silently under-counts any model with
        control flow, which then reports a residency fraction over a subset of the graph.
        """
        for op in block.operations:
            yield func_name, op
            for nested in getattr(op, "blocks", ()) or ():
                yield from walk(nested, func_name)

    rows = []
    for func_name, function in structure.program.functions.items():
        for fname, op in walk(function.block, func_name):
            name = op.operator_name
            if name == "const" and not include_const:
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            cost = plan.get_estimated_cost_for_mlprogram_operation(op)
            rows.append(
                {
                    "function": fname,
                    "operator": name,
                    "preferred": _short(usage.preferred_compute_device) if usage else "?",
                    "supported": sorted(
                        _short(d) for d in (usage.supported_compute_devices if usage else [])
                    ),
                    "cost": float(cost.weight) if cost is not None else None,
                }
            )
    return rows


def summarise(rows):
    by_count = Counter(r["preferred"] for r in rows)
    by_cost = defaultdict(float)
    total_cost = 0.0
    for r in rows:
        w = r["cost"] or 0.0
        by_cost[r["preferred"]] += w
        total_cost += w

    n = len(rows) or 1
    count_frac = {d: c / n for d, c in by_count.items()}
    cost_frac = {d: c / total_cost for d, c in by_cost.items()} if total_cost else {}

    # ops the ANE could have taken but did not get
    missed = Counter(
        r["operator"] for r in rows if r["preferred"] != "ANE" and "ANE" in r["supported"]
    )
    # Only meaningful when the ANE was actually on the table. Under CPU_ONLY or
    # CPU_AND_GPU the device is absent from every op's supported list by construction,
    # which says nothing about whether the op could run there.
    ane_offered = any("ANE" in r["supported"] for r in rows)
    unsupported = (
        Counter(r["operator"] for r in rows if "ANE" not in r["supported"])
        if ane_offered
        else Counter()
    )
    return {
        "ane_offered": ane_offered,
        "ops": len(rows),
        "total_cost": total_cost,
        "by_count": dict(by_count),
        "by_count_fraction": count_frac,
        "by_cost_fraction": cost_frac,
        "ane_fraction_by_cost": cost_frac.get("ANE", 0.0),
        "ane_fraction_by_count": count_frac.get("ANE", 0.0),
        "ane_capable_but_not_placed": dict(missed),
        "not_ane_capable": dict(unsupported),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help=".mlpackage or .mlmodelc path")
    ap.add_argument("--compute-units", default="CPU_AND_NE",
                    choices=[u.name for u in ct.ComputeUnit],
                    help="CPU_AND_NE is the honest way to ask the question (default)")
    ap.add_argument("--assert-ane-fraction", type=float, metavar="F",
                    help="exit 1 if ANE residency by cost is below F")
    ap.add_argument("--include-const", action="store_true",
                    help="count const ops (weights) as operations")
    ap.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    ap.add_argument("--per-op", action="store_true", help="print every operation")
    args = ap.parse_args()

    units = ct.ComputeUnit[args.compute_units]
    with tempfile.TemporaryDirectory() as workdir:
        compiled = ensure_compiled(args.model, workdir)
        rows = analyse(compiled, units, args.include_const)

    summary = summarise(rows)

    print(f"model         {args.model}")
    print(f"compute_units {units.name}")
    print(f"operations    {summary['ops']}"
          f"{'' if args.include_const else '  (const excluded)'}")
    print()
    print(f"{'device':>6}  {'by cost':>9}  {'by count':>9}")
    for dev in sorted(set(summary["by_count"]) | set(summary["by_cost_fraction"])):
        print(f"{dev:>6}  {summary['by_cost_fraction'].get(dev, 0.0):>8.1%}  "
              f"{summary['by_count_fraction'].get(dev, 0.0):>8.1%}")

    if summary["ane_capable_but_not_placed"]:
        print("\nANE-capable but placed elsewhere (the silent-fallback ops):")
        for opn, c in sorted(summary["ane_capable_but_not_placed"].items(),
                             key=lambda x: -x[1])[:12]:
            print(f"  {opn:24s} x{c}")

    if not summary["ane_offered"]:
        print("\nNote: the Neural Engine was not offered in these compute units, so"
              "\n      per-op ANE capability cannot be determined from this run.")

    if summary["not_ane_capable"]:
        print("\nNo ANE support at all:")
        for opn, c in sorted(summary["not_ane_capable"].items(), key=lambda x: -x[1])[:12]:
            print(f"  {opn:24s} x{c}")

    if args.per_op:
        print(f"\n{'operator':24s} {'pref':>5}  supported")
        for r in rows:
            print(f"  {r['operator']:22s} {r['preferred']:>5}  {','.join(r['supported'])}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"model": args.model, "compute_units": units.name,
                       "summary": summary, "operations": rows}, f, indent=2)
        print(f"\nwrote {args.json}")

    if args.assert_ane_fraction is not None:
        got = summary["ane_fraction_by_cost"]
        if got < args.assert_ane_fraction:
            print(f"\nFAIL: ANE residency by cost {got:.1%} < required "
                  f"{args.assert_ane_fraction:.1%}")
            return 1
        print(f"\nPASS: ANE residency by cost {got:.1%} >= "
              f"{args.assert_ane_fraction:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
