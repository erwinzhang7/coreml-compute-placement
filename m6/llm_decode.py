#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""LLM prefill and decode speed with MLX, against the memory-bandwidth ceiling.

Fixed prompt, greedy, exactly --gen tokens (generate_step does not stop at EOS, so every run
decodes the same count), median of --repeats after one warm-up run.

  prefill tok/s = prompt tokens / time to first token
  decode  tok/s = (gen - 1) / time from first token to last

Decode on a dense model reads every weight once per token, so its ceiling is
bandwidth / weight bytes. Weight bytes are summed from the loaded parameters (quantized
weights plus scales and biases, embeddings included; tied embeddings counted once).
"""
import argparse
import json
import platform
import statistics
import subprocess
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

PROMPT = ("You are reviewing a technical report on memory bandwidth in unified-memory "
          "laptops. Summarise, section by section, how bandwidth limits token generation "
          "for large language models, why prefill is compute-bound while decode is "
          "bandwidth-bound, and what that implies for choosing between a laptop with a "
          "small GPU and one with a large GPU. ") * 8


def run_once(model, prompt, gen, sampler):
    t0 = time.perf_counter()
    t_first = None
    n = 0
    for tok, _ in generate_step(prompt, model, max_tokens=gen, sampler=sampler):
        mx.eval(tok) if isinstance(tok, mx.array) else None
        n += 1
        if n == 1:
            t_first = time.perf_counter()
    t_end = time.perf_counter()
    return {"prompt_tokens": int(prompt.size), "generated": n,
            "ttft_s": t_first - t0,
            "prefill_tps": prompt.size / (t_first - t0),
            "decode_tps": (n - 1) / (t_end - t_first)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--bw-json", help="stream_bw.py output for this machine, for the ceiling")
    ap.add_argument("--published", type=float, help="published peak GB/s")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model, tok = load(a.model)
    weight_bytes = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True)
    prompt = mx.array(ids)
    sampler = make_sampler(temp=0.0)

    run_once(model, prompt, a.gen, sampler)  # warm-up: kernels, buffers
    runs = []
    for i in range(a.repeats):
        r = run_once(model, prompt, a.gen, sampler)
        runs.append(r)
        print(f"  run {i}: prefill {r['prefill_tps']:7.1f} tok/s ({r['prompt_tokens']} tok, "
              f"ttft {r['ttft_s'] * 1000:.0f} ms)  decode {r['decode_tps']:6.2f} tok/s  "
              f"({r['generated']} generated)", flush=True)

    dec = statistics.median(r["decode_tps"] for r in runs)
    pre = statistics.median(r["prefill_tps"] for r in runs)
    out = {"model": a.model, "weight_bytes": weight_bytes, "gen": a.gen, "repeats": a.repeats,
           "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip(),
           "macos": platform.mac_ver()[0], "mlx": mx.__version__,
           "peak_memory_GB": mx.get_peak_memory() / 1e9,
           "runs": runs, "median_decode_tps": dec, "median_prefill_tps": pre,
           "decode_spread_pct": (max(r["decode_tps"] for r in runs) -
                                 min(r["decode_tps"] for r in runs)) / dec * 100}
    gb = weight_bytes / 1e9
    line = f"decode {dec:.2f} tok/s, prefill {pre:.1f} tok/s, weights {gb:.2f} GB"
    if a.bw_json:
        bw = json.load(open(a.bw_json))["best_GBps"]
        out["measured_bw_GBps"] = bw
        out["ceiling_measured_bw_tps"] = bw / gb
        out["decode_fraction_of_measured_ceiling"] = dec / (bw / gb)
        line += f"; ceiling at measured {bw:.0f} GB/s = {bw / gb:.1f} tok/s ({dec / (bw / gb):.0%})"
    if a.published:
        out["ceiling_published_bw_tps"] = a.published / gb
        out["decode_fraction_of_published_ceiling"] = dec / (a.published / gb)
        line += f"; at published {a.published:g} GB/s = {a.published / gb:.1f} tok/s ({dec / (a.published / gb):.0%})"
    out["effective_decode_GBps"] = dec * gb
    print(line)
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
