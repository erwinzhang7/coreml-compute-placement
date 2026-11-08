#!/usr/bin/env bash
# Separate venv for the MLX bandwidth and LLM runs (pinned the same on every machine), plus the
# model download. Untimed: run it before the timed jobs so no download sits inside one.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}
if [[ ! -x .venv-llm/bin/python ]]; then
    /opt/homebrew/bin/python3.12 -m venv .venv-llm
    .venv-llm/bin/pip install --quiet --upgrade pip
    .venv-llm/bin/pip install --quiet mlx==0.32.4 mlx-lm==0.32.2
fi
.venv-llm/bin/python -c "import mlx.core as mx, importlib.metadata as m; print('mlx', mx.__version__, 'mlx-lm', m.version('mlx-lm'))"
.venv-llm/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL'))"
mkdir -p bin && cc -O3 -o bin/membw tools/membw.c
