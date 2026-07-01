#!/usr/bin/env bash
# Start a local vLLM server for development.
# Requires: Linux or WSL2 with a CUDA-capable GPU and vLLM installed.
# Install:  pip install vllm
# Usage:    bash scripts/start_vllm_local.sh
set -euo pipefail

MODEL="${LOCAL_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "Starting vLLM server: $MODEL on port 8000"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.40 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --enforce-eager
