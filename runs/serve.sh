#!/bin/bash
set -e

MODEL="${1:-Qwen/Qwen2.5-14B-Instruct}"
TP="${2:-$(nvidia-smi -L | wc -l)}"

docker run --runtime nvidia --gpus all --rm \
    --name vllm-eval \
    -v /root:/root \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.17.1 \
    --model "$MODEL" \
    --tensor-parallel-size "$TP"
