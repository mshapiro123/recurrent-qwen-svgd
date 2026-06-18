#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
SPLIT="${SPLIT:-6,18}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
IDENTITY_DTYPE="${IDENTITY_DTYPE:-float32}"
IDENTITY_ATTN="${IDENTITY_ATTN:-eager}"

pytest -q tests

python eval/eval_identity.py \
  --model_name "$MODEL_NAME" \
  --split "$SPLIT" \
  --dtype "$IDENTITY_DTYPE" \
  --attn_implementation "$IDENTITY_ATTN" \
  --device "$DEVICE" \
  --threshold 1e-3

python eval/eval_halting.py \
  --model_name "$MODEL_NAME" \
  --split "$SPLIT" \
  --max_loops 4 \
  --dtype "$DTYPE" \
  --device "$DEVICE"

python eval/eval_trajectories.py \
  --model_name "$MODEL_NAME" \
  --split "$SPLIT" \
  --max_loops 2 \
  --num_trajectories 2 \
  --dtype "$DTYPE" \
  --device "$DEVICE"
