#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
  # CUDA 12.1 wheels are a good default for current PyTorch GPU containers.
  pip install torch --index-url https://download.pytorch.org/whl/cu121
else
  pip install torch
fi

pip install -r requirements.txt
pip install sentencepiece safetensors

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
