# Colab Pro+ Runbook

Use Colab Pro+ as the primary first environment for this project.

## Runtime Settings

In `Runtime > Change runtime type`:

```text
Runtime type: Python 3
Hardware accelerator: H100 GPU if available
High-RAM: On
Runtime version: Latest (recommended)
```

Fallback order:

1. H100 GPU: best choice for identity gates, 0.5B/1.5B experiments, and longer
   trajectory runs.
2. A100 GPU: excellent fallback; use this freely if H100 is unavailable or too
   expensive in compute units.
3. L4 GPU: good for 0.5B identity and short 0.5B Phase 1/2 tests.
4. T4 GPU: smoke tests only; keep `max_length`, `max_loops`, and
   `num_trajectories` low.
5. TPUs: skip for this repo. The implementation is PyTorch/Hugging Face CUDA
   oriented, not JAX/TPU oriented.

If `G4 GPU` is offered, verify it with `nvidia-smi` before using it. Prefer
H100, A100, or L4 when the choice is available.

## First Cell Checks

Always run:

```python
!nvidia-smi

import psutil, torch
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("ram_gb", psutil.virtual_memory().total / 1e9)
```

If CUDA is false or `nvidia-smi` fails, go back to `Runtime > Change runtime
type` and select a GPU.

## Identity Gate

Run the strict Phase 0 identity gate in `float32` with eager attention:

```bash
python eval/eval_identity.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --split 6,18 \
  --dtype float32 \
  --attn_implementation eager \
  --device cuda \
  --threshold 1e-3
```

This checks the graph split itself. Half-precision optimized attention can show
larger numerical drift even when the layer ordering is correct.

## Project Defaults by GPU

H100 or A100:

```text
0.5B: max_length 512-1024, max_loops 4, num_trajectories 2
1.5B: max_length 512-768, max_loops 4, num_trajectories 2
```

L4:

```text
0.5B: max_length 512, max_loops 4, num_trajectories 2
1.5B: use only after 0.5B passes; reduce max_length first if OOM
```

T4:

```text
0.5B: max_length 256, max_loops 2, num_trajectories 1-2
```

## Stochastic Trajectory Diagnostics

Before training, the latent adapter is intentionally tiny. In fp16, the initial
latent delta can round away and show zero trajectory diversity. That is not a
Phase 2 failure by itself.

To verify the stochastic path is wired, run a diagnostic-only amplified eval:

```bash
python eval/eval_trajectories.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --split 6,18 \
  --max_loops 4 \
  --num_trajectories 2 \
  --dtype float16 \
  --device cuda \
  --diagnostic_latent_scale 1.0 \
  --diagnostic_adapter_std 0.02
```

This should produce nonzero hidden deltas. Do not use those diagnostic values
as training defaults.

## Pro+ Features to Use

- Background execution: safe for smoke training and longer Phase 1/2 runs as
  long as compute units remain available.
- High-RAM: turn it on for this project; tokenizer/model loading and notebooks
  are less brittle.
- VS Code Colab extension: useful if you prefer editing notebooks locally while
  executing on Colab.
- Drive mount: use it for `outputs/`, checkpoints, logs, and datasets.

## Do Not Spend H100 Time On

- CPU runtime.
- Full 7B/9B experiments before 0.5B and 1.5B gates pass.
- High `K` trajectory counts. Keep `num_trajectories=2` until the signal is
  real.
- Long no-cache generation loops unless you are intentionally measuring
  inference behavior.
