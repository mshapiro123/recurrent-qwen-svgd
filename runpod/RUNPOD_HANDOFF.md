# RunPod Handoff

RunPod is the best next environment if your local GPU is small. Use it for the
real Phase 0 identity gate and early Phase 1/2 experiments; keep Colab as a
portable demo or backup.

## GPU Choice

Start modest:

- T4 / L4 / A4000 class: good for Phase 0 identity and basic telemetry on 0.5B.
- A10 / RTX 3090 / RTX 4090 class with 24GB VRAM: better default for 0.5B and
  1.5B Phase 1/2 experiments.
- A100/H100 class: unnecessary until the small-model gates pass.

For this project, VRAM pressure grows with:

- `max_length`
- `max_loops`
- `num_trajectories`
- model size

So the first serious RunPod run should still use:

```text
Qwen/Qwen2.5-0.5B-Instruct
max_length: 256 or 512
max_loops: 2 for smoke, 4 for early real runs
num_trajectories: 2
```

## Pod Template

Use a PyTorch CUDA template with an attached persistent volume. Work under
`/workspace` so model caches, checkpoints, and logs survive pod restarts.

Suggested disk:

- 30-50GB for smoke tests
- 100GB+ once you start keeping checkpoints and datasets

## Setup

Upload or clone the repo into `/workspace`.

```bash
cd /workspace
git clone <YOUR_REPO_URL> gram-recurrent-qwen
cd gram-recurrent-qwen
bash scripts/runpod_setup.sh
```

If you upload a zip instead:

```bash
cd /workspace
unzip gram-recurrent-qwen.zip -d gram-recurrent-qwen
cd gram-recurrent-qwen
bash scripts/runpod_setup.sh
```

## First Gates

```bash
bash scripts/run_smoke_gates.sh
```

The critical gate is:

```text
max_abs_diff < 1e-3
PASS: identity wrapper drift is within threshold
```

If identity fails, stop. Do not train Phase 1 or Phase 2 until the split wrapper
preserves the base model logits.

## Tiny Training Smoke Run

Create a tiny JSONL file:

```bash
mkdir -p data
cat > data/smoke_train.jsonl <<'EOF'
{"prompt":"Solve: 2 + 3 = ","completion":"5","cot_tokens":16}
{"prompt":"If x + 2 = 5, x = ","completion":"3","cot_tokens":24}
{"prompt":"Find one valid 4-queens placement: ","completion":"[2, 4, 1, 3]","cot_tokens":64}
EOF
```

Then temporarily lower config values if needed:

```bash
python training/train_phase1_ponder.py \
  --config config/qwen_0_5b_phase1.yaml \
  --train_jsonl data/smoke_train.jsonl \
  --device cuda
```

For Phase 2:

```bash
python training/train_phase2_stochastic.py \
  --config config/qwen_0_5b_phase2.yaml \
  --train_jsonl data/smoke_train.jsonl \
  --device cuda
```

Trainable-only checkpoints are saved under `outputs/`.

## Practical Notes

- Use `tmux` or RunPod's persistent terminal for long runs.
- Keep `/workspace/.cache/huggingface` on the persistent volume if possible.
- If you hit CUDA OOM, lower `max_length` first, then `max_loops`, then
  `num_trajectories`.
- Do not use 9B until 0.5B and ideally 1.5B pass identity, halting, and
  trajectory gates.
