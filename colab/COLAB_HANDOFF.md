# Colab Handoff

Use this when moving the project into Google Colab.

## Recommended Path

1. Zip this project directory.
2. Open `colab/GRAM_Recurrent_Qwen_Colab.ipynb` in Colab.
3. Choose `Runtime > Change runtime type > H100 GPU` and turn High-RAM on.
4. Run the notebook setup cells and upload the project zip when prompted.
5. Run tests.
6. Run the Phase 0 identity gate.
7. Continue to Phase 1/2 smoke runs only after identity passes.

For the exact Pro+ runtime choices, see `colab/COLAB_PRO_PLUS_RUNBOOK.md`.

## Expected First Gate

The identity command should report:

```text
max_abs_diff < 1e-3
PASS: identity wrapper drift is within threshold
```

If that does not pass, stop. Do not train halting or latent trajectories until
the manual split wrapper preserves logits.

## Colab Hardware Defaults

For high-end Colab Pro+ GPUs, Colab is a reasonable primary environment for
the first real gates. RunPod is still useful when you need guaranteed hardware,
long-lived sessions, or persistent volumes, but it is not required just to start.

Good Pro+ path:

1. Pass Phase 0 on `Qwen/Qwen2.5-0.5B-Instruct`.
2. Run Phase 1/2 smoke tests on 0.5B.
3. Move to `Qwen/Qwen2.5-1.5B-Instruct` with `layer_split: auto`.
4. Stay at `num_trajectories: 2` until trajectory diversity is measurable.

For free or low-tier Colab GPUs:

- Start with `Qwen/Qwen2.5-0.5B-Instruct`.
- Use `dtype=float16`.
- Keep `max_loops=2` for smoke tests and `4` for early real tests.
- Keep `num_trajectories=2` for Phase 2.
- Keep `max_length` at `256` or `512` until memory is understood.

For Pro+:

- 0.5B: use `max_length` 512-1024, `max_loops` 4, `num_trajectories` 2.
- 1.5B: use `max_length` 512-768 first, `max_loops` 4, `num_trajectories` 2.
- 7B/9B: still wait until the small-model gates pass.

## Artifacts

Training scripts save trainable-only checkpoints under `outputs/` when
`output_dir` is set in the config. These include bridge, halting, latent, and
LoRA parameters that were trainable in that phase. They do not include the full
Qwen base model.

## Common Failures

- CUDA out of memory: lower `max_length`, `max_loops`, or `num_trajectories`.
- Identity drift: stop and check `transformers` version and Qwen layer API.
- Slow generation: expected. Recurrent generation is intentionally no-cache.
- Hugging Face download errors: rerun the cell or authenticate with a token if
  your environment requires it.
