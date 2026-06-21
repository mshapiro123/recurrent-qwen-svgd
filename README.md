# GRAM-Inspired Recurrent-Particle Qwen

This repository studies whether a pretrained dense Qwen model can be surgically
converted into a recurrent latent-reasoning model, recovered with a small
trainable parameter budget, and then extended with multiple particle-like latent
reasoning trajectories. The current base is
`Qwen/Qwen2.5-0.5B-Instruct`; the project deliberately stays small until the
identity, recurrence, recovery, and particle-selection gates are reproducible.

The work is organized like a model-surgery paper rather than a conventional SFT
run. The central question is:

> Can we preserve most of a trained model's competence after converting it into
> a recurrent-depth architecture, then use learned depth and latent particles to
> surpass the original model on hard reasoning?

The implementation has three stages:

1. **Identity-preserving surgery.** Split Qwen into Prelude, Recurrent Block,
   and Coda. With one recurrent pass, the wrapper must reproduce the original
   logits exactly under strict float32/eager settings.
2. **Deterministic recurrent recovery.** Train only LoRA adapters in the
   recurrent block, a gated identity bridge, and a sequence-level PonderNet
   halting head. The frozen base remains low precision; trainable controllers
   stay fp32 for stability.
3. **Recurrent particles.** Add stochastic latent trajectories and SVGD-style
   particle updates over recurrent hidden states, then test whether a selector
   can convert candidate diversity into accuracy.

For the current manuscript-style status, evidence, negative results, and next
gates, see [docs/PROJECT_STATUS_PAPER.md](docs/PROJECT_STATUS_PAPER.md). The
program-level strategy is tracked in [docs/PROGRAM_TRACK.md](docs/PROGRAM_TRACK.md).

## Current Result

The project has not yet shown a release-grade recurrent/SVGD win over base
Qwen. It has shown a more useful intermediate result: a trained dense Qwen 0.5B
model can be converted into a recurrent-depth architecture with exact one-pass
identity preservation, stable learned halting, and recoverable benchmark
competence under small-parameter adapter/controller training.

Latest balanced ARC assessment for the selected recurrent checkpoint:

| Benchmark | Base Qwen | Recurrent Phase 1 | Delta |
|---|---:|---:|---:|
| ARC-Easy | `421/570` | `412/570` | `-9` |
| ARC-Challenge | `167/299` | `169/299` | `+2` |
| Combined | `588/869` | `581/869` | `-7` |

The next gate is competence-preserving recurrent SFT that keeps the
ARC-Challenge gain while closing the ARC-Easy gap. Phase 2/SVGD work resumes
after deterministic recurrent recovery is competitive with base.

## What Has Been Achieved

- **Exact identity gate passed.** The manually wrapped Qwen path can reproduce
  the base model with `max_abs_diff=0.0` under the strict identity setting.
- **Stable learned depth.** Sequence-level halting remains non-collapsed after
  adapter/controller stabilization, with expected loop depth around 2.9 on the
  current recurrent checkpoints.
- **Recoverable competence.** The recurrent model has recovered to near-base
  performance on balanced ARC and slightly exceeds base on ARC-Challenge, while
  still trailing on ARC-Easy.
- **Particle mechanism signal.** SVGD and within-group particle geometry improve
  candidate density on controlled exact-task suites, but have not yet beaten the
  strongest deterministic recurrent checkpoint on non-toy benchmarks.
- **Automation and gates.** Colab runners now emit planner-readable summaries so
  GPU jobs can be bounded, resumed, assessed, and stopped instead of becoming
  open-ended notebook sweeps.

## What Is Still Required To Beat Base

The next scientific result must come from deterministic recurrent recovery, not
more particle geometry. The active recipe is:

1. Continue Phase 1 from the best balanced checkpoint.
2. Mix Opus/TraceInversion reasoning traces with benchmark-style ARC rows.
3. Weight ARC-Easy more heavily to close the easy-regression gap while keeping
   ARC-Challenge non-negative.
4. Preserve answer competence with a small response-level base-distillation
   signal.
5. Re-run the full balanced ARC assessment only after a bounded proxy gate is
   non-negative.

Only after the recurrent model is base-competitive should Phase 2/SVGD, GPQA
Diamond, 1.5B/3B scaling, or Hugging Face release work consume serious A100
time.

## Active Next A100 Action

Credits are tight, so A100 work is deliberately gate-based. The latest
credit-saver ARC-mix proxy gate completed:

```text
run_id = stage5_arc_agi_colab_continue_20260621_232031_plan_arc_mix_probe
status = proxy_lift
base proxy = 68/128
start proxy = 66/128
best recurrent proxy = 67/128
best checkpoint = outputs/stage5/stage5_arc_agi_colab_continue_20260621_232031_plan_arc_mix_probe/arc_mix_response_w005_lr2e6/phase1/phase1_step_100.pt
```

The lift is real but weak: `+1` versus the recurrent starting checkpoint and
`-1` versus base on the 128-example proxy. The planner's next action, if we
choose to spend the A100 time, is the full balanced ARC-Easy/ARC-Challenge
assessment for that checkpoint:

```bash
STAGE5_ARC_AGI_NEXT_PLAN_SOURCE_SUMMARY=outputs/stage5/stage5_arc_agi_colab_continue_20260621_232031_plan_arc_mix_probe/summary.json \
python colab/plan_stage5_next_run.py
```

Do not run GPQA, Phase 2/SVGD, or scale-up jobs before this deterministic
recurrent recovery question is resolved.

## A100 Credit Discipline

Use GPU time only for bounded training/evaluation actions that emit summaries,
checkpoints, and planner-readable next steps. The maintained Colab continuation
entrypoint is:

```bash
python colab/run_stage5_colab_continue.py
```

By default it now runs in `credit_saver` mode: one allowlisted planner action,
post-run summaries, safe text-artifact commit, then stop. Set
`STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE=gate` or `throughput` only when you
intentionally want a three-action loop. Set `same_recipe` or `claim` only after
the preceding evidence gate has landed and the extra A100 spend is deliberate.

## Phase 0 Identity Gate

```bash
python eval/eval_identity.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --split 6,18 \
  --dtype float32 \
  --attn_implementation eager \
  --threshold 1e-3
```

The wrapper path is manual, so a pass means:

```text
embed -> prelude -> recurrent block once -> coda -> norm -> lm_head
```

matches the original model logits with dropout disabled.

## Phase 1 Halting Telemetry

Do not run Phase 1 training until the Phase 0 identity gate passes.

```bash
python eval/eval_halting.py --max_loops 4 --split 6,18
```

For training on JSONL rows with `prompt` + `completion` or `text`:

```bash
python training/train_phase1_ponder.py \
  --config config/qwen_0_5b_phase1.yaml \
  --train_jsonl data/train.jsonl
```

## Hugging Face Reasoning Data

Convert common Opus/Qwen reasoning datasets to this project's JSONL format:

```bash
python training/prepare_hf_reasoning_jsonl.py \
  --dataset_id lordx64/reasoning-distill-opus-4-7-max-sft \
  --tokenizer_name Qwen/Qwen2.5-0.5B-Instruct \
  --output_jsonl data/opus47_train.jsonl \
  --val_jsonl data/opus47_val.jsonl \
  --limit 1000 \
  --max_total_tokens 2048
```

The converter supports datasets with `text`, `messages`, or
`thinking`/`response`-style fields, TraceInversion-style
`inverted_reasoning` rows, explicit Hugging Face JSONL files, and
Complete-FABLE-style `row_json` wrappers. It writes `prompt`, `completion`, and
`cot_tokens`.

For the Jackrong Opus TraceInversion rows, use the dedicated adapter so the
inverted reasoning trace is preserved instead of falling back to answer-only
chat messages:

```bash
python training/prepare_hf_reasoning_jsonl.py \
  --dataset_id Jackrong/Claude-opus-4.7-TraceInversion-5000x \
  --adapter trace_inversion \
  --output_jsonl data/trace_inversion_train.jsonl \
  --val_jsonl data/trace_inversion_val.jsonl \
  --limit 1000 \
  --max_total_tokens 2048
```

For Fable's flat merged trace file, address the file explicitly:

```bash
python training/prepare_hf_reasoning_jsonl.py \
  --dataset_id Glint-Research/Fable-5-traces \
  --hf_file fable5_cot_merged.jsonl \
  --adapter fable_flat \
  --output_jsonl data/fable5_flat_train.jsonl \
  --val_jsonl data/fable5_flat_val.jsonl \
  --limit 1000 \
  --max_total_tokens 4096
```

Audit unfamiliar trace datasets before mixing them into training:

```bash
python training/inspect_hf_reasoning_dataset.py \
  --dataset_id Glint-Research/Fable-5-traces \
  --hf_file fable5_cot_merged.jsonl \
  --adapter fable_flat \
  --limit 1000 \
  --output_json outputs/dataset_audits/fable5_flat.json
```

Known candidate trace sources and their intended roles are tracked in
`config/reasoning_dataset_registry.yaml`. Opus-style reasoning traces are the
current fine-tuning source. Jackrong Opus TraceInversion is an immediate audit
candidate for easy/hard recurrent curriculum work. Fable/Pi-agent traces are
treated as later agent/tool-diversity material unless an audit and filter
explicitly promote them into a training mix.

## Phase 2 Stochastic Trajectories

Do not run Phase 2 training until deterministic halting is non-collapsed.

```bash
python eval/eval_trajectories.py \
  --max_loops 4 \
  --num_trajectories 2 \
  --split 6,18
```

If untrained fp16 trajectories report zero diversity, run a diagnostic-only
amplified check:

```bash
python eval/eval_trajectories.py \
  --max_loops 4 \
  --num_trajectories 2 \
  --split 6,18 \
  --diagnostic_latent_scale 1.0 \
  --diagnostic_adapter_std 0.02
```

```bash
python training/train_phase2_stochastic.py \
  --config config/qwen_0_5b_phase2.yaml \
  --train_jsonl data/train.jsonl
```

## Phase 2 SVGD Particles

SVGD mode keeps the frozen Qwen base path deterministic but treats each
trajectory as a particle after the recurrent block. The update combines the
ordinary recurrent transition with an RBF-kernel repulsion term across the
trajectory axis. This is the preferred next experiment when testing whether
particle diversity can create useful answer candidates without relying on large
latent noise.

Start with the smoke config:

```bash
python training/train_phase2_stochastic.py \
  --config config/qwen_0_5b_phase2_svgd.yaml \
  --train_jsonl data/opus47_train.jsonl \
  --device cuda
```

Validate with the SVGD switches enabled:

```bash
python eval/eval_jsonl.py \
  --data_jsonl data/opus47_val.jsonl \
  --checkpoint outputs/qwen_0_5b_phase2_svgd_smoke25/phase2_step_25.pt \
  --max_loops 4 \
  --num_trajectories 4 \
  --particle_update_mode svgd \
  --particle_init_noise 0.02 \
  --svgd_repulsion_scale 0.5 \
  --svgd_repulsion_max_norm 1.0 \
  --max_length 512 \
  --beta 0.08 \
  --rho 1e-3 \
  --dtype bfloat16 \
  --adapter_dtype float32 \
  --device cuda
```

Compare the Phase 1 checkpoint against a Phase 2 checkpoint on small exact
generation tasks:

```bash
python eval/eval_best_of_k_jsonl.py \
  --phase1_checkpoint outputs/qwen_0_5b_phase1_a100_beta008_continue_150/phase1_step_150.pt \
  --phase2_checkpoint outputs/qwen_0_5b_phase2_svgd_smoke25/phase2_step_25.pt \
  --phase2_num_trajectories 4 \
  --phase2_particle_update_mode svgd \
  --particle_init_noise 0.02 \
  --svgd_repulsion_scale 0.5 \
  --svgd_repulsion_max_norm 1.0 \
  --max_new_tokens 64 \
  --dtype bfloat16 \
  --adapter_dtype float32 \
  --device cuda
```

To initialize Phase 2 from a Phase 1 trainable checkpoint, set `resume_from` in
the Phase 2 config, for example:

```yaml
resume_from: outputs/qwen_0_5b_phase1/phase1_step_100.pt
```

Validate a trainable checkpoint on held-out JSONL:

```bash
python eval/eval_jsonl.py \
  --data_jsonl data/opus47_val.jsonl \
  --checkpoint outputs/qwen_0_5b_phase1_opus47_200/phase1_step_200.pt \
  --max_loops 4 \
  --max_length 1024
```

For constrained Colab GPUs such as G4, use the 50-step stability profile first:

```bash
python training/train_phase1_ponder.py \
  --config config/qwen_0_5b_phase1_g4_stability.yaml \
  --train_jsonl data/opus47_train.jsonl \
  --device cuda
```

Then validate the 50-step checkpoint before any longer run:

```bash
python eval/eval_jsonl.py \
  --data_jsonl data/opus47_val.jsonl \
  --checkpoint outputs/qwen_0_5b_phase1_g4_stability_50/phase1_step_50.pt \
  --max_loops 4 \
  --max_length 512 \
  --dtype float16 \
  --adapter_dtype float32
```

This keeps trainable adapters/controllers in fp32 and aborts on nonfinite loss,
metrics, gradients, or trainable parameters. Do not resume from the discarded
NaN checkpoint at
`outputs/qwen_0_5b_phase1_opus47_beta005_lr2e5_200/phase1_step_200.pt`.

## Slow Recurrent Inference

```bash
python infer_recurrent.py \
  --prompt "Find one valid 4-queens placement." \
  --max_loops 4 \
  --num_trajectories 2 \
  --sample_latents
```

Generation is intentionally no-cache and slow because recurrent passes reuse the
same sequence. KV cache is only valid on the single-pass identity-shaped path.

## GPQA-Style Multiple Choice

```bash
python eval/eval_gpqa.py \
  --data_jsonl data/gpqa_lite.jsonl \
  --max_loops 4 \
  --split 6,18
```

Rows should contain `question`, `choices` or `options`, and `answer`.

## Colab

Open [colab/GRAM_Recurrent_Qwen_Colab.ipynb](colab/GRAM_Recurrent_Qwen_Colab.ipynb)
in Google Colab, set the runtime to GPU, upload this project as a zip, and run
the notebook from top to bottom through the Phase 0 identity gate first.

For Colab Pro+, use [colab/COLAB_PRO_PLUS_RUNBOOK.md](colab/COLAB_PRO_PLUS_RUNBOOK.md):
select H100 GPU if available, turn High-RAM on, and keep `num_trajectories=2`
until the small-model gates pass.

## RunPod

For a more stable GPU workbench, use [runpod/RUNPOD_HANDOFF.md](runpod/RUNPOD_HANDOFF.md).
The short path is:

```bash
bash scripts/runpod_setup.sh
bash scripts/run_smoke_gates.sh
```
