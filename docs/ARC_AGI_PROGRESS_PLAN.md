# ARC-AGI Progress Plan

## Current State

The project has not yet measured ARC-AGI. Current reported numbers are on
`allenai/ai2_arc` ARC-Challenge, used as a cheap multiple-choice proxy for
general reasoning recovery after the recurrent architecture surgery.

Latest proxy ladder:

- Base Qwen/Qwen2.5-0.5B-Instruct: 72/128.
- Stage 4 Phase1 deterministic recurrent: 70/128.
- Stage 4 Phase2/SVGD recurrent: 69/128.

Interpretation: deterministic recurrent recovery is working; current
particle/SVGD settings have not shown reliable lift.

## Near-Term Goal

Before claiming progress toward ARC-AGI, establish this sequence:

1. Recover deterministic recurrent performance to match or beat base Qwen on
   ARC-Challenge proxy slices.
2. Prove whether particles/SVGD provide non-negative lift over that recovered
   recurrent baseline.
3. Build an ARC-AGI-1/2 public-eval harness and report zero-shot or
   few-shot puzzle-solving results separately from ARC-Challenge.
4. Add ARC-AGI-specific training only after the harness is in place.

## Why This Order

The current recurrent model is a surgically altered pretrained LM. Some
regression is expected. The first win is preserving base competence with the new
architecture. The second win is proving recurrent depth or particles add lift.
Only then is it meaningful to pursue ARC-AGI-specific SOTA claims.

## Next Experiments

### Stage 5A: Deterministic Recovery Ladder

Run `colab/run_stage5_phase1_recovery_ladder.py`.

This continues Phase1 from the Stage 4 checkpoint, saves intermediate
checkpoints, and evaluates each checkpoint against:

- base Qwen on ARC-Challenge proxy,
- the Stage 4 Phase1 starting checkpoint,
- Opus validation CE and loop telemetry.

Proceed if the best checkpoint improves over Stage 4 Phase1 or closes the base
gap.

### Stage 5B: Particle Value Gate

Run `colab/run_stage4_particle_value_gate.py`.

This uses float32 particle arms to avoid bfloat16 K-copy numerical drift. Do not
continue large Phase2 training unless at least one particle arm is non-negative
against deterministic Phase1 and has helped examples greater than or equal to
harmed examples.

### Stage 5C: True ARC-AGI Harness

Build a separate harness for ARC-AGI public tasks. ARC-Challenge numbers should
not be described as ARC-AGI numbers.

Initial harness files:

- `eval/arc_agi_utils.py`: task loading, prompt rendering, grid parsing, and
  exact-grid scoring.
- `eval/eval_arc_agi.py`: base/recurrent exact-grid evaluator.
- `eval/arc_agi_symbolic.py`: small deterministic ARC candidate generator for
  simple geometry, consistent color-map, and constant-output transformations.
- `colab/run_stage5_arc_agi_smoke.py`: Colab smoke runner that can clone public
  ARC-AGI data and compare base Qwen against the recurrent Phase1 checkpoint.
- `colab/run_stage5_arc_agi_candidate_gate.py`: one-shot candidate-source
  value gate for symbolic-only, model-only, and model+symbolic hybrid ARC-AGI
  exact-grid evaluation.
- `training/prepare_arc_agi_sft_jsonl.py`: supervised ARC-AGI JSONL
  preparation with leave-one-out task rows and safe color-permutation
  augmentation. It also supports `--trace_mode symbolic`, which prepends a
  compact `<think>` transformation trace when the small symbolic solver can
  exactly explain the target grid.
- `colab/run_stage5_arc_agi_sft.py`: smoke fine-tune runner for adapting
  recurrent Phase1 on public ARC-AGI training tasks and evaluating held-out
  ARC-AGI evaluation tasks.
- `colab/run_stage5_arc_agi_trace_sft_gate.py`: matched two-arm SFT runner for
  grid-only supervision versus symbolic-trace supervision.
- Grid output formats: JSON, compact row strings, and tagged row strings are
  supported. Colab ARC-AGI runners default to compact row strings because they
  are shorter and easier for a 0.5B model to emit reliably.
- Evaluation reports first-candidate, selected-candidate, and oracle best-of-K
  exact-grid accuracy. Selected-candidate accuracy uses only demonstration
  shape heuristics and valid-grid parsing; best-of-K is diagnostic, not a
  deployable score.
- Evaluation can optionally include symbolic candidates with
  `--include_symbolic_candidates`. Colab runners expose this as
  `STAGE5_ARC_AGI_INCLUDE_SYMBOLIC=1` and
  `STAGE5_ARC_AGI_SYMBOLIC_POSITION=after_model|before_model|only`.
  Use this to separate model-only ability, symbolic-only transform coverage,
  and hybrid candidate selection.

The harness should and now does:

- load ARC-AGI-1 and ARC-AGI-2 public/evaluation JSON tasks,
- render train/test grids into model prompts,
- produce candidate output grids,
- parse and validate grids strictly,
- score exact-grid accuracy,
- support K candidates and verifier/reranker selection.

Next experiment:

- run `colab/run_stage5_arc_agi_candidate_gate.py` before spending more GPU on
  particles. This produces a compact table for:
  - symbolic-only transform coverage;
  - base Qwen model-only;
  - base Qwen plus symbolic candidates;
  - recurrent Phase1 model-only;
  - recurrent Phase1 plus symbolic candidates.

Next upgrades after that gate:

- add programmatic grid-edit/action traces rather than plain text grid output;
- add a verifier/reranker for K candidates;
- add synthetic ARC-style trace generation for recurrent fine-tuning;
- report ARC-AGI-1 and ARC-AGI-2 separately.

### Stage 5D: ARC-AGI SFT Smoke

Run `colab/run_stage5_arc_agi_sft.py`.

This creates supervised rows from public ARC-AGI training tasks:

- original task test pairs when outputs are public;
- leave-one-out examples from the task's train pairs;
- color-permutation augmentations applied consistently to every grid.
- dihedral geometry augmentations applied consistently to every grid.

It then fine-tunes the recurrent Phase1 checkpoint and compares exact-grid
generation against base and the pre-SFT recurrent checkpoint on the ARC-AGI
evaluation split.

Set `STAGE5_ARC_AGI_TRACE_MODE=symbolic` to train on explicit transformation
traces for covered examples. Keep `STAGE5_ARC_AGI_TRACE_MODE=none` as the
grid-only control.

Run `colab/run_stage5_arc_agi_trace_sft_gate.py` to execute both controls under
the same settings and write one comparison summary.

Gate:

- valid-grid rate should improve materially;
- exact-grid score should not regress against recurrent Phase1;
- symbolic-trace SFT should beat or match grid-only SFT before scaling this
  recipe;
- selected-candidate score should be reported separately from oracle best-of-K;
- if exact-grid remains near zero, next work is representation/traces, not more
  blind SFT steps.

## Training Direction After 5A/5B

If Phase1 improves but particles fail:

- continue deterministic recurrent training;
- add base-logit distillation to preserve base Qwen behavior;
- defer SVGD training.

If particles pass the value gate:

- train particles with set/coverage objectives rather than plain diversity;
- only reward diversity among correct or verifier-approved candidates;
- evaluate K=1, K=2, and K=4 separately.

If symbolic or hybrid candidates help while particles do not:

- treat this as evidence that candidate diversity can matter, but current
  latent particles are not yet producing the right alternatives;
- train the recurrent model on explicit transformation/action traces and
  use particles as proposal mechanisms only after those traces are learned;
- compare particle candidates against symbolic candidate coverage rather than
  against random output diversity.

If Phase1 does not improve:

- reduce learning rate;
- add base-logit distillation;
- improve modified reasoning traces so loop-depth targets and answers are
  cleaner.
