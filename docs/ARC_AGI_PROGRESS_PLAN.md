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
  exact-grid evaluation. It also writes symbolic coverage summaries before
  running model inference.
- `eval/analyze_arc_agi_symbolic.py`: cheap coverage analyzer for the small
  symbolic candidate generator.
- `training/prepare_arc_agi_sft_jsonl.py`: supervised ARC-AGI JSONL
  preparation with leave-one-out task rows and safe color-permutation
  augmentation. It also supports `--trace_mode symbolic`, which prepends a
  compact `<think>` transformation trace when the small symbolic solver can
  exactly explain the target grid, and `--trace_mode symbolic_program`, which
  emits a tiny program-style trace such as transform/recolor/return steps.
  It also supports `--trace_mode symbolic_state_trace`, which emits the same
  program operations plus compact intermediate grid states. This is the most
  recurrent-specific target: it teaches a visible transformation trajectory
  rather than only a final grid or abstract program.
- `training/generate_arc_agi_synthetic_tasks.py`: synthetic ARC-style task
  generator for geometry/color-map, non-background object crop, crop-then-recolor,
  crop-then-transform-then-recolor, and constant-output tasks that are exactly
  covered by the symbolic trace solver.
  This is a controlled curriculum for testing whether the recurrent
  architecture can learn clean transformation traces before we spend more time
  on particle mechanisms.
- `colab/run_stage5_arc_agi_sft.py`: smoke fine-tune runner for adapting
  recurrent Phase1 on public ARC-AGI training tasks and evaluating held-out
  ARC-AGI evaluation tasks. It can append the synthetic curriculum with
  `STAGE5_ARC_AGI_SYNTHETIC_TASKS`, `STAGE5_ARC_AGI_SYNTHETIC_SEED`, and
  `STAGE5_ARC_AGI_SYNTHETIC_MODES`.
- `colab/run_stage5_arc_agi_trace_sft_gate.py`: matched two-arm SFT runner for
  grid-only supervision versus symbolic-program trace supervision. It reports
  the best checkpoint in each child SFT ladder when available, not only the
  final checkpoint.
- `colab/run_stage5_arc_agi_distill_sft_gate.py`: matched two-arm SFT runner
  for base-logit distillation off versus on. It also compares best-in-ladder
  checkpoints when child SFT runs enable checkpoint-ladder evaluation.
- `colab/run_stage5_arc_agi_autopilot.py`: overnight runner that executes the
  candidate gate, conditionally runs trace-SFT, and conditionally runs the
  distillation gate using configurable thresholds.
- `colab/run_stage5_arc_agi_recovery_particle_gate.py`: controlled runner that
  first executes synthetic symbolic ARC SFT for deterministic recurrent
  recovery, then evaluates low-noise K-particle/SVGD variants on the tuned
  checkpoint. When checkpoint-ladder evaluation is enabled, it selects the best
  recovered checkpoint before running the particle arms. Use this to separate
  "more targeted training helped the recurrent model" from "particles add value
  over the recovered recurrent model."
- Grid output formats: JSON, compact row strings, and tagged row strings are
  supported. Colab ARC-AGI runners default to compact row strings because they
  are shorter and easier for a 0.5B model to emit reliably.
- Evaluation reports first-candidate, selected-candidate, and oracle best-of-K
  exact-grid accuracy. Selected-candidate accuracy uses only demonstration
  shape heuristics, valid-grid parsing, and a tiny program verifier when
  candidates emit executable symbolic programs. Best-of-K is diagnostic, not a
  deployable score.
- Gate 1 selector work now includes offline and live `reliability_vote`
  selection in addition to `heuristic`, `self_consistency`, and
  `symbolic_priority`. `reliability_vote` is target-free: it votes over parsed
  grid claims using candidate provenance, symbolic source, demonstration-fitting
  programs, source diversity, and output-shape consistency. Use
  `colab/run_stage5_arc_agi_rescore_selectors.py` on saved candidate JSONLs
  before tuning more kernel geometry.
- Evaluation can execute the tiny `symbolic_program` DSL as a fallback when a
  candidate does not contain a directly parseable output grid. Reports include
  a parse-method summary so literal-grid exactness and program-executed
  exactness remain visible. Use `--program_parse_mode fallback` for conservative
  grid-first scoring, and `--program_parse_mode prefer` when diagnosing whether
  the model learned executable transformations before it learned reliable final
  grid formatting.
- Executable program candidates are also checked against every demonstration.
  Candidates whose program reproduces all training outputs are preferred during
  selected-candidate ranking before shape heuristics. Reports include a
  `program_verifier_summary` so verifier lift remains separate from oracle
  best-of-K.
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

Set `STAGE5_ARC_AGI_TRACE_MODE=symbolic_program` to train on explicit
program-style transformation traces for covered examples. Use
`STAGE5_ARC_AGI_TRACE_FILTER=covered` for the clean curriculum arm, and keep
`STAGE5_ARC_AGI_TRACE_MODE=none` as the grid-only control. The older
`symbolic` mode remains available for prose traces.
Set `STAGE5_ARC_AGI_TRACE_MODE=symbolic_state_trace` for the recurrent
state-trajectory arm. Compare it against `symbolic_program` before scaling:
state traces may help the loop learn intermediate transformations, but they
also lengthen completions and should be checked by `training_signal.md`.

Set `STAGE5_ARC_AGI_SYNTHETIC_TASKS=200` or higher to append symbolically covered
synthetic ARC-style tasks to the public ARC SFT rows. This is the first
controlled test for whether more targeted training can recover or improve the
surgically altered recurrent model. Keep particle/SVGD claims separate until
the deterministic recurrent baseline improves under this curriculum.

Set `STAGE5_ARC_AGI_EVAL_CHECKPOINT_LADDER=1` with a non-final
`STAGE5_ARC_AGI_SAVE_EVERY` to evaluate every saved checkpoint. Use this when
the question is how much targeted ARC SFT is needed before recurrent recovery
peaks or crosses the base model on the chosen exact-grid slice.

Run `colab/run_stage5_arc_agi_recovery_particle_gate.py` when you want both
questions answered in one A100 session. Its report has two independent gates:

- deterministic recurrent recovery versus the pre-SFT recurrent checkpoint;
- particle/SVGD lift versus the tuned recurrent checkpoint.

Only treat particles as promising if the second gate clears. If only the first
gate clears, keep improving deterministic recurrent training before returning
to Phase2. This runner defaults to `symbolic_program` traces; override with
`STAGE5_ARC_AGI_RECOVERY_TRACE_MODE=symbolic` only when comparing against the
older prose-trace curriculum. It also defaults to
`STAGE5_ARC_AGI_PROGRAM_PARSE_MODE=prefer`, so program execution is measured
explicitly; set it to `fallback` for conservative grid-first scoring.
It also generates a disjoint synthetic holdout controlled by
`STAGE5_ARC_AGI_SYNTHETIC_EVAL_TASKS`, `STAGE5_ARC_AGI_SYNTHETIC_EVAL_SEED`,
and `STAGE5_ARC_AGI_SYNTHETIC_EVAL_PARSE_MODES`. Use that section to separate
operation-family generalization from public ARC evaluation noise.
Set `STAGE5_ARC_AGI_EVAL_CHECKPOINT_LADDER=1` and a non-final
`STAGE5_ARC_AGI_SAVE_EVERY` when using this combined runner; particle variants
will then be evaluated against the best recovered checkpoint rather than the
last training step.

Run `colab/run_stage5_arc_agi_trace_sft_gate.py` to execute both controls under
the same settings and write one comparison summary. By default it compares
grid-only SFT against symbolic-trace SFT on trace-covered examples.

Set `STAGE5_ARC_AGI_DISTILL=1` to add frozen-base next-token KL distillation
inside `training/train_phase1_ponder.py`. Run
`colab/run_stage5_arc_agi_distill_sft_gate.py` to compare the selected ARC SFT
recipe with distillation off versus on.

For overnight runs, prefer `colab/run_stage5_arc_agi_autopilot.py`. It always
runs the candidate gate, then proceeds to trace-SFT when symbolic coverage and
hybrid best-of-K deltas clear thresholds, then proceeds to distillation when
trace-SFT matches or beats grid-only SFT. Thresholds:

- `STAGE5_ARC_AGI_AUTOPILOT_MIN_SYMBOLIC_EXACT` default `1`;
- `STAGE5_ARC_AGI_AUTOPILOT_MIN_HYBRID_BEST_DELTA` default `0`;
- `STAGE5_ARC_AGI_AUTOPILOT_MIN_TRACE_BEST_DELTA` default `0`.

Gate:

- valid-grid rate should improve materially;
- exact-grid score should not regress against recurrent Phase1;
- symbolic-trace SFT should beat or match grid-only SFT before scaling this
  recipe;
- distillation should preserve or improve exact-grid results while reducing
  recurrent-vs-base regression;
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
