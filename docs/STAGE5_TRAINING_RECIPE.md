# Stage 5 Training Recipe: From Recurrent Recovery to Structural Diversity

## Framing

The fair comparison is not an untrained recurrent graft versus the intact base
model. The fair research program is:

1. Surgically convert base Qwen 0.5B into a recurrent-depth model.
2. Recover the competence lost by the surgery with minimal additional training.
3. Test whether recurrent depth and particle trajectories can improve over the
   recovered recurrent baseline.
4. Only then test whether the recurrent model can surpass unmodified Qwen 0.5B
   on non-toy benchmarks.

As of 2026-06-20, Stage 4 Phase1 already nearly closes the ARC-128 base gap:
base Qwen scored 72/128, trained Phase1 recurrent scored 70/128, and the current
Phase2/SVGD candidate scored 69/128. This means recurrent recovery is working,
but particle training is not yet adding reliable lift.

## Stage 5A: Recurrent Competence Recovery

Goal: train the deterministic recurrent model until it matches or beats base
Qwen 0.5B before spending heavily on particle training.

Use a mixed objective:

- Answer CE on modified Opus reasoning traces.
- Optional base-logit distillation on short prompts and benchmark-style MCQs, so
  the recurrent model does not drift away from base capabilities unnecessarily.
- PonderNet halting KL with non-collapsed loop depth.
- Small bridge/LoRA regularization to preserve the surgery identity path.

Recommended run ladder:

- Phase1 500 steps, already done.
- Phase1 1k steps on 3k Opus rows.
- Phase1 2k steps on 5k-8k filtered Opus rows if validation CE and ARC do not
  regress.
- Evaluate each checkpoint against base on ARC-128, ARC-512, GSM8K-mini, and the
  exact smoke suite.

Gate to proceed:

- Phase1 recurrent matches or beats base on at least one non-toy slice, or
  remains within a small gap while improving exact reasoning and loop telemetry.
- Mean loop depth remains non-collapsed.

## Stage 5B: Particle Mechanism Screening

Goal: decide whether the current particle mechanism is worth training around.

The current negative result is important: K=4, noise=0.05 particles hurt ARC
relative to Phase1, even with within-group projection. Therefore, do not start a
large Phase2 run until a lower-noise or trained-selector setting is at least
non-negative against Phase1.

Screen:

- Zero-noise K=4 control: should match Phase1. If not, debug trajectory scoring.
- K in {2, 4}.
- Noise in {0.005, 0.01, 0.02}.
- Repulsion in {0, 0.5, 2}.
- Aggregation in {mean, max, vote}.
- Selector in {self_consistency, reliability_vote}; do not spend more A100 time
  on kernel geometry until selector-rescored selected-answer metrics are
  reported.
- Report helped/hurt/tied against Phase1, not only aggregate accuracy.

Gate to train particles:

- A particle setting matches or beats Phase1 on ARC-128.
- Helped examples >= hurt examples.
- Exact-task diversity improves without MCQ collapse.

## Stage 5C: Spectrum Distillation For Particles

Use this only after Stage 5B passes.

Goal: make particles carry distinct useful solution paths rather than arbitrary
hidden perturbations.

Training data should contain multiple correct solution traces per problem where
available. For synthetic or verifiable domains, generate multiple strategies:

- arithmetic: algebraic, unit-cancellation, direct formula;
- combinatorics: constructive and constraint-checking traces;
- code/math: different valid proof or program approaches;
- multi-solution tasks: distinct valid final objects.
- ARC-style grids: compact symbolic state traces that show each transformation
  step and intermediate grid before the final answer. Use
  `STAGE5_ARC_AGI_TRACE_MODE=symbolic_state_trace` as the recurrent-specific
  curriculum arm, with `symbolic_program` and grid-only as controls.

Loss shape:

- Set/coverage CE: each problem has a set of valid traces; particles are matched
  to traces with min-over-K, Hungarian assignment, or soft set likelihood.
- Standard answer CE remains present to preserve correctness.
- Tiny diversity term only on correct or high-likelihood particles.
- Penalize diversity that lowers verifier score.

This is different from the current Phase2 objective, which injects particle
variation without forcing that variation to align with distinct correct
solution modes.

## Stage 5D: Spectrum-To-Signal / VibeThinker-Style Program

This is the scale-up program, not the immediate next experiment.

The proposed adaptation is:

1. Diversity-exploring distillation with the particle mechanism active.
2. Verifiable RL with maximum-entropy guidance or a GRPO/MGPO-like objective.
3. Offline self-distillation that preserves a diverse correct set, not only the
   modal answer.
4. Optional instruction RL after the diversity-survival question is answered.
5. Claim-level or verifier-based candidate selection over particle outputs.

Required control:

- Train a standard non-recurrent baseline and recurrent-particle model through
  the same recipe.
- Start with `colab/run_stage5_arc_agi_dense_sft.py` as the dense LoRA control:
  it prepares the same ARC-AGI rows, trains an unmodified Qwen adapter with
  `training/train_dense_lora.py`, and lets the planner launch the matched
  recurrent SFT arm.
- Judge that pair with `colab/assess_stage5_recipe_control.py`; it is the
  paired hard-tail assessment for whether recurrence adds value beyond the
  standard dense recipe. Treat `needs_selector_conversion` as a real diagnostic:
  recurrence may be creating better candidates before the current selector can
  choose them.
- Cross the signal phase with entropy on/off.
- Measure whether structural diversity survives the signal phase better than
  weight-only diversity.

Do not implement MGPO or CLR from memory. Pull the VibeThinker technical reports
before coding:

- VibeThinker-3B technical report / arXiv 2606.16140.
- VibeThinker-1.5B technical report / arXiv 2511.06221.

## Stage 5E: Benchmark And Release Gate

Only move toward Hugging Face packaging and GPQA Diamond when:

- Phase1 recurrent is competitive with base on ARC/GSM8K-style slices.
- Phase2 or a selector gives positive lift over Phase1.
- Checkpoints are backed up to Drive and metadata is committed.

Initial benchmark set:

- ARC-Challenge 512 or full validation.
- GSM8K subset, then full if stable.
- GPQA-lite before GPQA Diamond.
- Exact smoke suite for regression.
- Token/sec and VRAM telemetry for K=1, K=2, K=4.

Release candidate definition:

- Explicitly state whether the released checkpoint is Phase1 deterministic or
  Phase2 particle.
- Report both internal recurrent lift and gap to base Qwen.
- Package the latest same-recipe dense-vs-recurrent architecture assessment in
  the adapter metadata/model card when available, including whether the result
  is `passed`, `needs_selector_conversion`, or not yet established.
- Run `colab/assess_stage5_release_gate.py` before broader benchmark claims. It
  checks ARC benchmark confirmation, same-recipe architecture evidence, and HF
  export metadata from saved artifacts without using the GPU.
- If that gate returns `ready_for_broader_benchmarks`, run
  `colab/run_stage5_benchmark_suite.py`. The suite compares unmodified base Qwen
  and the recurrent artifact on ARC-Challenge plus GPQA-lite, writes sanitized
  result artifacts with paired recurrent-vs-base sign-test evidence under
  `outputs/stage5`, and leaves prepared question data under ignored `data/`
  paths.
- Run `colab/assess_stage5_benchmark_suite.py` on the suite summary before
  GPQA Diamond or public claims. A negative paired recurrent-vs-base result
  routes back to deterministic recurrent recovery; insufficient coverage routes
  to a larger benchmark-suite confirmation run.
- Run `colab/build_stage5_claim_packet.py` after the broader benchmark gate
  passes. It synthesizes release, architecture, benchmark, HF export, and
  ARC-AGI claim evidence while explicitly distinguishing release-candidate
  status from any SOTA ARC-AGI claim.
- Run `colab/build_stage5_arc_agi_sota_comparison.py` to create the
  authoritative SOTA-comparison artifact consumed by the claim packet. It
  requires a sourced `config/arc_agi_same_size_baselines.json`; the example file
  is only a schema and must not be used for claims.
- Run `colab/validate_arc_agi_baseline_registry.py` before any SOTA-facing
  comparison. The validator requires non-placeholder same-size baseline names,
  sourced URLs/DOIs/arXiv references, dates, metrics, and parameter counts inside
  the declared model-size band.
- Include the modified-architecture caveat and training recipe.
