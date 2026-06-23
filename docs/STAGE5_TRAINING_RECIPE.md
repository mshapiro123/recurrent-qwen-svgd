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

As of 2026-06-20, Stage 4 Phase1 already nearly closed the ARC-128 base gap:
base Qwen scored 72/128, trained Phase1 recurrent scored 70/128, and the then
current Phase2/SVGD candidate scored 69/128. Subsequent Stage 5 ARC-mixed
training produced a stronger deterministic Phase 1 checkpoint that beats base
on a bounded 256-example ARC-Easy/ARC-Challenge content-question confirmation
slice and remains non-negative under cyclic option-permutation scoring. This
means recurrent recovery is working. Particle training is still not yet adding
reliable lift over the strongest deterministic checkpoint.

The latest ARC-mix recovery results refine the framing. The key control problem
is not only "more recurrence" or "more particles"; it is allocation of depth
and width. Depth is the learned recurrent loop count. Width is particle spread.
Direct/easy tasks should use little of either, deep deterministic tasks should
use depth without particle spread, wide tasks should use particle coverage, and
hard multi-approach tasks may need both. The detailed training handoff is
[DEPTH_WIDTH_ROUTING_RECIPE.md](DEPTH_WIDTH_ROUTING_RECIPE.md).
The typed data-generation contract for external strong-model curriculum traces
is [CURRICULUM_DATA_PIPELINE.md](CURRICULUM_DATA_PIPELINE.md).

## Stage 5A: Recurrent Competence Recovery

Goal: train the deterministic recurrent model until it matches or beats base
Qwen 0.5B before spending heavily on particle training.

Use a mixed objective:

- Answer CE on modified Opus reasoning traces.
- Optional base-logit distillation on short prompts and benchmark-style MCQs, so
  the recurrent model does not drift away from base capabilities unnecessarily.
- PonderNet halting KL with non-collapsed loop depth.
- Small bridge/LoRA regularization to preserve the surgery identity path.
- Direct-mode pressure: easy/base-known rows should halt shallowly rather than
  being forced through latent computation that can shift answer calibration.

Recommended run ladder:

- Depth-1 preservation pass on base-known direct rows. Force or strongly target
  loop `1`, include base-logit/answer-margin distillation, and measure whether
  the recurrent wrapper stops regressing on examples the original Qwen 0.5B
  already solves.
- Capability-ladder pass on verified rows that separate base misses from
  stronger-model successes:
  - Qwen 0.5B correct -> depth `1` rehearsal/preservation.
  - Qwen 0.5B misses but Qwen 1.5B succeeds with independent verification ->
    depth `2` deterministic recurrent upgrade.
  - Qwen 0.5B and 1.5B miss but Qwen 3B or a stronger non-student solver
    succeeds with independent verification -> depth `3-4` deterministic
    recurrent upgrade.
  - unresolved or unverified rows -> selector/verifier/error-analysis data, not
    positive SFT.
- Phase1 500 steps, already done, remains the historical baseline.
- Phase1 1k steps on a filtered direct/deep mix only after the depth-1
  preservation check is non-negative.
- Phase1 2k steps on 5k-8k filtered Opus or capability-ladder rows if
  validation CE, ARC, and direct-mode calibration do not regress.
- Evaluate each checkpoint against base on ARC-128, ARC-512, GSM8K-mini, and the
  exact smoke suite.
- For generated width/depth curriculum shards, run
  `colab/run_stage5_curriculum_sft.py` only after
  `training/check_curriculum_sft_gate.py` reports `go=true`. This is the
  strong-model curriculum handoff: it trains Phase 1 deterministic recurrence
  from verified `positive_*` traces, keeps non-positive traces out of SFT, and
  leaves particle/SVGD training for a later mechanism gate.

Gate to proceed:

- Phase1 recurrent matches or beats base on at least one non-toy slice, or
  remains within a small gap while improving exact reasoning and loop telemetry.
- Any bounded surpass-base result replicates on an independent offset or larger
  split before it is treated as a robust benchmark claim.
- Mean loop depth remains non-collapsed.
- Loop-depth telemetry is sensible by task type: direct/easy rows are shallower
  than deep deterministic rows.
- Correct-answer margins and answer priors do not drift materially versus base
  on MCQ rows.

Do not add permutation-invariance training during this recovery stage. Debiased
MCQ evaluation should neutralize option-label artifacts for measurement, while
the depth-one direct route stays matched to base. If a later reasoning-path
objective needs order robustness, use a small per-item content-consistency loss
only on depth-two-plus rows and mix it with broad replay.

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
- When same-recipe architecture returns `needs_selector_conversion`, run
  `colab/run_stage5_arc_agi_rescore_selectors.py`, then
  `colab/assess_stage5_recipe_selector_conversion.py`. The conversion gate
  compares recurrent selector-selected outputs directly against the dense
  control, instead of only comparing selectors against the recurrent source
  heuristic.
- Treat a passed selector-conversion gate as selector-converted architecture
  evidence, not as a raw recurrent win. It can unlock release-candidate and
  broader-benchmark packaging, but the report must state that the lift came
  from recurrent candidate coverage plus the claim-level selector.
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

- Independent ARC-Easy/ARC-Challenge offset confirmation for the current
  positive ARC-mix checkpoint.
- ARC-Challenge 512 or full validation after offset confirmation.
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
  is `passed`, `needs_selector_conversion`, or not yet established. If selector
  conversion is the evidence that advances the artifact, package that
  selector-conversion summary too and state the lift as recurrent candidate
  coverage plus selector conversion.
- Run `colab/assess_stage5_release_gate.py` before broader benchmark claims. It
  checks ARC benchmark confirmation, same-recipe architecture or
  selector-conversion evidence, and HF export metadata from saved artifacts
  without using the GPU.
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
- Run `colab/assess_stage5_selector_replication.py` after two Gate 1 selector
  assessments exist. A selector setting is only replicated if the same
  comparison passes Gate 1 on both discovery and confirmation slices.
- Run `colab/build_stage5_claim_packet.py` after the broader benchmark gate
  passes. It synthesizes release, architecture, benchmark, HF export, and
  ARC-AGI claim evidence while explicitly distinguishing release-candidate
  status from any SOTA ARC-AGI claim.
- Run `colab/build_stage5_arc_agi_sota_comparison.py` to create the
  authoritative SOTA-comparison artifact consumed by the claim packet. It
  requires a sourced `config/arc_agi_same_size_baselines.json`; the example file
  is only a schema and must not be used for claims.
- For internally reproduced same-size controls, run
  `colab/build_stage5_arc_agi_reproduced_baseline_registry.py` against a saved
  Stage 5 ARC-AGI summary, usually with `--labels base`, to generate a
  validator-passing `reproduced_eval` registry row from the audited local
  artifact. The planner automatically chooses this step when a SOTA comparison
  has a candidate summary path but no registry. Treat this as
  reproduced-control evidence; prefer external official, paper, or model-card
  sources for public SOTA claims.
- `colab/build_stage5_arc_agi_sota_comparison.py` now reports
  `comparison_scope`. A reproduced-only registry can pass the recovery/control
  comparison as `passed_reproduced_control`, but only `status: passed` with
  `comparison_scope: public_sota` can unlock SOTA readiness in the claim packet.
- Run `colab/validate_arc_agi_baseline_registry.py` before any SOTA-facing
  comparison. The validator requires non-placeholder same-size baseline names,
  sourced URLs/DOIs/arXiv references, dates, metrics, row-level ARC
  version/split, evidence type, and parameter counts inside the declared
  model-size band. For `evidence_type: reproduced_eval`, a row may cite an
  existing local JSON `source_artifact` instead of an external URL only when it
  also records the `reproduction_command` and `git_commit` that produced that
  artifact.
- Include the modified-architecture caveat and training recipe.
