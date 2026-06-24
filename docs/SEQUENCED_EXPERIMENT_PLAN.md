# Sequenced Experiment Plan And Hypothesis Gates

## Purpose

This document is the operational experiment queue for the recurrent-particle
Qwen program. It is meant to be shared with a strategy or deep-research agent
as the current empirical map: what we have learned, what each next experiment
tests, what success or failure means, and what decisions each result should
trigger.

The central scientific question is:

> Can a pretrained dense Qwen model be surgically converted into a recurrent
> latent-computation model, preserve base competence at depth 1, and then use
> learned depth and particles to beat the original model on harder reasoning?

The current evidence says the architecture is promising but not yet proven:

- one-pass identity preservation is solved under strict settings;
- loop-1 recurrent behavior is close to, and sometimes above, base Qwen 0.5B;
- targeted ARC-mixed Phase 1 recovery has crossed the base line on a bounded
  256-example ARC-Easy/ARC-Challenge confirmation slice;
- unconditional deeper loops are harmful on easy tasks;
- deeper loops contain unique correct answers on hard ARC-Challenge examples;
- the next bottleneck is routing/selection and depth supervision, not more
  kernel geometry;
- MCQ label/position bias is a real confound, so ARC/GPQA-style results must
  use debiased scoring.

## Strategy Review Update: Foundation First, Spine Second

The numbered experiments below are components, not a strict serial chain. A
strategy review identified the main weakness in the first version of this plan:
it over-serialized cheap diagnostics that do not need to wait for each other.
The corrected program shape is two-layered:

1. **Foundation instruments, run in parallel where possible.** Build the
   selector, debiased benchmark harness, trace/data audit, and no-training
   1.5B viability probe up front. These are cheap relative to training and
   decide whether later GPU work is interpretable.
2. **Decisive spine.** Run depth-1 preservation, capability-ladder depth
   supervision, and the benchmark gate as one connected test of whether
   recurrence can substitute for scale.

This reordering keeps the validity gates but removes unnecessary waiting. The
selector and debiased harness are instruments. The data audit is an input
quality gate. The 1.5B probe is an information-value check on whether 0.5B is
too small to show the payoff. None of those requires completing the others.

The decisive test should be named directly:

> On examples that base Qwen 0.5B misses but Qwen 1.5B solves, can the
> recurrent 0.5B model trained with depth-ladder supervision improve at depth 2
> while preserving depth-1 behavior on examples base Qwen 0.5B already solves?

That is the central "recurrence substitutes for scale" experiment. The
particles/SVGD branch opens only after this deterministic depth spine produces
selector-convertible signal.

## Current Evidence Anchor

### Identity And Surgery

The model split into Prelude / Recurrent Block / Coda can exactly reproduce the
base model when run with a single recurrent pass under strict float32/eager
conditions. This means the architecture can represent the original model; later
regression is a training/routing issue, not an unavoidable consequence of the
wrapper.

### Deterministic Recurrent Depth

Phase 1 training with fp32 trainable adapters and low-precision frozen base is
numerically stable. The recurrent halting distribution does not collapse, and
mean expected loop depth around 2-3 has been observed on reasoning traces.

On direct ARC preservation checks, loop 1 is not broadly worse than base:

- ARC-Easy direct preservation confirmation: recurrent loop 1 `191/256` versus
  base `186/256`.
- ARC-Challenge direct preservation confirmation: recurrent loop 1 `150/256`
  versus base `148/256`.

This is the most important positive result so far: a surgically altered model
can still behave like the base model through the direct path.

### Depth Sweep Signal

The first loop-depth sweep showed a sharp allocation problem:

- loop 1 was near or above base;
- loops 2-4 harmed aggregate accuracy;
- however, oracle-over-depth was much higher than loop 1 alone, especially on
  ARC-Challenge.

For ARC-Challenge on the initial 256-example slice:

- loop 1: `150/256`;
- any recurrent depth correct: `191/256`;
- deeper unique over base plus loop 1: `38`;
- a simple score selector over loops 1-3 reached `155/256`.

This does not prove deployed performance. It proves that deeper loops contain
useful answers and harmful answers. The problem is selection.

A follow-up selector split analysis reinforces the same conclusion. Choosing a
simple score selector on one half of the original depth-sweep examples and
testing it on the other half produces a small but repeatable ARC-Challenge lift
and an easy-task cost:

- ARC-Challenge, five deterministic 50/50 splits: mean selected delta versus
  loop 1 `+1.4/128`, with `4` positive splits, `1` tied split, and `0`
  negative splits.
- ARC-Easy, the same splits: mean selected delta versus loop 1 `-1.8/128`,
  with `0` positive splits, `1` tied split, and `4` negative splits.

Interpretation: the deeper-loop signal is not random, but a single global
selector is not sufficient. The next selector needs task difficulty or
confidence conditioning so hard examples can route deeper while easy examples
stay on the preserved direct path.

### Held-Out Depth Sweep Observation

A later held-out sweep was observed in Colab but has not yet landed as a GitHub
artifact at the time of this writing. Treat these numbers as provisional until
the artifact is recovered.

Observed held-out tail:

- ARC-Easy loop 1: recurrent `235/314`, base `239/314`, delta `-4`.
- ARC-Easy loop 2: delta `-26`.
- ARC-Easy loop 3: delta `-35`.
- ARC-Challenge loop 1: recurrent `21/43`, base `19/43`, delta `+2`.
- ARC-Challenge loop 2: recurrent `22/43`, base `19/43`, delta `+3`.
- ARC-Challenge loop 3: recurrent `17/43`, base `19/43`, delta `-2`.

Interpretation: the held-out signal is directionally consistent with the
allocation story, but ARC-Challenge tail size is too small for strong claims.
Depth 2 may help hard examples, but easy tasks strongly prefer shallow routing.

### ARC-Mix Content-Surface Confirmation

The latest targeted ARC-mix run is now the active deterministic recovery
candidate:

```text
source = outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/summary.json
checkpoint = outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt
confirmation = outputs/stage5/stage5_content_arcmix_qonly_optiontext_arc256_check_20260623_123424/summary.json
```

On the 256-example confirmation slice:

- ARC-Easy cyclic: recurrent `204/256`, base `202/256`, delta `+2`;
- ARC-Easy content-question-only: recurrent `155/256`, base `146/256`, delta
  `+9`;
- ARC-Challenge cyclic: recurrent `154/256`, base `154/256`, delta `0`;
- ARC-Challenge content-question-only: recurrent `97/256`, base `87/256`,
  delta `+10`.

This is the first bounded non-toy recurrent-vs-base win after model surgery.
It is not yet a robust benchmark claim. The per-slice paired sign tests are
structurally underpowered because most rows are ties, so these small recovery
slices should be read by sign, magnitude, and replication rather than by a
confirmatory `p < 0.05` rule. The discipline that replaces the p-value gate is
independent offset confirmation and debiased scoring. Content-question gains are
the leading indicator that the answer surface recovered; cyclic-debiased scores
are the survival gate that says the gain is not just another option-order or
label-prior artifact.

The independent offset check has since been reclassified as
`offset_confirmed_flat_debiased`: content-surface gains replicated on
ARC-Easy, and cyclic-debiased scoring stayed effectively flat rather than
materially negative. That is enough to justify a bounded depth-routing probe,
but not enough to make a robust performance claim. The post-depth benchmark
must therefore be read first on the cyclic-debiased surface, with content gains
kept as secondary evidence.

The learned-loop 512/validation follow-up completed on the step-50
depth-routing checkpoint:

- ARC-Easy content-question-only: recurrent `316/512`, base `298/512`, delta
  `+18`, W/L/T `43/25/444`, sign-test p `0.0385`;
- ARC-Easy cyclic-debiased: recurrent `406/512`, base `406/512`, delta `0`,
  W/L/T `4/4/504`;
- ARC-Challenge content-question-only: recurrent `108/299`, base `98/299`,
  delta `+10`, W/L/T `28/18/253`;
- ARC-Challenge cyclic-debiased: recurrent `177/299`, base `177/299`, delta
  `0`, W/L/T `6/6/287`.

After fixing the assessment coverage rule to recognize that ARC-Challenge
validation contains 299 paired rows, this passes the current cyclic-debiased
survival gate. The honest claim is recovery plus preservation under debiased
scoring, not a debiased surpass-base win.

### Local-HF Capability-Ladder Trace SFT

The next curriculum experiment used locally generated Qwen-7B traces from a
small verified capability-ladder shard:

```text
summary = outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_194543/summary.json
direct rows:       26
deep-narrow rows:  37
target loops:      1:26, 2:28, 3:9
validation:        validation_sane
direct loops:      ~= 1.35
deep loops:        ~= 1.99
```

This is not enough data for a broad claim, but it is the first concrete test
of the "model-scale gap as depth label" idea. The immediate benchmark readout
was mixed:

```text
ARC-Easy content:      recurrent 68/128 vs base 75/128, delta -7
ARC-Easy cyclic:       recurrent 100/128 vs base 95/128, delta +5
ARC-Challenge content: recurrent 46/128 vs base 43/128, delta +3
ARC-Challenge cyclic:  recurrent 69/128 vs base 68/128, delta +1
```

The larger confirmation after direct-preservation repair still did not clear
the broader non-negative gate:

```text
assessment = outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm_assessment/summary.json
status = needs_recurrent_recovery
ARC-Easy content:      recurrent 140/256 vs base 148/256, delta -8
ARC-Easy cyclic:       recurrent 203/256 vs base 201/256, delta +2
ARC-Challenge content: recurrent 86/256 vs base 87/256, delta -1
ARC-Challenge cyclic:  recurrent 151/256 vs base 156/256, delta -5
```

The follow-up surface diagnostics changed the interpretation. Most ARC-Easy
content losses are not knowledge losses; they are near-miss content/cyclic
surface mismatches:

```text
content losses: 16
cyclic rescues: 14/16
stable cyclic rescues: 8/16
unrescued losses: 2/16
content-loss answer rank under content scoring: rank 2 on 14/16, rank 3 on 2/16
recommendation: prioritize_content_cyclic_surface_alignment
```

Therefore the current front-of-queue GPU action is not a new capability-ladder
run and not SVGD. It is a bounded content/cyclic surface-alignment repair from
the latest traced-SFT direct-preservation checkpoint. The purpose is to recover
ARC-Easy content calibration without erasing the hard-slice signal.

### Particle / SVGD Status

SVGD and latent particles generate measurable diversity, but the current
evidence does not yet show reliable selector-converted benchmark lift. Earlier
kernel-geometry work taught two useful lessons:

- global PCA was dominated by between-task structure and was not the right
  calibration space for within-prompt particle repulsion;
- within-group projection is more principled, but kernel geometry is not the
  current blocker.

The next particle question is now a dynamical-systems question, not another
repulsion-scale sweep: does the deterministic recurrent map preserve more than
one latent pathway for a fixed prompt when particles are initialized apart and
SVGD/latent sampling are turned off? The new diagnostic is
`eval/eval_effective_pathways.py`, which computes Leinster-Cobbold
similarity-sensitive effective pathway counts over q in `{0,1,2,inf}`, using a
nearest-neighbor local bandwidth and per-prompt particle states.

Readout:

- effective pathway count near 1 plus shrinking final-vs-initial spread means
  the map is in a single-attractor/contractive regime; pause kernel tuning and
  work on the recurrent regime or pathway supervision;
- effective pathway count meaningfully above 1 means the dynamics can support
  multiple pathways; resume kernel/selector work and ask whether the selector
  converts that breadth into accuracy.
- effective pathway count above 1 plus very large final/initial spread and many
  next-token argmaxes means the map is not collapsed, but the variation may be
  expansive/chaotic rather than clean multistable breadth; run a lower-noise
  sensitivity sweep before treating the pathway count as useful reasoning
  diversity.

Particles should return only after deterministic depth routing and selector
metrics are working, and particle-kernel tuning should return only after this
effective-pathway gate says the recurrent dynamics are not immediately
collapsing the pathways.

## Experiment 1: Complete Held-Out Depth Sweep Artifact Recovery

### Hypothesis

The depth signal observed on the first ARC split replicates on held-out items:
loop 1 preserves base behavior, and deeper loops contain some hard-tail lift.

### Design

- Recover and commit the held-out run artifacts.
- Run or inspect `eval/analyze_depth_sweep.py` output for the held-out run.
- Compare base, loop 1, loop 2, loop 3, any-depth oracle, and simple selectors.
- Report ARC-Easy and ARC-Challenge separately.

### Runtime

No GPU if artifacts exist. CPU/local/GitHub task only.

### Success

- Loop 1 is within a small gap of base on easy tasks.
- ARC-Challenge shows positive loop-1 or loop-2 signal.
- Any-depth oracle beats loop 1 on ARC-Challenge.
- Simple held-out selectors are non-negative versus loop 1.

### Failure

- Loop 1 regresses materially.
- Deeper loops add no unique correct answers.
- Initial depth-sweep oracle gain disappears out of sample.

### Decision

If oracle gain survives, prioritize selector training. If it does not, return
to depth-1 preservation before any deeper-depth SFT.

## Experiment 2: Selector From Existing Loop Outputs

**Foundation instrument.** This is not just another experiment; it is the
readout used by depth sweeps, depth-ladder training, and later particle runs.
Once built, the selector/oracle read should be reused rather than reconstructed
inside every downstream run.

### Hypothesis

The current model already produces useful deeper answers; selected-answer
accuracy is limited by routing, not by absence of correct candidates.

### Design

Train and test lightweight selectors on saved loop outputs:

- loop-1 margin;
- base margin;
- answer likelihood by loop;
- agreement or disagreement across loops;
- halting telemetry;
- confidence bucket;
- debiased MCQ scores.

Use the first ARC split for selector fitting and held-out tail for evaluation.
Include dumb baselines: loop 1 only, loop 2 only, max score, mean score,
weighted loop-1-plus-deeper score, and margin threshold routers.

### Runtime

CPU/local unless regeneration of logits is required. L4/T4 is enough for small
regeneration.

### Success

- Held-out selector beats base and loop 1 on ARC-Challenge.
- ARC-Easy degradation is near zero.
- Helped examples exceed harmed examples under paired sign test.
- The selector's win is not caused by bare label bias.

### Failure

- Oracle gain is high but selector cannot recover it.
- Selector gains vanish under held-out evaluation.
- Selector helps aggregate but hurts hard-tail or easy preservation.

### Decision

Selector success unlocks depth-supervised SFT. Selector failure means the model
needs stronger depth training or richer verifier features before particle work.

## Experiment 3: Depth-1 Preservation SFT

### Hypothesis

Before teaching the model to think longer, the recurrent model should be trained
to behave like base Qwen at depth 1 on examples base Qwen already solves.

### Design

Build a depth-1 preservation shard:

- examples Qwen 0.5B answers correctly;
- high-confidence direct examples where possible;
- MCQ examples scored with cyclic/debiased evaluation;
- short answer and exact tasks where the answer is independently verified.

Train LoRA/bridge/halting with:

- target depth 1 or strong shallow-depth prior;
- answer CE;
- optional base-logit or answer-margin distillation;
- replay from direct/easy rows.

Keep particles off.

### Runtime

Short SFT can run on L4/T4 for 0.5B. Use A100 only for a longer confirmation
run after the smoke result is non-negative.

### Success

- Loop 1 equals or beats base on ARC-Easy and ARC-Challenge slices.
- Expected loop depth on direct rows moves toward 1.
- Correct-answer margin and label prior do not drift away from base.
- No loss instability or NaNs.

### Failure

- Training improves CE but worsens benchmark accuracy.
- Depth-1 rows still route deeper or become less calibrated.
- Easy-task accuracy drops while hard tasks do not improve.

### Decision

If depth-1 preservation passes, proceed to capability-ladder depth supervision.
If it fails, fix data quality and distillation before doing any depth-2/3 work.

## Experiment 4: Capability-Ladder Depth Supervision

### Hypothesis

The Qwen model scale ladder can provide an approximate training signal for
recurrent depth: examples solved by larger models but missed by smaller models
are better candidates for deeper recurrence.

### Design

Create verified buckets:

| Bucket | Selection Rule | Training Role | Target Depth |
|---|---|---|---:|
| Base preservation | Qwen 0.5B correct | preserve direct competence | 1 |
| Shallow upgrade | 0.5B misses, 1.5B correct | teach modest recurrence | 2 |
| Deeper upgrade | 0.5B and 1.5B miss, 3B or stronger solver correct | teach deeper recurrence | 3 |
| Unresolved | teacher disagreement or unverified | error analysis only | none |

Correctness must be independently verified. Larger-model agreement alone is not
enough for positive SFT.

Implementation note: `training/build_capability_ladder_curriculum.py` accepts a
generic ordered ladder through `--model_ladder`, for example:

```bash
python training/build_capability_ladder_curriculum.py \
  --input_jsonl data/curriculum/scored_capability_rows.jsonl \
  --work_dir data/curriculum/capability_ladder_qwen_scales \
  --model_ladder qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3,qwen_7b:4
```

This keeps the depth-labeling contract modular for 1.5B, 3B, 7B, or stronger
Qwen-style checkpoints. The first correct rung supplies the positive trace and
the target loop count; unresolved rows are skipped rather than used for
positive SFT.

Before training, run the SFT gate with both mode-row and target-loop-row
requirements. The target-loop check is the important anti-collapse guard for
the depth curriculum: it verifies that the shard really contains the explicit
`target_loop_count` rungs implied by the ladder instead of only a coarse
`direct` / `deep_narrow` mode mix.

For the bounded Colab scoring probe, set the matching environment variables:

```bash
STAGE5_CAPABILITY_LADDER_MODELS='qwen_0_5b=Qwen/Qwen2.5-0.5B-Instruct,qwen_1_5b=Qwen/Qwen2.5-1.5B-Instruct,qwen_3b=Qwen/Qwen2.5-3B-Instruct,qwen_7b=Qwen/Qwen2.5-7B-Instruct'
STAGE5_CAPABILITY_LADDER_MODEL_LADDER='qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3,qwen_7b:4'
```

Train deterministic Phase 1 only:

- answer CE on verified rows;
- halting supervision or KL toward target depth;
- depth-1 replay throughout;
- no particle diversity.

### Runtime

Dataset scoring can run on CPU/network jobs or low-cost GPU depending on model
availability. 0.5B SFT can start on L4/T4. A100 is reserved for a longer run
only after the bucketed smoke run shows the expected depth gradient.

### Success

- Mean expected loop depth increases by bucket.
- Depth-1 accuracy stays near base.
- Depth-2/3 buckets improve versus the untrained recurrent model.
- Held-out hard-tail selector performance improves.

### Failure

- The model ignores depth labels.
- All tasks collapse to max depth.
- Hard examples improve only by memorizing answer priors.
- Depth-2/3 gains erase depth-1 preservation.

### Decision

If successful, this becomes the main training recipe for making recurrence
substitute for scale. If unsuccessful, investigate whether the scale ladder is a
bad proxy or whether the model lacks capacity at 0.5B.

## Experiment 5: Conditional Invariance And MCQ Debias Training

**Foundation instrument plus later auxiliary.** Debiased/cyclic scoring belongs
at the front as measurement infrastructure. Conditional invariance as a training
loss belongs later and should be restricted to the reasoning path after
depth-1 recovery is stable. It should not become a broad depth-1 recovery
regularizer.

### Hypothesis

Some benchmark losses are caused by nuisance sensitivity to option order,
letter labels, and answer-position priors rather than reasoning failure.

### Design

For MCQ items:

- generate cyclic option permutations;
- score by content answer, not raw `A/B/C/D`;
- measure consistency of content answer across permutations;
- optionally train a light consistency loss only on depth-2-plus rows.

Do not use this as a standalone MCQ cleanup stage. Use it as a nuisance
collapse objective after depth-1 preservation is stable.

### Runtime

Mostly CPU/L4/T4. A100 is unnecessary unless combined with a larger SFT run.

### Success

- Label-position bias drops.
- Cyclic, content, and raw scores become less divergent.
- ARC-Challenge performance improves under debiased scoring.
- Open-ended reasoning does not regress.

### Failure

- The model learns option-format tricks.
- Open-ended generation gets worse.
- Gains appear only under raw labels and disappear under content scoring.

### Decision

If successful, include as a light auxiliary regularizer. If not, rely on
debiased evaluation and keep training capacity focused on reasoning.

## Experiment 6: Wide-Versus-Deep Curriculum Split

### Hypothesis

Depth and width are different resources. Deterministic multistep tasks need
depth; multi-solution tasks need candidate width. Mixing them without labels
teaches the wrong allocation.

### Design

Create typed curriculum rows:

- direct: base-known, shallow, no width;
- deep narrow: deterministic chain, high depth, low width;
- wide: multiple valid answers, moderate depth, high width;
- deep plus wide: hard tasks with multiple plausible methods.

Use programmatic tasks and verified teacher traces. Trace length alone is not a
depth label; use verifier difficulty, base misses, model-ladder gaps, and
solution structure.

### Runtime

Dataset generation and validation can run CPU/local. SFT smoke can run L4/T4.
A100 is reserved for the first mixed-allocation confirmation run.

### Success

- Direct rows stay shallow.
- Deep rows increase loop depth without particle spread.
- Wide rows increase candidate diversity only when correctness is preserved.
- Mixed hard rows benefit from selector conversion.

### Failure

- Width appears on deep narrow tasks and hurts accuracy.
- Depth appears on direct tasks and causes drift.
- Diversity is measured only as text variety, not correct candidate coverage.

### Decision

Only after this split should particles return as a training objective rather
than a diagnostic perturbation.

## Experiment 7: Phase 2 Particle / SVGD Re-Test

### Hypothesis

Particles help after deterministic recurrent depth has become meaningful; they
are premature while the direct/deep route is still unstable.

### Design

First run the effective-pathway diagnostic with SVGD off and latent sampling
off:

- K in {16, 32} particles initialized by embedded-input noise;
- max loops in {4, 8} to check whether depth increases contraction;
- q in `{0,1,2,inf}` for similarity-sensitive effective pathway counts;
- final/initial particle spread and Lyapunov-proxy readout;
- optional within-group projection versus raw hidden state if the raw state is
  dominated by nuisance scale.

Only if the diagnostic shows non-collapsed pathways, compare:

- deterministic loop selector;
- K=2 particles;
- K=4 particles;
- zero-noise particle control;
- low-noise particle settings;
- SVGD on/off;
- selector over candidate claims.

Evaluate separately on wide tasks, deep tasks, and deep-plus-wide hard-tail
tasks.

### Runtime

0.5B K=2/K=4 tests can run on L4/A100 depending on batch size. Avoid long A100
jobs until a small particle setting is non-negative.

### Success

- Effective pathway count is meaningfully greater than 1 on at least the
  prompts where particle breadth is expected to help, without uncontrolled
  next-token chaos.
- Candidate-hit rate rises versus deterministic recurrence.
- Unique correct candidates appear that deterministic loops miss.
- Selector converts candidates into selected-answer accuracy.
- Gains concentrate on wide or hard-tail tasks, not easy/direct rows.

### Failure

- Effective pathway count is near 1 across prompts and spread contracts with
  loop depth; this points to single-attractor dynamics, not a kernel issue.
- Effective pathway count is above 1 only at large perturbation scale while
  final/initial spread explodes and next-token argmaxes fragment; this points to
  unstable sensitivity, not yet useful breadth.
- Diversity rises but correct-candidate coverage does not.
- Results are seed-fragile.
- Particles help only toy prompts or formatting tasks.

### Decision

If the effective-pathway gate fails, pause SVGD and test regime-change or
method-anchored pathway supervision before further kernel geometry. If the gate
passes and selected-answer metrics improve, train particles with set/coverage
objectives. If the gate passes but selected metrics do not, focus on selector
and candidate-scoring rather than recurrent dynamics.

## Experiment 8: Trace Dataset Audit And Conversion

**Foundation instrument.** This must happen before capability-ladder SFT, not
after several training stages have already consumed noisy data. Bad traces and
bad measurement have already been recurring sources of confusion.

### Hypothesis

Opus, Fable, and other reasoning traces are useful only after filtering into
verified recurrent-compatible buckets.

### Design

Audit candidate datasets:

- Opus distilled reasoning traces;
- Fable-5 traces;
- GSM8K-style traces;
- ARC/GPQA-like MCQ traces;
- synthetic constraint, maze, N-Queens, and symbolic tasks.

For each row, extract:

- prompt;
- final answer;
- verification status;
- trace style;
- estimated depth;
- estimated width;
- source model or teacher;
- whether row is suitable for SFT, selector training, or only diagnostics.

### Runtime

CPU/local/network task. Do not spend A100 on dataset inspection.

### Success

- A clean JSONL curriculum exists with typed rows.
- Positive SFT rows are answer-verified.
- Wide/deep/direct labels are sufficiently reliable for training.
- Contaminated or unverified traces are excluded from positive SFT.

### Failure

- Too many traces are unverifiable.
- Trace style teaches formatting rather than reasoning.
- Dataset labels are too noisy for depth supervision.

### Decision

Only verified typed rows feed SFT. Ambiguous rows become selector or error
analysis data.

## Experiment 9: Matched Dense-Control Recipe

### Hypothesis

The architecture earns its place only if it beats a dense Qwen model trained
with the same recipe.

### Design

For any recurrent SFT recipe that looks promising:

- train a dense LoRA control on the same rows;
- use the same base model;
- match trainable budget as closely as practical;
- evaluate on the same debiased benchmark suite;
- compare hard-tail performance and aggregate harm.

### Runtime

L4/T4 for short 0.5B controls. A100 only for larger or longer matched runs.

### Success

- Recurrent model beats dense control on hard-tail selected-answer accuracy.
- Aggregate performance is not materially worse.
- Candidate coverage or selector lift explains the gain.

### Failure

- Dense control gets the same or better gains.
- Recurrent wins only through more parameters or more compute.
- Hard-tail gains are offset by broad easy-task losses.

### Decision

If recurrent does not beat the same-recipe dense control, do not claim
architecture lift. Continue as a training-recipe project or revise the
architecture.

## Experiment 10: Scale Probe At 1.5B

**Front-loaded information-value probe.** This should move earlier than its
number suggests. It is a no-training viability probe, not a commitment to move
development to 1.5B. Its purpose is to tell us whether the tiny 0.5B model is
too capacity-limited for the recurrent-depth payoff to be visible.

### Hypothesis

Qwen 0.5B may be below the capacity floor where recurrent depth can visibly
substitute for scale. A 1.5B model may preserve direct behavior while exposing
more useful depth signal.

### Design

Repeat only the minimal gates:

- identity wrapper;
- loop-1 direct preservation;
- small depth sweep on ARC-style slices;
- no particle training;
- no long SFT until these pass.

Implementation note: the generic no-training model viability launcher is
`colab/STAGE5_MODEL_VIABILITY_PROBE_CELL.py`, exposed through bootstrap target
`model_viability_probe`. It defaults to Qwen 1.5B and can probe 3B or larger
Qwen checkpoints by overriding `STAGE5_MODEL_PROBE_MODEL_NAME`.

### Runtime

A100 or H100 preferred. Do not use A100 for debugging the wrapper; first make
the 0.5B path and scripts robust.

### Success

- Identity passes.
- Loop 1 is close to base.
- Any-depth oracle and simple selector gains are larger than 0.5B.
- Memory and throughput are manageable.

### Failure

- Surgery regression grows with scale.
- Depth signal does not improve.
- K/depth compute is too expensive for the available budget.

### Decision

If 1.5B passes, it becomes the main model for serious training. If not, continue
0.5B mechanism work and defer scale.

## Experiment 11: Benchmark Gate For Base Surpass

### Hypothesis

After preservation SFT, capability-ladder depth supervision, and selector
training, recurrent Qwen 0.5B can beat unmodified Qwen 0.5B on hard reasoning
subsets.

### Design

Evaluate:

- ARC-Challenge and ARC-Easy with cyclic/debiased MCQ scoring;
- GPQA-lite or GPQA-Diamond-style MCQ with debiased scoring;
- GSM8K hard subsets;
- exact synthetic reasoning tasks;
- ARC-AGI-style programmatic tasks where answer verification is available.

Compare:

- base Qwen 0.5B;
- recurrent loop 1;
- recurrent selected depth;
- dense LoRA same-recipe control;
- recurrent particles only after Phase 2 passes.

### Runtime

L4/T4 for 0.5B eval where possible. A100/H100 for larger models, K>4, or long
benchmark runs.

### Success

- Recurrent selected-depth model beats base on hard-tail strata.
- Easy-task regression is controlled.
- Improvement survives held-out tasks and debiased scoring.
- Dense-control comparison supports architecture contribution.

### Failure

- Gains are only from SFT, not recurrence.
- Gains vanish against dense control.
- GPQA/ARC-AGI performance remains below base.

### Decision

Only after this gate should the project claim surpass-base reasoning
performance.

## Experiment 12: Hugging Face And Paper Release Gate

### Hypothesis

Once the benchmark gate passes, the model and method can be released as a
reproducible model-surgery result.

### Design

Prepare:

- HF adapter/checkpoint card;
- exact training configs;
- benchmark scripts;
- summary tables with paired tests;
- ablations: base, loop 1, selected depth, dense control, particles;
- paper draft with claims separated by evidence strength.

### Runtime

Mostly CPU/local. GPU only for final reproducibility checks.

### Success

- Checkpoint loads from HF.
- Reproduction notebook runs from clean Colab.
- Paper claims match evidence.
- The stronger surpass-base claim is backed by held-out benchmark data.

### Failure

- Release depends on local-only artifacts.
- Evaluation scripts are not reproducible.
- Claims exceed the actual evidence.

### Decision

Publish only the strongest claim supported by reproducible evidence. If
surpass-base is not yet proven, release as a model-surgery and recurrence
recovery study, not a benchmark-superiority result.

## Cost Discipline

Use CPU/local for:

- dataset inspection;
- JSONL conversion;
- selector analysis from saved outputs;
- markdown/docs;
- GitHub/Drive/auth repair;
- statistical summaries.

Use L4/T4 for:

- 0.5B short SFT smoke runs;
- 0.5B benchmark regeneration;
- small deterministic depth sweeps.

Use A100/H100 for:

- longer 0.5B confirmation SFT;
- 1.5B or 3B wrapper/identity/depth probes;
- K>4 particle runs;
- final benchmark confirmation.

Stop or avoid GPU when:

- the job is waiting on Google auth;
- the task is GitHub plumbing;
- a previous summary has not been read;
- the next action is not tied to a clear pass/fail gate.

## Current Next-Best Order

The ARC-mix offset confirmation, learned-loop benchmark, scale64 local-HF
traced-SFT run, and score-level surface repair moved the project from "can
surgery recover?" to "can depth-labeled curriculum improve without stealing
from the direct route?" The current order is:

1. Treat the score-level surface repair as a diagnostic, not the next source.
   It improved ARC-Challenge content from `86/256` to `91/256`, now `+4` over
   base, but did not recover ARC-Easy content (`140/256` to `139/256`, still
   `-7` versus base). Do not rerun the same 75-step score repair unchanged.
2. Return to the stronger bounded ARC-mix recovered checkpoint as the next
   competence-preserving source. Its 256-example confirmation remains the best
   non-toy recurrent-vs-base result so far: ARC-Easy content `+9`,
   ARC-Challenge content `+10`, ARC-Easy cyclic `+2`, ARC-Challenge cyclic `0`.
3. Run `STAGE5_CURRENT_A100_TARGET=arc_mix_offset_then_depth_chain`. This first
   re-confirms the ARC-mix checkpoint on the offset-256 slice, then launches the
   learned-depth ARC-mix continuation only if the offset gate passes. This is
   the next GPU target because it directly tests the current mechanism
   hypothesis: depth 1 for easy/direct rows, depth 3 for harder ARC-Challenge
   rows.
4. Assess the post-depth checkpoint on the same ARC-Easy/ARC-Challenge content
   and cyclic surfaces. Success means ARC-Easy content/cyclic remain
   non-negative while ARC-Challenge keeps or improves the hard-content gain.
5. If post-depth improves selected/hard-tail behavior without damaging the
   easy route, run the matched dense same-curriculum MCQ control. This answers
   whether recurrence contributes beyond the trace data itself.
6. If dense control matches or beats recurrent, do not chase more surface
   repairs. Improve the depth-label curriculum and rerun the recurrent-vs-dense
   comparison.
7. If recurrent beats dense on hard-tail surfaces, package that as the first
   architecture contribution: small-parameter recurrent surgery plus
   depth-labeled traces produces useful behavior that the same dense LoRA
   recipe does not.
8. Keep the CE8 depth curve as the fixed mechanism readout for later runs:
   [STAGE5_CE8_DEPTH_CURVE_2026_06_23.md](STAGE5_CE8_DEPTH_CURVE_2026_06_23.md).
9. Build or improve a selector/training objective that can preserve depth 1 on
   easy/direct rows while using depth 2-3 on hard/ambiguous rows.
10. Keep debiased/cyclic MCQ scoring as the default benchmark harness, but keep
   content-question-only scoring as a guardrail because ARC-Easy content
   calibration has been the main failure surface.
11. Audit and type trace data before more SFT consumes it.
12. Run depth-conditional preservation SFT:
   depth 1 for base-correct/direct/easy rows, depth 2-3 for verified hard rows.
13. Evaluate fixed depths, learned router, and selector on the same balanced
   ARC slices.
14. Compare against a same-recipe dense LoRA control.
15. Re-test particles/SVGD only after deterministic selected depth is useful.
16. Run broader GPQA/ARC-AGI-style benchmark gates.
17. Package HF artifacts and paper claims only after held-out surpass-base or
    same-recipe architecture evidence exists.

The immediate strategic question for the deep-research agent is not whether
SVGD is the right kernel. It is whether the depth-ladder curriculum and
selector can turn the observed fixed-depth split into selected-answer
improvement without sacrificing base-preservation behavior.
