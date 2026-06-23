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

### Particle / SVGD Status

SVGD and latent particles generate measurable diversity, but the current
evidence does not yet show reliable selector-converted benchmark lift. Earlier
kernel-geometry work taught two useful lessons:

- global PCA was dominated by between-task structure and was not the right
  calibration space for within-prompt particle repulsion;
- within-group projection is more principled, but kernel geometry is not the
  current blocker.

Particles should return only after deterministic depth routing and selector
metrics are working.

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

Compare:

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

- Candidate-hit rate rises versus deterministic recurrence.
- Unique correct candidates appear that deterministic loops miss.
- Selector converts candidates into selected-answer accuracy.
- Gains concentrate on wide or hard-tail tasks, not easy/direct rows.

### Failure

- Diversity rises but correct-candidate coverage does not.
- Results are seed-fragile.
- Particles help only toy prompts or formatting tasks.

### Decision

If successful, train particles with set/coverage objectives. If not, pause SVGD
and continue deterministic recurrence and selector work.

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

Run the cheap foundation layer first, in parallel where practical:

1. Recover or rerun the held-out depth-sweep artifact.
2. Promote selector/oracle analysis into a fixed readout used by all later
   runs.
3. Keep debiased/cyclic MCQ scoring as the default benchmark harness.
4. Audit and type trace data before more SFT consumes it.
5. Add and run a no-training 1.5B recurrent viability probe: identity,
   loop-1 preservation, and a tiny depth sweep. This is an information-value
   probe only; 0.5B remains the cheap mechanism workbench unless 1.5B clearly
   changes the signal.

Then run the decisive deterministic spine:

6. Run depth-1 preservation SFT on base-correct rows.
7. Build the capability-ladder depth dataset from verified 0.5B/1.5B/3B or
   strong-solver results.
8. Run deterministic depth-supervised SFT and evaluate with the fixed selector
   and debiased harness.
9. Compare against a same-recipe dense LoRA control.

Only then open the branch work:

10. Re-test particles/SVGD after deterministic depth routing is useful.
11. Run broader GPQA/ARC-AGI-style benchmark gates.
12. Package HF artifacts and paper claims only after held-out surpass-base or
    same-recipe architecture evidence exists.

The immediate strategic question for the deep-research agent is not whether
SVGD is the right kernel. It is whether the depth-ladder curriculum and selector
can turn the already observed oracle-depth signal into selected-answer
improvement without sacrificing base-preservation behavior.
