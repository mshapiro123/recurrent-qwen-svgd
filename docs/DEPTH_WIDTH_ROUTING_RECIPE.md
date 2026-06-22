# Depth-Width Routing Recipe

## Purpose

This note reframes the current Stage 5 blocker as a resource-allocation problem.
The recurrent architecture already exposes two separate control surfaces:

- **Depth:** recurrent loop count, governed by sequence-level halting.
- **Width:** particle spread, governed by stochastic trajectories and SVGD-style
  repulsion.

The training problem is not just to make either surface stronger. It is to teach
the model when to spend depth, when to spend width, when to spend both, and when
to spend neither.

## Evidence Behind The Reframe

The current results are consistent with a real deep-versus-wide split:

- Deep narrow tasks, such as arithmetic procedures, can degrade as diversity
  rises. The correct behavior is convergence on one chain, not particle spread.
- Wide or multi-solution tasks improve more naturally from candidate diversity.
  The correct behavior is coverage of several valid approaches or answers.
- Multiple-choice calibration drift is plausibly a direct-mode failure: the
  recurrent model spends latent computation on items where the base model should
  answer directly.

The latest ARC-mix recovery proxy supports this caution. It did not produce a
competence lift and worsened margin calibration. More generic ARC-mix SFT is not
the next obvious answer; the next question is whether halting and routing are
learning the right computation budget for the problem type.

## Four Training Modes

The taxonomy is a data and evaluation scaffold, not a hard runtime classifier.
Real prompts lie on a continuum, and the model should learn a soft allocation.
Construct typed data by measurement rather than by trusting dataset labels:

1. sample several teacher solutions per problem;
2. keep only verified correct solutions where a verifier or answer check exists;
3. estimate **depth** from necessary reasoning steps, teacher trace structure,
   verifier/search difficulty, or recurrent loss by loop;
4. estimate **width** from distinct correct solution clusters, using explicit
   method labels where available or solution embeddings plus manual spot checks;
5. bin examples into direct, deep narrow, wide, or deep plus wide for curriculum
   and evaluation.

Trace length alone is not a depth label. It can help, but should be combined
with correctness, base confidence, verifier difficulty, and problem family.

| Mode | Desired allocation | Example families | Main risk |
|---|---|---|---|
| Direct | depth low, width low | easy MCQ, factual short answers, base-known ARC items | recurrent drift changes an already-correct answer |
| Deep narrow | depth high, width low | arithmetic, multi-step symbolic reasoning, deterministic proof traces | particles push away from the one correct chain |
| Wide | depth low/moderate, width high | multi-solution construction, valid examples, alternative plans | diversity becomes unsupported noise |
| Deep plus wide | depth high, width high | hard math/code/STEM with multiple plausible solution paths | selector cannot identify the correct claim |

## Curriculum Order

### 1. Direct And Deep Recovery

Train deterministic Phase 1 before particle training. The first gate is not
"can particles help?" but "can the surgically recurrent model answer directly
when no extra computation is needed, and think longer when the problem actually
requires depth?"

Recommended objectives:

- answer CE on deterministic reasoning traces;
- base-logit or answer-label distillation on direct and benchmark-style rows;
- halting supervision or regularization that rewards shallow depth on direct
  rows and non-collapsed depth on deep rows;
- no particle repulsion.

Diagnostics:

- expected loop depth by task type;
- accuracy by direct/deep split;
- correct-answer margin delta versus base;
- answer-prior shift by option label for MCQ.

Gate:

- recurrent Phase 1 is base-competitive on direct/easy rows;
- recurrent Phase 1 improves or stays non-negative on deep rows;
- direct rows show low loop depth rather than forced latent computation.

### 2. Width On Top Of A Depth-Competent Base

Only add particle width after Phase 1 is no longer broken on direct/deep rows.
Width on a model that cannot execute the narrow chain is just diversification of
a weak policy.

Recommended objectives:

- multi-trace or set-coverage supervision where multiple correct solutions are
  available;
- diversity reward only among candidates that remain correct or verifier-safe;
- low-noise particle screening before any long Phase 2 training;
- no reward for hidden-state distance by itself.

Diagnostics:

- candidate coverage by wide task family;
- selected-answer lift after a selector, not only oracle best-of-K;
- helped/hurt/tied against deterministic Phase 1;
- diversity among correct candidates, not raw output uniqueness.

Gate:

- particles improve candidate coverage without harming direct/deep selected
  answers;
- selector conversion is non-negative versus deterministic Phase 1.

### 3. Soft Allocation Training

After depth and width work separately, train mixed allocation. This is the least
proven and most important capability.

Inputs that can supervise or proxy allocation:

- base correctness and base margin;
- recurrent loss by loop depth;
- trace length and structure, but never trace length alone;
- task family labels for synthetic or curated data;
- candidate-set entropy and verifier disagreement;
- whether a problem admits multiple valid final answers.

Possible training signals:

- direct rows: penalize unnecessary loop depth and particle spread;
- deep rows: reward correct answer with sufficient loop depth and low particle
  disagreement;
- wide rows: reward diverse correct candidate coverage;
- deep-plus-wide rows: reward both depth and correct diverse coverage, then
  selected-answer conversion.

## Conditional Width Gate

The first allocation mechanism should be cheap and reversible. Do not build a
new learned routing head before this is tested.

Version 1 gates particle repulsion with an existing signal:

- halt-depth estimate from the deterministic recurrent path;
- early-state uncertainty, such as option-margin softness or candidate entropy;
- base confidence on direct/easy MCQ items.

Expected behavior:

- base-confident/direct rows: repulsion off or near zero;
- deep narrow rows: repulsion off, depth allowed;
- wide rows: repulsion on when candidate coverage is valuable;
- deep-plus-wide rows: depth and repulsion both on.

If the cheap gate cannot separate direct/deep/wide behavior, then introduce a
small allocation predictor over the prelude pooled state. That predictor should
emit a depth budget and width budget per problem, and it should be treated as
the novel and least certain component.

## Evaluation Design

Future reports should stratify by type as well as difficulty:

- direct;
- deep narrow;
- wide;
- deep plus wide.

The decisive metric is paired selected-answer performance. Useful secondary
metrics are:

- oracle best-of-K;
- candidate hit count;
- within-type helped/hurt/tied;
- mean loop depth by type;
- particle diversity among correct candidates;
- answer-prior drift and correct-answer margin deltas.

The architecture should not be claimed to beat dense sampling on width alone. A
dense model with temperature sampling is already a strong width baseline. The
recurrent-particle claim is strongest in the deep-plus-wide quadrant, where
latent recurrent computation might produce several well-developed approaches
without paying all reasoning cost in output tokens.

The dense baseline must be strong:

- same base model family;
- same training data where possible;
- same or explicitly reported trainable-parameter budget;
- output-space depth through chain-of-thought or longer generation;
- output-space width through temperature sampling or best-of-K;
- the same selector/verifier applied to both dense and recurrent candidates.

Weak dense baselines do not test the architectural claim.

## Immediate No-GPU Repair Work

Before any more A100 spend:

1. Add loop-depth diagnostics to the ARC MCQ reports, grouped by easy/direct
   versus hard/deep slices.
2. Inspect examples where base and recurrent start are correct but the latest
   ARC-mix continuation flips the answer.
3. Tag or construct a small typed evaluation set with direct, deep, wide, and
   deep-plus-wide rows.
4. Re-plan the next Phase 1 run as direct/deep recovery, not generic ARC-mix
   continuation.
5. Keep Phase 2/SVGD and scale-up deferred until deterministic routing is
   healthy.

Current implementation status:

- `eval/eval_mcq.py` can now emit loop-depth diagnostics for recurrent MCQ
  scoring rows.
- `eval/analyze_mcq_regressions.py` summarizes paired base-versus-recurrent
  losses and wins by routing bucket.
- `colab/run_stage5_benchmark_suite.py` carries those routing buckets into the
  benchmark-suite `summary.json` and `summary.md`, without storing prompt text in
  the aggregate suite report.
- `colab/run_stage5_routing_diagnostic.py` runs the bounded ARC-Easy /
  ARC-Challenge diagnostic that decides whether the next repair is direct-mode
  halting or deep-narrow recovery.
- `colab/run_stage5_routing_repair.py` consumes that diagnostic and launches
  one bounded deterministic Phase 1 repair profile with particles/SVGD off.

The next GPU run should be bounded and diagnostic. Use an L4 or T4 for
diagnostic-only benchmark scoring if it fits; reserve A100/H100 for training
runs that have already passed a local measurement gate. The immediate useful
GPU spend is a small Stage 5 MCQ eval with loop diagnostics enabled, followed by
a direct/deep recovery Phase 1 run only if the diagnostics confirm that shallow
direct rows are over-looping or drifting from base calibration.

## Open Questions For Deep Research

1. What is the best low-cost supervision for "direct mode" halting at or near
   zero additional depth?
2. Can loop-depth targets be inferred reliably from teacher traces, or should
   they come from verifier/search difficulty instead?
3. Which benchmark families cleanly instantiate deep narrow versus wide versus
   deep-plus-wide reasoning?
4. What selector is strong enough to compare recurrent latent width against
   dense temperature-sampling width fairly?
5. At what model scale, likely 1.5B or 3B, does the deep-plus-wide quadrant
   become meaningful enough to test the architecture's real advantage?
