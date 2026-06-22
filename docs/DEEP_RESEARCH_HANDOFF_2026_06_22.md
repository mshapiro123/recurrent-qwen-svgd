# Deep Research Handoff: MCQ Debias Before Direct-Route Training

## Purpose

This handoff is for the strategy/deep-research agent. It captures the current
state after the latest Stage 5 ARC-mix preservation failure and the follow-up
MCQ debias diagnostic. It frames the next questions around evidence rather than
enthusiasm for any one mechanism.

The project is still aimed at a compact recurrent-particle reasoning model, but
the current blocker is even more basic than direct-route training: the apparent
ARC direct-regression signal is confounded by multiple-choice option-label and
position bias. Before we train to fix the recurrent model, we need to know
whether the model is actually wrong or whether the scoring surface is wrong.

## Current Thesis

The scientific question is:

> Can a pretrained dense Qwen model be surgically converted into a recurrent
> latent-depth model, recover the base model's depth-1 competence with a small
> trainable budget, and then use learned depth plus latent particles to exceed
> the dense base on harder reasoning?

This splits the program into three claims that must be proven in order:

1. **Preservation:** at depth 1, the recurrent model should behave like the
   original dense model on base-known examples.
2. **Depth lift:** at depth 2+, the recurrent model should improve on examples
   the 0.5B base misses but larger same-family models or verified teachers
   solve.
3. **Width lift:** once the deterministic recurrent path is competent, particles
   and SVGD should improve useful candidate coverage, especially on hard-tail or
   multi-solution tasks, and a selector should convert that coverage into
   selected-answer accuracy.

The current evidence supports claim 0, that the surgery can be made
identity-preserving in the one-pass gate. It does not yet prove claim 1 on
benchmarks after training.

## Evidence Snapshot

### Architecture And Identity

The Qwen wrapper can reproduce the base model path exactly under the strict
identity gate when configured with float32/eager attention. This makes the
architecture surgery credible: the recurrent wrapper is not inherently unable
to represent the base computation.

### Stable Recurrent Training

Phase 1 recurrent training became numerically stable after moving trainable
adapters/controllers to fp32 while keeping the frozen Qwen base in low
precision. Sequence-level halting can learn non-collapsed expected loop counts.

### SVGD Mechanism Signal

SVGD and within-group particle geometry produced useful smoke-test signals:
within-group projected particle updates improved oracle and candidate-hit
coverage on controlled exact-task suites. This remains valuable mechanism
evidence, but it is not the current bottleneck.

### Non-Toy ARC Recovery

The best Stage 5 recovery work recovered much of the surgery-induced regression
but did not establish base-model superiority. A balanced ARC/Opus-style
checkpoint showed:

- ARC-Easy: recurrent remained below base.
- ARC-Challenge: recurrent was slightly above base in one balanced assessment.

This is promising but not enough for a benchmark claim.

### Latest Blocker: MCQ Option-Prior Drift

The latest conservative ARC-mix preservation probe failed to recover direct
base behavior. The diagnosis artifact is:

`outputs/stage5/stage5_arc_agi_next_action_20260622_181850_plan_conservative_direct_preservation/answer_prior_diagnosis.json`

ARC-Easy 128 result:

| model | correct |
|---|---:|
| base Qwen 0.5B | 87/128 |
| recurrent start | 82/128 |
| conservative probe best | 81/128 |

The label prior shifted badly:

| model | A | B | C | D |
|---|---:|---:|---:|---:|
| base | 36 | 33 | 42 | 17 |
| recurrent start | 58 | 15 | 22 | 33 |
| conservative probe best | 56 | 18 | 26 | 28 |

The failed probe barely changed the inherited prior drift. The correct reading
is not "train longer"; it is "the objective did not target the actual failure."

The next diagnostic tested whether that failure is a genuine competence loss or
a multiple-choice scoring artifact:

```text
outputs/stage5/stage5_mcq_debias_direct_20260622_194346/summary.json
status = selection_bias_likely
ARC config = ARC-Easy
```

Under bare `A/B/C/D` scoring, the loop-4 recurrent checkpoint trailed base by
`-5`. Under cyclic option-permutation aggregation, it was `+1` versus base.
The conservative best checkpoint was `-6` under bare labels and `0` under
cyclic aggregation. This strongly suggests that repeated recurrent computation
is amplifying a known MCQ option-label/position bias and that bare-label ARC
results are not a sufficient training target.

## Current Best Next Experiment

The next bounded GPU action is **not training**. It is the ARC-Challenge version
of the same debias diagnostic:

```text
STAGE5_CURRENT_A100_TARGET=arc_challenge_mcq_debias_confirm
```

This experiment should:

1. compare base and recurrent on ARC-Challenge;
2. report bare-label, content-only, and cyclic option-permutation scores;
3. push a summary artifact;
4. disconnect the runtime;
5. decide whether direct-route training is still warranted.

This is intentionally narrow. It asks whether the recurrent model really lost
ARC competence or whether the apparent regression is mostly a measurement
artifact. Direct-route preservation training comes next only if cyclic scoring
still shows a material gap.

## Strategic Questions For Deep Research

### 1. What is the right MCQ scoring policy?

The research agent should treat this as the immediate question. Investigate
whether all future MCQ benchmarks should report and gate on:

- cyclic-permutation aggregated likelihood;
- content-only likelihood as a secondary diagnostic;
- prediction-count drift by label;
- edge-minus-middle drift;
- paired wins/losses against base.

The key decision is whether cyclic scoring is the primary fair metric for ARC,
GPQA-style tasks, and any future MCQ benchmark claims.

### 2. What is the right preservation objective if a debiased gap remains?

The current failure suggests answer CE alone, or mixed ARC SFT, is not enough.
Investigate whether preservation needs one or more of:

- base-logit KL over all answer options;
- correct-answer margin preservation;
- label-prior regularization;
- hard `max_loops=1` direct-route training;
- direct-route adapter isolation;
- freezing or reducing LoRA updates that perturb coda/readout calibration.

The key design goal is not just accuracy. It is base-like calibration on
base-confident rows under debiased scoring.

### 3. Is the direct route distinct enough architecturally?

The recurrent wrapper may need an explicit direct mode, not just a shallow
halting prior. Questions:

- Should depth-1 execution bypass the bridge/halting-transformed state more
  aggressively?
- Should there be a learned residual gate that stays exactly zero on direct
  rows?
- Is answer-prior drift coming from recurrent-block LoRA, bridge, halt-weighted
  output mixing, or coda interaction?

The research agent should propose ablations that identify where the direct
route loses calibration.

### 4. How should capability-ladder depth targets be built?

The promising curriculum is:

| tier | selection rule | target depth |
|---|---|---:|
| base-preservation | Qwen 0.5B correct/high margin | 1 |
| shallow upgrade | Qwen 0.5B wrong, Qwen 1.5B correct, answer verified | 2 |
| deeper upgrade | 0.5B/1.5B wrong, Qwen 3B or strong teacher correct | 3-4 |
| unresolved | teacher disagreement or no verifier | no positive SFT |

The agent should refine how to obtain the labels cheaply and how to avoid
training on larger-model hallucinations. Stronger-model success is routing
evidence, not proof.

### 5. Which data sources belong in each lane?

Opus/Fable/TraceInversion should not be mixed blindly. A useful split is:

- **Direct preservation:** short MCQ, ARC-Easy, base-correct rows, high-margin
  dense-base answers.
- **Deep narrow:** verified math/science traces, Opus-style reasoning, tasks
  with one final answer and little legitimate diversity.
- **Wide:** multi-solution construction, valid examples, alternate proofs or
  strategies.
- **Deep plus wide:** hard verifiable problems where several plausible solution
  paths can lead to a checked answer.

Fable-style tool traces may be valuable later for wide or agentic trajectories,
but they may confound direct ARC/GPQA recovery if used too early.

### 6. When does model size become the bottleneck?

Qwen 0.5B is probably a mechanism testbed, not the model that proves the
frontier-tail thesis. The agent should reason about the earliest scale at which
the architecture can fairly show upside:

- 0.5B: surgery, preservation, routing, cheap curriculum plumbing.
- 1.5B: first serious depth-ladder test.
- 3B: likely minimum for a credible compact frontier-reasoning track.

The cost question matters: particles multiply runtime, so scale should happen
only after preservation and depth routing are proven at 0.5B.

### 7. What is the fair dense baseline?

The decisive experiment is not recurrent versus raw base only. It is:

- dense model trained with the same spectrum-to-signal recipe;
- recurrent model trained with the same recipe plus depth/width mechanisms;
- same selector/verifier;
- hard-tail stratified comparison.

The agent should specify the minimal dense-control experiment that prevents us
from attributing a training-recipe gain to the architecture.

### 8. When should SVGD come back?

SVGD should re-enter after deterministic recurrent recovery clears direct-route
preservation. The right question then is not raw diversity; it is:

- does particle width improve candidate coverage on examples where width is
  appropriate?
- does it avoid harming direct and deep-narrow rows?
- does a selector convert the coverage into selected-answer lift?

Until then, more kernel geometry is likely lower leverage than direct-route
repair and depth-ladder data construction.

## Recommended Research-Agent Output

The most useful response would be a concrete plan for:

1. MCQ scoring policy after cyclic ARC-Challenge confirmation;
2. direct-route preservation losses and ablations if debiased gaps persist;
3. a capability-ladder data builder using Qwen 0.5B/1.5B/3B comparisons plus
   answer verification;
4. a minimal dense-control baseline;
5. a decision gate for when to resume particles/SVGD;
6. a compute-aware path from 0.5B to 1.5B or 3B.

Do not recommend a large A100 run unless it directly answers one of these
questions. The current high-value GPU run is the bounded ARC-Challenge MCQ
debias confirmation; everything else should either be CPU-side data/eval work
or a carefully gated follow-up.
