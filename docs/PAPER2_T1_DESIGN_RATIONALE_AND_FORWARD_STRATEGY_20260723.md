# Paper Two T1 Design, Gate Rationale, and Forward Strategy

> **Draft 3 amendment, 2026-07-23:** The registered experiment is now
> full-block-only T1-lite, an actuator qualification for D0. The two-lineage
> capacity comparison below is historical design rationale and is not
> authorized. The governing documents are
> `PAPER2_EXPERIMENTAL_PLAN_DRAFT3_20260723.md` and
> `PHASE_T1_LITE_PREREGISTRATION_DRAFT4_20260724.md`, which supersedes Draft 3.

**Date:** 2026-07-23  
**Status:** design and strategy memo; it does not lock T1 or authorize training  
**Program:** causal control of recurrent computation

## 1. Executive reading

T1 asks a deliberately bounded causal question:

> Can a recurrent Qwen model learn an explicit internal continue/stop decision
> that selects the required recurrent depth without materially degrading the
> already demonstrated transition mechanism?

The experiment does not yet ask the model to infer open-ended problem
difficulty. The synthetic prompt states the required number of transitions.
T1 is therefore an information-path and actuator test: can an explicit token
position read the instruction and current loop state, emit a correct internal
decision, and causally change execution?

This distinction is important. A T1 pass establishes a controllable recurrent
machine. It does not establish natural adaptive reasoning. A T1 failure
localizes a failure of this joint-training interface. It does not establish
that adaptive recurrence, post-hoc stopping, convergence exits, budget
conditioning, or content-determined halting are impossible.

The main design is sound, but four details should be made explicit before the
preregistration is locked:

1. Define gate 3 as row-level exact selected-depth accuracy, not raw
   transition accuracy.
2. Balance continue and stop supervision so an always-continue policy cannot
   exploit label frequency.
3. Define the intervention in gate 4 at the control logits, rather than by
   directly overwriting the loop counter.
4. State the paired statistical readout and integer decision rules attached to
   the three-point margins and the per-depth 90 percent threshold.

## 2. Evidence entering T1

### 2.1 The recurrent computation is real under forced depth

The full-block reference answered `1005/1024 = 98.14%` of the frozen trained-
depth rows correctly. The R16-plus-bridge reference answered
`1021/1024 = 99.71%`. Both use the same 1,024 Phase A rows, depths 1 through 8,
with forced loop count equal to row depth. T1 therefore starts from a task
family on which recurrent computation is known to work.

### 2.2 The old pooled halting route failed even when depth was stated

The bounded selector's supervised S1 control selected the stated depth on only
`70/768 = 9.11%` of held-out rows. Its outcome-only Ponder arm selected depth
12 on all 768 rows. This ruled out more tuning of that same pooled head; it did
not rule out a new information path.

T1 changes that path. A reserved token position participates in the sequence
computation, and continue/stop are represented by dedicated rows in the tied
language-model vocabulary. The decision is intercepted internally and is
never emitted to the user.

### 2.3 Arm G failed at terminal re-entry control

Even true next-symbol information could not reliably control the frozen model
through terminal additive or FiLM re-entry conditioners. On all 1,899 training
variants, non-default transition control was `20.18%` and `23.30%`. This
localizes the tested failure to the terminal interface rather than held-out
generalization.

T1 is complementary. It does not ask a terminal residual to steer a branch.
It gives the recurrent computation an explicit discrete decision and an
actuator that controls whether another loop executes.

### 2.4 T0 proved that the interface can be installed neutrally

T0 added exactly three model rows (`2,688` tied parameters), masked all three
from visible generation, preserved every original embedding and output row,
and produced an inactive one-loop logit difference of exactly `0.0`. Forced
requested, executed, and selected loop counts agreed. The remaining question
is learning, not basic wiring or identity.

## 3. Experimental design

### 3.1 Two training-budget lineages

Both lineages begin from fresh Qwen2.5-0.5B surgery, not from a promoted
keeper:

1. **Full-block T1:** recurrent block, repaired split bridge, and three control
   rows train.
2. **Adapter T1:** rank-16 LoRA over recurrent-block projections, repaired
   split bridge, and three control rows train; pretrained Qwen weights remain
   frozen and hash-checked.

The two lineages answer whether controllability depends on the 180.6M
full-block training budget or survives at the approximately 6.0M adapter
budget. Either asymmetric outcome is informative.

### 3.2 Data and targets

Training uses the controlled synthetic transition family at depths 1 through
8. At the reserved control position after loop `t`:

- target `continue` when `t < d`;
- target `stop` when `t = d`;
- `d` is the exact row depth stated in the prompt.

The answer path retains full-symbol, question-only, first-completed-response
scoring. Control symbols cannot be generated visibly.

The current proposal mixes 70 percent control-target examples with 30 percent
mechanism rehearsal. Before lock, the loss contract should state whether a
control-target example also receives answer/chain loss. The recommended
contract is:

```text
L_control_row = balanced CE over continue/stop decisions
L_mechanism   = existing per-loop chain plus outcome objective
training mix  = 70% rows carrying L_control_row + L_mechanism
                30% pure mechanism-rehearsal rows
```

This avoids training the controller on examples whose recurrent computation
is allowed to drift freely.

### 3.3 Class-balanced control loss

Uniform depths 1 through 8 contain 28 continue transitions and only 8 stop
transitions. An always-continue predictor therefore reaches `28/36 = 77.78%`
raw transition accuracy. Raw micro accuracy is not an acceptable primary
control metric.

For each row with depth greater than one, assign half of the row's control-loss
mass to its stop decision and divide the other half across its continue
decisions. A depth-1 row assigns all mass to stop. This gives every row equal
total weight and prevents deeper rows or the majority continue label from
dominating the objective.

Report transition-level balanced accuracy, continue recall, stop recall, and
macro F1 as diagnostics. The gate itself should use exact selected depth per
row.

### 3.4 Curriculum and optimizer

The proposed 10,500-step ceiling mirrors the curriculum that installed the
known mechanism:

| Stage | Support | Steps | LR |
|---|---|---:|---:|
| Primitive | 1 | 500 | `2e-5` |
| Chain | 1-2 | 2,000 | `1e-5` |
| Chain | 1-4 | 4,000 | `1e-5` |
| Chain | 1-8 | 2,000 | `1e-5` |
| Dose/consolidation | 1-8 | 2,000 | `1e-5` |

The rationale is not that 10,500 is intrinsically optimal. It is a matched
dose inherited from the positive mechanism program, large enough to avoid
calling a dose-limited run an architecture failure. Checkpoints and frozen
validation readouts should land at every stage boundary and every 1,000 steps.
The registered endpoint remains fixed; intermediate curves diagnose learning
but do not permit post-hoc checkpoint selection.

Use AdamW as the primary optimizer, matched to the successful mechanism
installation. Muon would add an optimizer-interface variable to a causal-path
experiment and previously required separate gradient-path investigation in
this project. Muon can be a later optimizer ablation only if the primary path
works or a specific AdamW failure motivates it.

### 3.5 Evaluation sets and outputs

Primary evaluation uses the immutable Phase A rows, 128 per depth for depths
1 through 8 (`n=1,024`). Each row receives two paired evaluations:

1. **Forced depth:** execute exactly the gold row depth.
2. **Self halted:** allow the internal token decision to stop execution.

Record:

- answer correctness under forced and self-halted execution;
- exact selected loop count;
- absolute loop-count error;
- under-compute and over-compute rates;
- continue/stop confusion by loop index and row depth;
- executed loops and wall-clock latency;
- control-logit margins and calibration;
- paired row IDs and predictions.

Depths 9 through 14 should be evaluated descriptively with no gate. This costs
only inference and tests whether the controller extrapolates beyond its
trained support. It must be labeled out-of-support.

## 4. Gate rationale and exact readings

All four gates are conjunctive. Passing one cannot compensate for failing
another because each excludes a different false explanation.

### Gate 1: preserve forced-depth chain accuracy

**Rule:** forced-depth T1 accuracy must be within three absolute percentage
points of the matched non-halting reference.

The three-point margin is the project's standing bounded non-inferiority
tolerance. It was used for preservation, adapter parity, canaries, and K=1
checks. Here it means the controller cannot buy routing by materially damaging
the computation it routes.

At `n=1,024`, the count floors are:

- full block: reference `1005/1024`; floor `95.1445%`; minimum `975/1024`;
- adapter: reference `1021/1024`; floor `96.7070%`; minimum `991/1024`.

This is a practical non-inferiority margin, not a universal statistical
constant. Report the paired T1-versus-reference discordant counts and a
one-sided paired confidence interval for the accuracy difference. The binary
count rule remains the preregistered gate.

**Good:** both lineages preserve their references.  
**Bad:** forced-depth performance falls outside margin. This means joint
control training damaged mechanism competence; it does not show that the
controller cannot select depth on a frozen healthy backbone.

### Gate 2: self-halted answers match forced-depth answers

**Rule:** self-halted accuracy must be within three absolute points of the
same checkpoint's paired forced-depth accuracy.

This isolates the cost of allocation from the quality of the trained
computation. The comparison is within checkpoint and row, so initialization
and mechanism quality cannot explain the delta.

**Good:** the learned decisions preserve almost all answer competence.  
**Bad:** forced depth works but self-halting loses more than three points. The
substrate is healthy and the policy is the failure. Separate under-halting
from over-halting; they imply different repairs.

### Gate 3: strong exact depth selection

**Recommended exact definition:** at every trained depth, at least 116 of 128
rows select the exact gold depth (`116/128 = 90.625%`). If the program applies
the standing nearest-integer convention for small-sample gates, it may lock
115/128 instead, but that choice must be written before training.

Why 90 percent? The task states the depth explicitly and is intentionally an
easy control-path test. Chance or a coarse fixed-depth policy is not an
interesting success. Ninety percent requires the new interface to work on the
large majority of each depth stratum, while gate 2 separately requires that
its remaining errors do not destroy answers.

This is a quality threshold, not a claim that the population accuracy has a
90 percent lower confidence bound. Requiring the observed threshold at all
eight depths is much stricter than it looks: if true per-depth accuracy were
exactly 90 percent, the probability of observing at least 116/128 at all eight
depths is about `0.28%`; at true 95 percent it is about `90.8%`. The gate is
therefore operationally close to demanding a 95-percent-quality controller.

Two defensible choices exist before lock:

1. **Strict claim gate:** retain 116/128 at every depth and report near misses
   descriptively. This maximizes claim strength.
2. **Noise-robust gate:** require at least 90 percent pooled exact-depth
   accuracy and at least 85 percent at every depth, with balanced stop and
   continue recall at least 90 percent. This reduces the eight-stratum
   multiplicity penalty.

The first is consistent with the current draft. The second is statistically
less brittle. Neither should be chosen after seeing T1 data.

### Gate 4: causal override

**Rule:** intervene on the control logits before the decision is sampled or
argmaxed. On a model-continue event, force stop to be the winning control logit
and verify termination at that loop. On a model-stop event, force continue and
verify at least one additional loop executes within the maximum.

Run this on all 1,024 primary rows. Do not satisfy the test by writing directly
to `selected_loop` or `max_loops`; that would test the outer loop, not the
token decision. Report both intervention directions, eligible-event counts,
and execution compliance.

Why no numeric threshold below complete compliance? This is primarily a
software and causal-actuation invariant. Once an eligible control decision is
overwritten, execution should obey it deterministically. Any noncompliance is
an interface bug or an unregistered decision path.

## 5. Interpretation matrix

| Forced chain | Self-halted | Exact depth | Override | Reading |
|---|---|---|---|---|
| Pass | Pass | Pass | Pass | Full T1 positive: explicit token control is accurate, competence-preserving, and causal. |
| Fail | Any | Any | Pass | Joint training damaged the mechanism; test frozen/post-hoc control or increase preservation, not another selector sweep. |
| Pass | Fail | Fail | Pass | Actuator works, learned policy is weak; analyze under/over-halting and training exposure. |
| Pass | Pass | Fail | Pass | Answers are depth-robust, but the model did not learn precise allocation; no learned-depth claim. |
| Pass | Pass | Pass | Fail | Decisions correlate with depth but are not the execution cause, or the override hook is wrong. |
| Pass in one budget only | Varies | Varies | Pass | Controllability is training-capacity-sensitive; report the budget interaction. |
| Fail in both budgets | Varies | Varies | Pass | This joint token-pathway recipe fails on both budgets; alternative interfaces remain open. |

### Strong positive

Both lineages pass all gates, with low loop-count error, calibrated stop
probabilities, and no canary regression. This supports an existence claim at
two training budgets. A second seed is still needed before a robustness claim.

### Useful partial positive

Only full block passes. The control operation is learnable but requires more
adaptation capacity than R16 provides. Only R16 passes. Parameter-efficient
training may protect the installed computation better than full-block joint
training. Either outcome is scientifically useful.

### Clean negative

Both lineages preserve forced-depth accuracy and pass override, but fail exact
selection and self-halted parity by large margins. This is a learned-policy
negative on a healthy, causal actuator. It justifies changing how depth
robustness and allocation are trained, not abandoning recurrent computation.

### Inconclusive negative

Forced-depth competence collapses, gradients are absent, stop/continue labels
are unbalanced, or the causal override does not obey intervention. Those are
training or implementation failures and cannot support a controllability
boundary claim.

## 6. What the external literature changes

### 6.1 Learned halting is historically fragile

[Adaptive Computation Time](https://arxiv.org/abs/1603.08983) established
learned recurrent step counts. [Universal Transformers](https://arxiv.org/abs/1807.03819)
applied dynamic per-position halting to a weight-tied Transformer.
[PonderNet](https://arxiv.org/abs/2107.05407) replaced ACT's deterministic
accumulation with a probabilistic halt distribution and a KL penalty to a
geometric compute prior. PonderNet explicitly describes ACT as unstable and
shows that its own behavior depends on a compute prior. These precedents argue
for causal and competence gates, not for dismissing halting after one recipe.

### 6.2 Explicit tokens can carry computation, but exposure matters

[Pause-token training](https://arxiv.org/abs/2310.02226) showed that learned
special tokens can support extra hidden computation, with its strongest gains
when models were exposed to delays in both pretraining and fine-tuning. T1 is
not pause-token replication: its controls select recurrent execution rather
than merely add token positions. The result nevertheless warns that three new
rows introduced only at fine-tuning may not transfer naturally without a
larger exposure curriculum.

### 6.3 Depth robustness and depth allocation may need separation

[Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) jointly trains
lightweight routers with a shared recursive stack and demonstrates adaptive
token-level depth. In contrast, a very recent single-seed preprint,
[Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers](https://arxiv.org/abs/2607.14427),
reports that a training-free convergence rule on a frozen randomized-depth
backbone outperformed a learned linear router. The latter should be treated as
new evidence, not settled consensus, but its separation principle is directly
relevant: first train an elastic backbone, then allocate compute.

### 6.4 Variable-depth training changes the computation learned

[Looped Transformers for Length Generalization](https://arxiv.org/abs/2409.15647)
uses step-dependent supervision and adaptive loop counts, and reports that
input re-injection and variable depths matter. A 2026 preprint on
[learned stochastic stopping](https://arxiv.org/abs/2606.29983) finds that
randomizing training depth reduces run-to-run and loop-count sensitivity, but
can stabilize a suboptimal computation. This is a direct warning against
equating stable halting with good reasoning.

[LoopFormer](https://arxiv.org/abs/2602.11451) takes another route: condition
each loop on time and step size and use shortcut-consistency training so one
checkpoint remains useful across multiple budgets. This is a plausible repair
if T1's joint controller disrupts or over-specializes the trajectory.

### 6.5 Confidence exits are valid comparators, not substitutes for causality

[CALM](https://arxiv.org/abs/2207.07061) calibrates local early-exit decisions
against global output-quality tolerances. The length-generalization work also
finds maximum-confidence stopping effective on convergent tasks and weaker on
non-converging ones. T1 should therefore record confidence, state-change, and
successive-logit-change baselines. They can reveal whether a learned token is
adding information beyond a free convergence heuristic.

## 7. Forward pathways

### Path A: T1 passes

1. Replicate the passing lineage with at least one additional seed.
2. Measure the accuracy-compute Pareto curve, not just exact-depth accuracy.
3. Run a content-determined synthetic T1b before natural traces. Good task
   families include pointer chasing to an unstated sentinel, iteration until a
   predicate becomes true, and first-cycle detection. Match depth frequencies
   and remove explicit numeric depth cues.
4. Compare the learned token policy with fixed depth, oracle depth, answer
   confidence, successive-logit KL, and state-change exits.
5. Only then open T2 natural traces, with segmented step counts and independently
   verified answers.
6. Revisit stochastic width after WP4. If reopened, keep halting decisions per
   trajectory and preserve K=1 identity.

### Path B: policy fails but forced computation survives

This is the cleanest reason to separate backbone training from allocation:

1. Freeze the healthy T1 or Arm A/E backbone.
2. Train only the token readout/controller on cached per-loop states.
3. Test a training-free convergence exit using successive output KL, answer
   stability, and state delta.
4. Train an elastic backbone with randomized loop depths, then repeat the
   frozen-controller comparison.
5. If loop states stagnate, add time/step conditioning or a shortcut-consistency
   objective before training another controller.

### Path C: joint training damages forced-depth competence

1. Increase mechanism supervision within control-target examples rather than
   merely increasing total steps.
2. Freeze more of the substrate and train control rows/readout first.
3. Use gradient-conflict diagnostics between control and chain losses; apply
   loss projection only if conflict is measured.
4. Try staged training: install the mechanism first, freeze or lower its LR,
   then train the controller.
5. Do not switch optimizers until gradient liveness and objective conflict are
   measured.

### Path D: control predicts correctly but self-halted execution fails

This suggests exposure bias: training sees gold continue/stop histories while
inference follows its own decisions.

1. Add scheduled sampling or DAgger-style roll-ins using the model's own
   decisions while retaining gold corrective labels.
2. Train on one-step early and late perturbations so recovery behavior is
   represented.
3. Distinguish irreversible early stops from harmless extra loops; penalize
   them asymmetrically.
4. Calibrate stop decisions on held-out rows against a registered quality
   tolerance, following the principle used by CALM.

### Path E: the explicit-depth control passes but content-depth fails

The interface is healthy; the remaining problem is learning a stopping
criterion. Candidate signals are:

- a sentinel or verified terminal predicate;
- answer stability over successive loops;
- successive-output KL;
- state-delta norm or fixed-point residual;
- verifier confidence;
- disagreement between independently trained readouts;
- expected value of one more loop, trained from paired loop outcomes.

The last formulation turns halting into a value-of-computation problem:
continue when the expected accuracy gain from another loop exceeds its cost.
It is more general than predicting a gold depth label and naturally supports
latency-aware deployment.

### Path F: width eventually reopens

WP1 says the tested terminal routes never fit their command mapping. If width
reopens, do not repeat additive versus FiLM training on the same frozen terminal
interface. The next causal probe should move conditioning inside the recurrent
block or permit bounded adaptive substrate training. Oracle control must pass
before variational width, coverage, selection, particles, or SVGD return.

## 8. Recommended preregistration amendments

Before T1 is locked:

1. State the exact multi-task loss and class balancing.
2. Define gate 3 as exact selected depth and choose strict or noise-robust form.
3. Use all 1,024 rows for logit-level causal override.
4. Lock AdamW and the parameter-group learning rates.
5. Lock the final checkpoint as primary and intermediate checkpoints as
   diagnostic only.
6. Add descriptive depths 9-14, calibration, loop-error, and compute metrics.
7. Predeclare a replication rule. Budget-conscious recommendation: seed 0 for
   both lineages, followed by seed 1 for any passing or near-threshold lineage.
   Strongest recommendation: two seeds for both from the outset.
8. Keep the four-gate positive sentence unchanged. Add explicit labels for
   `interface_positive`, `policy_negative`, `mechanism_regression`, and
   `implementation_inconclusive` so no failure is overgeneralized.

## 9. Recommended immediate sequence

1. Strategy review of this memo and selection-gate form.
2. Lock the T1 preregistration, including loss weighting, optimizer, integer
   thresholds, override semantics, and replication rule.
3. Implement with red/green unit tests and a tiny one-batch GPU smoke.
4. Run the two seed-0 lineages in parallel only after the smoke passes.
5. Bank each stage independently; do not select checkpoints post hoc.
6. Convene WP4 with T1, WP1, and all descriptive controls before reopening T2
   or width.

The central discipline is to treat adaptive computation as a stack of causal
requirements: a useful recurrent computation, a readable stopping signal, a
correct actuator, and a policy that preserves quality under its own decisions.
T1 is the first experiment in this program that tests all four together.
