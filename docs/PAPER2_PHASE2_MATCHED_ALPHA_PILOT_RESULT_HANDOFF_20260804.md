# Handoff: Matched-Alpha Pilot Result - No Qualifying Arm

Date: 2026-08-04

Audience: strategy and research review

Status: complete terminal pilot receipt; no alpha selected; no rerun authorized

## 0. Executive verdict

The corrected matched-alpha target completed a technically valid training run.
All six registered arms passed the finite-loss preflight, executed optimizer
updates, respected the frozen-lineage and zero-loop identity contracts, and
landed terminal receipts. Across the six arms, 1,769 optimizer steps ran.

The experiment did not select an alpha. Every arm hit a registered stopping
boundary before the adequacy condition could be established:

- `alpha=0.0` trained for 500 and 600 steps, then failed the point-estimate
  upper-model-quality floor at two consecutive evaluations;
- `alpha=0.5` trained for 193 and 184 steps, then crossed the registered
  trust-saturation count;
- `alpha=1.0` trained for 146 steps in both seeds and crossed the same
  trust-saturation count.

No arm improved verified mean accepted length. The result is therefore not
"the three alphas were equivalent" and not "alpha=0.5 won by default." It is a
constrained feasibility negative for the current jointly trained module and
objective: no tested scaling produced an acceptance gain while remaining
inside all locked quality, trust, and adequacy conditions.

E1 remains blocked. The conditional `alpha=0.25` and `alpha=0.75` refinement
arms are not triggered because their trigger requires a valid qualifying-arm
comparison. Any further training requires an amendment reviewed before the
next optimizer step.

## 1. Question and registered design

The pilot asked which shared-basis scaling, `alpha in {0.0, 0.5, 1.0}`, gives
the best verified acceptance after training the complete student module under
a matched DEV-only budget, without violating identity, frozen-lineage, tube,
preservation, or upper-model-quality constraints.

The sole alpha-dependent operation was:

```text
lambda_eff ** (-alpha / 2)
```

in one frozen PCA basis. Canonical mean, learned layer mixture, RRR projector,
retained rank, eigenvalues, architecture, initialization stream, batches,
evaluation rows, optimizer, and stopping rules were matched. The registered
matrix was three alphas by two seeds.

Key constants:

- canonicalizer: `learned_mixture_rrr`, seed `20260814`;
- optimizer: AdamW, learning rate `3e-4`, weight decay `0.01`;
- batch size `128`, nominal budget `1,000` steps;
- warmup `100` steps, evaluation every `100` steps;
- fp32 recurrent-gradient paths;
- trust ceiling `0.5` on the amended canonical update ratio;
- trust stop: penalty active on more than `50` of any rolling `100`
  post-warmup steps;
- quality floor: baseline-correct retention at least `0.997` and Wilson 95%
  lower bound at least `0.990`;
- quality stop: failure at two consecutive evaluations;
- adequacy: at least four of six runs improve flow loss by at least 20% from
  step 100 and open their gates beyond initialization.

The one-time extension to 2,000 steps applies to completed but inadequate
arms. It does not override registered trust or quality aborts.

## 2. Provenance and engineering validity

| Item | Receipt |
|---|---|
| Protocol lock | `cf6747264e48e2de657eb2a1646f1e7c4f152ea5` |
| Identifying launcher | `72082031b50e054a2677e281e7eb96cad8113432` |
| Published result | `a63ca9201e4710bf76feaa853294fa2a2325b8b9` |
| Result directory | `outputs/stage5/stage5_paper2_phase2_matched_alpha_20260804/` |
| Decision | `no_selection` |
| Run status | `blocked_with_receipts` |

The identifying launcher repaired the earlier sparse-support defect by
reconstructing finite residual seeds for valid sparse candidates while keeping
the authoritative zero-loop distribution unchanged. This run, unlike the
superseded attempts, reached real optimizer updates:

- all six loss preflights were finite;
- every preflight reported reconstructed support of `343` candidates;
- zero-loop hidden states and logits were bit-exact;
- frozen LM-head parameters had `requires_grad=False` and received no gradient;
- no arm stopped for a non-finite loss, frozen mutation, or assertion failure;
- all source, data, constants, and canonicalizer hashes were recorded.

The stop reasons below are therefore experimental outcomes under the locked
protocol, not repetitions of the earlier engineering failures.

## 3. Primary results

### 3.1 Arm-level terminal accounting

Quality and acceptance for trust-aborted arms are the last scheduled
evaluation at step 100, not an evaluation at the exact abort step. This
distinction is material and is carried throughout the handoff.

| Alpha | Seed | Optimizer steps | Stop | Last evaluated step | Retention | Wilson lower | Acceptance delta |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0.0 | 0 | 500 | quality, two consecutive | 500 | 0.996569 | 0.995738 | -0.000530 |
| 0.0 | 1 | 600 | quality, two consecutive | 600 | 0.996738 | 0.995925 | -0.003086 |
| 0.5 | 0 | 193 | trust saturation | 100 | 0.998390 | 0.997791 | -0.000083 |
| 0.5 | 1 | 184 | trust saturation | 100 | 0.998009 | 0.997354 | -0.000094 |
| 1.0 | 0 | 146 | trust saturation | 100 | 0.998348 | 0.997742 | -0.000328 |
| 1.0 | 1 | 146 | trust saturation | 100 | 0.998348 | 0.997742 | -0.000344 |

All six acceptance deltas were negative. Their absolute magnitudes were small,
but the registered currency did not improve in any arm.

### 3.2 Quality boundary for alpha 0

The quality stop was driven by the point floor, not the confidence bound. With
23,606 baseline-correct positions, the `0.997` floor requires at least 23,536
retained positions and permits at most 70 losses.

| Seed | Retained | Losses observed | Losses permitted | Excess losses | Wilson result |
|---:|---:|---:|---:|---:|---|
| 0 | 23,525 | 81 | 70 | 11 | pass |
| 1 | 23,529 | 77 | 70 | 7 | pass |

This is near the boundary but replicated. It is not a small-sample confidence
artifact. Seed 0 failed at steps 400 and 500. Seed 1 fluctuated around the
floor, then failed at steps 500 and 600. The registered two-evaluation rule
worked as intended and prevented a single noisy checkpoint from ending the
arm.

### 3.3 Trust boundary for whitened arms

Each trust-aborted arm stopped on the first step at which the rolling rule was
violated:

| Alpha | Seed | Abort step | Trust-active steps recorded |
|---:|---:|---:|---:|
| 0.5 | 0 | 193 | 51 / 193 total observed |
| 0.5 | 1 | 184 | 51 / 184 total observed |
| 1.0 | 0 | 146 | 51 / 146 total observed |
| 1.0 | 1 | 146 | 51 / 146 total observed |

The decision rule operates on the rolling post-warmup window, so the total-run
fractions above are descriptive only. The exact count of 51 confirms that the
runner stopped at the locked boundary rather than after a delayed failure.

The evaluation-time trust rents were tiny, suggesting boundary-hugging rather
than a grossly explosive state. The receipts save the per-step active/inactive
flags but not the magnitude distribution of the excess ratio. We can say the
constraint was persistently active. We cannot yet say how far above the
ceiling those training batches were.

### 3.4 Flow and acceptance did not move together

| Alpha | Seed | Flow at step 0 | Flow at step 100 | Last flow | Reading |
|---:|---:|---:|---:|---:|---|
| 0.0 | 0 | 0.094675 | 0.087648 | 0.070541 | 19.5% improvement from step 100 |
| 0.0 | 1 | 0.094673 | 0.087557 | 0.068674 | 21.6% improvement from step 100 |
| 0.5 | 0 | 0.115993 | 0.094059 | 0.094059 | 18.9% improvement by step 100 |
| 0.5 | 1 | 0.115938 | 0.093187 | 0.093187 | 19.6% improvement by step 100 |
| 1.0 | 0 | 0.305692 | 0.336871 | 0.336871 | 10.2% worse by step 100 |
| 1.0 | 1 | 0.305427 | 0.337313 | 0.337313 | 10.4% worse by step 100 |

Only alpha 0 seed 1 met the numerical 20% flow-improvement threshold before
its quality stop. Alpha 0 seed 0 missed by about half a percentage point.
Partial whitening approached a 20% reduction from initialization but stopped
before a post-step-100 adequacy interval existed. Full whitening made the flow
objective worse immediately.

None of these latent improvements became an acceptance gain. Alpha 0 seed 1
is the sharpest illustration: flow loss and probe KL continued to improve as
the mean draft gate opened from about 0.029 at initialization to 0.507 at step
600, while accepted length fell and quality crossed its floor. This repeats
the earlier program-level warning that latent or probe improvement is not a
substitute for verified decision improvement.

### 3.5 Optimization pressure was high

| Alpha | Bridge clip fraction | Head clip fraction | Refiner clip fraction |
|---:|---:|---:|---:|
| 0.0 | 0.000 | 1.000 | 1.000 |
| 0.5 | 0.000 | 1.000 | 1.000 |
| 1.0 | 0.000 | 0.308-0.315 | 0.966-0.993 |

The head and refiner were clipped on essentially every step for alpha 0 and
0.5, and the refiner was almost always clipped for alpha 1. The gradient atlas
shows the functional-probe KL gradient dominating several other loss
gradients by orders of magnitude. At step zero for alpha 0 seed 0, for example,
the head/refiner norms from functional-probe KL were approximately 446/729,
versus 0.71/1.05 for flow and 0.19/0.28 for final CE.

This does not invalidate the locked run. It does show that the nominal loss
weights did not translate into balanced effective optimization. Because
clipping was nearly continuous, the realized update direction was frequently
the clipped composite direction. Loss balance, gating dynamics, and trust
pressure are therefore part of the result, not incidental telemetry.

## 4. What the result supports

1. The corrected complete module is trainable in the narrow engineering sense:
   its losses are finite, gradients propagate, and parameters update.
2. Alpha materially changes the binding constraint. No whitening avoids trust
   activation but eventually loses quality; partial and full whitening
   preserve quality at step 100 but saturate trust shortly afterward.
3. Full whitening is disfavored under this recipe. It starts with much larger
   flow error, worsens that error by step 100, and reaches the trust stop first.
4. Partial whitening remains the most plausible whitening prior, but it is not
   selected. Its two arms preserved quality at step 100 and improved flow from
   initialization, yet did not improve acceptance and did not remain inside
   the trust rule.
5. Canonical or probe convergence is not sufficient. The system can improve
   flow loss, probe KL, and probe top-1 while verified accepted length remains
   flat or declines.
6. The current joint objective and dynamics do not expose a qualifying alpha
   under the registered constraints.

## 5. What the result does not support

Do not claim that:

- the alphas are equivalent;
- alpha 0.5 won by the protocol's equivalence default;
- whitening is generally harmful;
- the architecture is incapable of improving acceptance;
- the trust ceiling itself is wrong;
- the quality floor should be relaxed because alpha 0 was close;
- a 2,000-step continuation is authorized;
- the step-100 quality of trust-aborted arms describes their abort checkpoints;
- cached teacher-forced accepted length is serving throughput;
- this DEV pilot is E1 confirmation.

The numerical equivalence intervals in `decision.json` are non-decisive
because every pooled arm is marked invalid and the arms stopped at different
steps. They are retained as receipt arithmetic, not as an alpha comparison.

## 6. Impact on the Phase-2 plan

### Immediate status changes

1. **Alpha selection remains open.** No production canonical scaling can be
   locked from this run.
2. **E1 remains blocked.** The prerequisite matched-pilot selection did not
   produce a qualifying arm.
3. **No grid refinement opens.** Neither alpha 0.25 nor 0.75 is authorized.
4. **No automatic extension opens.** All arms were safety-aborted rather than
   completed-but-inadequate.
5. **The next GPU job should not be E1.** Any new training run must follow a
   written amendment naming the mechanism it changes and the evidence that
   motivates it.

### Strategic reframing

The next decision is no longer "which alpha is best?" The evidence moves the
question one level down:

> Can the complete module produce verified acceptance improvement before its
> gate dynamics consume the quality margin or its canonical updates persist at
> the trust boundary?

Alpha is one contributor, but the receipts point to a three-way interaction:

1. canonical geometry controls flow scale and trust pressure;
2. the composite loss produces heavily clipped, imbalanced gradients;
3. the arbitration gates open without producing a global acceptance gain.

A denser alpha sweep would not isolate any of these and is therefore low value
at this point.

## 7. Recommended closure before an amendment

These jobs are read-only and do not require an A100 training session.

### 7.1 Exact-abort checkpoint evaluation

Evaluate the saved alpha 0.5 and 1.0 checkpoints at steps 184, 193, and 146 on
the same fixed DEV slice. Report quality, acceptance, flow, probe metrics,
gate state, and trust-ratio distribution. This closes the current receipt gap
between the last step-100 evaluation and the actual trust stop.

### 7.2 Row-level trade-off audit

Using the saved scheduled rows, compare:

- alpha 0's last quality-passing checkpoint with its abort checkpoint;
- partial whitening at step 100 with alpha 0 at matched step 100;
- rows whose accepted length improves, rows whose final token changes, and
  rows whose quality is lost;
- workload, position, teacher-disagreement, and oracle-help strata;
- gate magnitude and opening against each outcome class.

This determines whether the harm is diffuse or concentrated in a predictable
arbitration subset.

### 7.3 Gradient-atlas synthesis

Aggregate the existing atlas by loss and module rather than running new
training. Quantify effective gradient dominance, clipping, and the relevant
conflict cosines. The key question is whether acceptance-aligned losses are
being overwhelmed before clipping or whether their directions directly
conflict with flow/probe fitting.

## 8. Amendment options for strategy review

The following are alternatives, not cumulative recommendations. None is
authorized by this handoff.

### Option A: optimization-pressure amendment

Retain alpha 0.5 and the trust/quality rules, but change how gradients reach the
module. Candidate mechanisms include loss-gradient normalization, a staged
schedule that fits the canonical flow before opening arbitration paths, or a
lower learning rate. The amendment must state which diagnosis it addresses and
must not tune against the current DEV outcomes without a fresh development
split or explicit exploratory classification.

This is the leading option if the atlas confirms that one auxiliary loss and
continuous clipping dominate the effective update.

### Option B: trust-constrained update amendment

Retain alpha 0.5 but replace repeated penalty-boundary contact with a
construction that projects or parameterizes the canonical update inside the
trust region. The trust ceiling need not be relaxed. This tests whether the
problem is penalty enforcement rather than required update magnitude.

This is the leading option if exact-abort evaluation shows useful behavior and
only marginal trust excess with preserved quality.

### Option C: arbitration amendment

Keep the latent flow but delay, freeze, or separately calibrate the draft and
bridge gates. Require the gate to demonstrate row-level acceptance benefit
before it can open materially. This addresses the alpha 0 trajectory in which
the draft gate opens while acceptance and quality worsen.

This is the leading option if the row-level audit finds a separable safe-use
subset or systematic gate misclassification.

### Option D: bank the boundary and simplify

If the read-only audits show diffuse harm, no useful behavior at the exact
trust stops, and irreducible gradient conflict, bank this as a constrained
negative for the complete joint module. Return to a simpler path or close this
version of the Phase-2 architecture rather than spending on a multi-variable
rescue sweep.

## 9. Questions for the strategy agent

1. Do the three read-only closure analyses run before any amendment? Coding
   recommendation: yes, with exact-abort evaluation first.
2. Does the near-boundary alpha 0 quality failure justify another attempt?
   Coding recommendation: not by itself. Both seeds crossed the locked point
   floor and neither improved acceptance.
3. Should partial whitening remain the provisional design prior despite no
   selection? Coding recommendation: yes as a prior only, because it preserved
   quality at step 100 and improved flow, but label it unselected.
4. Is persistent but possibly marginal trust contact better addressed by a
   constrained parameterization than by relaxing the ceiling? Coding
   recommendation: decide after measuring the exact-abort checkpoints and
   trust-ratio magnitudes.
5. Does the loss-gradient imbalance justify staged training or gradient
   normalization before changing architecture? Coding recommendation: likely,
   subject to the atlas synthesis.
6. If a new training attempt is approved, what makes it scientifically
   identifying? It should change one diagnosed mechanism, retain the locked
   safety outcomes, use a fresh DEV partition where possible, and pre-state
   the reading for both success and failure.

## 10. Recommended next sequence

1. Bank this result as `no_selection`, not alpha equivalence.
2. Run exact-abort checkpoint evaluation on the four trust-aborted arms.
3. Run the row-level trade-off audit and gradient-atlas synthesis from existing
   artifacts.
4. Return those receipts to strategy.
5. Select at most one amendment mechanism: optimization pressure, constrained
   trust enforcement, or arbitration calibration.
6. Write and lock the amended DEV protocol before training.
7. Do not open E1 until a qualifying alpha/configuration exists under the
   amended protocol.

## 11. Plain-language summary

The experiment finally trained the intended machine, but it exposed a real
trade-off rather than choosing a whitening strength. With no whitening, the
model learned the latent target most steadily but slowly damaged too many
previously correct outputs. With partial or full whitening, the outputs still
looked healthy at the first evaluation, but the internal updates spent too
many training steps pressed against the allowed trust boundary. Full whitening
was the weakest of the three because its latent loss worsened almost
immediately.

Most importantly, none of the six runs made speculative acceptance better.
That means the next problem is not finding a more precise alpha. It is learning
why the training signal improves internal proxies without improving the actual
decision metric, and why the gates open without safely using that internal
progress. The existing checkpoints and row-level receipts can answer much of
that without another full training run. E1 should wait for that diagnosis and
one targeted amendment.

## 12. Strategy r3 amendment of record

Strategy revision r3 is the governing interpretation for the closure audit
(Drive `1ByHrPod0MVjVzgrMQcBZub_PM3Mgp-gL`, 15,760 bytes, reported SHA-256
`b4d8adfd...091012`; the full digest should be copied from Drive before any
future preregistration cites it as an exact lock).

It introduces a standing distinction between **tripwires** and **shapers**.
Tripwires remain hard stops for genuine catastrophe, including non-finite
loss, frozen-lineage mutation, and identity failure. Shapers, including trust
rent and gradient clipping when their constants have not been empirically
grounded, run in observe-and-log mode with only a generous catastrophe stop.
The old `0.997` point-retention and `0.990` Wilson rules remain endpoint
qualification criteria. They are not retrospectively relabeled as catastrophe
tripwires. A future quality-collapse tripwire requires its own empirically
grounded threshold.

This amendment ratifies the pilot as a constrained-feasibility negative,
authorizes the read-only exact-checkpoint, row-tradeoff, gradient-attribution,
and loss-scale audits, and leaves alpha unselected and E1 blocked. It does not
authorize a re-pilot. Any re-pilot must first lock one identifying amendment
using the audit results and run ungrounded shapers in observation mode.

## 13. Canonical artifacts

- `docs/PAPER2_PHASE2_MATCHED_ALPHA_PILOT_PROTOCOL_DRAFT_20260804.md`
- `training/paper2_phase2_matched_alpha_preregistration.json`
- `outputs/stage5/stage5_paper2_phase2_matched_alpha_20260804/summary.json`
- `outputs/stage5/stage5_paper2_phase2_matched_alpha_20260804/decision.json`
- `outputs/stage5/stage5_paper2_phase2_matched_alpha_20260804/alpha_0p0_seed_0.json`
- `outputs/stage5/stage5_paper2_phase2_matched_alpha_20260804/alpha_0p0_seed_1.json`
- `outputs/stage5/stage5_paper2_phase2_matched_alpha_20260804/alpha_0p5_seed_0.json`
- `outputs/stage5/stage5_paper2_phase2_matched_alpha_20260804/alpha_0p5_seed_1.json`
- `outputs/stage5/stage5_paper2_phase2_matched_alpha_20260804/alpha_1p0_seed_0.json`
- `outputs/stage5/stage5_paper2_phase2_matched_alpha_20260804/alpha_1p0_seed_1.json`
