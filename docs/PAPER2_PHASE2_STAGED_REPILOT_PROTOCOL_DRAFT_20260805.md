# Phase-2 Staged Optimization Re-Pilot Protocol

Date: 2026-08-05  
Status: `locked_before_training`

This protocol implements the strategy response at Drive
`1IiaXH7n1Pi-YYIHaQUjs1L-VOFZvaQb3`, 8,872 bytes, SHA-256
`ce2bfa5772891f425fa662a778d58a63930a7642e2cf591868ccb06bbe079e6b`.
It supersedes the active recipe of the matched-alpha pilot for this staged DEV
re-pilot without altering the historical pilot registration or verdict.

## 1. Question

Can the alpha-0.5 complete student first construct a functionally useful state
and then use that frozen state to improve verified accepted length when loss
pressure is mechanically calibrated and previously ungrounded shapers are
reduced to telemetry plus catastrophe tripwires?

This is a repair test of one optimization recipe, not an alpha comparison.

## 2. Arms, lineage, and sequence

- Alpha: `0.5` only, explicitly unselected.
- Seeds: `0` and `1`.
- Batch size: `128`.
- A1 nominal budget: `1,000` optimizer steps per seed.
- A2 nominal budget: `1,000` optimizer steps per seed.
- One extension per stage: nominal budget may double once to `2,000`.
- Calibration: 100 gradient-sampling batches before each stage, uncounted and
  with no optimizer updates. This interpretation is required because the
  calibration outputs define the weights used by the first optimizer step.
- A2 cannot launch until A1 receipts have been reviewed under the locked A1
  gate. The A1 launcher contains no path that can enter A2.
- Same frozen DEV-C source, document-isolated training/evaluation split,
  learned-mixture RRR canonicalizer, architecture, initialization contract,
  source hashes, optimizer family, and learning rate as the matched-alpha
  pilot.
- Frozen E1 partitions remain untouched.
- V1d constants are banked: `c=0.15` and p99 state-RMS cap
  `0.5508932316303252`.

## 3. Calibration contract

Each stage samples 100 deterministic batches from its training partition.
Batches 1-49 are settle-in measurements and are retained as telemetry. Batches
50-100 determine constants. Model parameters and optimizer state remain
unchanged throughout calibration.

For each active loss, measure its gradient on the refiner path. Gradient dot
products and squared norms are accumulated in fp64. Let `g_i` be the arithmetic
mean per-loss refiner-gradient norm over batches 50-100. A non-finite value or
`g_i < 1e-12` is a blocked calibration, not an invitation to invent a weight.

Weights are static within a stage:

```text
w_i = (s_i / s_flow) * (g_flow / g_i), with w_flow = 1.0
```

For A2, `final_ce` is the algebraic anchor at weight 1.0 and replaces `flow`
in the formula. The name `s_flow` in the strategy expression is therefore read
as the stage anchor share.

Target shares:

- A1: flow `0.60`, functional probe KL `0.20`, preserve KL `0.20`.
- A2: final CE `0.35`, cumulative KL `0.35`, local CE `0.10`, preserve KL
  `0.20`.

The A1 execution gates remain closed. Its preserve KL is a training-only
counterfactual read through the frozen initialized bridge: the bridge is
differentiable with respect to the state but none of its parameters update and
its output is never used as the executed A1 prediction. This is the only way
to make the registered nonzero A1 preserve share compatible with closed
execution gates; the receipt reports it separately as
`counterfactual_preserve_kl`.

Huber delta is the exact p75 of absolute per-coordinate target increments over
the A1 calibration batches. It is computed from the training partition only
and frozen for A1. A2 inherits the fitted state and does not use flow Huber as
an active loss.

Per-module clip ceilings equal 10 times the p99 norm of the calibrated weighted
total gradient. The p99 is reconstructed from the stored per-loss gradient
Gram matrices after static weights are known. Clip activation is telemetry and
an alarm above 1% of optimizer steps; it is not an automatic stop or dynamic
loss change.

Realized weighted-gradient shares are measured every 100 steps. A share outside
`[target/2, 2*target]` is an alarm only. At optimizer step 200, every active
loss must be within 10 percentage points of its target share; otherwise the
run stops as `static_weight_contract_miss` and is classified as a protocol
implementation failure.

## 4. Stage A1: state construction

- Trainable parameters: `module.flow` only.
- Frozen parameters: initializer, bridge, control state, draft head, student
  embedding, teacher embedding, and all cached targets.
- Executed bridge and draft gates are forced closed.
- Active losses: flow, functional probe KL, counterfactual preserve KL.
- Final CE, cumulative KL, local CE, and trust penalty are inactive.
- Trust penalty weight is exactly zero.
- Logged every 100 steps: validation flow MSE, registered flow loss,
  functional-probe KL, accepted length under the gate-closed execution path,
  both trust-ratio definitions, static and realized loss shares, clip activity,
  endpoint error, probe quality, and flow slope.

A1 state-construction gate:

1. Mean functional-probe KL on the fixed DEV slice improves by at least `0.10`
   nats versus step zero.
2. Flow validation MSE is lower than at step zero.

At step 1,000, A1 extends to 2,000 if either the gate is not yet met or flow
validation loss improved by more than 0.5% relative from step 900 to step
1,000. At step 2,000, a gate miss is `a1_negative`; a gate pass with positive
terminal slope is reported as `a1_pass_budget_limited`, but no second extension
occurs. A gate pass otherwise is `a1_pass`.

## 5. Stage A2: state use

A2 is locked here but implemented and launched only after strategy banks A1.

- The A1 flow is frozen exactly and its hash is asserted before and after A2.
  It may retain gradients for refiner-path share telemetry but is excluded from
  every optimizer group.
- Trainable parameters: bridge, control state, and draft head. The initializer
  remains frozen.
- Active losses: final CE, cumulative KL, local CE, preserve KL.
- Local CE is horizon-wise NLL of cumulative draft logits against the cached
  teacher top-1 candidate, averaged over valid horizons.
- Gates and acceptance-facing heads train. Flow and functional-probe losses are
  telemetry only.
- A flow learning-rate multiplier of 0.1 is a future preregistered fallback,
  not part of the primary A2 run.
- One draft-head-only zero-loop control runs per seed with no flow or writeback.
  Its recipe, batches, budget, and acceptance scorer match A2.

A2 extends from 1,000 to 2,000 if oracle headroom is below its adequacy gate or
mean accepted length improved by more than `0.002` tokens from step 900 to step
1,000. No second extension occurs.

A2 state-use gates:

1. The identical hindsight oracle procedure on the final A2 rows reaches at
   least `+2%` relative to the common zero-loop mean accepted length.
2. The full system's always-on or quality-safe-selected accepted-length delta
   exceeds the matched draft-head-only control.
3. Endpoint quality remains qualified under the preservation criterion.

If A1 passes but final A2 oracle headroom remains below 2%, the bounded sidecar
feasibility premise fails at this substrate and scale. The decision returns to
strategy with the drafter-only j-axis, D1, and E4 alternatives.

## 6. Tripwires and shapers

Hard tripwires:

- non-finite loss, state, or gradient telemetry;
- frozen-lineage, source-hash, or stage-frozen-parameter mutation;
- zero-loop identity failure;
- amended endpoint-referenced trust ratio above 5 on more than 50 of any
  rolling 100 post-warmup optimizer steps;
- endpoint quality below both-tier non-inferiority criterion on two consecutive
  100-step evaluations.

The optimizer warmup is 100 steps. Trust is logged from step zero and its
rolling stop begins after optimizer step 100. A single quality miss is a
receipt warning. Quality uses point retention `>=0.997` and Wilson 95% lower
bound `>=0.990`.

Observation-mode shapers:

- trust loss weight `0`;
- endpoint-referenced and state-referenced ratios logged every step;
- state-referenced ratio is telemetry only;
- calibration-derived clip ceilings, with activation-rate alarms but no
  dynamic threshold changes.

## 7. Verdicts and boundaries

Every stage writes complete receipts before returning `complete`, `negative`,
`budget_limited`, `blocked`, or `protocol_bug`. No threshold or constant may
change after this lock.

Do not claim:

- alpha 0.5 was selected;
- oracle headroom is achievable by a learned router;
- observation-mode trust establishes a deployment-safe threshold;
- DEV accepted length is serving throughput;
- the historical alpha matrix compared geometry alone;
- an A1 pass demonstrates useful state use.

## 8. Canonical strategy and receipt resources

- Filled lock fields: Drive `1IiaXH7n1Pi-YYIHaQUjs1L-VOFZvaQb3`.
- Headroom handoff: Drive `1qsxfxR-HJH3ppRvHz2SFOXsD2BzoUNHA`.
- V1d handoff: Drive `1zH20VEuuc4myQl9pvFgv56iQ4tNXa4iQ`.
- Historical pilot result handoff: Drive
  `1bfGT1ufxIRE0Ol9O32vGtdFH1az4iCxs`.
- Historical audit handoff: Drive `1VXk1NlYHiublmYCTioXrqQMrRaZJ2Fvs`.

The git commit containing this protocol and its machine-readable registration
is the staged re-pilot lock. The training launcher must descend from it.
