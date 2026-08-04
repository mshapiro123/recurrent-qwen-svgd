# Phase-2 Staged Optimization Re-Pilot Protocol

Date: 2026-08-05  
Status: `draft_not_locked_no_training_authorized`

This draft implements the strategy response at Drive
`1XjiWs4zQy7o_ygipy7oNA53BW-gMtP2u`. It supersedes the active recipe of the
matched-alpha pilot for the next DEV re-pilot but does not alter the historical
pilot registration or its verdict.

## 1. Question

Can the alpha-0.5 complete student learn useful state construction and then
use that state to improve verified accepted length when the two learning
problems are separated, gradient pressure is mechanically calibrated, and
previously ungrounded trust/clipping shapers are reduced to telemetry plus
catastrophe tripwires?

This is a repair test of one optimization recipe, not an alpha comparison.

## 2. Arms and lineage

- Alpha: `0.5` only, explicitly unselected.
- Seeds: `0` and `1`.
- Same frozen DEV-C source, document-isolated training/evaluation split,
  learned-mixture RRR canonicalizer, model architecture, initialization
  contract, and source hashes as the matched-alpha pilot.
- Frozen E1 partitions remain untouched.
- V1d constants are banked design evidence: `c=0.15`, p99 state-RMS cap
  `0.5508932316303252`.

## 3. Calibration

Before A1 weights are frozen, each seed runs 100 calibration steps/batches on
the training partition. Calibration records per-loss refiner-path gradients,
per-module total gradients, target-increment magnitudes, both trust ratios, and
gate state.

Mechanical outputs:

1. Static A1 loss weights satisfy flow at least 50% and functional probe at
   most 25% of measured refiner-path post-clip gradient norm.
2. Huber delta is the p75 of absolute per-coordinate target increments on the
   calibration/training rows, never on the fixed evaluation slice.
3. Per-module catastrophe clips are approximately 10 times the calibration
   p99 total gradient norm. Clip-active fraction is telemetry; above 1% raises
   an alarm and does not silently change the optimizer.

No dynamic per-step normalization is used in the primary run. It is a named
fallback only.

`[LOCK-BLOCKER]` Specify whether calibration is gradient-only or contains
optimizer updates. Coding recommendation: gradient-only, preserving identical
fresh initialization for A1.

`[LOCK-BLOCKER]` Specify the static-weight solver, including target shares for
preserve KL, floors/caps for near-zero calibration norms, and verification of
the achieved shares on a held-out calibration subset.

## 4. Stage A1: state construction

- Flow parameters train.
- Writeback, drafter, and arbitration gates are forced closed.
- Active losses: rebalanced flow, rebalanced functional probe, preserve KL.
- Acceptance-facing final CE, cumulative KL, and local CE are inactive.
- Trust penalty weight is zero.
- Both trust ratios, loss shares, gradient norms, clip activity, endpoint
  error, probe quality, and flow slope are logged.

`[LOCK-BLOCKER]` Specify the A1 step budget and its stage-completion/extension
rule.

## 5. Stage A2: state use

- The A1 flow parameters are frozen exactly.
- Acceptance-facing final CE, cumulative KL, and scheduled local CE activate.
- Gates, draft heads, and bridge/writeback controls train.
- Flow/probe state-construction losses are telemetry only unless the strategy
  response explicitly keeps one as a frozen-flow consistency term.
- Gate features are re-measured; the prior zero-correlation finding does not
  preclude learning under the repaired objective.
- A flow learning-rate multiplier of 0.1 is a preregistered fallback only if
  frozen-flow evidence shows use misalignment. It is not part of the primary
  run.

`[LOCK-BLOCKER]` Specify the A2 step budget and where the one-time extension is
allocated.

## 6. Tripwires and shapers

Hard tripwires:

- non-finite loss or state;
- frozen-lineage or source-hash mutation;
- zero-loop identity failure;
- registered quality-collapse threshold once grounded below.

Observation-mode mechanisms:

- trust loss weight `0`;
- amended and state-referenced ratios logged every step;
- catastrophe trust threshold `r > 5` over a 100-step window;
- calibration-derived high clip ceilings with clip-active telemetry.

`[LOCK-BLOCKER]` Define "sustained" mathematically. Coding recommendation:
strictly more than 50 of the last 100 post-calibration optimizer steps exceed
5, preserving the old window arithmetic while changing only the catastrophe
threshold.

`[LOCK-BLOCKER]` Ground and lock the quality-collapse tripwire separately from
the 0.997/0.990 endpoint qualification rule.

## 7. Verdicts

The re-pilot must separately report:

- A1 state-construction adequacy and achieved gradient shares;
- A2 verified accepted-length change;
- endpoint quality qualification;
- natural trust-ratio and gradient-norm distributions;
- clip-active rate;
- gate-benefit correlations and oracle-selector gap.

`positive`: both seeds qualify on endpoint quality and produce positive
accepted-length change with the registered paired interval/slope rule.

`negative`: the repaired staged mechanism reaches its registered budget with
no positive accepted-length result and no qualifying extension condition.

`budget_limited`: quality is healthy and the preregistered terminal slope is
positive enough to trigger the one allowed extension. The extension outcome
resolves the label.

`blocked`: a genuine tripwire fires or calibration fails its mechanical
contracts. Receipts are written before exit.

`[LOCK-BLOCKER]` Transcribe the numerical slope trigger and one-time extension
budget from the governing strategy revision; they are not stated numerically
in the delivered response.

## 8. Post-re-pilot sequence

- The CPU-only perfect-selector ceiling is reported with baseline receipts.
- A positive repaired recipe reopens the scale-controlled A35 alpha matrix.
- Alpha selection occurs only after that matrix.
- E1 remains blocked on a qualifying repaired recipe, rerun alpha matrix,
  alpha selection, resource note, and V1d delivery acknowledgement.
- V1d itself has run and passed; its existing receipt must be delivered to
  strategy rather than rerun.

## 9. Do-not-claim boundaries

- Alpha 0.5 was selected by the failed pilot.
- Oracle-selector headroom is achievable by a learned router.
- Observation-mode trust establishes a safe production threshold.
- DEV accepted length is serving throughput.
- A calibration-derived constant is confirmatory evidence.

No training launcher may be created until every `[LOCK-BLOCKER]` is resolved,
the machine-readable JSON is committed, and the commit records
`locked_before_training`.
