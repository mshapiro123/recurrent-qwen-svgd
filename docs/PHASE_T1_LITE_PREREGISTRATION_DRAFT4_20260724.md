# Phase T1-Lite Preregistration - Internal Control-Token Halting

**Draft 4, 2026-07-24. Status: `locked_before_training`.** Mark ratified the
five decisions in the P0 strategy handoff on 2026-07-24. This document and the
matching machine-readable `preregistration.json` are binding from their lock
commit. No gate, constant, curriculum stage, seed policy, or evaluation set may
change after this lock. The uncitable P0 pilot is complete and informs only the
loss constants recorded below.

This document governs over the earlier T1 design memo and T0 spec wherever
they differ.

## 1. Registered Question And Scope

T1 asks whether an explicit internal continue/stop token decision at each
recurrent transition can read a stated required depth, select that depth
accurately, preserve the installed recurrent computation, and causally change
loop execution.

Depth is stated in the prompt. T1 is an information-path and actuator test,
not a difficulty-inference test. No T1 result supports a claim about inferred
difficulty, content-determined depth, or natural-language halting.

## 2. Substrate And T1-Lite Lineage

The single registered lineage starts fresh from `Qwen/Qwen2.5-0.5B-Instruct`, base SHA
`960f8bf265ba2850c9cdd60a388a00f8f366464babe0507521f010cb7f34971f`,
with Prelude 0-6, weight-tied Recurrent Block 6-18, Coda 18-24, the
identity-preserving one-loop path, and repaired split re-entry bridge.

The recurrent block, repaired split bridge, and three new control-token rows
train. The earlier R16-plus-bridge registered lineage is descoped. T1-lite is
an actuator qualification for the D0 program, whose substrate trains the full
recurrent block. This forfeits the proposed T1 capacity comparison.

The T0 contracts are prerequisites: exactly three reserved symbols,
`<|recur_continue|>`, `<|recur_stop|>`, and `<|recur_readout|>`; tied
embedding policy preserved; controls masked from visible generation; control
read at the private per-loop readout position; one-loop identity below `1e-3`;
and exact loop accounting.

## 3. Task, Curriculum, And Training

- Train depths 1-8 on the controlled synthetic transition family.
- Target `continue` before the required depth and `stop` exactly at that depth.
- Use 30% unchanged mechanism rehearsal.
- Train 10,500 steps: 500 at depth 1, 2,000 at support 1-2, 4,000 at support
  1-4, and two 2,000-step stages at support 1-8.
- Use AdamW, batch size 1, gradient accumulation 1, weight decay 0, gradient
  cap 0.5, bfloat16 recurrent-block weights, and float32 bridge and control rows.
- Use learning rate `2e-5` in the primitive stage and `1e-5` thereafter. The
  bridge Prelude LR multiplier is 1 in the primitive stage and 10 thereafter.
- Primary training seed is 0. Section 10 governs seed 1.
- Maintain EMA 0.999. Evaluate raw and EMA, with final-step EMA primary.

Recipe receipt and configuration:
`outputs/stage5/stage5_support8_dose_arm_20260706_153028/summary.json` and
`outputs/stage5/stage5_support8_dose_arm_20260706_153028/chain_continuation_train_config.yaml`.
The chain-stage learning rate `1e-5`, AdamW settings, gradient cap, and Prelude
multiplier are copied from that mechanism-installation receipt. The primitive
`2e-5` rate is the registered first-stage rate from the same staged
installation lineage.

The earlier proposal for a standalone 1,500-step full-block confirmation cell
is withdrawn. It would end inside the support-1-2 stage and would therefore be
structurally mismatched to the question it was intended to answer.

### Stage-boundary liveness readouts

At steps 500, 2,500, 6,500, and 8,500, log the control-loss trajectory over
the completed stage and measure stop and continue recall on the P0 pilot slice
restricted to depths trained so far: 1, 1-2, 1-4, and 1-8. These readouts are
descriptive and cannot change constants, curriculum, gates, or the final-step
primary analysis.

For an executable definition, control loss is flat when the ordinary
least-squares slope across all logged points in the completed stage is at
least `-1e-5` loss units per step. Abort for diagnosis only when that condition
and stop recall exactly zero on the trained-depth pilot rows both hold. The
aborted run writes its receipts and does not consume the registered attempt.
There is no other boundary abort rule.

## 4. Loss And Pilot P0

Control examples retain mechanism and answer supervision. Total loss is
mechanism/answer CE plus lambda times class-balanced control CE.

For uniform depths 1-8 there are 28 continue labels and 8 stop labels. The
default inverse-frequency ratio is therefore 3.5 stop to 1 continue. Weights
are normalized to mean one over realized control labels.

### Pilot P0 (complete before lock, never citable)

- Adapter lineage only, seed 9999, 1,500 steps per cell.
- Ten cells: lambda in `{0.5, 1, 2}` crossed with stop-to-continue ratio in
  `{1, 3.5, 7}`, plus a lambda-zero mechanism/answer reference.
- Dedicated 256-row slice, 32 per depth, excluded from every registered set.
- Readouts at steps 500, 1,000, and 1,500: control loss, stop recall, continue
  recall, answer accuracy versus lambda zero, and gradient norms.

Retain cells with both recalls at least 0.60 at step 1,500, then choose the
smallest answer-accuracy drop versus lambda zero over all nine non-reference
cells. Break ties toward lambda 1 and then ratio 3.5. If no eligible cell
qualifies, reassess openly before lock. Do not extend the grid silently.

Under the Draft 3 pivot, P0 is a loss-feasibility and hyperparameter-calibration
pilot only. Its adapter lineage is not matched to registered full-block
T1-lite. A selected lambda and ratio may be transferred only by locking that
choice before T1-lite; the full-block run must independently clear all four
gates. P0 cannot support a T1-lite efficacy or capacity claim.

The fixed selection rule chose `lambda0p5_ratio1`. Registered T1-lite therefore
uses control-loss lambda `0.5` and equal normalized class weights, continue
`1.0` and stop `1.0`. At P0 step 1,500 the selected cell had stop recall
`177/256 = 0.6914`, continue recall `885/896 = 0.9877`, exact selected-depth
accuracy `166/256 = 0.6484`, and answer accuracy `151/256 = 0.5898`, compared
with `136/256 = 0.5312` for lambda zero. The depth-structured exact-selection
result is a risk signal. It does not change Gate 3 and does not authorize a P0
extension.

## 5. Frozen Evaluation Sets

- Gated: 1,024 rows, 128 at each depth 1-8. Forced and self-halted runs are paired.
- Extrapolation: 128 rows at each depth 9-14. Descriptive only.
- Calibration: 512 rows, 64 at each depth 1-8, disjoint from gated rows. Used
  only for thresholds of the descriptive baselines in Section 8.

The gated and extrapolation rows are the canonical Phase A frozen rows in
`stage5_synthetic_depth_frozen_eval_v2_depth14`. Their locked row-ID hashes are
`7aa673d0...1fdcbe` for depths 1-8 and `74c56235...14b48` for depths 9-14.
The prior draft's `14482ca4...` gated hash was a stale placeholder and did not
match the canonical rows; it was corrected before this lock. The calibration
set is generated independently with seed `2026072401`, prefixed IDs, and
row-ID hash `ebc17c10...6d6b2`. Full row hashes and paths are in the locked
machine-readable preregistration.

Self-halting uses `max_loops=12` on gated rows and 16 on extrapolation rows. A
row that never stops is a selection failure and its answer is scored at the
last executed loop. Exhaustion is reported by depth.

## 6. Four Gates

T1-lite passes only if every gate passes at the final-step checkpoint.

### Gate 1 - Substrate Preservation

Forced-depth answer accuracy on the gated rows must be within 3 points of the
matched non-halting reference.

- Full block: reference 1005/1024; floor 975/1024. Receipt:
  `outputs/stage5/stage5_phase_a_surpass_receipt_20260714/summary.json`,
  checkpoint SHA `dc00f7b694ce32427eb13b0b85d365bc15e0c0317130bd22d4bbc3568544f71b`.
### Gate 2 - Allocation Does Not Cost Competence

Self-halted answer accuracy must be within 3 points of paired forced-depth
accuracy on the same 1,024 rows.

### Gate 3 - Exact Depth Selection

The selected stop loop must equal the stated required depth. Both conditions
are required:

- at least 115/128 at every depth 1-8;
- at least 922/1024 pooled.

Transition micro-accuracy is never a gate or headline metric because the
always-continue policy already gets 28/36 transitions, or 77.78%, correct.
Balanced accuracy, stop recall, continue recall, and macro F1 are descriptive.

The count gate is intentionally strict. Approximate pass probabilities are
1.6% at true row accuracy 0.90, 59% at 0.93, and 96% at 0.95.

### Gate 4 - Full Causal Override Sweep

Intervene on control logits, never on `max_loops` or the loop counter.

- Force stop at every loop `k` from 1 through required depth `d`; execution
  must terminate exactly at `k`. Total: 4,608 executions.
- Force continue at `d`; execution must reach `d+1`. Total: 1,024 executions.

All 5,632 interventions must agree exactly. A miss is an implementation or
actuator finding, not a scientific negative. Fix and rerun Gate 4 only.

## 7. Descriptive Analyses

- Depths 9-14 selection and answer accuracy.
- Overshoot/undershoot confusion by depth.
- Self-halt loop-count distributions and exhaustion.

None is gated.

## 8. Descriptive Baselines

Fit thresholds only on the calibration set and evaluate on the gated set:

1. Fixed depth `K`, for K 1-8.
2. Answer-logit margin exit.
3. Successive-loop output KL exit.
4. Hidden-state update-norm exit.

No superiority claim against these baselines is preregistered.

## 9. Expected Readings

- Full pass: explicit token control is accurate, competence-preserving, and causal on the tested full-block substrate.
- Gate 1 passes and Gate 2 fails: healthy substrate, weak allocation.
- Gates 1 and 2 pass and Gate 3 fails: robustness masks imprecise routing.
- Gate 1 fails: joint training damaged the mechanism; next question is staged
  or frozen-substrate controller training, not a rerun.
- Gate 4 fails: implementation finding; apply its repair rule.
- A miss with healthy forced computation: this full-block joint-controller recipe fails.

## 10. Replication

Seed 0 runs first. Seed 1 runs for a passing or near-threshold T1-lite result.
Near-threshold means Gates 1, 2, and 4 pass and pooled Gate 3 is at least 0.85,
or Gate 1 or Gate 2 misses by no more than 1.5 points while all other gates
pass.

A positive headline requires T1-lite to pass at seed 1. A strong negative
boundary requires the result to be confirmed at seed 1.

## 11. Checkpoint Policy

The final step is primary; EMA is primary and raw is reported. Intermediate
checkpoints are diagnostic only. Do not select a keeper by intermediate peak.

## 12. Stop Policy

The uncertainty-aware small-sample stop policy adopted at Paper One closure
governs any mid-run guardrail event.

## 13. Do Not Claim

- Inferred difficulty, content-determined depth, or natural-language halting.
- More than description at depths 9-14.
- Any cross-budget capacity or protection claim; T1-lite has one lineage.
- Superiority or inferiority to Section 8 baselines.
- Transition micro-accuracy as depth-selection evidence.
- Any intermediate-checkpoint number as a primary result.
- That a Gate 4 failure answers the scientific question.

## 14. Artifacts

At lock: `preregistration.json` with gates, seeds, selected P0 weights, and set
hashes. Per run: training traces, frozen-parameter and gradient-liveness
assertions, evaluation summaries, `gate.json`, starting-point and checkpoint
SHA-256 values, and final commit hashes under `outputs/stage5`.

## 15. Lock Record

The P0 constants, learning rates, curriculum, references, stage-boundary
liveness rule, seeds, gates, and frozen set manifests are complete. The
machine-readable contract is
`outputs/stage5/stage5_paper2_t1_lite_preregistration_20260724/preregistration.json`.
Registered launcher creation is authorized only in a commit after the commit
that locks this document and that JSON file.
