# Paper Two T1 P0 Calibration - Strategy Handoff

**Date:** 2026-07-24  
**Status:** P0 complete; full-block T1-lite remains unregistered and unrun  
**Decision requested:** ratify the selected loss constants and lock Draft 3

## 1. Executive Result

The authorized ten-cell P0 calibration completed. The fixed selection rule
chose `lambda0p5_ratio1`:

- control-loss lambda: `0.5`;
- stop-to-continue class-weight ratio: `1.0`;
- realized normalized class weights: continue `1.0`, stop `1.0`;
- stop recall: `177/256 = 0.6914`;
- continue recall: `885/896 = 0.9877`;
- exact selected-depth accuracy: `166/256 = 0.6484`;
- answer accuracy: `151/256 = 0.5898`;
- lambda-zero answer reference: `136/256 = 0.5312`;
- answer-accuracy drop versus reference: `-0.0586`, meaning a `+5.86`
  percentage-point improvement rather than a loss.

Both P0 recall floors of `0.60` cleared. Seven of nine controlled cells
qualified. The selected cell had the smallest answer-accuracy drop, exactly as
specified before the grid ran. No cell or checkpoint was selected by looking
at the registered T1 evaluation sets.

This is a loss-feasibility and coefficient-calibration result only. P0 used
the R16 adapter-plus-bridge lineage, while registered T1-lite uses a fresh
full-block lineage. P0 is uncitable and cannot support a learned-halting,
capacity, or cross-budget claim.

## 2. Why P0 Was Run

The continue/stop labels are structurally imbalanced. With depths 1-8 sampled
uniformly, one row at depth `d` contributes `d-1` continue labels and one stop
label. Across one row at each depth this is 28 continue labels and eight stop
labels. In the realized 1,400-row control stream, this corresponds to 4,900
continue and 1,400 stop transitions.

The grid tested whether a usable control gradient existed without sacrificing
the installed transition mechanism, and whether inverse-frequency weighting
was actually necessary. It was not a search over architectures or training
budgets. The grid was fixed at:

- lambda in `{0.5, 1.0, 2.0}`;
- stop-to-continue class-weight ratio in `{1.0, 3.5, 7.0}`;
- one lambda-zero reference;
- seed `9999`;
- 1,500 steps per cell;
- evaluations at steps 500, 1,000, and 1,500;
- a dedicated 256-row pilot set, 32 rows per depth, excluded from every
  registered set.

The training mixture was exactly 70% control-bearing rows and 30% unchanged
mechanism rehearsal, balanced by depth. The A-P output surface matched Phase A
and passed the loop-target/token-position alignment preflight. Pretrained
embedding rows remained hash-identical. The three control rows, R16 adapters,
and repaired bridge were the only trainable components, totaling 6,010,113
forward-active parameters.

## 3. Complete Step-1,500 Grid

| Cell | Lambda | Stop:continue | Stop recall | Continue recall | Exact depth | Answer accuracy | Qualifies |
|---|---:|---:|---:|---:|---:|---:|---|
| `lambda0_reference` | 0.0 | 1.0 | 0.000 | 1.000 | 0.000 | 0.531 | reference |
| `lambda0p5_ratio1` | 0.5 | 1.0 | 0.691 | 0.988 | 0.648 | 0.590 | yes, selected |
| `lambda0p5_ratio3p5` | 0.5 | 3.5 | 0.969 | 0.693 | 0.145 | 0.480 | yes |
| `lambda0p5_ratio7` | 0.5 | 7.0 | 1.000 | 0.121 | 0.125 | 0.465 | no |
| `lambda1_ratio1` | 1.0 | 1.0 | 0.520 | 0.915 | 0.273 | 0.340 | no |
| `lambda1_ratio3p5` | 1.0 | 3.5 | 0.750 | 0.943 | 0.551 | 0.316 | yes |
| `lambda1_ratio7` | 1.0 | 7.0 | 0.848 | 0.922 | 0.574 | 0.312 | yes |
| `lambda2_ratio1` | 2.0 | 1.0 | 0.641 | 1.000 | 0.641 | 0.293 | yes |
| `lambda2_ratio3p5` | 2.0 | 3.5 | 0.801 | 0.956 | 0.648 | 0.266 | yes |
| `lambda2_ratio7` | 2.0 | 7.0 | 0.750 | 0.964 | 0.625 | 0.285 | yes |

The ratio manipulation worked but was nonlinear. Strong stop up-weighting at
lambda `0.5` caused an early-stop collapse: ratio `7.0` reached perfect stop
recall but only `0.121` continue recall. Larger lambda values eventually
produced balanced recalls under several ratios, but with substantially worse
answer accuracy. Equal weighting at lambda `0.5` gave the best joint result.

## 4. Dynamics And The Important Warning

The selected cell was not already solved at step 1,000. Its trajectory was:

| Step | Stop recall | Continue recall | Exact depth | Answer accuracy |
|---:|---:|---:|---:|---:|
| 500 | 0.375 | 0.893 | 0.125 | 0.254 |
| 1,000 | 0.125 | 1.000 | 0.125 | 0.340 |
| 1,500 | 0.691 | 0.988 | 0.648 | 0.590 |

The late change argues against shortening registered training merely because
the transition recalls can look superficially stable. It also shows why
transition recall is not the T1 gate.

Exact selected-depth accuracy remained only `0.648`, and it was highly
nonuniform by depth: `0.000, 0.563, 1.000, 0.969, 1.000, 1.000, 0.000,
0.656` for depths 1-8. Thus P0 demonstrates that the explicit token pathway
can learn both decisions without destroying answer performance under one
calibration. It does not demonstrate precise general depth allocation.

This distinction is central. Registered Gate 3 remains row-level exact depth,
requiring at least `115/128` at every trained depth and `922/1024` pooled.
Transition micro-accuracy, balanced accuracy, and class recall remain
descriptive.

## 5. Recommended Lock

I recommend locking the following without another calibration sweep:

1. `control_loss_lambda = 0.5`.
2. Stop-to-continue class-weight ratio `1.0`, yielding normalized weights
   continue `1.0` and stop `1.0`.
3. The existing 10,500-step full-block curriculum: 500 steps at depth 1,
   2,000 at support 1-2, 4,000 at support 1-4, and two 2,000-step stages at
   support 1-8.
4. AdamW, batch size 1, no weight decay, gradient cap `0.5`, EMA `0.999`,
   primary seed 0, final-step EMA primary.
5. The full-block non-halting reference `1005/1024`, Gate 1 floor `975/1024`,
   checkpoint SHA
   `dc00f7b694ce32427eb13b0b85d365bc15e0c0317130bd22d4bbc3568544f71b`.
6. All four gates unchanged, including the exact-depth and full causal-override
   requirements.
7. The existing seed-1 replication rule for a pass, near-threshold result, or
   strong negative boundary.

The calibration should transfer as a candidate setting, not as matched-lineage
evidence. The full-block experiment must independently clear all four gates.

## 6. Expected Readings After Lock

### Full pass

Forced computation is preserved, self-halting matches forced accuracy, exact
depth selection clears every depth and pooled threshold, and logit-level
overrides control execution exactly. This supports explicit token-pathway
halting on the tested stated-depth synthetic family. A positive headline still
requires the specified seed-1 replication.

### Healthy forced computation, imprecise selection

Gates 1 and possibly 2 pass while Gate 3 fails. The internal decision pathway
is trainable but this joint controller does not allocate depth precisely. The
next mechanistic options are frozen post-hoc control, randomized-depth
backbone training, exposure-bias repair, or convergence-based exits. It is not
license for an unregistered coefficient sweep.

### Substrate damage

Gate 1 fails. Joint control training damaged the installed operation. The next
question is staged or frozen-substrate controller training, not repeating the
same run.

### Causal override mismatch

Gate 4 failure is treated as an actuator or implementation finding. Its
registered remedy is to repair and rerun Gate 4 only; it is not a scientific
negative.

## 7. Decisions Requested From Strategy Review

1. Ratify lambda `0.5` and equal class weights as the transferred P0 constants.
2. Ratify the 10,500-step curriculum and existing seed/replication policy.
3. Confirm that the low, depth-structured P0 exact-selection result is a risk
   signal, not grounds to alter Gate 3 or extend P0.
4. Confirm the full-block reference construction and hash as binding.
5. Authorize conversion of Draft 3 to `locked_before_training` and creation of
   the registered T1-lite launcher only after that lock commit.

My recommendation is yes on all five. The grid did its assigned job. More P0
tuning would spend information from the pilot set to optimize the registered
run and weaken the evidentiary boundary.

## 8. Canonical Artifacts

- Overall P0 receipt:
  `outputs/stage5/stage5_paper2_internal_token_t1_p0_letter_v2_20260724/summary.json`
- Human-readable grid:
  `outputs/stage5/stage5_paper2_internal_token_t1_p0_letter_v2_20260724/summary.md`
- Per-cell receipts:
  `outputs/stage5/stage5_paper2_internal_token_t1_p0_letter_v2_20260724/cells/*/summary.json`
- Drive checkpoints:
  `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_paper2_internal_token_t1_p0_letter_v2_20260724/cells/`
- Locked governing preregistration:
  `docs/PHASE_T1_LITE_PREREGISTRATION_DRAFT4_20260724.md`
- Draft 3 program amendment:
  `docs/PAPER2_EXPERIMENTAL_PLAN_DRAFT3_20260723.md`

The P0 receipt is marked `citable: false`, `registered_t1_training: false`,
and `phase_t1_remains_unlocked: true`. Those boundaries remain in force until
the strategy decisions above are committed in the final preregistration.
