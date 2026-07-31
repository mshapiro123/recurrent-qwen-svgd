# Strategy Handoff: DC1 Follow-ups Complete, Stage A Lock Required

**Date:** 2026-07-30
**Status:** read-only follow-ups complete; Stage A training blocked only on the separate Drive-locked preregistration
**Training authorized by this document:** no
**Evaluation surfaces touched:** DEV-C only; EVAL-C and EVAL-B remain untouched

## 0. Executive verdict

The two read-only jobs authorized after the DC1 preflight are complete.

First, the population-matched parity ledger corrects the earlier pre/post-D0 comparison. Severe forced-depth harm was already present before D0. D0 did not create the asymmetry; it modestly reduced it. On the same 50,108 DEV-C positions, forced depth 2 had net utility of -6,625 before D0 and -5,325 after D0. Fixed depth 2 is therefore not a router-free solution on either tested checkpoint.

Second, the scale-response probe rejects the proposed residual-crossover explanation under its pre-stated test. The accuracy trough occurred at 10 times embedding RMS, while the receipt's nearest measured cosine crossover was at 3 times. Raw hidden-state scale was not a performance plateau: 1.5 and 2 times raw improved accuracy and reduced harm, but every tested scale remained strongly net harmful. The pathway is signal-bearing, but the untrained forced-append actuator remains unsafe.

These results do not change the authorized Stage A design. The strategy handoff explicitly locked raw hidden-state scale as an operational initialization, not as a global optimum, and stated that the scale probe could not alter Stage A. Stage A now has one blocker: its separate governing preregistration has not been delivered to the Drive research folder. The existing strategy document says that preregistration will arrive separately.

## 1. Lineage and integrity

### 1.1 Parity ledger

| Field | Receipt |
|---|---|
| Run | `stage5_paper2_dc1_parity_ledger_20260730` |
| Landing commit | `2ff10729` |
| Pre-D0 checkpoint | `93d2e5f9a941bbe79a0b2fc3f9bf43d582bf054990c14b1a93ff67024140062d` |
| Post-D0 checkpoint | `8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf` |
| Population | 113 identical DEV-C rows, 50,108 positions |
| Teacher targets | identical cached 7B targets pre/post |
| Evaluation partition | DEV-C reusable diagnostic surface |
| Training | none; zero optimizer steps |
| EVAL-C touched | false |
| Checkpoint mutation | false for both checkpoints |
| Tie policy | fp32 logits; exact ties choose lowest token ID |
| Applied branch | archived floor was rejected-only, so exact same-row GPU fallback |

### 1.2 Scale-response probe

| Field | Receipt |
|---|---|
| Run | `stage5_paper2_dc1_scale_response_20260730` |
| Landing commit | `a0d10e1e` |
| Checkpoint | post-D0 EMA `8245cabf...1086cf` |
| Population | same 113 DEV-C rows, 50,108 positions |
| Training | none; zero optimizer steps |
| EVAL-C touched | false |
| Checkpoint mutation | false |
| Status | `complete_descriptive_non_gating` |
| Mechanism language authorized by receipt | false |

The parity job uses row-identical all-position populations and identical teacher targets. Its accepted/rejected subsets are defined independently for each checkpoint by each checkpoint's own depth-1 agreement, so their counts differ slightly. The all-position comparison is the primary population-matched result.

## 2. Population-matched parity result

### 2.1 Forced transition 1 to 2

| Measure | Pre-D0 | Post-D0 | Post minus pre |
|---|---:|---:|---:|
| Depth-1 accuracy | 72.266% | 72.166% | -0.100 points |
| Depth-2 accuracy | 59.044% | 61.539% | +2.495 points |
| Helps | 1,924 | 2,083 | +159 |
| Hurts | 8,549 | 7,408 | -1,141 |
| Net utility, helps minus hurts | -6,625 | -5,325 | +1,300 |
| Harm/help ratio | 4.443 | 3.556 | improved, still harmful |

D0 improved the forced 1-to-2 transition by 1,300 net positions, or 2.594 percent of the scored population. Most of the improvement came from 1,141 fewer harmed predictions rather than from additional recovery.

### 2.2 Accepted and rejected decomposition

| Cohort | Pre-D0 | Post-D0 | Reading |
|---|---:|---:|---|
| Rejected positions helped at depth 2 | 1,924 / 13,897 = 13.845% | 2,083 / 13,947 = 14.935% | modest recovery increase |
| Accepted positions harmed at depth 2 | 8,549 / 36,211 = 23.609% | 7,408 / 36,161 = 20.486% | meaningful harm reduction |

The rejected-only archaeology was directionally correct about recovery but incapable of measuring the dominant harm term. The earlier sentence that severe harm did not exist before D0 must be retired. The corrected statement is:

> On a population-matched DEV-C diagnostic, forced depth 2 was net harmful before and after D0. D0 reduced, rather than created, that harm asymmetry.

### 2.3 Forced transition 2 to 3

Depth-3 accuracy was 43.831% pre-D0 and 45.967% post-D0. However, the incremental 2-to-3 transition remained harmful: net -7,623 pre-D0 and -7,803 post-D0. The higher post-D0 depth-3 endpoint does not imply that the third step became safe; it starts from a stronger depth-2 state and still loses net correctness during the transition.

### 2.4 Strategic consequence

The parity contingency closes negatively. The pre-D0 checkpoint does not offer a net-positive fixed-depth-2 actuator, so no ungated shortcut replaces Stage A or revives D1 on that basis. This result also narrows the manuscript claim: the harm is demonstrated on two checkpoints of this recurrent substrate and this DEV-C diagnostic, not on all recurrent architectures or checkpoints.

## 3. Scale-response result

The labels `3x`, `10x`, `30x`, `100x`, and `300x` refer to multiples of embedding RMS. They are still far below raw hidden-state RMS until the upper end of the grid. The final two arms are multiples of raw hidden-state RMS and should not be conflated with the embedding-relative labels.

| Feedback scale | Accuracy | Helps | Hurts | Net utility | Median cosine to fed state |
|---|---:|---:|---:|---:|---:|
| Embedding-matched | 16.187% | 651 | 28,697 | -28,046 | 0.626 |
| 3x embedding RMS | 10.643% | 508 | 31,332 | -30,824 | 0.580 |
| 10x embedding RMS | 9.537% | 483 | 31,861 | -31,378 | 0.539 |
| 30x embedding RMS | 11.272% | 540 | 31,049 | -30,509 | 0.518 |
| 100x embedding RMS | 18.290% | 863 | 27,855 | -26,992 | 0.577 |
| 300x embedding RMS | 42.827% | 1,484 | 16,181 | -14,697 | 0.819 |
| Raw hidden-state RMS | 51.900% | 1,728 | 11,879 | -10,151 | 0.922 |
| 1.5x raw RMS | 54.786% | 1,798 | 10,503 | -8,705 | 0.947 |
| 2x raw RMS | 56.586% | 1,888 | 9,691 | -7,803 | 0.958 |

Registered k=0 accuracy on this execution path was 72.158%. No forced-append scale reached baseline accuracy or non-negative net utility.

### 3.1 Pre-stated prediction

The refined crossover hypothesis predicted that fed-state cosine would rise sigmoidally with scale and that the accuracy trough would coincide with the region where neither component dominated. The receipt reports:

- accuracy trough: `10x` embedding RMS;
- nearest measured cosine crossover: `3x` embedding RMS;
- trough coincides with crossover: false;
- non-decreasing fed-cosine adjacent pairs: 5 of 8.

Under the pre-written rule, the crossover account is not supported and should be retired. The U-shaped accuracy response remains a measured observation without an established mechanism.

### 3.2 Layerwise readout

At raw scale, mean cosine to the fed state stayed high throughout the block, declining from 0.986 after the first layer to 0.900 near the final transformer layer and 0.881 after the final readout transformation. At 10 times embedding RMS, it fell to 0.127 after the first layer, remained near 0.07 to 0.18 through most layers, and recovered to 0.438 only at the final state. This supports a bounded descriptive statement that raw-scale feedback preserves its direction much more strongly through the block than the low-scale trough arm.

It does not establish the proposed competition mechanism. Cosines to the fed state and to the registered k=0 state are nearly identical at every scale, so the two references do not provide independent directions with which to identify competing components.

### 3.3 Strategic consequence

Raw is not globally optimal among tested scales. Two times raw was the least harmful tested arm. This does not amend Stage A because the governing strategy explicitly locked raw as an operational identity initialization and stated that the scale probe was non-gating. The bridge is trainable and may move away from raw. No post-hoc scale change should enter the preregistration without an explicit strategy amendment.

## 4. Updated scientific record

### Supported, bounded

1. The horizontal append pathway is signal-bearing: changing scale changes tens of thousands of outcomes and produces large representation changes.
2. Raw and above-raw scaling preserve the fed-state direction substantially better than embedding-relative scales.
3. Forced append is unsafe at every untrained scale tested.
4. Forced in-place depth 2 is net harmful on both the pre-D0 and post-D0 checkpoints on the population-matched DEV-C diagnostic.
5. D0 modestly improved the forced 1-to-2 transition, primarily by reducing harm.

### Refuted or retired

1. The claim that D0 created the severe harm asymmetry.
2. The strict monotone copy-through account.
3. The refined claim that the accuracy trough is located at the measured cosine crossover.
4. The possibility that ungated fixed depth 2 is already net-positive on the pre-D0 checkpoint.

### Still unknown

1. Why the embedding-relative scale curve has a U-shaped trough.
2. Whether bridge-only training can make forced append safe on untouched EVAL-C.
3. Whether a trained safe actuator can later support useful control-token routing.
4. Whether any result generalizes beyond this checkpoint lineage, text mixture, and single seed.

## 5. Sole blocker: missing Stage A preregistration

The Drive research folder contains `STRATEGY_TO_CODING_AGENT_DC1P_BANK_STAGE_A_PREP_20260730.md` (Drive ID `12wp0ovgsW83FW5LQOPV7Tf8YBVRZMFfl`). That document states: "The Stage A preregistration drafts now in the strategy lane and arrives as a separate document for lock." A search of the Drive research folder found no separate Stage A preregistration at the time of this handoff.

The coding lane has completed its prerequisite resource note:

| Field | Proposed locked value |
|---|---:|
| Runtime | A100-SXM4-80GB or equivalent |
| Step ceiling | 2,000 optimizer steps |
| Microbatch / accumulation | 1 / 1 |
| Effective batch | 1 row |
| Maximum sequence length | 512 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `0.0` |
| Gradient clipping | `0.5` global norm |
| Precision | full fp32 model, feedback boundary, gradients, optimizer |
| Passive checkpoints | 500, 1,000, 1,500, 2,000 |
| Primary checkpoint | final-step raw weights |
| Expected wall time | approximately 2 to 4 hours |

No Stage A launcher has been created because the standing policy prohibits launcher construction before the governing preregistration is stored in Drive with its SHA-256.

## 6. Exact request to the strategy agent

Please deliver the separate Stage A preregistration as raw markdown in Drive, with file ID, byte count, and SHA-256. It should include a machine-readable companion or an exact JSON appendix covering the following fields.

### 6.1 Lineage and data

- Post-D0 EMA initialization SHA `8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf`.
- DEV-C training manifest and cached 7B teacher hashes.
- EVAL-C frozen manifest and teacher-cache hashes, unread before the final registered pass.
- Single-seed policy and exact seed.
- Explicit statement that EVAL-B is never read again.

### 6.2 Architecture and trainable set

- Bridge-only trainable set, with complete parameter-name allowlist.
- Every non-bridge parameter frozen with start/end hash assertions.
- Identity initialization at raw hidden-state scale, described as operational rather than optimal.
- Advancing position IDs.
- Forced `k=1`, global `L=1`, hard asserted `k <= 3`.
- Recompute-only graph and full-fp32 execution.
- k=0 bit-identity contract before and after training.
- Control-read plumbing may be stage-C-ready but remains inactive in Stage A.

### 6.3 Objective and optimization

- Teacher cross-entropy at the appended-slot readout.
- Label-to-objective alignment statement for forced-append safety and fallback.
- The resource-note values above, or an explicit pre-lock amendment explaining any change.
- Final-step raw checkpoint primary; quarter-point checkpoints passive only.
- No DEV-C-guided early stopping or hyperparameter selection.

### 6.4 Single EVAL-C pass

The same pass must score:

1. registered k=0 baseline;
2. trained append, forced k=1;
3. untrained append, forced k=1;
4. in-place forced depth 2;
5. in-place forced depth 3;
6. code/general strata and matched-layer descriptive table.

The verdict computation must freeze row-cluster bootstrap resampling, confidence level, cluster definition, tie policy, and exact units before EVAL-C is read.

### 6.5 Locked decision mapping

- **Qualifies:** trained-append net utility at or above zero and row-cluster bootstrap 95% CI lower bound no worse than -0.25% of scored positions.
- **Partial domestication:** hurts at or below 50% of same-partition untrained-append hurts, helps not below untrained-append helps, and net utility remains negative.
- **No material improvement:** hurts reduction versus untrained append below 50%; transient append retires on this substrate.
- In-place depth-2 and depth-3 arms are descriptive anchors, not qualification gates.
- Blocked or aborted outcomes still write receipts and do not silently consume or replace the registered attempt.

## 7. Questions requiring strategy confirmation

1. Confirm that raw-scale initialization remains binding despite 2x raw being the least harmful descriptive arm. The existing strategy handoff says yes; the preregistration should make that explicit so no one later reads it as an oversight.
2. Confirm the proposed `2,000` steps, batch 1, AdamW `1e-4`, and full-fp32 resource note without modification, or issue a pre-lock amendment.
3. State the exact single seed.
4. Provide the EVAL-C row-cluster bootstrap seed and number of replicates.
5. Confirm whether the `-0.25%` confidence-bound floor is applied to the integer net count through a preregistered conversion or directly to the normalized row-cluster statistic.
6. Confirm that the single EVAL-C pass may compute all five system arms from one immutable cache without violating the read-once rule.

## 8. Execution sequence after delivery

1. Coding agent verifies the Drive bytes and SHA-256.
2. Reconcile the preregistration against the resource note and both landed receipts.
3. Commit the markdown and machine-readable registration with `locked_before_training` and all hashes.
4. Build the Stage A runner and Colab target with red/green unit tests, one-batch equivalence tests, frozen-parameter assertions, resume tests, and EVAL-C read-once enforcement.
5. Run startup and 20-step throughput/memory projection on an A100 80GB.
6. Complete 2,000 steps with passive receipts at quarter points.
7. Execute the one immutable EVAL-C pass and scripted verdict.
8. Bank the result before drafting Stage B or any amendment.

## 9. Do-not-claim boundaries

- Do not say append is safe, useful, or accuracy-positive.
- Do not say raw scale is optimal.
- Do not explain the U-shaped trough using the rejected crossover mechanism.
- Do not say D0 created the in-place harm asymmetry.
- Do not generalize the parity result beyond the tested checkpoints and DEV-C diagnostic.
- Do not use DEV-C numbers as registered headline evidence.
- Do not touch EVAL-C before the locked single pass.
- Do not build or launch Stage B, Stage C, Stage D, policy training, persistent scratchpad, RG-12, GRAM, or width work.
- Do not create the Stage A training launcher before the governing Drive preregistration is locked.

## 10. Canonical artifacts

| Artifact | Path |
|---|---|
| Parity public receipt | `outputs/stage5/stage5_paper2_dc1_parity_ledger_20260730/summary.json` |
| Scale-response public receipt | `outputs/stage5/stage5_paper2_dc1_scale_response_20260730/summary.json` |
| DC1 preflight handoff | `docs/PAPER2_DC1_PREFLIGHT_RESULT_HANDOFF_20260730.md` |
| Read-only follow-up build handoff | `docs/PAPER2_DC1_FOLLOWUPS_BUILD_HANDOFF_20260730.md` |
| Stage A resource note | `docs/PAPER2_DC1_STAGE_A_RESOURCE_NOTE_20260730.md` |
| Governing composite design | `docs/COMPOSITE_TRAINING_DESIGN_20260729.md` |
| Strategy preparation handoff | Drive ID `12wp0ovgsW83FW5LQOPV7Tf8YBVRZMFfl` |

## 11. Plain-language summary

The extra computation step is harmful whether measured before or after D0, although D0 made it somewhat less harmful. Increasing the strength of the appended hidden state also makes the pathway less destructive once the strength is high enough, but even twice the original hidden-state scale still breaks far more correct predictions than it fixes. The pathway carries useful information, but the model has not learned how to use it safely.

The next experiment is therefore still the right one: freeze the model, train only the small bridge that feeds the temporary thinking slot, and ask whether one forced thinking step can become safe. The engineering work and decision rules are ready. The only missing item is the formal preregistration document that freezes those rules before the untouched evaluation set is read.
