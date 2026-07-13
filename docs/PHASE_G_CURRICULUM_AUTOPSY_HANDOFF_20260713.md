# Handoff: Phase G Curriculum Autopsy and the Hidden Composition Gain

**Date:** July 13, 2026
**Run:** `stage5_phase_g_curriculum_autopsy_20260712`
**Status:** Read-only autopsy complete; Phase G-alpha remains closed; curriculum redesign is now the highest-value next step.

## 1. Executive conclusion

The previous final-diagonal read understated what the curriculum recovery learned. The recovery did not install full depth-8 inverse composition, but it produced a large, held-out, paired improvement at the second inverse step and a smaller significant improvement at the third. The final diagonal remained weak because depths 4-8 require later loop transitions that received progressively less effective supervision.

This is no longer best described as a failed continuation that merely redistributed errors. It is a **partial staircase install hidden by final-answer scoring**:

- one inverse step is essentially solved and depth-general;
- the second inverse step became functional and generalizes;
- the third step shows a weak but real signal;
- loops 4-8 remain unsupported or at chance;
- the current linear curriculum supplied only 19 raw loop-8 labels, about 2.4 full-row-equivalent CE weight, during recovery.

The deterministic substrate gate still fails, so abductive training and G-alpha remain blocked. But the mechanism line is not closed. The next run must correct the supervision distribution rather than repeat the same linear 2-to-8 ramp at greater nominal step count.

## 2. Integrity and construction findings

Both exact checkpoints were restored and verified:

- fixed-boundary step 1000: `0d6cf119bd66290a2c85686bf58fdc6f9363109c8fdae0ea625f32d13409a1a6`;
- curriculum-recovery step 2000: `fc98feb5d5bd450f7ecc4f6d43ce36fd436418d7ad2cd69df38a089d5ec453d1`.

The canonical data regenerated to the exact landed payload hashes. The train autopsy used 16 rows per depth selected from the exact seeded DataLoader prefix seen by both checkpoints, so "train" here means actually exposed, not merely a member of the training split. Held-out evaluation used the same 16-per-depth convention.

Static verification falsified the suspected end-read/hold construction error:

- the first run computed eight loops;
- loops beyond row depth had labels masked to `-100`;
- `per_loop_labels` averaged active labels only;
- evaluation read loop `d`, not loop 8;
- the recovery did ramp active compute from 2 to 8.

There was no supervised hold objective and no train/eval read-position mismatch. Fixed-eight compute was inefficient, but it did not impose the proposed wrong objective.

The saved prediction confusion also falsifies "invert once, then hold." One-step-preimage errors were only 4/128 before recovery and 6/128 after it. Most errors were other legal names.

## 3. Active-loop results

### 3.1 Accuracy aggregated by loop index

| Checkpoint / split | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Step 1000, seen train | 124/128 | 10/112 | 3/96 | 5/80 | 4/64 | 3/48 | 3/32 | 0/16 |
| Step 1000, held out | 121/128 | 5/112 | 1/96 | 3/80 | 7/64 | 0/48 | 4/32 | 0/16 |
| Recovery, seen train | 127/128 | 51/112 | 12/96 | 2/80 | 3/64 | 1/48 | 2/32 | 1/16 |
| Recovery, held out | 125/128 | 42/112 | 10/96 | 1/80 | 2/64 | 2/48 | 0/32 | 3/16 |

At N=20, chance is 5%.

### 3.2 Paired recovery effect

| Split / loop | Helped | Hurt | Tied | Exact two-sided sign p |
|---|---:|---:|---:|---:|
| Seen train L2 | 46 | 5 | 61 | `2.33e-9` |
| Held-out L2 | 40 | 3 | 69 | `3.02e-9` |
| Seen train L3 | 12 | 3 | 81 | `0.0352` |
| Held-out L3 | 10 | 1 | 85 | `0.0117` |

Held-out recovery accuracy was `42/112 = 37.5%` at loop 2, with a one-sided exact binomial tail versus 5% of approximately `8.0e-26`. Loop 3 was `10/96 = 10.4%`, with one-sided p approximately `0.0219` versus chance.

The seen-versus-held-out gaps are modest relative to the effect: 45.5% versus 37.5% at loop 2, and 12.5% versus 10.4% at loop 3. The primary failure is not table memorization.

### 3.3 Why the final diagonal looked almost unchanged

The diagonal asks depth `d` to be correct at loop `d`. A large loop-2 gain changes the final answer only for depth-2 rows. For depth 3 and beyond, that same gain is invisible unless every later transition also succeeds. The earlier overall comparison, 22/128 to 26/128, was valid as a gate result but incomplete as a mechanism diagnosis.

## 4. Exact curriculum exposure audit

The 2,000-step recovery used a rounded linear cap from 2 to 8. Its exact cap durations were:

```text
cap 2: 167 steps
cap 3: 333
cap 4: 333
cap 5: 334
cap 6: 333
cap 7: 333
cap 8: 167
```

Because rows were depth-balanced and a loop is active only when both row depth and current cap reach it, raw active labels fell sharply:

```text
L1 2000, L2 1749, L3 1377, L4 950,
L5 594, L6 310, L7 114, L8 19
```

The loss then averaged active loop labels within each batch-one row. Expressed as full-row-equivalent CE weight, recovery supplied approximately:

```text
L1 773.7, L2 522.7, L3 336.7, L4 194.4,
L5 105.4, L6 48.6, L7 15.9, L8 2.4
```

Including the first 1,000-step run, approximate cumulative effective weights were:

```text
L1 1119.0, L2 735.0, L3 488.5, L4 302.8,
L5 183.8, L6 102.8, L7 49.3, L8 18.6
```

The learned staircase tracks this exposure gradient. A nominal 4,000-6,000-step repeat with the same linear ramp would still allocate very little useful weight to the final loops. Nominal steps are therefore the wrong dose axis; active weighted labels per loop are the required accounting unit.

## 5. State and above-diagonal diagnostics

The exploratory state-query probe did not find literal prompt-manifold alignment. On eight held-out depth-2 rows, re-entry-to-paired-prompt top-1 was 1/8 before recovery and 0/8 after; the loop-2-output comparison showed the same top-1 pattern. These eight rows were not stratified by correctness, and all eight recovery rows happened to be loop-2 failures. The probe therefore cannot outweigh the large functional loop-2 gain. Its useful conclusion is narrower: successful repeated inversion need not recreate the embedding geometry of a fresh depth-1 prompt.

Above the labeled horizon, recovery remained mostly unstructured. On held-out rows, 34/448 cells continued the inverse orbit, 22/448 held the answer, and 392/448 were other. This was nearly unchanged from step 1000. The model learned supported transitions, not an autonomous out-of-support inverse algorithm.

## 6. Corrected sampling interpretation

Uniform sampling over N names has expected unique-answer coverage:

```text
1 - (1 - 1/N)^K
```

For N=20, the landed fixed-temperature answer-head sampler was above uniform at K=1, 2, 4, and 8 because it concentrated mass on the easy depth-1 answers, but below uniform by K=20:

| K | Observed | Uniform | Delta |
|---:|---:|---:|---:|
| 1 | 0.1797 | 0.0500 | +0.1297 |
| 2 | 0.2344 | 0.0975 | +0.1369 |
| 4 | 0.2969 | 0.1855 | +0.1114 |
| 8 | 0.3984 | 0.3366 | +0.0619 |
| 20 | 0.5703 | 0.6415 | -0.0712 |

This remains evidence against unguided answer-head sampling as useful width. It says nothing negative about GRAM-style guided latent width, which is still gated on deterministic competence.

## 7. Context within the recent program

The forward synthetic work established that the repaired recurrent substrate can learn repeated state transitions and carry them beyond one loop. The natural-surface rung established a green deterministic keeper with synthetic and verbal transfer guardrails. The GRAM audit then correctly made deterministic backward competence the prerequisite for multimodal coverage.

The abductive work now adds a more specific finding: the same substrate can learn repeated inverse transitions, but the current loss and curriculum install them in order of loop exposure. This localizes the current block to curriculum allocation and possibly reverse-search cost, not a global recurrence failure, missing re-injection, end-reader bug, or simple table memorization.

Phase G-alpha still cannot start because the deterministic model does not yet solve the full inverse task family. However, the expected value of one curriculum redesign is now higher than it was after the final-diagonal read.

## 8. Recommended next experiments

### Experiment A: inverse-table direction control

Use the already prepared inverse-table rendering on the same mappings, chains, targets, keeper, and frozen held-out rows. Restrict the first comparison to depths 1-4 and use identical supervision accounting for both arms.

- Forward-table arm: inverse search plus composition.
- Inverse-table-given arm: forward-style lookup plus identical composition.
- Primary metrics: held-out active accuracy at loops 2, 3, and 4.
- Interpretation: a large inverse-table advantage localizes the remaining cost to reverse retrieval; parity localizes it to recurrent composition/curriculum.

### Experiment B: loop-balanced stagewise restart from the locked keeper

Do not reuse the same linear 2-to-8 cap. Train from the keeper with explicit mastery stages:

1. introduce loop 2 with loop-2 emphasis and loop-1 rehearsal;
2. advance only after a locked held-out loop-2 threshold is reached;
3. repeat for loops 3 and 4 in the bounded first job;
4. continue to 8 only if the staircase advances monotonically and the guardrail stays green.

Dose must be logged as raw and weighted active labels per loop. A practical loss form is new-loop emphasis plus bounded prefix rehearsal, rather than averaging every active loop equally. Checkpoints should be scored on the active matrix, not only final diagonals.

### Experiment C: only if A and B disagree or stage 2 cannot master

Run a fixed-table or small-table micro-overfit and then the two-loops-per-logical-step control. The current data already show loop-2 composition is learnable and generalizes, so this micro-test is no longer the first branch.

## 9. Decisions requested from strategy

1. Should Experiment A precede the restart, or should A and the depth-1-to-4 stagewise restart run as matched arms in one job?
2. What held-out active-loop threshold should advance a mastery stage: 0.71, 0.80, or the existing 0.90 substrate standard?
3. Should new-loop emphasis allocate 50% of chain loss to the newest loop and 50% across the prefix, or should the objective equalize expected loss mass across loop indices?
4. Should the first redesigned job stop at depth 4 for a cheap causal read, or proceed conditionally to depth 8 when each stage passes?
5. Is literal state-query manifold alignment still scientifically important given the functional loop-2 gain, or should future probes focus on decodability and transition success instead?

## 10. Recommended position

Run the inverse-table and loop-balanced depth-1-to-4 comparison before a full depth-8 restart. It is the smallest experiment that both isolates operation direction and tests the corrected dose unit. If forward-table composition advances under balanced supervision, continue that checkpoint through depths 5-8. If only inverse-table composition advances, redesign the backward representation before Phase G. If neither advances despite balanced current-loop dose, then the fixed-table/two-loops-per-step autopsy becomes justified.

Do not weaken deterministic gates, start mixed abductive training, or construct stochastic heads yet.
