# Strategy Handoff: Inverse Composition Staircase, Rebase, and Phase A Controls

> **Superseded for forward planning:** This result synthesis remains the source record, but the current queue, Phase A claim boundary, and Phase G re-base are controlled by [TWO_LANE_SURPASS_REBASE_AMENDMENT_20260714.md](TWO_LANE_SURPASS_REBASE_AMENDMENT_20260714.md).

**Date:** 2026-07-14

**Status:** Prior experiment block complete; F9 multi-channel precursor battery running independently

**Decision state:** Phase G-alpha remains closed

## 0. Executive read

This block produced three useful results.

1. **The recurrent substrate can learn repeated composition.** When the relation is rendered as an inverse table so each recurrent transition is a forward lookup, the recurrent 0.5B model reached 62/64 at depth 2 and 63/64 at depth 3.
2. **The present backward-reasoning curriculum does not transfer that success to repeated inverse search over a forward table.** The canonical arm solved the first inverse transition on 55/64 depth-2 rows but only 3/64 after the second transition. The failure is localized to repeated inverse composition, not the first lookup.
3. **Explicit serialized traces are a much stronger dense-model training surface than direct answers on this synthetic family.** At step 4,000, the 0.5B scratchpad arm scored 952/1,792 (53.12%), versus 470/1,792 (26.23%) for 0.5B direct and 322/1,792 (17.97%) for 1.5B direct.

The negative result is retention. Extending the successful inverse-table control from cap 2 to cap 3 reached 63/64, but the locked synthetic guardrail minimum fell from 0.9375 to 0.8125. The model learned the new transition while damaging its prior deepest-horizon behavior. That blocks release, cap 4, abductive-injective qualification, and therefore G-alpha.

The immediate strategic problem is no longer whether recurrence can carry a learned update. It can. The problem is whether we can install a backward update repeatedly **without either losing the update at the next loop or overwriting the already learned forward operator**.

## 1. Program context

This block sits on the deterministic gate leading back to the original GRAM-inspired width question:

```text
working deterministic backward substrate
  -> non-injective abductive family
  -> frozen-block prior/posterior latent heads
  -> oracle coverage at K
  -> matched temperature-K and iso-compute depth controls
```

The early particle/SVGD negatives do not answer that question because they predated the loop-closure repair and lacked the target-conditioned posterior. G-alpha remains the correct width test, but it cannot open until the deterministic abductive-injective substrate passes with a green guardrail.

## 2. Experiment A: matched inverse-composition staircase

### 2.1 Question

Is the backward failure caused by recurrence itself, or by requiring each loop to recover a predecessor from a forward-rendered table?

### 2.2 Arms

- **F, canonical experiment:** forward table, reverse-search transition.
- **C, direction control:** identical latent rows rendered as inverse tables, making each transition a forward lookup.

The train and test row identities matched exactly. Both arms started from keeper SHA256 `0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f`, used AdamW, effective batch 8, 250 optimizer steps, weighted per-loop labels, and approximately 1,600 newest-loop weighted labels.

### 2.3 Locked results

| Arm | Depth-2 diagonal | Conditional second transition | Weighted labels | Synthetic guardrail minimum | Gate |
|---|---:|---:|---:|---:|---|
| C: inverse table / forward lookup | 62/64 | 62/64 | 1,598.4 | 0.9375 | Pass |
| F: forward table / inverse search | 3/64 | 3/55 | 1,603.2 | 0.21875 | Fail |

The most diagnostic F-arm row is the depth-2 path:

- Loop 1 recovered the first predecessor on 55/64 rows.
- Loop 2 reached the requested second predecessor on only 3/64 rows.
- The active target was not linearly decodable above permutation controls at either tested loop.
- Guardrail damage increased with depth, reaching 7/32 at guardrail depth 12.

The C arm showed the opposite signature:

- Depth-1 diagonal: 63/64.
- Depth-2 loop-1 state: 64/64.
- Depth-2 loop-2 state: 62/64.
- Above-diagonal behavior was almost entirely `iterate` (63/64).
- The synthetic guardrail remained just above the preregistered 0.93 floor.

### 2.4 Correct verdict

The embedded run summary contains the older label `non_native_position_cost`. That label was corrected in the experiment ledger to:

```text
experiment_stalled_at_matched_dose
```

The five-fold dose ratio required for the stronger preregistered interpretation was never observed. The supported conclusion is narrower: at the matched bounded dose, the inverse-table control learned the second transition and the forward-table experiment did not.

### 2.5 Interpretation

This rules out a blanket claim that the recurrent block cannot implement a repeated transition. It instead localizes the bottleneck to the combination of inverse retrieval, state carry, and repeated application. It does not yet distinguish among:

- insufficient curriculum for position-transfer of inverse lookup;
- prompt retrieval and carried query state competing in one re-entry channel;
- destructive parameter updates while learning the inverse operator;
- a larger dose requirement for canonical inverse composition.

The result is single-run evidence. The arm gap is very large, but the arms used different training seeds and have not been replicated with swapped or repeated seeds.

## 3. Experiment B: inverse-table cap-3 rebase

### 3.1 Question

Does the successful control continue to a third recurrent transition when initialized from the exact cap-2 checkpoint?

### 3.2 Lineage

- Start checkpoint SHA256: `bc1de1cd7d2a7acf30b9217c8d7054d805888c341b942ff0dab7691b4f995b01`.
- AdamW, 250 optimizer steps, effective batch 8.
- Newest-loop weighted labels to bar: 1,397.65.
- Cap-3 gate: at least 46/64 plus synthetic guardrail minimum at least 0.93.

### 3.3 Results

| Readout | Result |
|---|---:|
| Depth-1 diagonal | 64/64 |
| Depth-2 diagonal | 64/64 |
| Depth-3 diagonal | 63/64 |
| Conditional transition 2 | 128/128 |
| Conditional transition 3 | 63/64 |
| Overall synthetic guardrail | 373/384 (97.14%) |
| Synthetic guardrail minimum | 26/32 (81.25%) at depth 12 |

The capability gate passed decisively. The retention gate failed, so cap 4 correctly did not run.

### 3.4 Interpretation

The model can install and execute a third recurrent update. The failure is not aggregate collapse: guardrail accuracy remained 97.14%, and depths 1-7 were perfect. The damage is concentrated in the deepest old horizon, especially depth 12. This is a continual-learning/Pareto problem rather than a failure to acquire cap 3.

This suggests a low-EVPI next test: preserve the successful cap-3 target dose while adding explicit replay or another retention constraint. More narrow cap-3 steps without retention support are not justified.

## 4. Experiment C: dense Phase A controls

### 4.1 Question

How much of the synthetic composition task can ordinary dense Qwen models learn, and how much does explicit trace serialization matter?

### 4.2 Matched setup

- Train set: 2,048 frozen rows, SHA256 `cf61c14c2629f2caa7e1b6bd100adb122a468d5285b74970aaa4aebfbb56fd12`.
- Eval set: 1,792 frozen rows across depths 1-14, SHA256 `3de844669aba303063e6932f5852914ee0993e531c8e65c2a4c4b18e219b3fc8`.
- Full-model AdamW with FP32 parameters/moments and BF16 compute.
- Effective batch 8, learning rate `2e-6`, 4,000 steps, checkpoints at 2,000 and 4,000.
- Arm B: Qwen2.5-0.5B-Instruct, direct answers.
- Arm C: Qwen2.5-0.5B-Instruct, serialized orbit scratchpad.
- Arm D: Qwen2.5-1.5B-Instruct, direct answers.

### 4.3 Checkpoint comparison

| Arm | Step 2,000 | Step 4,000 | Delta |
|---|---:|---:|---:|
| B, 0.5B direct | 464/1,792 (25.89%) | 470/1,792 (26.23%) | +6 |
| C, 0.5B scratchpad | 930/1,792 (51.90%) | 952/1,792 (53.12%) | +22 |
| D, 1.5B direct | 350/1,792 (19.53%) | 322/1,792 (17.97%) | -28 |

Paired within-arm sign tests from step 2,000 to 4,000:

| Arm | Helped | Hurt | Tied | Two-sided sign p |
|---|---:|---:|---:|---:|
| B | 150 | 144 | 1,498 | 0.771 |
| C | 37 | 15 | 1,740 | 0.00319 |
| D | 172 | 200 | 1,420 | 0.161 |

The C gain is statistically detectable but operationally small: 1.23 percentage points for another 2,000 full-model steps. B is flat and D is directionally worse. The first saved checkpoint already contains almost all of the useful result.

### 4.4 Step-4,000 paired comparisons

| Contrast | Helped | Hurt | Tied | Net correct | Two-sided sign p |
|---|---:|---:|---:|---:|---:|
| C minus B | 639 | 157 | 996 | +482 | `1.02e-69` |
| C minus D | 743 | 113 | 936 | +630 | `2.24e-114` |
| D minus B | 231 | 379 | 1,182 | -148 | `2.23e-9` |

### 4.5 Depth profile at step 4,000

| Depth | B direct 0.5B | C scratchpad 0.5B | D direct 1.5B |
|---:|---:|---:|---:|
| 1 | 119 | 120 | 13 |
| 2 | 84 | 57 | 16 |
| 3 | 42 | 86 | 21 |
| 4 | 28 | 100 | 21 |
| 5 | 28 | 83 | 45 |
| 6 | 26 | 95 | 39 |
| 7 | 21 | 78 | 35 |
| 8 | 18 | 96 | 38 |
| 9 | 25 | 89 | 31 |
| 10 | 19 | 92 | 5 |
| 11 | 17 | 18 | 16 |
| 12 | 16 | 13 | 15 |
| 13 | 15 | 19 | 14 |
| 14 | 12 | 6 | 13 |

Each depth has 128 rows.

Key structure:

- C scored 896/1,280 (70.0%) across depths 1-10 despite training only through depth 8.
- C fell to 56/512 (10.94%) at depths 11-14, near the 1/16 symbol chance rate.
- B was better than C at depth 2 (84 versus 57), but C was much better from depths 3-10.
- C depth-2 errors were 57 correct, 17 one-step-early, and 54 other, with no parse failures.
- The scratchpad advantage is therefore a medium/deep composition advantage, not a uniform shallow-task improvement.

### 4.6 Optimization and repeatability

Arm C training loss was already approximately zero by step 50 and remained there. B and D also approached near-zero training loss by the end while generalization remained limited. The run is data/generalization limited, not training-loss limited; 4,000 steps were more than needed to establish the ranking.

B and C step-4,000 evaluations reproduced exactly. The independent D reload scored 322 rather than the prior 320. The discrepancy passed the preregistered GPU repeatability envelope: total delta 2, maximum depth delta 2, and zero parse-failure delta. This is a repeatability receipt, not a scientific effect.

### 4.7 Interpretation and caveats

The strong claim is about **supervision surface**: a compact dense model benefits greatly from explicit serialized state transitions on this task. The first observed checkpoint crossover is already present at step 2,000; the exact earlier crossing is unknown because no earlier checkpoints were saved.

The weak 1.5B direct result must not be read as evidence that scale hurts. The arms used the same learning rate, batch, and step count, not a scale-optimized recipe, and D displayed a highly non-monotonic depth profile. It is a valid result for the preregistered recipe, not a general 1.5B capability estimate.

These dense controls also do not establish that the recurrent wrapper beats dense Qwen. They establish the baseline behavior and the value of explicit traces. A model-level comparison needs identical task rows, scoring, compute accounting, and a matched recurrent checkpoint.

## 5. Cross-experiment synthesis

### 5.1 What is now supported

- The repaired recurrent architecture can learn a real iterative update through at least three transitions.
- The direction in which the relation is presented matters enormously at matched dose.
- The canonical backward arm learns one inverse lookup but fails to compose it a second time.
- Extending a successful update can damage an older deep-horizon operator even when aggregate guardrail accuracy remains high.
- Explicit intermediate traces are a major training lever for dense models and shift the accuracy frontier from shallow direct lookup toward depths 3-10.
- Additional training after the first saved checkpoint has sharply diminishing returns under the current fixed dataset.

### 5.2 What is not supported

- No conclusion that the architecture cannot do backward reasoning in principle.
- No five-fold position-cost estimate.
- No claim that 1.5B is intrinsically weaker than 0.5B.
- No claim that the recurrent model has surpassed its dense base on natural reasoning benchmarks.
- No GRAM-style stochastic-width result.
- No reason to reopen SVGD or particle geometry before a target-conditioned latent model and deterministic abductive gate exist.

## 6. Current running work: F9 multi-channel precursor

The running battery is eval-only and asks whether the single full-width re-entry pathway contains head/subspace specialization worth exploiting. It measures:

- M1: loop drift concentration by learned query-head write subspaces versus random matched subspaces;
- M2: table-line attention specialization and head identity stability;
- M3: sensitivity to removing only selected prelude-injection subspaces, with a bit-exact flag-off equivalence check.

This work may improve the diagnosis and the paper's mechanism characterization. It cannot authorize a multi-channel bridge by itself. F9 remains banked unless at least two measurements are positive **and** the staircase produces the preregistered reading one. The current staircase verdict is `experiment_stalled_at_matched_dose`, so that second condition is absent.

## 7. Questions for strategy review

1. **Retention recipe:** Is one mixed-replay cap-3 run the correct next deterministic experiment, starting from exact C cap 2 and matching the previous cap-3 target-label dose while adding old-operator replay?
2. **Canonical inverse curriculum:** Should we next train the inverse primitive across loop positions before composing it, rather than simply increasing canonical F dose?
3. **Guardrail objective:** Is the strict minimum-over-depth floor still the right release gate, or should it remain the hard gate while aggregate accuracy is reported as a secondary Pareto measure? My recommendation is to keep the hard minimum.
4. **G-alpha substrate:** Must G-alpha wait for the canonical forward-table abductive task, or can an inverse-rendered non-injective control be used as a preliminary width-only assay? The latter is cheaper but weakens the backward-reasoning claim.
5. **Dense Phase A:** Is C step 2,000 sufficient as the scratchpad control, given that another 2,000 steps bought only 22 rows and training loss had saturated?
6. **Scale control:** Is rerunning 1.5B with a scale-tuned recipe worth the GPU cost now, or should D remain a bounded negative for this exact recipe?
7. **Shallow/deep tradeoff:** Should the depth-2 regression in scratchpad C be treated as an explicit curriculum target, or accepted as the cost of gaining depths 3-10?
8. **Replication:** Before a causal paper claim about table direction, do we require one repeated-seed or seed-swapped F/C cap-2 confirmation?

## 8. Recommended next sequence

1. **Finish and score F9 as diagnostic-only.** Do not activate architecture work from it under the current staircase verdict.
2. **Land the Phase A figure and receipts.** Plot per-depth B/C/D at steps 2,000 and 4,000; report the paired tests and D repeatability envelope.
3. **Run one retention-repair cap-3 experiment if strategy approves.** Start from exact C cap-2 SHA, preserve the prior cap-3 target dose, add fixed old-operator replay, and require both `>=46/64` cap-3 performance and `>=0.93` guardrail minimum. Report added replay compute explicitly.
4. **If retention passes, attempt cap 4 once.** If it fails again, stop the inverse-table rebase rather than tuning indefinitely.
5. **Design the canonical inverse position-transfer micro-test.** Verify that a learned one-step inverse operation can be applied at the second recurrent position before spending on another full staircase.
6. **Open abductive-injective only after capability and retention are both green.** G-alpha remains immediately behind that deterministic gate.
7. **Keep Phase G implementation preparation CPU-only.** Frozen sets, exact coverage scorer, temperature-K control, posterior/prior spec, and preregistration can proceed without GPU, but latent training must not jump the gate.

## 9. Artifact lineage

- Staircase: `outputs/stage5/stage5_inverse_composition_staircase_20260713/summary.json`
- Rebase: `outputs/stage5/stage5_inverse_table_rebase_caps3_4_20260713/summary.json`
- Dense B/C: `outputs/stage5/stage5_phase_a_dense_full_bc_20260713/summary.json`
- Dense D: `outputs/stage5/stage5_phase_a_dense_full_d_20260713/summary.json`
- Checkpoint comparison: `outputs/stage5/stage5_phase_a_checkpoint_comparison_20260713/summary.json`
- Paired row record: `outputs/stage5/stage5_phase_a_checkpoint_comparison_20260713/paired_rows.jsonl`
- Running F9 specification: `docs/MULTICHANNEL_BRIDGE_PRECURSOR_SPEC.md`

## 10. Bottom line

This block did not open stochastic width, but it moved the program forward. It demonstrated a three-step learned recurrent computation, identified repeated inverse composition as the canonical bottleneck, exposed a deep-horizon retention conflict, and established explicit traces as a powerful dense baseline. The next useful GPU dollar should buy either retention-preserving cap-3 evidence or a tightly scoped inverse position-transfer diagnosis, not more unconstrained dose, particle noise, or kernel geometry.
