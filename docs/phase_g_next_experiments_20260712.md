# Phase G Next Experiments

> **Amendment of record, July 12, 2026:** The deterministic and stochastic tracks are now one dependency chain. See [PHASE_G_TRACK_REUNIFICATION_AMENDMENT.md](PHASE_G_TRACK_REUNIFICATION_AMENDMENT.md). Where this document specifies a fixed-temperature answer-head comparator or the earlier N=20 fan set, the amendment controls: the final G-alpha claim uses entropy matching and the frozen stratified N=24 arbitrary-function sets.

## Landed-result review, deterministic keeper lock, and G-alpha preregistration

**Date:** July 12, 2026  
**Repository head reviewed:** `ffffe31`  
**Primary decision:** Stop natural-surface dose escalation. Lock the original step-2000 keeper and move to the abductive/injective gate required for stochastic-width testing.

---

## 1. What landed

Three result families are complete:

1. `stage5_natural_surface_receipts_20260709_210151`
   - Full original-seed checkpoint receipts at steps 2000, 4000, and 6000.
   - Frozen synthetic guardrails, paired relay/pointer sets, robustness sets, and same-reader checks.
2. `stage5_natural_surface_followups_2_3_20260710`
   - Corrected unseen single-token names.
   - Passive versus fronted template placement.
   - Baton transfer.
   - Loop-index and state-envelope probes.
3. `stage5_natural_surface_replication_dose_seed931337_20260710`
   - Fresh training data seed.
   - Checkpoints at 1000, 1500, 2000, 2500, 3000, 4000, and 6000.
   - Evaluation on the original frozen rows.

All three summaries report `finished`.

---

## 2. Original-seed checkpoint curve

| Checkpoint | Relay min d1-8 | Relay min d9-12 | Pointer min d1-8 | Pointer min d9-12 | Synthetic min d1-12 |
|---|---:|---:|---:|---:|---:|
| Frozen N24 | 0.0625 | 0.0781 | 0.0781 | 0.0313 | 0.9688 |
| Step 2000 | **0.8281** | **0.5938** | **0.8125** | **0.5313** | **0.9531** |
| Step 4000 | 0.8438 | 0.6875 | 0.7500 | 0.2813 | 0.8906 |
| Step 6000 | 0.9375 | 0.3594 | 0.8438 | 0.2344 | 0.3750 |

The step-2000 checkpoint is the only balanced keeper:

- Relay total active accuracy: `0.89163`.
- Pointer total active accuracy: `0.87159`.
- Full-width synthetic minimum: `0.953125`, above the locked `0.93` guardrail.
- Later checkpoints improve some in-distribution natural cells but progressively damage the synthetic mechanism and tail transfer.

Keeper receipt:

```text
/content/drive/MyDrive/recurrent-qwen-svgd-backups/
  natural_surface_backup_20260709_180835/checkpoints/
  stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812/
  unfrozen_recurrent_step_2000.pt
```

```text
SHA256: 0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f
Bytes:  370811845
```

---

## 3. Independent-recipe replication

| Step | Relay min d1-8 | Relay min d9-12 | Pointer min d1-8 | Pointer min d9-12 | Synthetic full-width min |
|---|---:|---:|---:|---:|---:|
| 1000 | 0.5078 | 0.3125 | 0.4375 | 0.1953 | 0.9219 |
| 1500 | 0.7031 | 0.2031 | 0.6406 | 0.2344 | 0.9375 |
| 2000 | 0.8750 | 0.3438 | 0.7734 | 0.3203 | 0.3906 |
| 2500 | 0.6953 | 0.0625 | 0.6797 | 0.0859 | 0.0469 |
| 3000 | **0.9219** | **0.4609** | **0.7813** | **0.3906** | 0.5156 |
| 4000 | 0.7344 | 0.1797 | 0.7734 | 0.2422 | 0.7500 |
| 6000 | 0.8516 | 0.2891 | 0.7109 | 0.1719 | 0.5391 |

This replication confirms that the recipe can rapidly learn natural-surface transitions. It does not reproduce the original balance between natural transfer and mechanism preservation.

The curve is strongly non-monotonic. Step 2500 nearly destroys full-width synthetic performance, step 3000 partially recovers it, and later steps move again. Training loss or rehearsal-set accuracy therefore cannot select a keeper. The frozen full-width guardrail must remain an online checkpoint criterion.

One design flaw was found in the replication machinery: `train_seed` controlled generated data, but the trainer did not seed Python, PyTorch, CUDA, or DataLoader shuffling. The run remains a valid independent recipe replication because both data and optimization path changed. It is not a clean one-variable seed comparison. The trainer and natural-surface launchers have now been patched to record and apply an explicit training seed.

---

## 4. Surface and representation diagnostics

### 4.1 Transfer is real but position-sensitive

At the original step-2000 keeper:

| Follow-up set | Min d1-8 | Min d9-12 | Total active accuracy |
|---|---:|---:|---:|
| Corrected unseen-name relay | 0.6016 | 0.2344 | 0.7983 |
| Corrected unseen-name pointer | 0.6172 | 0.2891 | 0.7917 |
| Fronted relay | 0.5938 | 0.2813 | 0.7951 |
| Fronted pointer | 0.3750 | 0.1406 | 0.6895 |
| Passive baton | 0.6406 | 0.4375 | 0.7766 |
| Fronted baton | 0.0781 | 0.0313 | 0.4008 |

The model transfers to disjoint single-token names and to a passive baton surface. It is not invariant to where and how the transition relation is phrased. The severe fronted-baton failure means the result is a learned natural-surface transition family, not yet a general language-level relation-following algorithm.

### 4.2 Recurrent time is explicit in the state

Across frozen, step-2000, step-4000, and step-6000 checkpoints, the loop-index probe is `1.0` accurate with permutation p95 around `0.28-0.32`. Three leading directions account for most of that separability after deflation.

At the same time, late-loop states leave the early-loop envelope. For paired relay at step 6000, reconstruction MSE grows from about `0.125` at loop 4 to `18.13` at loop 8 and `132.74` at loop 12; feature norm grows from `29.0` to `49.9` to `82.3`.

This supports two conclusions:

1. The loop is doing structured, time-indexed computation rather than repeatedly returning one identical state.
2. The state transition is not a stable fixed-point iteration. Long-horizon training can amplify loop-index and norm structure while harming task-general recurrence.

This is acceptable for freezing a keeper. It is another reason not to continue deterministic dose escalation before G-alpha.

---

## 5. Decision

### Keeper

Lock the original natural-surface step-2000 checkpoint by path and SHA. Do not promote the step-6000 model or the fresh-seed step-3000 model.

### Claim discipline

Supported:

- One preregistered checkpoint demonstrates strong natural-surface transition learning while retaining the synthetic mechanism guardrail.
- A fresh training-data/optimization path also learns the natural tasks, confirming that the effect is not unique to one dataset.

Not supported:

- The exact keeper balance is reliably reproduced across training seeds.
- More natural-surface training monotonically improves the model.
- The learned transition is invariant to prompt position or paraphrase.
- This result itself establishes stochastic width or GRAM-style multimodality.

### Queue change

Natural-surface dose escalation stops here. The green step-2000 keeper satisfies the deterministic guardrail half of the amended Phase G gate. The remaining prerequisite is the paired abductive/injective task family.

---

## 6. Experiment 0: no-GPU G-alpha gate preparation

### Hypothesis

A paired finite-function task can isolate ordinary inverse reasoning from multimodal abduction while giving exact solution-set denominators.

### Design

- Symbols: 20 tokenizer-compatible names.
- Depths: 1-8.
- Injective control: a permutation mapping with exactly one depth-step preimage.
- Abductive arm: a constructive convergent fan with exactly 2, 3, or 4 valid preimages.
- Training: one sampled valid reverse chain per abductive row.
- Evaluation: retain every valid start and every corresponding forward orbit.
- Coverage: `unique valid starts sampled / exact valid starts`.

### Generated manifests

| Dataset | Rows | Solutions |
|---|---:|---|
| Train injective | 2048 | exactly 1 per row |
| Train abductive | 2048 | balanced 2-4 per row |
| Test injective | 1024 | exactly 1 per row |
| Test abductive | 1024 | balanced 2-4 per row |

Every row has passed exact exhaustive preimage recomputation. Train and test IDs are disjoint. The local gate reports `phase_g_alpha_ready=true` because the data gate and keeper guardrail are both green.

### Code

- `training/abductive_injective_task.py`
- `training/generate_abductive_injective_task.py`
- `colab/run_stage5_phase_g_gate_prepare.py`
- `tests/test_abductive_injective_task.py`
- `tests/test_stage5_phase_g_gate_prepare.py`

---

## 7. Experiment 1: deterministic task-learnability and answer-head baseline

This is the next paid experiment. Use an L4. Do not use particles, sampled latents, learned halting, or SVGD.

### Arms

1. **Injective deterministic control**
   - Initialize from the locked keeper.
   - Train on injective reverse chains.
   - Establish that the architecture can learn the inverse transition when the answer is unique.
2. **Abductive deterministic baseline**
   - Initialize independently from the same keeper.
   - Train on a 50/50 injective-abductive mix with one sampled valid chain per abductive row.
   - Measure greedy valid-answer rate and mode collapse.
3. **Answer-head temperature baseline**
   - No additional training beyond arm 2.
   - Sample `K=1,2,4,8,20` answers from the final symbol distribution at temperature `0.7`.
   - Report exact coverage, duplicate rate, and invalid-sample rate.

### Controls

- Explicit dataset and training seeds.
- Same keeper SHA for both training arms.
- Same optimizer, number of updates, and trainable parameter set.
- Checkpoints at 250-step intervals.
- Full-width synthetic guardrail at every promoted checkpoint.
- Short 16-row-per-depth smoke before the full 128-row-per-depth evaluation.

### Preregistered task gates

The task family is suitable for G-alpha only if:

1. Injective greedy validity is at least `0.90` pooled and at least `0.80` at every depth 1-8.
2. The mixed deterministic arm reaches at least `0.75` pooled valid-answer rate on abductive rows and at least `0.60` at every depth.
3. A promoted checkpoint retains synthetic full-width minimum at or above `0.93`.
4. Exact preimage validation remains green after serialization and tokenization.

Answer-head coverage is a locked comparator, not a pass condition. Its complete K curve is recorded regardless of quality.

### Failure interpretation

- Injective failure means the reverse task is not learnable under this substrate/recipe; do not build Phase G yet.
- Injective success plus abductive low validity means task SFT or target-chain construction needs repair.
- High validity but low coverage is the expected deterministic mode-collapse baseline and is suitable for G-alpha.
- Guardrail failure means checkpoint selection or trainable scope must be tightened before stochastic work.

The answer-head evaluator is implemented in `eval/eval_abductive_coverage.py`.

---

## 8. Experiment 2: G-alpha guided stochastic transition

This experiment begins only after Experiment 1 passes.

### Architecture

- Inject stochasticity at the re-entry bridge/state transition, not inside attention or MLP sublayers.
- Add learned prior `p_theta(epsilon_t | state_t, input)`.
- Add target-conditioned posterior `q_phi(epsilon_t | state_t, input, gold_next_state)`.
- Train with one sampled valid reverse chain on each multimodal row.
- Train from posterior samples; infer from independent prior samples.
- Freeze the deterministic keeper initially. Train only prior, posterior, and stochastic adapter modules.
- No SVGD.
- No learned selector.
- No learned per-trajectory halting.

### Stage gates

1. **K=1 parity:** prior inference retains deterministic valid-answer performance within 2 percentage points.
2. **Posterior/prior transfer:** posterior validity exceeds prior early in training and the gap narrows without KL collapse.
3. **Latent width:** independent `K=2,4,8,20` prior samples increase exact oracle coverage.
4. **Answer-head comparator:** latent sampling beats answer-head temperature sampling at matched K.
5. **Iso-compute comparator:** width beats deterministic extra depth at matched recurrent-transition count.

### Primary success criterion

On held-out abductive rows, latent sampling must improve paired mean exact coverage over both locked comparators, with:

- at least `+0.05` absolute mean coverage;
- paired bootstrap 95% confidence interval excluding zero;
- no more than a 2-point drop in valid-sample rate;
- no regression below the deterministic synthetic guardrail.

### Training diagnostics

- Transition KL by loop, depth, and solution count.
- Prior and posterior variance by loop.
- Posterior/prior mean-distance.
- Posterior-collapse fraction.
- Unique valid candidates by K.
- Invalid and duplicate candidate rates.
- EMA and non-EMA evaluations.
- KL balance initialized near `0.8`, with a small preregistered sweep if collapse occurs.

---

## 9. Experiment 3: G-beta, only after a G-alpha coverage win

If and only if Experiment 2 wins on oracle coverage:

1. Train an LPRM-style latent correctness/value model.
2. Test selector conversion of oracle coverage into top-1 accuracy.
3. Restore per-trajectory adaptive halting.
4. Test SVGD only as an ablation against independent prior sampling.

If Experiment 2 does not win, do not proceed to selector, halting, or SVGD. Diagnose prior/posterior mismatch once, then close or redesign the stochastic transplantation claim.

---

## 10. Immediate execution order

1. Land the explicit RNG fix, abductive/injective generator, gate runner, and exact coverage evaluator.
2. Run the CPU gate and publish manifests.
3. Build and run Experiment 1 on L4.
4. Review the deterministic and answer-head baselines.
5. Implement the posterior/prior transition only if Experiment 1 passes.

This sequence follows the amended gate. It does not spend GPU on selectors, halting, or particle geometry before the model demonstrates a learnable multimodal task and a coverage opportunity.
