# Handoff: Phase G Return, Deterministic Abduction Gate, and Curriculum Recovery

**Date:** July 12, 2026  
**Status:** Deterministic Phase G substrate gate remains blocked after both the valid first run and the bounded curriculum/dose recovery.  
**Recovery artifact commit:** `2c04b3c`  
**Next action:** No automatic training continuation. Run a cheap intermediate-checkpoint/train-split autopsy or preregister one clean curriculum-from-keeper restart after strategy review.

---

## 0. Executive read

The project has now returned to its original GRAM-inspired question under a much cleaner design. The early Gaussian-latent and SVGD experiments did not implement GRAM's target-conditioned posterior/prior training and were run before the recurrent loop was fully repaired. They are therefore evidence about naive noise and particle repulsion only, not evidence against guided stochastic recurrent width.

The deterministic substrate supporting the renewed test is real. On the established forward-transition task, the recurrent model learned a depth-indexed state update and reached strong held-out extrapolation beyond its trained support. It also transferred that transition to several natural-language surfaces while retaining a synthetic guardrail at the selected step-2000 keeper.

The first deterministic abduction run, however, exposed a new and specific bottleneck. After correcting a token-boundary bug, the model learned depth-1 inverse lookup perfectly (`16/16`) but scored only `6/112 = 5.36%` across depths 2-8, essentially the `1/20 = 5%` chance rate. The overall smoke result was `22/128 = 17.19%`, almost entirely explained by perfect depth 1. Thus:

> The model can learn one application of the inverse relation, but the first mixed-depth recipe did not teach it to compose that inverse operation recurrently.

This is not a stochastic-width result. Phase G-alpha remains closed. Because the valid run used only 1,000 updates over 2,048 training rows, disabled the recurrence curriculum, and began immediately at eight-loop compute, a bounded continuation was run from the exact step-1000 checkpoint: 2,000 additional updates with a `2 -> 8` loop curriculum, active-compute ramping, identical frozen rows, exact SHA checks, and deterministic `K=1` gates. It completed training but remained blocked at `26/128 = 20.31%`. The paired change against the first valid run was 10 helped, 6 hurt, and 112 tied (`p=0.4545`), so the apparent four-row gain is not reliable evidence of compositional learning.

The curriculum/dose correction did not establish compositional inverse competence. The strategy agent should now decide whether one clean curriculum restart from the locked keeper is warranted or whether to close this task family after a short train-versus-held-out and intermediate-checkpoint mechanism autopsy.

---

## 1. Scientific destination and corrected GRAM framing

### 1.1 Original project question

The intended system was not merely:

```text
input -> recurrent refinement -> learned depth -> one answer
```

It was:

```text
input -> sampled latent recurrent trajectories -> possibly different paths/depths
      -> several candidate solutions -> selection or verification
```

The intended claims were:

1. identical inputs can follow meaningfully different internal reasoning paths;
2. width can discover valid alternatives that depth alone misses;
3. useful diversity can be converted into better answer selection.

### 1.2 What the early implementation omitted

The early hybrid included Qwen, recurrent depth, PonderNet-style halting, LoRA/bridge training, a state-conditioned Gaussian latent, multi-trajectory sampling, and later SVGD repulsion. It did **not** include the central guidance mechanism verified in GRAM:

- learned conditional prior over stochastic transitions;
- target-conditioned posterior that sees the solution during training;
- posterior-to-prior transition KL;
- prior-only sampling at inference.

GRAM's own ablation supports the corrected interpretation. On N-Queens, full guided stochasticity scores `99.69`, while the stochasticity-only Gaussian variant scores `50.27`; guidance without stochasticity scores `0`. The component omitted here is the component GRAM reports as necessary on a multimodal task.

### 1.3 Current do-not-claim rule

The program has **not** shown that GRAM-style stochastic recurrence fails on Qwen. It has shown that naive Gaussian injection and SVGD repulsion did not reliably create correct-bearing alternatives on an early, partly miswired and barely trained recurrent substrate.

Phase G is therefore a transplantation test, not a GRAM reproduction: can guided stochastic transition learning work in a retrofitted pretrained recurrent language model whose deterministic transition has already been installed and characterized?

Primary references in the repository:

- `docs/gram_divergence_audit_20260711.md`
- `docs/PHASE_G_TRACK_REUNIFICATION_AMENDMENT.md`
- `docs/PHASE_G_ALPHA_GUIDED_STOCHASTIC_TRANSITION_SPEC.md`

---

## 2. Deterministic foundation already established

### 2.1 Recurrent loop repair

The project found and repaired structural issues that confounded earlier recurrence tests, including missing input/prelude re-injection and loop re-entry mismatch. Subsequent gradient-path audits confirmed that active per-loop losses reach the recurrent bridge and block. This changed the evidentiary status of all earlier negatives.

### 2.2 Forward synthetic mechanism

On the N=24 support-12 synthetic chain, the final 6,000-step checkpoint achieved the following active-label diagonal accuracy on frozen rows:

| Depth | Accuracy |
|---:|---:|
| 1 | 1.0000 |
| 4 | 1.0000 |
| 8 | 0.9766 |
| 12 | 0.9766 |
| 13 | 0.9609 |
| 14 | 0.9141 |
| 16 | 0.8594 |
| 17 | 0.8047 |
| 18 | 0.7031 |
| 20 | 0.4609 |
| 22 | 0.1094 |

The locked evaluation classified this as `strong_four_point_law`: training through depth 12 produced substantial held-out competence through depth 20 and remained above chance at depth 22. Artifact checks confirmed 22 observed loop states and 21 bridge re-entry calls.

This establishes a trainable, persistent, depth-indexed recurrent mechanism on a forward transition task. It does not establish arbitrary reasoning or stochastic width.

Artifact:

- `outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json`

### 2.3 Natural-surface transfer and keeper selection

The recurrent transition transferred to relay and pointer language. The original-seed step-2000 checkpoint was selected because it balanced natural transfer and mechanism preservation:

| Metric | Step 2000 |
|---|---:|
| Relay minimum, depths 1-8 | 0.8281 |
| Relay minimum, depths 9-12 | 0.5938 |
| Pointer minimum, depths 1-8 | 0.8125 |
| Pointer minimum, depths 9-12 | 0.5313 |
| Synthetic full-width minimum | 0.9531 |

Keeper SHA256:

```text
0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f
```

Later natural-surface training improved selected cells but damaged the synthetic mechanism. A fresh data/optimization replication also learned the language tasks but produced a non-monotonic capability/guardrail curve. The implication is that checkpoint selection must use frozen behavior gates rather than training loss.

Natural transfer was real but surface-sensitive. At step 2000, unseen-name relay/pointer and passive baton transfer were meaningful, while fronted baton collapsed. Loop index remained perfectly decodable and late-loop state norms/envelope error grew sharply, showing structured recurrent time but not a stable fixed-point dynamic.

Artifact summary:

- `docs/phase_g_next_experiments_20260712.md`
- `outputs/stage5/stage5_natural_surface_followups_2_3_20260710/summary.json`
- `outputs/stage5/stage5_natural_surface_replication_dose_seed931337_20260710/summary.json`

---

## 3. Phase G task and evaluation preparation

### 3.1 Why abduction is the gate

Width has little to measure on a unique-answer task. The Phase G task family therefore pairs:

- **injective control:** exactly one depth-step preimage;
- **non-injective abduction:** exactly 2-4 valid preimages in the constructive N=20 screen;
- **arbitrary N=24 calibration/test:** exact preimage strata `1`, `2-4`, and `>=5`.

The injective arm asks whether the deterministic substrate can perform backward inference at all. The abductive arm asks whether it can produce valid answers while a deterministic model is expected to collapse toward one mode. Only then does oracle coverage at K become attributable to stochastic exploration.

### 3.2 Frozen N=20 screen

All rows store the full mapping, exact valid starts, selected valid chain, and per-loop reverse-chain labels. Exhaustive forward recomputation verifies every denominator.

| Split | Rows | Depths | Solution count |
|---|---:|---:|---:|
| Train injective | 2,048 | 1-8 | exactly 1 |
| Test injective | 1,024 | 1-8 | exactly 1 |
| Train abductive | 2,048 | 1-8 | balanced 2-4 |
| Test abductive | 1,024 | 1-8 | balanced 2-4 |

The corrected row hashes are frozen and tested for resume compatibility.

### 3.3 Frozen N=24 claim sets

Calibration and test each contain 128 rows in each exact-preimage stratum:

- unique: exactly 1;
- small: 2-4;
- large: at least 5.

Calibration and test IDs are disjoint. Exact denominators are recomputed by applying the map from all 24 starts. The test split remains downstream of deterministic calibration competence and preregistration completion.

### 3.4 Locked Phase G comparators

1. Entropy-matched answer-head sampling at matched K.
2. Deterministic extra depth at matched recurrent-transition count.

Primary outcome: paired per-instance exact oracle coverage at `K=1,2,4,8,20`, stratified by exact preimage count.

Deferred until a G-alpha coverage win:

- LPRM/selector;
- per-trajectory learned halting;
- SVGD, strictly as an ablation.

Preparation status: green. The only preregistration blank is the powered numeric margin derived from the calibration split.

Artifacts:

- `outputs/stage5/stage5_phase_g_alpha_prepare_20260712/summary.json`
- `data/phase_g_alpha/manifest.json`
- `data/phase_g_alpha/preregistration.json`

---

## 4. Experiment 1A: invalid first launch

### 4.1 Intended experiment

Train the deterministic recurrent block and bridge on exact injective reverse chains, then require:

- smoke greedy validity at least `0.50`;
- full pooled greedy validity at least `0.90`;
- every depth 1-8 at least `0.80`.

No latent, learned halting, LPRM, or SVGD module was enabled.

### 4.2 Failure

The first launch at commit `98dfb8a` performed 1,000 no-op updates:

```text
loss = 0
per_loop_label_active = 0
all trainable gradients = 0
```

The generated prompt ended in `"Answer: "` while the completion lacked the leading space. Qwen boundary tokenization made the prompt token length equal the full sequence token length, so every completion token was masked.

The resulting `9/128` smoke score is an untrained baseline and is inadmissible as a model result.

### 4.3 Repair

- Canonical boundary changed to `"Answer:" + " Name"`.
- Real-tokenizer preflight confirmed one active outcome token and one active loop token.
- Training now aborts before paid compute if active supervision is zero.
- Training aborts after backward if every trainable gradient is zero.
- All 24 names are one completion token under the Qwen tokenizer.

Artifact:

- `outputs/stage5/stage5_phase_g_experiment1_20260712/invalid_training_diagnosis.md`

---

## 5. Experiment 1B: valid fixed-boundary injective run

### 5.1 Configuration

| Item | Value |
|---|---|
| Initialization | locked natural-surface step-2000 keeper |
| Keeper SHA | `0f657b...d2f` |
| Training rows | 2,048 injective rows |
| Depths | 1-8, balanced |
| Updates | 1,000 |
| Batch size | 1 |
| Optimizer | AdamW |
| Main LR | `1e-5` |
| Bridge-prelude LR multiplier | `10` |
| Loop loss | per-loop exact reverse-chain labels |
| Forward loops | fixed at 8 |
| Recurrence curriculum | disabled |
| Trainable | recurrent block plus bridge, about 182.2M parameters |
| Stochastic modules | all disabled |

Checkpoint SHA256:

```text
0d6cf119bd66290a2c85686bf58fdc6f9363109c8fdae0ea625f32d13409a1a6
```

### 5.2 Training validity

The repaired assertions passed:

```text
[assert-ok] active_supervision={'outcome_tokens': 1, 'loop_tokens': 1, 'active_loops': 1}
```

At step 0:

```text
loss = 18.1928
per_loop_label_active = 6
bridge prelude gradient RMS before clipping = 0.0554
bridge state gradient RMS before clipping = 0.2573
```

Loss subsequently varied with sampled depth and example, generally remaining around `2-3` on deeper rows. A depth-1 sample reached approximately zero loss at step 500, while deeper examples remained materially nonzero. The run was therefore active but not converged across the mixed-depth distribution.

### 5.3 Deterministic smoke result

| Depth | Correct / 16 | Greedy accuracy |
|---:|---:|---:|
| 1 | 16 / 16 | 1.0000 |
| 2 | 1 / 16 | 0.0625 |
| 3 | 0 / 16 | 0.0000 |
| 4 | 1 / 16 | 0.0625 |
| 5 | 1 / 16 | 0.0625 |
| 6 | 0 / 16 | 0.0000 |
| 7 | 3 / 16 | 0.1875 |
| 8 | 0 / 16 | 0.0000 |
| **Overall** | **22 / 128** | **0.1719** |

The key decomposition is:

```text
depth 1:    16/16  = 100%
depths 2-8: 6/112 = 5.36%
N=20 chance:       = 5.00%
```

This is a sharp one-step/composition split. It is not a diffuse partial failure.

The preregistered smoke floor was `0.50`, so the script recorded `blocked_injective_smoke` and intentionally returned exit code `2`. The notebook's `CalledProcessError` was the wrapper surfacing that scientific stop, not a runtime crash.

### 5.4 Provisional answer-head sampling readout

Although the task has one valid answer per row, temperature sampling sometimes recovered it:

| K | Full-coverage rate | Valid-sample rate | Duplicate rate |
|---:|---:|---:|---:|
| 1 | 0.1797 | 0.1797 | 0.0000 |
| 2 | 0.2344 | 0.1992 | 0.1602 |
| 4 | 0.2969 | 0.1836 | 0.2637 |
| 8 | 0.3984 | 0.1816 | 0.4092 |
| 20 | 0.5703 | 0.1801 | 0.6078 |

This curve says the correct symbol often remains somewhere in the output distribution even when greedy recurrent composition fails. It does **not** establish latent width, multimodal coverage, or useful reasoning diversity. The final comparator must be entropy-matched on the frozen N=24 set.

Artifact:

- `outputs/stage5/stage5_phase_g_experiment1_fixed_boundary_20260712/summary.json`

---

## 6. Interpretation

### 6.1 What is established

1. The corrected task data carry active supervision.
2. Per-loop losses reach the bridge and recurrent block.
3. The model can parse the forward table and learn one inverse lookup.
4. The first recipe did not teach repeated inverse application.
5. The deterministic substrate gate remains closed, so no GRAM-style stochastic training is authorized.

### 6.2 Most likely explanation

The initial recipe asked the model to solve balanced depths 1-8 immediately with fixed eight-loop computation. It omitted the staged recurrence curriculum that was important in the successful forward-chain program. It also used only 1,000 updates for 2,048 rows, less than half an update per row on average before repeats are considered.

The most economical explanation is therefore underdeveloped compositional recurrence, not graph disconnection:

- depth 1 proves basic inverse lookup is learnable;
- active gradients rule out the prior masking failure;
- correct reverse-chain ordering was statically and unit tested;
- chance-level depths 2-8 localize the missing behavior to reapplication of the inverse operator.

### 6.3 Alternative explanations still live

1. **Representation mismatch:** the state emitted after one inverse lookup may not be in a form that supports applying the same lookup again.
2. **Curriculum failure:** fixed eight-loop compute may let depth-specific shortcuts dominate before a reusable local inverse transition forms.
3. **Insufficient dose:** 1,000 updates may simply be inadequate for the larger and linguistically heavier table task.
4. **Generalization failure:** training rows may improve while unseen tables remain at chance; no train-split depth matrix was run before the smoke stop.
5. **Task complexity jump:** inverse lookup requires searching the rendered forward table, unlike the earlier forward transition, so successful forward recurrence does not guarantee backward recurrence.

### 6.4 What should not be inferred

- Guided stochastic width failed.
- GRAM does not transplant to Qwen.
- The repaired recurrent loop cannot compose any operation.
- Answer-head sampling is a substitute for latent trajectories.
- The smoke threshold should be weakened after seeing the result.

---

## 7. Completed recovery experiment

### 7.1 Purpose

Exercise one predeclared dose/curriculum lever before accepting a deterministic substrate negative.

### 7.2 Locked changes

| Variable | First valid run | Recovery |
|---|---|---|
| Initialization | locked keeper | exact step-1000 injective checkpoint |
| Additional updates | 1,000 initial | 2,000 continuation |
| Compute curriculum | none, fixed 8 | linear 2 -> 8 |
| Label target | row depth | row depth capped by scheduled loop count |
| Compute ramp | off | on |
| Data seed/hashes | frozen | unchanged |
| Stochastic modules | disabled | disabled |
| Gate sampling | K=1,2,4,8,20 | deterministic K=1 only |

The parent checkpoint is restored from Drive and must match SHA `0d6cf119...a1a6`. The curriculum target is exposed as `phase_g_injective_curriculum_recovery`. Full CPU validation before push: `1,758 passed`.

### 7.3 Decision behavior

1. Train for 2,000 additional updates under the 2->8 curriculum.
2. Run the same 16-row-per-depth injective smoke with deterministic K=1 scoring.
3. If smoke remains below `0.50`, record a blocked result and stop.
4. If smoke passes, run the full injective gate: pooled at least `0.90` and every depth at least `0.80`.
5. If injective passes, continue to the independently initialized mixed abductive arm and its locked gate.
6. Promote nothing unless the synthetic guardrail remains at least `0.93`.

### 7.4 Why continuation rather than restart

Continuation conserves GPU and retains a useful learned primitive: perfect depth-1 inversion. The curriculum then tests whether the same parameters can organize that primitive into a repeated transition.

The causal limitation is that a failed continuation would not prove a clean curriculum-from-keeper restart also fails. That is a strategy decision, not something to run automatically.

### 7.5 Landed result

Training completed all 2,000 additional updates and backed up the final checkpoint before evaluation. The final checkpoint SHA256 is:

```text
fc98feb5d5bd450f7ecc4f6d43ce36fd436418d7ad2cd69df38a089d5ec453d1
```

| Depth | First valid run | Curriculum recovery | Delta correct / 16 |
|---:|---:|---:|---:|
| 1 | 16/16 | 16/16 | 0 |
| 2 | 1/16 | 4/16 | +3 |
| 3 | 0/16 | 2/16 | +2 |
| 4 | 1/16 | 0/16 | -1 |
| 5 | 1/16 | 0/16 | -1 |
| 6 | 0/16 | 1/16 | +1 |
| 7 | 3/16 | 0/16 | -3 |
| 8 | 0/16 | 3/16 | +3 |
| **Overall** | **22/128 (17.19%)** | **26/128 (20.31%)** | **+4** |

On identical rows, the recovery helped 10, hurt 6, and tied 112. The exact two-sided sign-test p-value is `0.4545`. Across depths 2-8, recovery achieved `10/112 = 8.93%`; against an idealized independent N=20 chance rate of 5%, the one-sided binomial tail is approximately `0.054` before any correction. The depth pattern is non-monotonic and includes complete losses at depths 4, 5, and 7. This is best read as redistributed uncertainty, not a learned repeated inverse operator.

The notebook's final `CalledProcessError` again wrapped the runner's intentional exit code `2` after it wrote `blocked_injective_smoke`. It did not interrupt training.

---

## 8. Decision tree after the recovery

### Outcome A: injective gate passes

Proceed in order:

1. mixed injective/abductive deterministic training;
2. abductive pooled validity `>=0.75` and every depth `>=0.60`;
3. synthetic guardrail `>=0.93`;
4. arbitrary N=24 calibration competence;
5. calibration-split power calculation and numeric margin lock;
6. G-alpha with frozen deterministic block.

### Outcome B: smoke improves materially but remains below 0.50

Do not weaken the gate. First inspect:

- depth-wise train accuracy;
- held-out depth matrix;
- earliest depth at which performance returns to chance;
- intermediate-label accuracy by loop;
- state similarity between one inverse output and the next loop's expected input manifold.

Then decide whether the evidence warrants one clean restart from the keeper with curriculum active from step 0 and a dose near the established program rate.

### Outcome C: depth 1 remains perfect and depths 2-8 remain at chance

Treat this as a mechanistic composition failure. The highest-value autopsy is a small repeated-inverse micro-test:

- one or a few fixed tables;
- exact loop-by-loop labels;
- verify train overfit at depths 1, 2, 4, and 8;
- probe whether the predicted predecessor at loop k becomes the effective query at loop k+1.

If even the micro-test cannot overfit, the backward task is structurally mismatched to the recurrent state update or decode/re-entry pathway. Phase G should remain closed and this task family should be redesigned or abandoned.

### Outcome D: train depth composition succeeds but held-out tables fail

This is task generalization failure, not recurrence failure. Options include:

- more independently generated tables;
- stronger table-format randomization;
- explicit inverse-relation pretraining;
- moving to arbitrary N=24 deterministic continuation before stochastic heads.

Any added training must retain the frozen forward synthetic guardrail.

---

## 9. G-alpha design once the gate opens

### 9.1 Frozen architecture

Freeze the deterministic recurrent-Qwen substrate. Train only:

- conditional prior head;
- target-conditioned posterior head;
- small scalar/vector injection scale.

Assert frozen block parameters are non-trainable and receive zero gradients after every backward pass.

### 9.2 Transition

Inject stochasticity only at the high-level re-entry state:

```text
u_t = deterministic recurrent update(h_t, x)
epsilon_t ~ posterior during training, prior during inference
h_(t+1) = u_t + scale * epsilon_t
```

The posterior conditions on the gold next intermediate state. On multimodal rows, one valid chain is sampled uniformly and stored. This uses supervision unavailable in GRAM's terminal-target setup and must be treated as a design contribution, not a reproduction detail.

### 9.3 Objective and diagnostics

- per-loop chain/outcome loss;
- per-loop posterior-to-prior KL;
- KL balance initialized near `0.8`;
- coefficient sweep `1e-4`, `1e-3`, `1e-2`;
- EMA `0.999` and raw-weight evaluation;
- KL, means, variance, and collapse rate by loop/depth/solution count.

### 9.4 Success hierarchy

1. K=1 validity parity with deterministic keeper.
2. Increasing unique valid solutions with K.
3. Coverage win over entropy-matched output sampling.
4. Coverage win over deterministic depth at matched transitions.
5. Only then open G-beta selection, adaptive halting, and SVGD ablation.

---

## 10. Questions for the strategy and research agent

### Immediate recovery decision

1. Is continuation from the depth-1-competent checkpoint the right causal test, or should a clean curriculum-from-keeper restart be preregistered now as the only follow-up if continuation fails?
2. Is 2,000 additional updates sufficient? Total nominal dose becomes 3,000 updates over 2,048 rows (`1.46` updates/row), below the program's earlier approximate `1.95` steps/row convention. Should the recovery be extended to 3,000 additional updates before launch, or should 2,000 remain the bounded first look?
3. Should checkpoint evaluations be added at +500, +1000, and +2000 to distinguish late emergence from instability, despite extra inference cost?

### Mechanism interpretation

4. Does the exact `depth 1 = 100%`, `depths 2-8 = chance` pattern most strongly indicate curriculum failure, state-format mismatch after one inverse step, or insufficient task exposure?
5. What minimal probe best tests whether the first predicted predecessor becomes the next loop's effective query without conflating decode accuracy and hidden-state transition quality?
6. If a fixed-table micro-test overfits but random-table generalization fails, should the task be reframed as an inverse-operator pretraining problem rather than a recurrence problem?

### Gate and task family

7. Are the locked `0.90` pooled / `0.80` per-depth injective gates appropriately calibrated for substrate competence, or should their scientific role be clarified while leaving their values unchanged?
8. Should mixed abductive training initialize independently from the keeper, as currently designed for clean arm comparison, or from the passed injective checkpoint to isolate multimodality after inverse competence?
9. Does constructive N=20 competence remain a necessary screen before arbitrary N=24, or could the constructive generator encourage shortcuts that make it a weak predictor of arbitrary-table competence?

### Phase G

10. Should the posterior condition on the gold next symbolic embedding, the gold next hidden-state target, or both?
11. What prior/posterior capacity is large enough to represent multimodal transition residuals without becoming a second unfrozen reasoner?
12. How should entropy matching be defined when answer-head support includes invalid names but latent trajectories can fail through invalid intermediate states?
13. Should iso-compute compare recurrent transitions only, or also report sequential latency, batch-parallel latency, and total FLOPs separately?
14. What minimum validity floor should accompany coverage gains so stochastic width cannot win by spraying mostly invalid candidates?

---

## 11. Recommended position

The program should continue with the single bounded curriculum recovery. The corrected run is informative but underpowered as a negative because it omitted the recurrence curriculum and used less than one pass-equivalent of updates. The result is nevertheless valuable: it identifies exactly what the next experiment must repair and prevents premature construction of stochastic heads on a substrate that only performs one-step inversion.

Do not resume SVGD, latent noise, selector work, or adaptive halting yet. Do not interpret the K=20 answer-sampling recovery as evidence that stochastic width already works. Preserve the locked gates and frozen test sets.

If the recovery establishes deterministic inverse composition, proceed directly through mixed abductive competence and arbitrary N=24 calibration into G-alpha. If it fails without any improvement beyond depth 1, pause for the micro-test/state-query autopsy and decide whether one clean curriculum restart has sufficient expected value. That decision should be made before further GPU spend.

---

## 12. Artifact and commit ledger

| Item | Location / commit |
|---|---|
| GRAM divergence audit | `docs/gram_divergence_audit_20260711.md` |
| Track amendment | `docs/PHASE_G_TRACK_REUNIFICATION_AMENDMENT.md` |
| G-alpha architecture spec | `docs/PHASE_G_ALPHA_GUIDED_STOCHASTIC_TRANSITION_SPEC.md` |
| Prior experiment plan | `docs/phase_g_next_experiments_20260712.md` |
| N=24 preparation | `outputs/stage5/stage5_phase_g_alpha_prepare_20260712/summary.json` |
| Invalid first attempt diagnosis | `outputs/stage5/stage5_phase_g_experiment1_20260712/invalid_training_diagnosis.md` |
| Valid fixed-boundary result | `outputs/stage5/stage5_phase_g_experiment1_fixed_boundary_20260712/summary.json` |
| Boundary/data/evaluator repair | `bcd3028` |
| Valid run artifacts | `a0e42fc`, `e14c4a8`, `57f377c` |
| Curriculum recovery implementation | `d705ba8` |
