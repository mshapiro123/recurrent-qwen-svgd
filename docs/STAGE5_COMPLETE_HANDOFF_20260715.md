# Handoff: Recurrent-Qwen Deterministic Evidence, Inverse-Task Boundary, and the Phase G Decision

**Date:** July 15, 2026  
**Project:** `mshapiro123/recurrent-qwen-svgd`  
**Status:** Strategy pause after the single authorized W4 deterministic continuation  
**Repository state covered:** through commit `8011479`  
**Audience:** research strategy, architecture, curriculum, and implementation reviewers

---

## 0. Executive disposition

The program has established a real deterministic recurrent mechanism and a strong, narrowly scoped synthetic result. A recurrent Qwen2.5-0.5B system reached `1506/1792 = 84.04%` on a frozen depth-1-through-14 composition family, compared with `952/1792 = 53.13%` for the strongest evaluated dense 0.5B serialized-scratchpad control. The recurrent advantage was largest beyond depth 10: `272/512 = 53.13%` versus `56/512 = 10.94%`. The comparison is paired on identical rows and strongly significant. It is a system-level synthetic result, not yet a matched-training-lineage, matched-FLOP, natural-reasoning, or architecture-only claim.

The current inverse-task curriculum did not provide a competence-preserving path to the deterministic substrate required for guided stochastic width. Three related branches have now closed under their registered budgets:

1. Canonical forward-table reverse composition stalled at depth 2 under matched dose.
2. Explicit inverse-table composition reached depth 3, but either lost synthetic recurrence or crossed the natural-surface hard stop as training progressed; a four-checkpoint Pareto sweep found no joint-passing checkpoint.
3. The re-based inverse-rendered N=24 task nearly passed zero-shot calibration (`276/384` versus `288/384`) but failed depth 4 and retention. Its single authorized 200-step continuation made every decisive measure worse: calibration fell to `208/384`, synthetic retention to `0.125`, and the natural canary by `21.875` percentage points.

Therefore:

- Phase G-alpha remains **closed**.
- The held-out inverse-rendered test split remains **unopened**.
- Cap 4 remains **unauthorized**.
- The cap-3 symbolic-rehearsal branch is **closed**.
- The multi-channel bridge proposal remains **banked**, not active.
- No additional inverse-rendered dose run is authorized without a new strategy decision and a materially different causal hypothesis.

This is not evidence against GRAM-style guided stochastic width. That experiment has not been run. The early particle/noise experiments lacked the target-conditioned posterior that GRAM's own ablations identify as essential, and some predated correction of the recurrent loop closure. The correct conclusion is that the present deterministic substrate gate failed, so the principled width experiment remains gated.

The immediate next action is a strategy review, not another automatic GPU launch. The review must choose between:

- one remaining curriculum-science localization experiment, the loop-position transfer test, followed conditionally by a corrected primitive-first forward-table arm; or
- selecting a different deterministic multimodal substrate on which to test G-alpha without requiring the current model to learn canonical inverse search and preserve all existing capabilities in one small continuation.

In parallel, the deterministic mechanism and Phase A synthetic-surpass evidence should be consolidated for the first paper.

---

## 1. Original thesis and present claim boundary

### 1.1 Original deterministic proposal

The project began with a pretrained Qwen backbone converted into an adaptive recurrent-depth model:

```text
input
  -> recurrent latent loop
  -> learned halting depth
  -> one latent trajectory
  -> one answer
```

The intended contribution was economical latent computation: reuse a middle transformer block, preserve a persistent hidden state, and learn how much recurrent depth a problem requires.

### 1.2 GRAM-inspired extension

The intended breadth extension was:

```text
input
  -> learned stochastic recurrent transition
  -> K independently sampled latent trajectories
  -> possibly different depths
  -> K candidate answers
  -> oracle, consistency rule, or verifier
  -> selected answer
```

The hybrid design combined:

- Qwen as the pretrained language backbone;
- a recurrent bridge and reused middle block;
- PonderNet-style probabilistic halting;
- economical adaptation through LoRA and auxiliary heads;
- a variational latent transition;
- parallel trajectory sampling and, initially, SVGD-style anti-collapse pressure.

It was described as GRAM-inspired rather than a reproduction. The project had not reproduced GRAM's exact posterior, objective, transition hierarchy, or training recipe.

### 1.3 What the GRAM audit changed

The verified GRAM paper uses a learned conditional prior and a target-conditioned posterior. Training guides stochastic trajectories toward solution-bearing paths through posterior-to-prior KL terms; inference samples independent trajectories from the prior. GRAM's own ablation is directly relevant: stochastic Gaussian noise without learned guidance collapsed on the multimodal N-Queens task (`50.27` versus `99.69` for full GRAM).

Our early stochastic work was closest to that insufficient ablation. It used naive latent sampling and particle geometry on a weak checkpoint, lacked target-conditioned posterior guidance, and partly predated the loop-closure repair. Those runs establish that naive noise and repulsion did not create correct-bearing alternatives. They do **not** test the GRAM mechanism.

The program therefore redefined Phase G-alpha as a clean transplantation test:

- freeze a deterministic keeper;
- train only a conditional prior, target-conditioned posterior, and injection scale;
- inject stochasticity at the high-level re-entry state;
- require K=1 parity first;
- compare latent-K with entropy-matched output-head sampling;
- compare width with deterministic depth at matched recurrent-transition budget;
- score exact distinct-valid coverage on a multimodal task.

### 1.4 Current implemented system

The current positive deterministic system is substantially different from the original economical proposal:

- corrected prelude/input re-injection on every recurrent pass;
- a trainable bridge and re-entry path;
- a fully unfrozen 12-layer recurrent block, approximately 179M block parameters plus the bridge;
- fixed or supervised loop counts for the mechanism experiments;
- exact intermediate symbolic targets;
- one deterministic trajectory;
- no active latent sampling, learned halting objective, particle update, or SVGD in the current training line.

The strongest evidence therefore concerns **trainable deterministic latent recurrence after substantial recurrent-block adaptation**. Any claim that only a tiny fraction of parameters was updated, or that the current results already demonstrate stochastic width, would be incorrect.

---

## 2. Architecture integrity: why the later evidence is admissible

### 2.1 Initial structural confounds

The recurrent surgery initially contained two serious problems:

- the bridge path was effectively dead in relevant checkpoints (`bridge_gate=0`, zero bridge delta, and no useful bridge gradient path);
- recurrent passes did not correctly re-inject the prelude/input representation.

The missing prelude re-injection was corrected in commit `96efe39`. This materially changes interpretation of all earlier stochastic and recurrence failures: the pre-repair loop was not the intended grounded recurrent computation.

### 2.2 Re-entry repair result

The bounded repair run `stage5_reentry_repair_smoke_20260625_114554` established:

- bridge gate movement from `0.0` to approximately `1.0002`;
- live bridge and re-entry-adapter gradients;
- measurable bridge and adapter movement;
- no regression of the loop-1 smoke behavior.

The follow-on recovery `stage5_reentry_recovery_20260625_114836` reported `reentry_health_sane`, with loop-8 output/input RMS approximately `1.0061`. This resolved the immediate loop-closure liveness and norm-stability concern.

### 2.3 What remains architectural versus curricular

Later experiments showed that the repaired substrate can learn persistent state transitions, survive removal of intermediate supervision, and extrapolate beyond some trained depth support. This makes a global "recurrence is nonfunctional" explanation untenable.

The current failures are better localized to:

- operation direction and retrieval cost;
- per-loop supervision allocation;
- capability retention under narrow symbolic continuation;
- task/substrate compatibility;
- possible position-specific transition installation.

The multi-channel bridge precursor did not confirm a sufficiently head-aligned specialization signal, so no additional bridge architecture is presently authorized.

---

## 3. Established positive: deterministic recurrent depth

### 3.1 Synthetic chain mechanism

On the exact synthetic state-transition family, the corrected recurrent system learned a staged chain rather than merely memorizing final labels. A representative post-anneal active-label result was `625/640 = 97.7%` across trained depths 1-4 after the final 1,000 steps used outcome-only supervision. Above the target depth it usually continued applying the learned update (`357/384 = 93.0%`) rather than holding the answer (`1/384`).

This demonstrated:

- a trainable recurrent state update;
- persistence after intermediate-chain supervision was removed;
- genuine loop-indexed latent computation;
- a finite-horizon mechanism that could later be extended with additional support.

It did not by itself establish natural reasoning gains or indefinitely algorithmic recurrence.

### 3.2 Phase A design

The Phase A comparison used one frozen set of `1792` rows:

- depths 1-14;
- `128` rows per depth;
- identical row IDs across all systems;
- same final-symbol answer space;
- step-4000 dense checkpoints fixed before the final comparison.

The systems were:

| Arm | System | Training/evaluation role |
|---|---|---|
| A | Recurrent Qwen2.5-0.5B | Same-reader recurrent system at forced loop depth |
| B | Dense Qwen2.5-0.5B | Direct-answer SFT control |
| C | Dense Qwen2.5-0.5B | Serialized-scratchpad SFT control |
| D | Dense Qwen2.5-1.5B | Direct-answer scale control |

The comparison matched model family and frozen rows. It did not match total lineage, optimizer history, tokens, wall-clock, FLOPs, or exact inference compute.

### 3.3 Phase A results

| Arm | Correct | Accuracy | Depths 11-14 | Tail accuracy |
|---|---:|---:|---:|---:|
| A, recurrent 0.5B | `1506/1792` | `84.04%` | `272/512` | `53.13%` |
| B, dense 0.5B direct | `470/1792` | `26.23%` | `60/512` | `11.72%` |
| C, dense 0.5B scratchpad | `952/1792` | `53.13%` | `56/512` | `10.94%` |
| D, dense 1.5B direct | `322/1792` | `17.97%` | `58/512` | `11.33%` |

The preregistered primary comparison was A versus B. A passed the one-sided Fisher gate at all 14 depths. The stronger A-versus-C result is a labeled analysis extension and also passed at all 14 depths.

Paired row-level results:

| Comparison | Helped | Hurt | Tied | Net correct | Two-sided exact p |
|---|---:|---:|---:|---:|---:|
| A vs B | `1074` | `38` | `680` | `+1036` | `2.12e-264` |
| A vs C | `607` | `53` | `1132` | `+554` | `3.42e-120` |

Dense checkpoint saturation was also measured. From step 2000 to 4000:

- B: `+6` net rows, paired `p=0.771`;
- C: `+22` net rows, paired `p=0.00319`;
- D: `-28` net rows, paired `p=0.161`.

The direct 1.5B recipe underperformed here; that is a result about this recipe, not a general claim that 0.5B models outperform 1.5B models.

### 3.4 Permitted Phase A claim

> On this frozen synthetic composition family, the trained recurrent 0.5B system outperformed the evaluated dense 0.5B direct and serialized-scratchpad recipes, with the largest advantage beyond the scratchpad control's observed depth horizon.

Not permitted:

- superiority on GPQA, ARC-AGI, mathematics, coding, or broad natural reasoning;
- recurrence-only causality under matched training lineage;
- matched-FLOP or matched-latency superiority;
- superiority to a well-tuned 1.5B or 3B reasoning system;
- evidence for stochastic latent width.

---

## 4. Why deterministic inverse competence became the Phase G gate

Width is meaningful only where multiple valid solutions exist. The selected assay used non-injective mappings with exact preimage sets, allowing exact oracle coverage denominators. But stochastic exploration cannot be credited with solving a task the deterministic keeper cannot perform. Deterministic competence was therefore a prerequisite, as in GRAM's comparisons against deterministic models that already learn their tasks.

Two related task surfaces were separated:

- **Canonical forward-table abduction:** render the forward mapping and require repeated reverse lookup. This is the stronger non-native reasoning test.
- **Inverse-rendered abduction:** render the inverse relation explicitly, turning each transition into a forward-style lookup while preserving non-injective multimodality. This is a narrower but cleaner potential substrate for the first width assay.

These were assigned to separate lanes so an inverse-rendered width result could not be misrepresented as canonical backward-search competence.

---

## 5. Curriculum autopsy: the hidden partial inverse-composition gain

### 5.1 Initial deterministic screen and recovery

The first valid injective inverse run trained for 1,000 updates:

- depth 1: `16/16`;
- depths 2-8: `6/112`;
- total: `22/128`.

A 2,000-step continuation improved the final diagonal only to `26/128`; the paired delta was 10 helped, 6 hurt, 112 tied (`p=0.4545`). Final-diagonal scoring initially made the continuation look almost inert.

### 5.2 Active-loop autopsy

The read-only active-loop matrix showed a real partial staircase:

| Split / checkpoint | L1 | L2 | L3 | L4-L8 |
|---|---:|---:|---:|---:|
| Held out, step 1000 | `121/128` | `5/112` | `1/96` | unsupported |
| Held out, recovery | `125/128` | `42/112` | `10/96` | unsupported |

Paired recovery effects:

- held-out L2: 40 helped, 3 hurt, `p=3.02e-9`;
- held-out L3: 10 helped, 1 hurt, `p=0.0117`.

The continuation had learned inverse transitions at loops 2 and 3, but later loops remained unsupported. The final diagonal hid this because a depth-d answer requires all transitions through loop d.

### 5.3 Exposure audit

The linear 2-to-8 curriculum produced sharply decreasing raw active labels:

```text
L1 2000, L2 1749, L3 1377, L4 950,
L5 594, L6 310, L7 114, L8 19
```

After per-row active-label averaging, loop 8 received only about `2.4` full-row-equivalent CE weight during recovery. The learned staircase tracked this dose gradient. This justified a matched, loop-balanced experiment rather than another nominal-step extension.

---

## 6. Matched inverse-composition staircase

### 6.1 Design

The experiment used the same mappings, chains, keeper, optimizer, held-out rows, and active-label accounting in two arms:

- Arm C: explicitly rendered inverse table, making each recurrent transition a forward lookup.
- Arm F: forward table, requiring reverse retrieval and recurrent composition.

Locked settings included:

- AdamW;
- effective batch size 8;
- 250 optimizer steps at cap 2;
- newest-loop emphasis of 2x;
- inverse-exposure loop weights;
- approximately 1,500 newest-loop weighted labels;
- task gate `46/64`;
- synthetic active-diagonal floor `0.93`;
- stage advance only on gate pass.

Muon was deliberately excluded from this causal comparison.

### 6.2 Results

| Arm | Cap-2 task | Weighted newest-loop labels | Synthetic minimum | Decision |
|---|---:|---:|---:|---|
| C, inverse rendered | `62/64` | `1598.4` | `0.9375` | Advanced |
| F, forward rendered | `3/64` | `1603.2` | `0.21875` | Blocked |

Arm F's first inverse transition reached `55/64`, but the second reached only `3/64`. The failure therefore localized to repeated reverse composition under this curriculum, not failure to perform any inverse retrieval.

The correct verdict is `experiment_stalled_at_matched_dose`. The experiment did not produce a finite F-to-C dose ratio and does not justify the stronger claim that a fixed five-fold position cost was measured.

---

## 7. Explicit inverse-table rebase and retention trade-off

### 7.1 Cap-3 rebase

Starting from exact cap-2 checkpoint SHA:

```text
bc1de1cd7d2a7acf30b9217c8d7054d805888c341b942ff0dab7691b4f995b01
```

the rebase trained cap 3 with equalized active-label mass and 2x newest-loop emphasis. At step 250 it reached:

- cap-3 task: `63/64`;
- conditional third-transition success: `63/64`;
- source checkpoint SHA for later branches:

```text
83767ebff2c2a13a2f15fe8266f605fb8485985c3289c1f1720cd70c122a9ac5
```

But synthetic retention fell to `0.8125`, below the `0.93` floor. Cap 4 did not run.

### 7.2 Fixed 25% rehearsal replay

The preregistered rehearsal test asked whether 25% forward-synthetic rehearsal could retain the installed inverse task while restoring prior recurrence. At step 334:

- inverse task: `64/64`;
- synthetic minimum: `0.96875`;
- natural canary: `212/256 = 82.81%`;
- locked natural baseline: `227/256 = 88.67%`;
- natural delta: `-5.859` percentage points;
- hard-stop margin: `-3` percentage points.

The source cap-3 checkpoint itself scored `222/256 = 86.72%`, so rehearsal added another `-10/256 = -3.906` percentage points of natural harm beyond the inherited source regression.

### 7.3 Checkpoint Pareto sweep

Every saved checkpoint was evaluated on the same task, synthetic, and natural gates:

| Step | Task | Synthetic minimum | Natural delta | Joint gate |
|---:|---:|---:|---:|---|
| 100 | `61/64` | `0.90625` | `+0.391 pp` | No: synthetic fail |
| 200 | `62/64` | `0.8125` | `-10.156 pp` | No: synthetic and natural fail |
| 300 | `63/64` | `0.9375` | `-3.125 pp` | No: natural hard stop |
| 334 | `64/64` | `0.96875` | `-5.859 pp` | No: natural hard stop |

No checkpoint jointly passed. This closes the cap-3 symbolic-rehearsal branch. The data show a task-versus-retention Pareto path, but the path does not cross the permitted region.

---

## 8. Re-based inverse-rendered N=24 validity assay

### 8.1 Frozen-set construction

The deterministic Phase G prerequisite was re-based to a narrower task that explicitly renders the inverse relation while preserving multimodality.

For each split:

- `N=24` arbitrary non-bijective mappings;
- depths 1-4;
- `384` rows total;
- `96` rows per depth;
- `128` rows in each exact-preimage stratum:
  - unique: exactly 1 preimage;
  - small: 2-4;
  - large: 5 or more;
- exact predecessor chains and valid-preimage sets stored in the manifest;
- disjoint calibration and test row IDs;
- chain validity recomputed against the relation.

Locked deterministic gates:

- pooled calibration validity at least `288/384 = 75%`;
- every depth at least `58/96 = 60.42%`;
- synthetic retention at least `30/32` at every guarded depth;
- natural canary must avoid the `-3 pp` hard stop;
- test split remains unopened until all calibration and retention gates pass.

### 8.2 W3 zero-shot gate

Checkpoint SHA `83767ebf...9ac5` was evaluated without training:

| Depth | Result | Gate | Pass |
|---:|---:|---:|---|
| 1 | `95/96` | `58/96` | Yes |
| 2 | `82/96` | `58/96` | Yes |
| 3 | `61/96` | `58/96` | Yes |
| 4 | `38/96` | `58/96` | No |
| Pooled | `276/384` | `288/384` | No |

The result was materially competent but below gate, authorizing exactly one bounded deterministic continuation. Its inherited synthetic retention was already `0.8125`, below floor. The test split stayed closed.

### 8.3 W4 bounded continuation design

The only authorized W4 tune was deliberately bounded:

- initialization: exact SHA `83767ebff2c2a13a2f15fe8266f605fb8485985c3289c1f1720cd70c122a9ac5`;
- `768` newly generated inverse-rendered training rows;
- `256` rows per unique/small/large preimage stratum;
- `192` rows per depth 1-4;
- zero overlap with frozen calibration/test IDs;
- 200 AdamW optimizer steps;
- effective batch size 8;
- fixed max loops 4;
- no latent, halting, or loop-control objective;
- no forward rehearsal, because the registered rehearsal branch produced no joint-gate checkpoint;
- loop-label weights:

```text
[0.324324, 0.432432, 0.648649, 2.594595]
```

- calibration first; test only if calibration, synthetic retention, and natural canary all passed.

Training checkpoint:

```text
SHA256 32d4a6a954a2db8c25f9056862da5f4dd686fbfaecb9bfd48393b2f9b5055389
```

### 8.4 W4 results

| Depth | W3 zero-shot | W4 step 200 | Delta | Gate |
|---:|---:|---:|---:|---:|
| 1 | `95/96` | `94/96` | `-1` | Pass |
| 2 | `82/96` | `58/96` | `-24` | Bare pass |
| 3 | `61/96` | `35/96` | `-26` | Fail |
| 4 | `38/96` | `21/96` | `-17` | Fail |
| Pooled | `276/384` | `208/384` | `-68` | Fail |

Guardrails:

- synthetic active-diagonal minimum: `0.125` versus `0.93` floor;
- natural canary: `171/256 = 66.80%` versus locked `227/256 = 88.67%`;
- natural delta: `-21.875 pp`;
- natural verdict: `red_hard_stop`.

The test split correctly remained unopened. Phase G-alpha remained closed.

### 8.5 W4 interpretation

This is not an ambiguous under-dose result. The sole registered continuation moved calibration by `-68` rows and caused catastrophic retention loss. The training distribution was valid, balanced by preimage stratum, disjoint from frozen evaluation IDs, and completed all 200 steps. More of the same objective is not warranted.

The result supports a transfer-boundary interpretation:

> Under this initialization, objective, and 200-step bounded continuation, adapting the recurrent block toward the inverse-rendered N=24 distribution did not improve deterministic chain validity and severely damaged previously installed synthetic and natural-surface behavior.

It does not establish that no curriculum, regularizer, model scale, frozen-block adapter, or alternative multimodal task could provide a suitable deterministic substrate.

---

## 9. Multi-channel bridge precursor: banked result

The proposed bridge generalization used learned subspaces with channel-specific carry/injection behavior. Before training it, an eval-only precursor tested whether the existing dynamics showed basis-specific specialization beyond matched random orthogonal partitions.

Locked battery:

- M1: subspace drift concentration;
- M2: retrieval-head concentration and identity stability;
- M3: subspace injection-sensitivity intervention;
- at least 20 matched random controls;
- replication required on both the N24 checkpoint and backward-recovery condition;
- architecture activation required at least two confirmed measurements plus an independently priced staircase reading one.

Replication result:

| Measurement | N24 | Backward recovery | Confirmed |
|---|---|---|---|
| M1 | Fail/smeared | Fail/smeared | No |
| M2 | Pilot positive | Fail, zero stable heads | No |
| M3 | Not run | N/A | No |

The final decision was `remain_banked`. No multi-channel bridge training, M3 extension, or alternative-basis search is authorized by these data.

---

## 10. Gate registry and present state

| Gate | Registered requirement | Best/last admissible result | State |
|---|---|---|---|
| Re-entry integrity | live bridge/adapter; loop-1 preserved; bounded loop RMS | Passed; loop-8 RMS ratio ~`1.0061` | Green |
| Deterministic synthetic mechanism | accurate loop-indexed chain; survives scaffold removal | `625/640`, 97.7% active diagonal | Green on synthetic family |
| Phase A A vs B | A wins at >=3 consecutive depths, one-sided p<0.05 | A wins all 14 depths | Green |
| Phase A A vs C extension | labeled extension, paired reporting | `+554` net rows, p=`3.42e-120` | Green extension |
| Canonical forward-table cap 2 | `46/64` plus retention >=`0.93` | `3/64`, retention `0.21875` | Closed under tested curriculum |
| Explicit inverse-table cap 3 | `46/64` plus retention >=`0.93` | `63/64`, retention `0.8125` | Task pass, retention fail |
| Rehearsal Pareto | task + synthetic + natural all pass | no joint candidate at 4 checkpoints | Closed |
| W3 inverse-rendered calibration | `288/384`, every depth `58/96` | `276/384`; depth 4 `38/96` | Failed; W4 was authorized |
| W4 bounded continuation | calibration + synthetic + natural | `208/384`; `0.125`; `-21.875 pp` | Failed; branch closed |
| Phase G-alpha entry | deterministic calibration/test and retention green | prerequisite failed | Closed |
| F9 architecture activation | >=2 replicated measurements + staircase reading one | 0 replicated votes; no reading one | Banked |
| Cap 4 | prior cap and all guardrails green | prerequisites failed | Unauthorized |

---

## 11. Claims ledger

### 11.1 Established

- A pretrained Qwen2.5-0.5B can be surgically reorganized into a recurrent architecture with exact single-pass preservation and a corrected, trainable loop closure.
- The repaired recurrent block can learn persistent, loop-indexed latent state transitions under exact intermediate supervision.
- The learned chain can survive removal of the intermediate scaffold on the synthetic task family.
- On the frozen Phase A synthetic family, the recurrent 0.5B system substantially outperforms the evaluated dense 0.5B direct and scratchpad recipes, especially beyond depth 10.
- Repeated reverse composition is much harder than forward lookup on explicitly rendered inverse tables under matched dose.
- The present symbolic continuation recipes trace a task-versus-retention Pareto frontier with no joint-passing checkpoint.
- Naive stochastic noise and particle repulsion did not create useful correct-bearing diversity on the early substrate.

### 11.2 Not established

- That recurrence alone caused the Phase A advantage under matched lineage and compute.
- That the recurrent model beats base Qwen on broad natural reasoning.
- That learned halting selects useful depth on held-out hard reasoning.
- That the model has multiple stable correct-bearing latent pathways.
- That GRAM-style guided latent width transfers to recurrent Qwen.
- That SVGD helps once a proper learned posterior/prior exists.
- That the current model can perform canonical repeated backward search beyond the partial loop-2/3 signal.
- That a multi-channel bridge is useful.

### 11.3 Explicit do-not-claim statements

- Do not use the early Gaussian/SVGD negatives as evidence against GRAM.
- Do not describe the W3 fixed-temperature K=1 diagnostic as a stochastic-width experiment.
- Do not report the unopened inverse-rendered test split.
- Do not call the Phase A comparison matched-compute or matched-training-lineage.
- Do not call the inverse-table direction difference a measured five-fold dose ratio.
- Do not authorize cap 4, G-alpha, G-beta, learned trajectory selection, or SVGD from current results.

---

## 12. Remaining work: consolidation versus new experiments

### 12.1 Evidence consolidation, no GPU decision required

The following work is compatible with the current evidence and should continue:

1. Finalize the deterministic paper's architecture narrative:
   - pretrained Qwen surgery;
   - exact single-pass identity;
   - bridge/re-entry failure and repair;
   - intermediate-chain supervision;
   - outcome-only persistence;
   - Phase A frozen comparison;
   - explicit limitations on lineage and natural transfer.
2. Preserve the Phase A receipt and figure as the central positive evidence.
3. Add the inverse curriculum and W3/W4 outcomes as boundary/negative results.
4. Record the GRAM divergence audit so readers understand that guided width remains untested.
5. Keep all checkpoint hashes, frozen row hashes, test-closure decisions, and gate definitions in the paper appendix or artifact ledger.

### 12.2 One admissible curriculum-science experiment

The remaining experiment already justified by prior evidence is the loop-position transfer micro-test.

Purpose: determine whether a learned inverse primitive transfers across recurrent positions or must be reinstalled at every loop.

Design:

1. Start from the locked natural-surface keeper, not a failed inverse continuation.
2. Train the canonical inverse primitive at loop position 1 only.
3. Stop when held-out position-1 accuracy reaches the existing `46/64` bar, or at a fixed preregistered dose ceiling.
4. Freeze the checkpoint.
5. Evaluate the identical held-out mapping rows at forced positions 2, 3, and 4 without updates.
6. Keep prompt, answer reader, loop count, and rows identical across positions.
7. Report chance, accuracy, paired transitions, and synthetic/natural guardrails.

Interpretation:

- strong zero-shot transfer: per-position repurchase is not the primary block; revisit curriculum composition;
- monotonic transfer decay: position-specific installation cost is supported; a corrected primitive-first arm is justified;
- failure at position 1: the primitive is not consolidated and no deeper arm should run.

This is a localization test, not another attempt to open Phase G directly.

### 12.3 Conditional corrected forward-table arm

Only if the position-transfer result supplies a coherent repair hypothesis:

1. train the inverse primitive to the locked cap-1 bar with a fixed rehearsal mix;
2. save and hash the consolidated checkpoint;
3. open cap 2 under equalized weighted active-label accounting;
4. stop on task, synthetic, or natural failure;
5. advance only under existing gates;
6. use a seed-swapped comparison before any publication claim about direction cost.

### 12.4 Alternative deterministic substrate decision

If strategy does not expect the position-transfer result to change the curriculum, choose a different multimodal task for G-alpha. The substrate must satisfy all of these before stochastic heads are built:

- deterministic K=1 competence is already high;
- multiple valid solutions are intrinsic and exactly verifiable;
- exact solution-set or coverage denominators are available;
- the task exercises recurrent state transitions rather than only output sampling;
- frozen calibration/test splits and guardrails can be locked;
- competence does not require a separate non-native operation the model has not learned.

Possible task families should be screened on CPU and with eval-only keeper runs before training. The decision should be based on substrate competence and multimodal verifiability, not convenience alone.

---

## 13. If and only if a deterministic substrate turns green: frozen G-alpha plan

The G-alpha specification remains ready but gated.

### Trainable components

```text
phase_g_prior_head.*
phase_g_posterior_head.*
phase_g_injection_scale
```

The recurrent block, bridge, prelude, coda, and reader stay frozen. Every backward pass asserts that frozen gradients are absent or exactly zero.

### Transition

```text
u_l = frozen deterministic recurrent update
z_l ~ q(z_l | u_l, gold_next_l)       # training
z_l ~ p(z_l | u_l)                    # inference
h_l = u_l + softplus(s) R z_l
```

`R` is a fixed seeded projection. Stochasticity enters only at the high-level re-entry state.

### Objective and diagnostics

- per-loop symbolic CE;
- per-loop balanced KL from posterior to prior;
- initial KL balance near `0.8`;
- coefficient sweep `1e-4`, `1e-3`, `1e-2`;
- EMA `0.999`;
- prior/posterior variance, KL, collapse fraction, injection-to-state RMS, and frozen-gradient assertions by loop/depth/stratum.

### Required comparisons

1. K=1 parity with the frozen deterministic keeper.
2. Latent K=`1,2,4,8,20` versus entropy-matched answer-head sampling at the same K.
3. K trajectories at depth T versus one deterministic trajectory at depth K*T using actual bridge-call counts.
4. Exact oracle coverage: distinct valid solutions divided by exact solution-set size.

G-beta, learned selection, per-trajectory halting, and SVGD open only after latent-K beats both registered comparators.

---

## 14. Questions for strategy review

1. Does the strong Phase A synthetic result plus the current natural-transfer boundary justify closing paper one now around deterministic recurrence, rather than withholding it for a Phase G result?
2. Is the loop-position transfer micro-test likely to alter the canonical inverse curriculum decision, or has the inverse branch reached diminishing returns?
3. If the inverse branch closes, which multimodal task best satisfies deterministic competence, exact validity, and recurrent-transition relevance for G-alpha?
4. Should the next substrate use the current 0.5B keeper for continuity, or should G-alpha wait for a 1.5B/3B recurrent substrate with more capacity?
5. Which retention policy should replace narrow symbolic rehearsal: frozen-block adapters, explicit distillation to the keeper, replay at a larger ratio, gradient projection, or a larger base?
6. What additional matched-lineage dense control is necessary before presenting the Phase A result publicly?
7. Should the natural-surface capability goal remain a prerequisite for paper one, or be positioned as the next-stage transfer problem?
8. Is the project's next highest-value GPU hour better spent on position transfer, a new substrate screen, or a matched-lineage Phase A control?
9. What result would cause the program to close the 0.5B architecture line and move to 1.5B/3B?
10. What is the minimum deterministic gate that makes G-alpha attributable without making the gate so strict that it requires solving a harder problem than the width hypothesis itself?

---

## 15. Recommended decision

The evidence supports a two-part recommendation.

First, bank the deterministic result. The repaired recurrent model has demonstrated a real state-transition mechanism and a large synthetic depth advantage over the evaluated dense controls. That is the strongest result in the project and should no longer be treated as merely preliminary engineering.

Second, pause the current inverse-rendered training branch. W4 was the only authorized continuation and it produced a decisive deterioration, not a near miss. Do not spend another GPU run on the same objective, add SVGD, or weaken the gate.

The next strategy choice is between one final low-cost localization experiment and a clean substrate change:

- run the position-transfer micro-test if its result will determine a concrete corrected curriculum; otherwise
- select a deterministic multimodal substrate that the keeper already solves and use it to test the original GRAM-inspired width hypothesis directly.

Either route should preserve the deterministic paper as a separate deliverable. The width paper, if it opens, should stand on that characterized substrate rather than delay recognition of the deterministic contribution.

---

## 16. Source artifacts and lineage

### Controlling plans and audits

- `docs/PROGRAM_TRACK_MASTER_SEQUENCE.md`
- `docs/gram_divergence_audit_20260711.md`
- `docs/PHASE_G_TRACK_REUNIFICATION_AMENDMENT.md`
- `docs/TWO_LANE_SURPASS_REBASE_AMENDMENT_20260714.md`
- `docs/PHASE_G_ALPHA_GUIDED_STOCHASTIC_TRANSITION_SPEC.md`
- `docs/PHASE_G_CURRICULUM_AUTOPSY_HANDOFF_20260713.md`
- `docs/INVERSE_COMPOSITION_STAIRCASE_SPEC.md`
- `docs/INVERSE_TABLE_REBASE_SPEC.md`
- `docs/MULTICHANNEL_BRIDGE_PRECURSOR_SPEC.md`

### Machine-readable receipts

- `outputs/stage5/stage5_phase_a_surpass_receipt_20260714/summary.json`
- `outputs/stage5/stage5_phase_a_surpass_receipt_20260714/phase_a_depth_profile.svg`
- `outputs/stage5/stage5_inverse_composition_staircase_20260713/summary.json`
- `outputs/stage5/stage5_inverse_table_rebase_caps3_4_20260713/summary.json`
- `outputs/stage5/stage5_inverse_table_cap3_rehearsal_20260714/summary.json`
- `outputs/stage5/stage5_inverse_table_cap3_rehearsal_20260714/checkpoint_pareto/summary.json`
- `outputs/stage5/stage5_inverse_rendered_width_gate_20260714/summary.json`
- `outputs/stage5/stage5_inverse_rendered_n24_continuation_20260715/summary.json`
- `outputs/stage5/stage5_multichannel_bridge_precursor_replication_20260714/summary.json`

### Critical checkpoint identities

| Checkpoint | SHA256 | Role |
|---|---|---|
| Phase A recurrent arm A | `dc00f7b694ce32427eb13b0b85d365bc15e0c0317130bd22d4bbc3568544f71b` | Main deterministic synthetic result |
| Inverse staircase C cap 2 | `bc1de1cd7d2a7acf30b9217c8d7054d805888c341b942ff0dab7691b4f995b01` | Explicit inverse cap-2 source |
| Inverse rebase C cap 3 | `83767ebff2c2a13a2f15fe8266f605fb8485985c3289c1f1720cd70c122a9ac5` | W3/W4 source |
| Rehearsal step 334 | `d662829d0d3c4f3f1521aaa567728203712152d5fd81bab17d78ad63b0ce6c00` | Task/synthetic pass, natural fail |
| W4 continuation step 200 | `32d4a6a954a2db8c25f9056862da5f4dd686fbfaecb9bfd48393b2f9b5055389` | Failed bounded continuation |

---

## 17. Final handoff state

```text
Architecture repair                         GREEN
Deterministic synthetic recurrence          GREEN
Phase A synthetic surpass receipt           GREEN, narrow claim
Canonical repeated backward composition     BLOCKED under tested curriculum
Explicit inverse-table cap-3 retention       NO JOINT-GATE CHECKPOINT
Inverse-rendered N24 W3                      BELOW GATE
Inverse-rendered N24 W4                      DECISIVE REGRESSION
Multi-channel bridge                         BANKED
GRAM-style guided latent width               NOT YET TESTED
Phase G-alpha                                CLOSED
Next automatic GPU job                       NONE
Next action                                  STRATEGY REVIEW
```

