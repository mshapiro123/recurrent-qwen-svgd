# Retrofitting Recurrent Depth into a Pretrained Language Model: A Forensic Study of Identity, Iteration, Transfer, and Retention

**Part 1 manuscript draft**  
**Status:** deterministic program closed; results frozen July 15, 2026  
**Model:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Evidence ledger:** [part1_claim_evidence_ledger.json](part1_claim_evidence_ledger.json)

## Abstract

Can a pretrained dense language model be converted into a recurrent-depth model without pretraining a new architecture from scratch? We study a surgical retrofit of Qwen2.5-0.5B-Instruct that partitions the transformer into a Prelude, a weight-tied Recurrent Block, and a Coda. The one-loop path is identity preserving; later loops combine the persistent state with a re-injected Prelude representation through a trainable bridge. A sequence-level controller can allocate loop depth, while exact intermediate-state labels provide direct supervision for iterative computation.

The study produced three positive findings and one boundary. First, after repairing a silent loop-closure error, intermediate supervision installed a genuine loop-indexed state transition: after 1,000 subsequent outcome-only steps, the active-label diagonal remained 625/640 (97.7%), and 357/384 (93.0%) above-diagonal states continued the transition. Second, support-depth scaling extended the frozen same-reader frontier: on an N24 transition family, the final checkpoint scored 91.4% at depth 14 and 70.3% at depth 18, before falling to 10.9% at depth 22. Third, on a frozen 1,792-row depth-1-through-14 synthetic composition family, the recurrent 0.5B system scored 1,506/1,792 (84.0%), versus 952/1,792 (53.1%) for the strongest evaluated dense 0.5B serialized-scratchpad recipe. The paired result was 607 helped, 53 hurt, and 1,132 tied (two-sided exact p = 3.42e-120). This is a system-level, synthetic-family comparison; training lineage and inference compute were not matched.

The boundary was retention. New inverse operations were learnable in isolation, including a 63/64 task result, but the tested full-block continuations did not retain both the new operation and the consolidated recurrent mechanism. Controlled verbalized transfer was positive but tail-limited, and broad natural-benchmark superiority was not established. These results show that a pretrained language model can host persistent latent iterative computation after architecture surgery, while also identifying loop closure, reader alignment, support depth, and catastrophic interference as first-order constraints. Guided stochastic width remains open and is separated into a subsequent study.

## 1. Introduction

Transformers normally obtain additional computation by adding layers during design or tokens during generation. Recurrent-depth models offer a third axis: repeatedly apply a shared transformation to hidden states. This can increase effective depth without increasing the number of distinct block parameters, and it creates a natural place for adaptive computation and latent iterative reasoning.

Most recurrent-depth work trains the recurrent architecture as such. Our question is different:

> Can a pretrained dense language model be surgically converted into a recurrent-depth system, recover a stable iterative mechanism, and outperform registered dense recipes on tasks that reward depth?

The conversion problem is not solved by wrapping a loop around transformer layers. The repeated block receives states from a distribution it did not see during pretraining. The input context must remain available on later iterations. Credit must reach the re-entry pathway. Intermediate predictions must be read at the correct point in the loop. Additional computation can also destroy a correct state through overthinking or catastrophic interference.

We therefore treat model conversion as a forensic engineering and measurement problem. The program uses exact identity checks, gradient-path audits, forced-depth evaluations, frozen row manifests, same-reader scoring, paired statistical tests, checkpoint hashes, and guardrail hard stops. Negative results are retained when they localize a boundary instead of being discarded as failed tuning runs.

### Contributions

1. **An identity-preserving recurrent retrofit of Qwen2.5-0.5B.** The model is split into Prelude, Recurrent Block, and Coda regions. The one-loop route reproduces the base computation when recurrent additions are inactive.
2. **A repaired and audited re-entry mechanism.** We identify a silent omission of Prelude re-injection, repair the loop closure, and verify bridge liveness, forced-loop execution, and per-loop loss connectivity.
3. **Evidence for persistent loop-indexed latent computation.** Exact intermediate-state supervision installs an iterative transition that survives outcome-only annealing and extends beyond its directly supported horizon.
4. **A registered dense-control comparison.** The recurrent system substantially exceeds direct and serialized-scratchpad dense recipes on identical frozen synthetic rows, especially beyond depth 10.
5. **A mapped retention boundary.** Three inverse-task branches distinguish acquisition from retention and show where full-block continuation fails on this substrate.
6. **An auditable claim discipline.** Every paper claim is linked to a durable local receipt in a machine-readable ledger; unsupported natural-benchmark and stochastic-width claims are explicitly excluded.

## 2. Related Work

### Recurrent depth and adaptive computation

Adaptive Computation Time lets recurrent networks learn how many internal steps to take before emitting an output [Graves, 2016](https://arxiv.org/abs/1603.08983). Universal Transformers apply a self-attentive transition recurrently over depth and add dynamic per-position halting [Dehghani et al., 2018](https://arxiv.org/abs/1807.03819). PonderNet formulates learned computation depth as a probabilistic halting process with a regularized distribution over step counts [Banino et al., 2021](https://arxiv.org/abs/2107.05407). Deep Equilibrium Models instead solve directly for the fixed point of an effectively infinite weight-tied network [Bai et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html).

Our system uses explicit recurrent iterations rather than a root solver, and sequence-level probabilistic halting rather than per-token halting. Its main distinction is the retrofit setting: a pretrained causal language model is partitioned and looped after pretraining, making identity and re-entry distribution alignment central experimental concerns.

### Retrofitting pretrained transformers

Kasai et al. convert pretrained transformers into recurrent alternatives through a swap-then-finetune procedure, targeting efficient autoregressive attention [Kasai et al., 2021](https://aclanthology.org/2021.emnlp-main.830/). LoRA freezes pretrained weights and adds low-rank trainable matrices for downstream adaptation [Hu et al., 2021](https://arxiv.org/abs/2106.09685). Our surgery differs in purpose and location: the original attention implementation remains, while a middle transformer region is reused over computation depth.

Qwen2.5 provides the pretrained dense substrate and a family of model sizes [Qwen Team, 2024](https://arxiv.org/abs/2412.15115). Early project stages emphasized adapters and controllers, but capacity localization later required unfreezing the 12-layer recurrent block. The final deterministic training budget contains 182,163,457 trainable parameters. Claims in this paper therefore concern architectural conversion and recurrent computation, not parameter-efficient recovery.

### Latent iterative and stochastic reasoning

Recent recurrent-depth studies investigate implicit multi-hop composition and depth extrapolation, including the risk that excessive loops degrade predictions [Kohli et al., 2026](https://arxiv.org/abs/2604.07822). GRAM models recursive reasoning as a distribution over stochastic latent trajectories and trains a conditional prior against a target-conditioned posterior [Baek et al., 2026](https://arxiv.org/abs/2605.19376). Our early Gaussian and particle experiments omitted that guided prior/posterior construction and predated the final loop repair. They therefore do not test the GRAM hypothesis. This manuscript closes the deterministic substrate study only.

## 3. Architecture

### 3.1 Prelude, Recurrent Block, and Coda

Let the pretrained transformer contain ordered layers split at indices `(a, b)`:

```text
tokens -> Prelude[0:a] -> Recurrent Block[a:b] -> Coda[b:L] -> LM head
```

The Prelude computes an input-grounded representation `p`. The recurrent state begins from that representation and is updated by the shared middle block. After the selected number of loops, the Coda maps the final state into the pretrained output space.

For one loop, the wrapper executes each pretrained layer exactly once in its original order. Attention masks, position identifiers, normalization, dtype, and causal semantics are preserved. Recurrent additions are inactive on this identity route.

### 3.2 Corrected loop closure

The intended later-loop update is input-injected recurrence:

```text
u_t = Reentry(p, h_t)
h_(t+1) = RecurrentBlock(u_t)
```

The implemented split bridge learns separate projections of the persistent state and Prelude representation before recombination. An identity-biased gate and optional re-entry normalization stabilize the transition. The Prelude contribution is applied only on re-entry, so the one-loop identity path is unchanged.

The original implementation fed the recurrent output back without correctly re-injecting the Prelude representation. That omission changed the dynamical system being tested. The repair was followed by static graph inspection, per-loop gradient matrices, finite-difference checks, covariance and norm diagnostics, and forced-loop artifact checks.

### 3.3 Halting and forced-depth evaluation

A sequence-level halting head produces a distribution over loop counts. The training objective combines task loss, optional intermediate-state losses, and a regularizer toward a centered geometric prior. Mechanism experiments also use forced loop counts. This separates two questions:

- can the block execute a useful transition at depth `t`;
- can a router select the appropriate depth.

Part 1 primarily resolves the first question. Reliable learned selection remains incomplete.

## 4. Training and Measurement Protocol

### 4.1 Exact intermediate-state supervision

Synthetic transition rows provide the state expected after each loop. For loop `t`, the model is decoded with the same symbol reader and receives cross-entropy against the registered intermediate target. Outcome-only rows supervise only the final state. Staged training first installs the transition with dense intermediate labels, then removes that scaffold to test persistence.

### 4.2 Frozen same-reader evaluation

The same-reader protocol holds the prompt surface, output symbol vocabulary, row IDs, and decoder constant across loop counts and systems. This avoids a failure mode discovered during the program: a model could represent the correct state under the active symbol reader while a separately formatted multiple-choice reader selected a different option label.

The primary synthetic family has 128 rows at each depth from 1 through 14, for 1,792 rows total. The Phase A arms use identical row IDs. Paired exact sign/McNemar tests compare correctness row by row; a count-based Fisher gate was preregistered for the primary comparison.

### 4.3 Lineage and guardrails

Every promoted checkpoint has a SHA-256 receipt. Continuations resolve both the exact source checkpoint and its source metrics before loading the model. A continuation cannot start below a registered retention floor. Full-block exploratory runs are marked `disposable_measurement`, cannot produce successor lineage, and must delete non-promotable checkpoints after their measurement completes.

### 4.4 Comparison limits

The recurrent and dense arms share the model family, frozen rows, and output reader, but they do not share identical training histories, token counts, optimizer trajectories, FLOPs, latency, or inference compute. Phase A is consequently a registered system comparison, not an architecture-only causal estimate.

## 5. Forensic Repair Results

### 5.1 Why early negatives were not final evidence

The missing input re-injection meant that later loops were not repeatedly applying a context-grounded transition. Early particle experiments therefore combined stochasticity with a malformed deterministic substrate. Later audits also found that global norm rescaling could alter geometry without improving correct-candidate conversion, and that output diversity was not equivalent to useful latent alternatives.

The corrected architecture was required before interpreting recurrence or width. This distinction is recorded in the claim ledger and Phase G specification.

### 5.2 Persistent chain after scaffold removal

After staged chain training, the final 1,000 steps used outcome supervision only. The post-anneal checkpoint retained:

| Measure | Result |
|---|---:|
| Active-label diagonal, depths 1-4 | `625/640 = 97.7%` |
| Above-diagonal states that continued iterating | `357/384 = 93.0%` |
| Above-diagonal states that held | `1/384 = 0.3%` |

The model did not simply preserve the answer once reached. It usually continued to apply the learned transition, which is the expected signature of an installed loop operator. This also implies that uncontrolled extra loops can be harmful and motivates a separate selector problem.

## 6. Support-Depth Scaling

The N24 program increased transition support and evaluated every depth from 1 through 22 on a frozen set. The final step-6,000 checkpoint produced:

| Depth | Accuracy |
|---:|---:|
| 10 | `97.7%` |
| 12 | `97.7%` |
| 14 | `91.4%` |
| 16 | `85.9%` |
| 18 | `70.3%` |
| 20 | `46.1%` |
| 22 | `10.9%` |

At step 2,000, depth-14 accuracy was 69.5% and depth-18 accuracy was 10.2%. Additional support moved the frontier substantially, but did not create indefinite algorithmic extrapolation. The result is best described as a support-dependent extrapolation frontier with a measurable tail ceiling.

## 7. Registered Dense-Control Comparison

### 7.1 Systems

| Arm | System | Recipe |
|---|---|---|
| A | Recurrent Qwen2.5-0.5B | forced-depth same-reader recurrent system |
| B | Dense Qwen2.5-0.5B | direct-answer SFT |
| C | Dense Qwen2.5-0.5B | serialized-scratchpad SFT |
| D | Dense Qwen2.5-1.5B | direct-answer scale control |

### 7.2 Aggregate and tail results

| Arm | Correct / 1,792 | Accuracy | Depths 11-14 | Tail accuracy |
|---|---:|---:|---:|---:|
| A, recurrent 0.5B | `1,506` | `84.04%` | `272/512` | `53.13%` |
| B, dense 0.5B direct | `470` | `26.23%` | `60/512` | `11.72%` |
| C, dense 0.5B scratchpad | `952` | `53.13%` | `56/512` | `10.94%` |
| D, dense 1.5B direct | `322` | `17.97%` | `58/512` | `11.33%` |

The primary preregistered comparison was A versus B. A cleared the count-based gate at all 14 depths. A versus C was a labeled extension against the strongest dense control and also favored A at every depth.

| Paired comparison | Helped | Hurt | Tied | Net | Two-sided exact p |
|---|---:|---:|---:|---:|---:|
| A vs B | `1,074` | `38` | `680` | `+1,036` | `2.12e-264` |
| A vs C | `607` | `53` | `1,132` | `+554` | `3.42e-120` |

The strongest evidence is the tail separation. The serialized scratchpad remained competitive through parts of the trained range but fell to 10.9% over depths 11-14, while the recurrent system retained 53.1%.

### 7.3 Dense checkpoint saturation

From step 2,000 to step 4,000, dense direct arm B added six net rows (`p=0.771`), scratchpad arm C added 22 (`p=0.00319`), and 1.5B direct arm D lost 28 (`p=0.161`). The D result is specific to this recipe and is not evidence that smaller dense models are generally stronger.

## 8. Controlled Natural-Surface Transfer

The installed transition was evaluated on generated verbal relay and pointer surfaces with the same symbol reader. At step 6,000:

| Surface | Correct / 1,536 | Accuracy |
|---|---:|---:|
| Relay | `1,321` | `86.0%` |
| Pointer | `1,213` | `79.0%` |

Performance was high at shallow and middle depths but declined in the tail. Relay depth 10 remained 82.8%, then fell to 57.0% at depth 11 and 32.0% at depth 12. Pointer depth 9 was 77.3%, then fell to 55.5%, 44.5%, and 19.5% at depths 10-12.

These surfaces show that the transition is not confined to one symbolic rendering. They are nevertheless generated controlled tasks, not broad natural reasoning benchmarks. The surface experiments also showed that retrieval organization depends on training history: head-level specialization observed on the consolidated N24 checkpoint did not replicate on the backward-recovery checkpoint.

## 9. Acquisition-Retention Boundary

Three inverse-task branches tested whether the system could install a non-native reverse operation while preserving the forward mechanism.

### 9.1 Canonical forward-table inverse

Several curricula stalled at matched dose. Active-loop analysis revealed partial gains at loops 2 and 3 that final-answer scoring had hidden, but later loops remained unsupported. This was not a total optimization null; it was an incomplete staircase.

### 9.2 Explicit inverse-table branch

The isolated task reached 63/64, directly demonstrating acquisition. A rehearsal continuation reached 64/64 and retained the synthetic mechanism above its floor (`0.96875 >= 0.93`), but failed the natural canary with an accuracy delta of `-0.0586`, beyond the `-0.03` hard-stop margin. No checkpoint on the tested Pareto sweep satisfied all retention constraints.

### 9.3 Inverse-rendered W3/W4

W3 reached 288/384 on its nominal calibration metric, but a source audit placed synthetic retention below its required floor (`0.8125 < 0.93`). The authorized W4 continuation worsened all relevant measures:

| Measure | Before | After |
|---|---:|---:|
| Calibration | `288/384` | `208/384` |
| Synthetic retention minimum | `0.8125` | `0.125` |
| Natural canary | `227/256` reference | `171/256` |

Together, these branches separate learning from retention. The operation is representable and trainable, but the tested full-block adaptation regime cannot preserve both old and new behavior on the 0.5B substrate.

## 10. Closed Architecture Alternative: Multi-Channel Re-entry

A preregistered precursor battery tested whether the bridge should be split into learned channels. Activation required two of three positive measurements plus a priced staircase reading. M1 subspace drift was smeared on both checkpoints. M2 retrieval-head concentration was positive on N24 but failed replication on the backward-recovery checkpoint. M3 could therefore produce at most one vote, making the activation gate unsatisfiable. The intervention was closed without training it.

This negative avoids an unpriced architectural branch while retaining useful attention-capture instrumentation and the finding that retrieval structure is training-history specific.

## 11. Final Localization and Width-Substrate Gate

One shared post-closeout session ran the final non-promotable deterministic localization measurement and the two preregistered width-substrate screens.

### 11.1 Loop-position transfer was stopped by retention

At step 500, the disposable loop-position arm cleared its trained-position prerequisite: inverse position 1 scored `46/64 = 71.9%` and position 2 scored `64/64 = 100%`. At step 1,000 those trained-position scores were unchanged, but the frozen synthetic guardrail minimum fell to `0.8125`, below the registered `0.93` floor. The runner hard-stopped and deleted both disposable checkpoints. Transfer positions 3 and 4 were therefore never evaluated.

The correct reading is inconclusive. The experiment neither establishes position-invariant transition reuse nor confirms per-position installation. It does demonstrate that the lineage and hard-stop policy worked: a potentially damaging full-block branch produced no successor checkpoint.

### 11.2 A verbal multi-valued forward substrate passed

The branching-relations task stores an exact reachable set and scores the same-reader argmax as valid when it belongs to that set. Both screens used 128 frozen rows at each depth from 1 through 4.

| Keeper and surface | D1 | D2 | D3 | D4 | Pooled | Gate |
|---|---:|---:|---:|---:|---:|---|
| Natural step-2,000, N20 verbal | `127/128` | `95/128` | `87/128` | `80/128` | `389/512 = 75.98%` | Pass |
| N24 step-6,000, symbolic | `128/128` | `86/128` | `67/128` | `74/128` | `355/512 = 69.34%` | Fail |

The gate required pooled validity at least `0.70` and every depth at least `0.55`. The verbal keeper passed with a minimum depth accuracy of `0.625`. The symbolic keeper missed both pooled validity and the depth-3 floor (`0.5234`). Because one frozen keeper passed, no adapter touch-up is needed. The deterministic prerequisite for Phase G-alpha is satisfied; only the powered numeric margin remains to be locked before launch.

This result establishes substrate competence, not useful stochastic width. No latent prior/posterior head was trained in this session.

## 12. Discussion

### What the model learned

The combined evidence is inconsistent with a pure final-label lookup account. The model predicts registered intermediate states, continues the transition beyond the target state, retains the operator after intermediate losses are removed, improves its depth frontier when support is increased, and separates from a serialized dense scratchpad most sharply beyond the scratchpad's horizon.

### What remains confounded

The recurrent arm had a longer and different training lineage than the dense controls. It also receives forced loop compute proportional to depth. The result therefore establishes that this recurrently trained system can realize the task family more effectively than the evaluated dense recipes. It does not isolate weight tying, supervision format, optimization history, or compute as the sole cause.

### Why the retention boundary matters

The inverse experiments show that a converted pretrained model is not an empty substrate. New training competes with a consolidated transition and with retained language-model behavior. Larger substrates, detachable adapters, routing, or explicit modularity may move this boundary. Weakening guardrails would not answer the scientific question because it would trade one capability for another.

### Why width is a separate paper question

Early stochastic noise and SVGD-style repulsion increased diversity in some settings without reliably producing new correct candidates. Those runs lacked the repaired recurrence and the target-conditioned variational guidance that distinguishes GRAM from unguided noise. Guided stochastic width remains open. Its next valid test freezes the deterministic keeper, trains only prior/posterior latent heads and an injection scale, and compares exact coverage at matched K against answer-head sampling and matched transition compute.

## 13. Limitations

- Results center on Qwen2.5-0.5B and controlled synthetic or generated surfaces.
- Phase A does not match training tokens, optimizer history, FLOPs, latency, or inference compute.
- Forced-depth evaluation measures the transition mechanism separately from learned depth selection.
- Full-block unfreezing uses a substantial 182M-parameter adaptation budget.
- External benchmarks such as GPQA Diamond, ARC-AGI, mathematics, and coding have not established a superiority claim.
- The natural surfaces are controlled renderings and do not substitute for broad language understanding.
- The inverse retention boundary may move with scale, adapters, routing, or a different curriculum.
- No post-repair target-conditioned stochastic-width result exists yet.

## 14. Reproducibility and Artifact Map

| Result | Canonical artifact |
|---|---|
| Part 1 decision | `docs/PART1_DETERMINISTIC_PROGRAM_CLOSEOUT_20260715.md` |
| Claim-to-evidence ledger | `docs/part1_claim_evidence_ledger.json` |
| Persistent chain | `outputs/stage5/stage5_chain_anneal_20260703_160250/summary.json` |
| N24 depth frontier | `outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json` |
| Frozen depth-22 matrix | `outputs/stage5/stage5_n24_support12_rung_20260707_140139/eval/frozen_depth22_step_6000/active_summary.json` |
| Phase A paired comparison | `outputs/stage5/stage5_phase_a_surpass_receipt_20260714/summary.json` |
| Natural same-reader transfer | `outputs/stage5/stage5_natural_surface_receipts_20260709_210151/summary.json` |
| Explicit inverse rehearsal | `outputs/stage5/stage5_inverse_table_cap3_rehearsal_20260714/summary.json` |
| Inverse-rendered W3 | `outputs/stage5/stage5_inverse_rendered_width_gate_20260714/summary.json` |
| Inverse-rendered W4 | `outputs/stage5/stage5_inverse_rendered_n24_continuation_20260715/summary.json` |
| Multi-channel bridge battery | `outputs/stage5/stage5_multichannel_bridge_precursor_replication_20260714/summary.json` |
| Loop-position and branching gate | `outputs/stage5/stage5_part1_closeout_pivot_20260715/summary.json` |
| GRAM divergence audit | `docs/gram_divergence_audit_20260711.md` |
| Phase G preregistration | `docs/PHASE_G_ALPHA_GUIDED_STOCHASTIC_TRANSITION_SPEC.md` |

The Phase A receipt records identical row-ID hashes and verified checkpoint hashes. The repository test suite validates the claim ledger, key arithmetic, wrapper contracts, continuation floors, and Phase G frozen-parameter contract.

## 15. Conclusion

Part 1 answers the deterministic substrate question. A pretrained Qwen2.5-0.5B model can be converted into an identity-preserving recurrent-depth architecture, trained to execute a persistent loop-indexed state transition, and scaled to a depth regime where it substantially outperforms the registered dense recipes on a frozen synthetic family. The same program also found the limits of that result: general natural reasoning remains unproven, learned routing remains incomplete, and full-block continuation could not retain newly acquired inverse operations alongside the consolidated mechanism.

The deterministic program is closed. The clean keeper is frozen, and the verbal N20 branching substrate has passed the deterministic validity gate. The next action is to lock a powered coverage margin, then test whether target-conditioned stochastic latent trajectories add exact multi-solution coverage beyond output sampling and additional depth.

## References

1. Baek, J., Jo, M., Kim, M., Ren, M., Bengio, Y., and Ahn, S. (2026). [Generative Recursive Reasoning](https://arxiv.org/abs/2605.19376).
2. Bai, S., Kolter, J. Z., and Koltun, V. (2019). [Deep Equilibrium Models](https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html). NeurIPS.
3. Banino, A., Balaguer, J., and Blundell, C. (2021). [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407).
4. Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., and Kaiser, L. (2018). [Universal Transformers](https://arxiv.org/abs/1807.03819).
5. Graves, A. (2016). [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983).
6. Hu, E. J. et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
7. Kasai, J. et al. (2021). [Finetuning Pretrained Transformers into RNNs](https://aclanthology.org/2021.emnlp-main.830/). EMNLP.
8. Kohli, H., Parthasarathy, S., Sun, H., and Yao, Y. (2026). [Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers](https://arxiv.org/abs/2604.07822).
9. Qwen Team (2024). [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115).
