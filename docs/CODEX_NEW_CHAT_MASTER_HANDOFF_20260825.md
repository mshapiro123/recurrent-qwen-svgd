# New-Chat Master Handoff - Latent-Space Reasoning Models / Paper Two

**Date:** 2026-08-25
**Purpose:** restart the research and coding collaboration in a new Codex chat without losing scientific, governance, repository, or execution context
**Repository:** `mshapiro123/recurrent-qwen-svgd`
**Working branch:** `codex/bicameral-stage0`
**Current result commit before this handoff:** `4e215a0d7238d3452d06f565684ba4cbedd64cc7`
**Current scientific status:** TM-0 complete; registered keys `STITCH-DEAD` and `ROTATION-ABSENT`
**Compute status at closeout:** no active Colab sessions
**Sealed material:** CONFIRM and EVAL-E remain unscored
**Decision status:** do not launch training or injection; strategy review and an honest-options memo are next

---

## 0. Copy-ready bootstrap message for the new chat

Paste the following message into the new chat and attach or point it to this file:

```text
Continue the latent-space reasoning-model program from the attached new-chat master handoff. Read the entire handoff before modifying code, launching compute, or proposing a new experiment.

Treat all registered thresholds, machine keys, sealed partitions, stop rules, authority documents, hashes, and do-not-claim boundaries as binding. Do not reopen a closed arm, substitute an estimator, choose a favorable seed/checkpoint, or touch CONFIRM/EVAL-E without a new written authorization.

The current branch is codex/bicameral-stage0 at the TM-0 closeout. TM-0 is banked as STITCH-DEAD plus ROTATION-ABSENT. No Colab compute should be active. The immediate task is to verify the local/Drive receipts and then help prepare or implement the next strategy decision. Do not launch another GPU experiment merely because a possible follow-up is described in the handoff.

Work autonomously when the governing documents settle an implementation detail. Ask questions only for genuine contract conflicts or scientifically consequential ambiguity. Use charts and figures when they improve interpretation. Complete implementation, verification, publication, compute teardown, and a standard coding-to-strategy handoff before stopping.
```

---

## 1. Read this first: exact current state

The latest completed wave is TM-0 trajectory-memory reconnaissance and jet geometry. It was forward-only. It constructed no optimizer, performed no training, and performed no state injection. It used a frozen 6,144-row prompt-only panel, Qwen2.5-0.5B student states, and Qwen2.5-7B/14B teacher states.

Two preregistered gates failed:

1. **`STITCH-DEAD`:** a cross-fitted linear ridge stitch could not reconstruct teacher states in the student's usable whitened metric. Raw-space reconstruction was near-perfect because broad scale and mean structure were shared, but the best whitened relative-MSE improvements were only 0.119 for 7B and 0.083 for 14B, below the 0.20 gate.
2. **`ROTATION-ABSENT`:** successful teacher trajectories did not exhibit reproducible turning planes under the primary active-token-mean `v wedge a` jet estimator. Curvature existed, but success-conditioned plane consistency and pivot structure did not clear `D_none` and smooth-noise controls.

The binding disposition is:

- bank both machine keys;
- do not construct TM-1-prime from these artifacts;
- do not run TM-2 displacement injections under the current charter;
- preserve the last-active-token jet contrast only as a possible fresh-estimator hypothesis;
- do not let that secondary signal delay closure of the present TM line;
- return to strategy for an honest-options memo or a genuinely new hypothesis.

The complete result handoff is [CODING_TO_STRATEGY_TM0_RESULT_HANDOFF_20260825.md](CODING_TO_STRATEGY_TM0_RESULT_HANDOFF_20260825.md).

---

## 2. Collaboration and research operating contract

### 2.1 How to work with Mark

- Read the codebase and governing documents before proposing implementation details.
- When implementation is authorized, build it. Do not stop at a plan unless Mark explicitly asks only for analysis.
- Give short, informative progress updates during long work.
- Ask questions only when a contract conflict cannot be resolved from the repository, Drive authority files, or receipts.
- Report failures with the exact underlying status/traceback receipt, not only the outer `CalledProcessError`.
- When reviewing results, explain them first in plain language, then provide the technical evidence and decision implications.
- Use charts, graphs, and figures when they reveal structure that tables obscure. Visually QC every figure for missing series, hidden legends, clipping, and blank panels.
- At the end of a wave, prepare the standard comprehensive coding-to-strategy handoff and publish it with receipts.

### 2.2 Scientific discipline

- Preserve preregistered estimators, populations, thresholds, bands, seeds, readers, and checkpoints.
- Never silently replace a missing or incompatible estimator. Stop and request a ruling.
- Do not select the best checkpoint, seed, ceiling, arm, or population after seeing results unless the design explicitly authorizes an exploratory selection and labels it accordingly.
- Distinguish development evidence, causal diagnostics, and capability estimates. Oracle-assisted results are causal contrasts, not model capability.
- Match calibration and training estimators exactly. The program has already caught several bugs where weights or guardrails were calibrated under a different depth mixture or reader.
- Treat the execution schedule and serving graph as part of evaluator identity. Batch-concat, runtime, reader precision, and graph differences have changed results materially.
- Keep sealed evaluations sealed until their written opening rule is satisfied.

### 2.3 Tripwires versus shapers

The program adopted a standing distinction:

- **Tripwires** are hard stops for genuine catastrophes: non-finite loss, frozen-lineage mutation, major quality collapse, identity failure, or a sealed-data violation. They stay armed.
- **Shapers** are continuous penalties, caps, loss-share constraints, and trust rents that redirect learning. Their constants must be empirically grounded. During exploration they should often observe and log, with only a generous catastrophe boundary, rather than become the experiment.

The matched-alpha pilot demonstrated why: an ungrounded trust region and an overpowered functional probe dominated the optimization and obscured the intended question.

### 2.4 Standard wave completion

A wave is not complete until all applicable items are done:

1. implementation and focused tests;
2. preflight identity/integrity gates;
3. registered execution;
4. durable row-level and aggregate receipts;
5. independent analysis or scripted verdict;
6. figure generation and visual QC;
7. hash and byte-size manifest;
8. publication to the project Drive folder;
9. coding-to-strategy handoff;
10. paid-compute teardown and an orphan-session check.

---

## 3. Program thesis and architecture evolution

The program studies whether a small pretrained language model can gain useful, economical reasoning computation through recurrent latent processing, conditional state correction, memory, or structured multi-branch computation.

The architecture family evolved through several increasingly targeted questions:

1. **Deterministic recurrent depth:** reuse a middle Qwen block, preserve latent state, and vary internal loop count.
2. **Adaptive depth/router:** learn when additional loops are useful.
3. **Bounded sidecar writeback:** predict a small correction and gate it into the frozen model.
4. **Memory:** retrieve teacher-derived or literal content and deliver it through the validated write channel.
5. **Ordered depth:** train later loops to add useful computation rather than merely repeat or drift.
6. **Bicameral processing:** split the recurrent substrate into two conditioned branches and test whether their difference carries complementary correction information.
7. **Trajectory memory:** test whether teacher computation supplies transportable state trajectories or stable jet geometry.

The program's strongest positive result remains the early deterministic synthetic mechanism: on a frozen depth-1-through-14 composition family, the recurrent 0.5B system reached 1,506/1,792 = 84.04%, compared with 952/1,792 = 53.13% for the strongest evaluated dense 0.5B serialized-scratchpad control. Beyond depth 10 it scored 272/512 = 53.13% versus 56/512 = 10.94%. This is a synthetic system-level result, not a natural-reasoning, matched-FLOP, tiny-adapter, or learned-halting claim.

The natural-task program has repeatedly shown that **actuation is easier than useful direction**. Write paths, gates, and recurrent loops can be live and behavior-changing while failing to produce reliable net corrections.

---

## 4. Banked empirical arc

| Date / wave | Registered result | Core evidence | Program consequence |
|---|---|---|---|
| 2026-07-15 Stage 5 | deterministic synthetic mechanism positive; inverse curriculum closed | 84.04% recurrent versus 53.13% dense control on frozen synthetic composition; inverse continuations failed retention/safety | preserve the deterministic evidence; do not claim natural reasoning or stochastic width |
| 2026-07-27 D0 adaptive depth | `not_recoverable_at_pilot_scale` | pooled recoverable fraction 1.003 points; adaptive teacher agreement 70.459% below plain drafter 72.617%; forced depth had capacity but the policy allocated poorly | actuation works, allocation/target design failed; pause this D0 recipe |
| 2026-08-08 Phase 2 Option B | `curve_supports_E1_recipe_transfer` | full system learned with fresh data and retained a small reproducible advantage, but neither seed met the 1% endpoint target | retain writeback for E1; do not buy another unchanged extension |
| 2026-08-11 P3.3 | replicated partial control | canonical direction capture about 15-16%, perfect registered retention, zero observed collateral; gate learned better than aim | one bounded iteration authorized; direction learning became the bottleneck |
| 2026-08-12 P3.3 i1 | `SAFE_PLATEAU` | aim-only output projection received >99.998% gradient share but mean `pi_dir` stayed 14.95%, far below 25% | duration and objective competition ruled out for this trainable set; capacity/alignment next |
| 2026-08-14 P3.4 A2 | `REPLICATED_POSITIVE_BELOW_TRIGGER_B` | DEV endpoints +5 and +10 rows, mean +7.5/1,024; CONFIRM trigger required +10 mean | positive exploratory task conversion, but no sealed confirmation; move to lever work |
| 2026-08-15 P3.5 | Branch C | stabilized endpoints +4 and +6, mean +5 below +8; EMA reduced churn but not effect size | return to Stage 2A; do not select favorable ceiling/checkpoint post hoc |
| 2026-08-18 Stage 2A T3 | `SCREEN_BELOW_PROCEED_THRESHOLD` | teacher-fingerprint memory +3/+8 versus frozen base, mean +5.5 below +8; later crossed-value audit did not resolve a content-identity effect | do not open T3-full; memory content did not earn promotion |
| 2026-08-19 Stage 2B-D | `REPLICATED_DEV1_HARD_FLOOR_STOP_AT_STEP_1000` | both seeds lost about 20 points; K1 margin 2.742 fell to about -0.088 at K4; pass-one identity remained exact | stop was mandatory; later loops learned a harmful direction |
| 2026-08-20/21 autopsy and reconciliation | objective-interface and evaluator-provenance boundary | smaller amplitude and component removals did not rescue; the apparent K4 recovery was traced to an evaluator mismatch and withdrawn | match serving graph exactly; do not preserve the invalid 162/10/2/160 story |
| 2026-08-23 2B-S final cell | `SCHEDULE-NEUTRALIZED_MARGIN_BANKED` | no-reentry writes accumulated to about 18.65% of hidden RMS and changed about 45% of answers, but net accuracy and teacher margin stayed flat | scheduling was not hiding useful computation; close this implementation line |
| 2026-08-23 Bicameral Stage 0 | structured clusters, low diagonal reach, correlated residual field | silhouettes about 0.76; diagonal banks explained about 10% of correction energy; residual structure exceeded the simple sizing model | conditional map estimand replaced fixed-direction sizing; desk gates required before training |
| 2026-08-24 Bicameral W1 | `TARGETS-NOT-ANSWER-GRADE`; `H-noise` | row-specific oracle direction changed answers; global/cluster fixed targets failed; random direction often beat population targets | write interface is live, but fixed targets are not deployable corrections |
| 2026-08-25 Bicameral W2-prime | `HEMISPHERES-UNINFORMATIVE` | prompt-only conditional cosine about 0.31, but current hemispheres did not improve held-out risk over matched single-stream features; nuisance-deflated residual cosine about 0.01-0.02 | do not run Phase G under that charter; bicameral split did not earn GPU scoring |
| 2026-08-25 TM-0 | `STITCH-DEAD` plus `ROTATION-ABSENT` | whitened cross-scale stitch failed; success-specific turning planes failed controls | close current trajectory-memory route; strategy options memo next |

This table is an orientation layer. The cited handoffs below remain authoritative for exact populations, estimators, exceptions, and hashes.

---

## 5. Detailed current result: TM-0

### 5.1 Authority and scope

- TM-0 r2 charter: Drive `1MVnzhL0oYoxm_B5tFJDzfFNAsKajmIXg`, 20,221 bytes, SHA-256 `3103ca7f81367f3a47cea0ec1b2f92de73ce0eae27240aae46e57eaa044e9460`.
- Ratified r3 execution order: Drive `1gQHhrUHlRN_l2cf3GfC1_icQnKIU009a`, 14,769 bytes.
- Preflight rulings: Drive `1w-rRoDDkhUQKvAGkQluqcMMtan-OywRF`, 9,867 bytes.
- Ratified r4 jet amendment: Drive `1GDZE-YnYU-RNHoBcMWBKcuWXjW3pxyaH`, 10,910 bytes, SHA-256 `aa354b8bd6735d2780ff7afb25925e9cb08cc325898495f6dd22146dd880080a`.
- r4 implementation commit: `1492ea65`.
- final result commit: `4e215a0d`.

r4 replaced the prior geometric section. The primary object was `v wedge a`; scalar Gram invariants were computed stitch-free in each model's whitened frame; 256 frozen Gaussian plane probes replaced the Q-sketch; full per-layer jet profiles replaced coarse windows; the pivot signature was tested against `D_none` and a step-norm-matched smooth-noise null.

### 5.2 Population and caches

- Frozen panel: 6,144 prompt-only rows.
- Battery counts: 4,963 GSM8K; 1,128 ARC-Challenge; 40 MBPP; 6 ARC-Easy; 6 MMLU; 1 Tier-1.
- Partition counts: 6,034 verified-train; 110 DEV.
- Panel SHA-256: `e108b0a92fdc69b9cb27274ac420908b65303213307f9d8dfc1f4ba73d58b5ca`.
- Student cache bundle: `6cf589410562eb23e6ec7aaa5f322301fc583b0133a30f68300f0c51a60429ab`.
- 7B cache bundle: `fbba8216b379965a43dc357e95c6806936e7dbf0b4ec7e4475cde55fa33da752`.
- 14B cache bundle: `5a373ccf410758c1a892b5a173dca56addb0baf37f6d8baaf41066d7ba7b92d3`.
- Each model cache: 96 shards, 6,144 rows, prompt-only, sequential execution.
- Merged 7B correctness score SHA-256: `e884180f8545fd964c444bbad304506216ea124498042dcc074ae13b07f766f9`.
- The 7B score recovery merged three deterministic sources; 648 overlapping rows agreed exactly.

### 5.3 TM-1 stitch result

Linear-CKA calibration was stable across two disjoint 512-row subsets:

| Teacher | Subset A selected layer | Subset B selected layer | Stability |
|---|---:|---:|---|
| 7B | 7 | 8 | pass, difference 1 |
| 14B | 10 | 10 | pass, difference 0 |

The calibrated and neighboring layers then failed the whitened cross-fitted reconstruction gate:

| Teacher | Best whitened relative-MSE improvement | Gate | Best raw-space improvement | Result |
|---|---:|---:|---:|---|
| 7B | 0.119 | 0.200 on both halves | 0.995 | fail |
| 14B | 0.083 | 0.200 on both halves | 0.995 | fail |

The map is not random: cosine-over-random was roughly 0.32-0.55. The failure is that the apparently excellent raw reconstruction does not survive putting directions on equal footing. Do not cite raw 0.995 as transportability.

### 5.4 TM-2g-J primary jet result

| Teacher | Success stratum | Rows | Plane consistency minus `D_none` (95% CI) | Minus smooth noise | Two-half balanced accuracy | Result |
|---|---|---:|---:|---:|---:|---|
| 7B | `D_7>0.5` | 1,902 | +0.0012 [-0.0005, +0.0030] | -0.0821 | 0.586 / 0.595 | fail |
| 7B | `D_14>0.5` | 1,426 | +0.0005 [-0.0013, +0.0023] | -0.0827 | 0.638 / 0.662 | fail |
| 14B | `D_7>0.5` | 1,902 | -0.0023 [-0.0037, -0.0009] | -0.0354 | 0.604 / 0.590 | fail |
| 14B | `D_14>0.5` | 1,426 | -0.0027 [-0.0040, -0.0012] | -0.0355 | 0.635 / 0.628 | fail |

Curvature was nonzero on all rows and layers, but that is a calibration fact, not success geometry. The last-active-token secondary showed a small positive contrast in three of four cells, about +0.0014 to +0.0026. It was not the primary estimator and cannot change the registered key.

### 5.5 R-1 rider

The no-refit GSM8K conditional relation survives weakly:

| Seed | GSM8K conditional cosine (95% CI) | Pooled cosine |
|---:|---:|---:|
| 0 | 0.290 [0.252, 0.329] | 0.306 |
| 1 | 0.313 [0.276, 0.349] | 0.323 |

This says a weak prompt-state-to-correction relation remains. It does not show that the present hemispheres improve it, that it is causal, or that it rescues trajectory memory.

### 5.6 Current claim boundary

Supported:

- broad representations correlate across scale;
- a weak deployable prompt-state relation exists on GSM8K;
- teacher trajectories have nonzero curvature;
- the tested write interfaces are mechanically live in prior experiments.

Not established:

- usable cross-scale teacher-state transport;
- success-specific reusable rotation planes;
- a beneficial fixed correction target;
- complementary information from the current hemispheric split;
- a capability gain on sealed data;
- any conclusion about 32B internal geometry.

Do not claim that teacher/student states are unrelated, that all nonlinear stitches are impossible, that teachers do not reason geometrically, or that all trajectory memory is impossible.

---

## 6. Closed arms, live hypotheses, and authorization boundary

### 6.1 Closed under current evidence

- D0's binary teacher-disagreement router recipe.
- Another unchanged Option B extension.
- P3.4/P3.5 confirmation from the observed DEV effects.
- T3-full from the Stage 2A memory screen.
- Stage 2B-D continuation after the step-1,000 hard stop.
- Further scheduling/tuning of the 2B-S implementation.
- Fixed global, cluster-mean, and tested residual-direction correction targets as answer-grade mechanisms.
- Bicameral Phase G under W2-prime.
- TM-1-prime and TM-2 injection under the TM-0 charter.

### 6.2 Scientifically live but not authorized

- A parameter-matched **single-stream conditional state-to-correction map**, because the base state weakly predicted corrections and matched or beat the current hemispheres.
- A **different specialization objective or exchange mechanism** that creates complementary branch information before re-testing bicamerality.
- A **fresh last-active-token jet estimator** on a balanced, independently frozen population. This must be newly preregistered and cannot inherit the active-token-mean positive bar.
- A **nonlinear or jointly trained cross-scale map**, provided it is motivated by a mechanism that addresses the whitened stitch failure and is tested against proper controls.
- A **32B internal-geometry rung**. It was not run; no conclusion should be extrapolated to it.
- A design that trains corrections against **ordered margin improvement or explicit helps/harms labels**, rather than expecting schedule changes alone to make writes useful.

These are options for strategy, not coding authorization. Do not launch any of them without a new governing document.

---

## 7. Seals, data governance, and teacher roles

### 7.1 Sealed partitions

- **CONFIRM:** remains sealed and unscored.
- **EVAL-E:** remains sealed and unscored.
- The TM-0 hermetic screening-index job used salted exact-match hashes and MinHash/LSH signatures to screen contamination without releasing plaintext or model scores. It was infrastructure protection, not an evaluation read.
- Do not materialize, score, inspect, or backfill sealed populations without a written rule that explicitly authorizes the operation.

### 7.2 Standing teacher ladder

The teacher ladder has had different roles at different phases:

- **14B:** principal cached teacher lattice, token target, and TM-0 state source.
- **32B:** verifier/concurrence role where available; its internal TM geometry was deferred.
- **7B:** reserve/intermediate teacher in strategy and a full state/correctness source in TM-0.

Teacher agreement is not truth. D0 measured that the 14B endorsed the drafter token on about 16.567% of cached 7B rejections, quantifying noise/disagreement in a teacher-defined target. Wherever possible, retain programmatic answer checks and teacher-family concurrence for write-positive labels.

### 7.3 Development versus capability language

- Teacher-token margin and oracle-direction capture are mechanism estimators.
- Oracle-routed or oracle-target-assisted generation is a causal contrast, not deployable capability.
- DEV changes are exploratory and cannot become confirmation by favorable narrative.
- Sealed capability claims require the exact registered serving graph, reader, estimator, and opening rule.

---

## 8. Repository and workspace state

### 8.1 Canonical local checkout

```text
C:\Users\mshap\Documents\Codex\2026-06-15\below-is-a-codex-ready-handoff\.work\bicameral-stage0
```

Current Git state before this handoff was created:

- branch: `codex/bicameral-stage0`;
- HEAD: `4e215a0d7238d3452d06f565684ba4cbedd64cc7`;
- remote branch: `origin/codex/bicameral-stage0` at the same commit;
- relation to `origin/main`: 26 commits ahead, 0 behind;
- tracked files: clean;
- untracked files: numerous execution archives, caches, temporary transport files, and recovered artifacts.

**Do not delete or reset untracked artifacts.** They include durable recovery material from Bicameral W1/W2-prime and TM-0. Treat them as user/generated evidence unless a cleanup order names exact paths and confirms their publication hashes.

### 8.2 Important code and artifact roots

- `models/`: recurrent, sidecar, memory, and Bicameral model code.
- `training/`: locks, runners, and training contracts.
- `eval/`: task and diagnostic evaluators.
- `analysis/`: reproducible result analyzers and figure generators.
- `colab/`: bootstrap targets and runtime launchers.
- `docs/`: handoffs, strategy documents, figures, and public receipts.
- `artifacts/tm0_20260825/`: complete TM-0 local evidence tree.
- `artifacts/bicameral_w1_20260824/`: W1 ladder, generation, and recovery artifacts.
- `artifacts/bicameral_w2p_20260825/`: W2-prime CPU desk-gate artifacts.

### 8.3 Current validation

- 61 relevant TM-0 and inherited Bicameral tests passed in 9.37 seconds.
- Independent end-to-end analysis reproduced `STITCH-DEAD` and `ROTATION-ABSENT`.
- PNG/SVG figure pairs were visually checked; no missing series, obscured legends, clipping, or blank panels were found.
- At closeout, `colab sessions` reported no active sessions.

---

## 9. Compute, authentication, and execution lessons

### 9.1 Colab operating practice

- Prefer the `colab` CLI for repeatable launch, status, file transport, and teardown.
- Authentication may expire independently from a live remote process. Before restarting anything, distinguish CLI-token failure from VM/process failure.
- On reconnect, perform a read-only health check first: endpoint identity, process PID, log mtime, file-size growth over 60 seconds, GPU utilization, and mount health.
- Never cancel or replace a live session merely because the desktop app or CLI froze.
- Use `/mnt/local-scratch` when available. It avoids DriveFS bottlenecks and has previously exposed roughly 368 GiB of local scratch.
- Mirror durable checkpoints/receipts to Drive incrementally and validate hashes before reusing completed cells.
- Select GPU class by measured memory and runtime need. Do not insist on 80 GB when a 40 GB or L4 run is scientifically and operationally sufficient, but do not quantize or alter the model to fit without authorization.
- The user has explicitly stated that a roughly two-hour GPU run costing only a few dollars is acceptable when the information value warrants it. Cost discipline still means dry-run pricing, caps, and no idle orphan sessions.

### 9.2 Authentication hygiene

- Google ADC and Colab CLI authentication have required browser approval during this program.
- Hugging Face device authentication has also been used.
- Never place authorization codes or access tokens in repository files, handoffs, logs, or final responses.
- If Drive mount authorization fails, report that infrastructure failure separately from the scientific process.

### 9.3 Repeated failure classes already solved

- Outer Colab `CalledProcessError` messages hide the actual failure. Always inspect durable `status.json` or the unbuffered child log.
- DriveFS can truncate or delay sync. Use atomic writes, local scratch, file-size checks, and SHA manifests.
- Checkpoint container hashes can change after reserialization even when tensor state is identical. Use exact tensor-state digests when the contract concerns model state.
- Reader precision and execution graph matter. BF16/FP32, sequential/batch-concat, and native/mismatched serving graphs have changed conclusions.
- A successful runtime must still be explicitly torn down.

---

## 10. Drive publication and canonical receipts

Project research folder:

```text
Drive folder ID: 1aSbU2i8JZ37g5bJpyweLuvFaV0y92Qjr
```

Latest TM-0 publication:

- Result handoff: Drive `1jhGiiHMtVX0tCJEpOwDXhOJGFkRpqZDz`, 12,205 bytes, SHA-256 `caded0a6a3da1e26eb2eb74a1a7fe2efda0912962b55d629fd81be7c0404bae5`.
- Evidence bundle: Drive `1dwNWY7BmXEGFck7mnuodQgv-iyO7geDW`, 837,867 bytes, SHA-256 `b55c4cbb8049cfa049d702049776bedf8a3314a625c41a01dd75b6d088364be0`.
- Assertions: Drive `17_nRY9wEsXLNw_Kfbmyniyt4AOG-qDWQ`, 1,182 bytes.
- Artifact manifest: Drive `12-B7mHioiNTwducGZzwdi8BxexC7S5Ed`, 5,952 bytes.

The handoff and evidence bundle were downloaded from Drive and byte-verified against the local committed files.

Primary local TM-0 receipts:

- `artifacts/tm0_20260825/results/tm0_cpu_pipeline_status.json`
- `artifacts/tm0_20260825/results/tm0_result_assertions.json`
- `artifacts/tm0_20260825/results/tm0_result_artifact_manifest.json`
- `artifacts/tm0_20260825/results/tm1_stitch_summary.json`
- `artifacts/tm0_20260825/results/tm2g_jet_summary.json`
- `artifacts/tm0_20260825/results/tm0_r1_receipt.json`
- `artifacts/tm0_20260825/results/teacher_7b_transport_merge_receipt.json`

Primary figures:

- [TM-1 CKA calibration](figures/paper2_tm0_cka_calibration_20260825.png)
- [TM-1 stitch gates](figures/paper2_tm0_stitch_gates_20260825.png)
- [TM-2g jet profiles](figures/paper2_tm0_jet_profiles_20260825.png)
- [TM-2g decisive contrasts](figures/paper2_tm0_jet_decisive_contrasts_20260825.png)

---

## 11. Canonical reading list

Read these in order when reconstructing the program. Do not infer current authorization from an older document if a later handoff supersedes it.

### 11.1 Foundational deterministic evidence

1. [STAGE5_COMPLETE_HANDOFF_20260715.md](STAGE5_COMPLETE_HANDOFF_20260715.md)
2. [PAPER2_D0_PILOT_RESULT_HANDOFF_20260727.md](PAPER2_D0_PILOT_RESULT_HANDOFF_20260727.md)
3. [PAPER2_D0_ROUTER_FEASIBILITY_HANDOFF_20260727.md](PAPER2_D0_ROUTER_FEASIBILITY_HANDOFF_20260727.md)

### 11.2 Phase 2 and Phase 3 writeback program

4. [PAPER2_PHASE2_OPTION_B_FINAL_RESULT_HANDOFF_20260808.md](PAPER2_PHASE2_OPTION_B_FINAL_RESULT_HANDOFF_20260808.md)
5. [PAPER2_PHASE3_P33_RESULT_HANDOFF_20260811.md](PAPER2_PHASE3_P33_RESULT_HANDOFF_20260811.md)
6. [PAPER2_PHASE3_P33_I1_RESULTS_HANDOFF_20260812.md](PAPER2_PHASE3_P33_I1_RESULTS_HANDOFF_20260812.md)
7. [PAPER2_PHASE3_P34_A2_RESULTS_HANDOFF_20260814.md](PAPER2_PHASE3_P34_A2_RESULTS_HANDOFF_20260814.md)
8. [PAPER2_PHASE3_P34_STABILITY_AND_REFRESH_HANDOFF_20260815.md](PAPER2_PHASE3_P34_STABILITY_AND_REFRESH_HANDOFF_20260815.md)
9. [PAPER2_PHASE3_P35_RESULTS_HANDOFF_20260815.md](PAPER2_PHASE3_P35_RESULTS_HANDOFF_20260815.md)

### 11.3 Stage 2A memory and Stage 2B depth

10. [PAPER2_STAGE2A_T3_SCREEN_RESULTS_HANDOFF_20260818.md](PAPER2_STAGE2A_T3_SCREEN_RESULTS_HANDOFF_20260818.md)
11. [PAPER2_STAGE2A_CV1_D5_RESULTS_HANDOFF_20260818.md](PAPER2_STAGE2A_CV1_D5_RESULTS_HANDOFF_20260818.md)
12. [PAPER2_STAGE2B_DEPTH_REGISTERED_STOP_HANDOFF_20260819.md](PAPER2_STAGE2B_DEPTH_REGISTERED_STOP_HANDOFF_20260819.md)
13. [PAPER2_STAGE2B_AUTOPSY_RESULT_HANDOFF_20260820.md](PAPER2_STAGE2B_AUTOPSY_RESULT_HANDOFF_20260820.md)
14. [PAPER2_STAGE2BS_PRELUDE_RESULT_HANDOFF_20260821.md](PAPER2_STAGE2BS_PRELUDE_RESULT_HANDOFF_20260821.md)
15. [PAPER2_STAGE2BS_RECONCILIATION_RESULT_HANDOFF_20260822.md](PAPER2_STAGE2BS_RECONCILIATION_RESULT_HANDOFF_20260822.md)
16. [PAPER2_STAGE2BS_FINAL_CELL_RESULT_HANDOFF_20260823.md](PAPER2_STAGE2BS_FINAL_CELL_RESULT_HANDOFF_20260823.md)

### 11.4 Bicameral and trajectory-memory program

17. [PAPER2_BICAMERAL_STAGE0_RESULT_HANDOFF_20260823.md](PAPER2_BICAMERAL_STAGE0_RESULT_HANDOFF_20260823.md)
18. [PAPER2_BICAMERAL_W0_RESULT_HANDOFF_20260824.md](PAPER2_BICAMERAL_W0_RESULT_HANDOFF_20260824.md)
19. [PAPER2_BICAMERAL_W1_RESULT_HANDOFF_20260824.md](PAPER2_BICAMERAL_W1_RESULT_HANDOFF_20260824.md)
20. [STRATEGY_BICAMERAL_W2P_CONDITIONAL_MIXER_CHARTER_20260825.md](STRATEGY_BICAMERAL_W2P_CONDITIONAL_MIXER_CHARTER_20260825.md)
21. [CODING_TO_STRATEGY_BICAMERAL_W2P_PHASE_D_RESULT_HANDOFF_20260825.md](CODING_TO_STRATEGY_BICAMERAL_W2P_PHASE_D_RESULT_HANDOFF_20260825.md)
22. [STRATEGY_TM0_R3_EXECUTION_ORDER_20260825.md](STRATEGY_TM0_R3_EXECUTION_ORDER_20260825.md)
23. [STRATEGY_TM0_PREFLIGHT_RULINGS_20260825.md](STRATEGY_TM0_PREFLIGHT_RULINGS_20260825.md)
24. [CODING_TO_STRATEGY_TM0_RESULT_HANDOFF_20260825.md](CODING_TO_STRATEGY_TM0_RESULT_HANDOFF_20260825.md)

---

## 12. First-turn checklist for the new coding agent

Before doing substantive work:

1. Read this entire document and the latest TM-0 result handoff.
2. Confirm the working tree and branch. Do not reset, clean, or delete untracked artifacts.
3. Confirm `origin/codex/bicameral-stage0` contains the TM-0 result commit.
4. Confirm the TM-0 handoff and bundle hashes if Drive access is available.
5. Run `colab sessions` only as a read-only check; expected result is no active sessions.
6. Verify that CONFIRM and EVAL-E remain unscored in the latest machine receipts.
7. Do not construct an optimizer, launch a GPU, or create an injection run.
8. Ask Mark whether a new strategy response/handoff has landed since TM-0 closeout.
9. If no new authority exists, help prepare the honest-options memo from the accumulated boundary rather than inventing an experiment.
10. If a new strategy document exists, byte-verify it, reconcile it against this handoff and the current machine keys, surface any contradiction, then implement only the authorized scope.

The first status message in the new chat should state, in one short paragraph:

- repository/branch/commit found;
- whether tracked files are clean;
- whether Colab is inactive;
- whether the TM-0 receipts are present;
- whether a newer strategy authority exists;
- what action is being taken next.

---

## 13. Standard coding-to-strategy handoff structure

Use this structure for future result handoffs:

1. **Metadata and status:** date, wave, branch/commit, authorization, compute state, machine key.
2. **Executive result:** one paragraph and a concise result table.
3. **Question and rationale:** what uncertainty the wave was designed to resolve.
4. **Authority and lock:** Drive IDs, hashes, ratification, amendments, exact scope.
5. **Population and lineage:** datasets, partitions, model/checkpoint hashes, reader and runtime identity.
6. **Design:** arms, estimators, thresholds, controls, stop rules, and deviations.
7. **Results:** primary first, then secondary and mechanism telemetry.
8. **Interpretation:** what is supported, what is not, and how the result changes the plan.
9. **Limitations and do-not-claim list.**
10. **Registered disposition and questions for strategy.**
11. **Receipts:** local paths, Drive IDs, byte sizes, hashes, tests, figures, session teardown.
12. **Plain-language close.**

Every handoff should make it possible for another agent to reproduce the decision without relying on chat memory.

---

## 14. Immediate decision frame for strategy

The accumulated evidence now rules out several convenient explanations:

- the write path is not simply dead;
- extra depth is not merely being averaged away;
- smaller amplitude alone does not rescue the harmful Stage 2B direction;
- fixed global or cluster directions are not answer-grade;
- the current hemispheres do not add held-out predictive value beyond the base state;
- raw cross-scale similarity does not imply usable state transport;
- visible trajectory curvature does not imply success-specific reusable planes.

The remaining decision is architectural and theoretical, not a request for more tuning of the current implementation. A defensible next program should explain **how it will learn row-specific, answer-grade direction using deployment-available information**, and it should earn that claim on a cheap desk or causal gate before a full campaign.

The cleanest live comparison is likely between:

1. a parameter-matched single-stream conditional correction map;
2. a new specialization mechanism that creates genuinely complementary internal views;
3. a different supervision target based directly on realized helps/harms or ordered margin improvement;
4. closing the natural-task architectural search for Paper Two and writing the accumulated boundary as the result.

No choice among these has been authorized in the current record.

---

## 15. Final plain-language orientation

The project has repeatedly built mechanisms that are alive: they change internal states, open gates selectively, accumulate writes, and alter answers. The persistent obstacle is not whether the system can intervene. It is whether the system can infer a **question-specific correction that helps more often than it harms** using information available at deployment.

The latest experiments sharpen that boundary. Two inherited branches did not provide complementary information. Fixed teacher-derived directions did not translate into useful answers. Teacher trajectories did not supply a reliable cross-scale map or a stable success-specific rotation pattern. The correct next move is therefore not another unmodified training extension. It is a strategy decision about a new source of conditional direction, or an honest close to the current architectural search.

That is the state the new chat must preserve.
