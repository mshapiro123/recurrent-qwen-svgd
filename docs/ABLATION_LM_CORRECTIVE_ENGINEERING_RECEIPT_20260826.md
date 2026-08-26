# Ablation LM corrective engineering receipt

**Date:** 2026-08-26
**Scope:** architecture substrate, tests, and repository baseline triage only
**Base commit:** `85bb8eae5cb70f7660b817af0b21b09b090782ef`
**Authority:** build-only. This receipt does not authorize tokenizer fitting or freezing, optimizer construction, training, evaluation-panel contact, or sealed-partition access.

## Outcome

The strategy review's R-1 arithmetic objection is confirmed and corrected. R-2 identifies a real naming and future-use hazard, but no hyperbolic boost was active in the model forward path. R-4 does not describe the current parameter graph: no learned `mu`/`delta` pair was split across optimizers. The patch nevertheless closes the future enforcement gap with a Transformer-only Muon allowlist and keeps every coupled mode matrix on AdamW.

The ablation-focused tests pass. The full repository suite remains intentionally red at exactly three legacy nodes: `3 failed, 3089 passed, 19 warnings`. A strict engineering runner executes every test and passes only when that exact set, with no additions, removals, or renames, is observed. This is not a repo-wide green claim, and training remains blocked.

## Code-level rulings

| Ruling | Actual implementation finding | Corrective disposition |
|---|---|---|
| R-1 target accounting | The reported 41.55% and 36.25% shares reconstruct the `d=512` 4/6-core μProxy exactly. They are not target-scale shares. | `d=512` is now named μProxy. The decision reference is `d=1024`, approximately 290M, where the 32K matrix is 33,554,432 parameters or 11.57%. Selection rederives public rows from their stored typed contract. Fixed-total contracts bind all four registered vocabularies on one registered rung to one feasible common budget/tolerance; fixed-body contracts derive candidate totals from one rung's reference topology. A final freeze still requires both 4- and 6-core rulings, so it remains blocked. |
| R-2 Clifford signature | The removed helper implemented `Cl(1,0)` with a positive square, but it had no model call site. The live graph used only the exact lane mean/difference coordinate transform. No rotor, exponential, `cosh`, `sinh`, or boost was present. | Lane coordinates are no longer called Clifford. The first typed Clifford primitive is an explicit Euclidean `Cl(2,0)` rotor with `B^2=-1`, and FP64 T5 checks relative norm error below `1e-6`. Angles must be scalar or exactly match the vector batch shape. Hyperbolic carrier maps are prohibited. |
| R-4 optimizer ownership | In the reviewed all-pillar tiny graph, 77 unique trainables partitioned into 28 Transformer dense matrices eligible for Muon and 49 AdamW auxiliaries. There were no named `mu` or `delta` parameters; those are activations. Packed scratch matrices were already whole-parameter AdamW assignments. | Coupled dense/factored paths use the `COUPLED_MODE` role and stay together on AdamW. A wrapper-aware closed allowlist permits only Transformer attention and SwiGLU weights. The legacy rank-only splitter rejects full and filtered ablation inventories while preserving the existing recurrent-Qwen path. Deepcopy, device transforms, and assign-load restore safety markers; meta/assign also restores tied vocabulary pointer identity. Mode-wise Muon remains unimplemented. |

## Added stability and diagnostic gates

- The two-lane carrier now enforces the exact horizon-wide minimum retention bound `(1 - 2 rho)^K >= 0.9`; the default `rho` is 0.005 and every diagnostic forward reports the bound.
- Hadamard routing reports valid-token logit mean `m`, logit scale `s`, expert load, load dispersion, and routing entropy. Calibration cannot freeze at step zero and requires two consecutive nonzero windows to pass finite registered tolerances.
- The optional z-loss uses the same valid shifted-token mask as cross-entropy; coefficient zero is a structural loss-identity path that does not execute another full-vocabulary `logsumexp`.
- Recurrent diagnostics retain `z_0` and every post-visit state; all visit velocities, acceleration, curvature, Gram eigenvalue ratio, and deterministic plane probes have the required three-state history and an explicit valid-token mask. Stationary and colinear jets report zero Gram ratio rather than epsilon-induced pseudo-rank.
- Valid-token hidden, composite-update, and terminal scratch-mode RMS are reported. Hidden-state adjoints expose valid-token per-visit RMS and adjacent cosines after backward. These are not mislabeled as shared-parameter contribution cosines.
- Local recurrent-visit Jacobian probes use a dimension-balanced joint hidden/scratch metric. The full unroll reports hidden-to-hidden gain. Both project onto valid tokens, and the explicit math-attention path makes all-masked rows forward-mode safe.
- The `Cl(2,0)` rotor is angle-parameterized, computes low-precision angles in FP32, has an exact batch-shape broadcasting contract, and has a pure-tensor JVP/vmap/compile path; its scalar and bivector properties therefore remain unit-normalized after angle updates.
- Parameter accounting discovers the tied vocabulary through transparent wrappers. Meta-device transforms and `assign=True` checkpoint loads restore the tied embedding/head object identity, unique-parameter count, and optimizer-safety provenance; disagreeing checkpoint aliases fail before mutation.

## Repository baseline disposition

Ten initially failing node IDs were raw-byte transport failures under `core.autocrlf=true`. Their canonical LF payloads matched `HEAD` and their registered authorities. The repair materialized the exact `HEAD` blobs and added path-specific `text eol=lf` attributes. No governing content, registered hash, or expected test value changed. No sealed file was accessed or modified.

The first eleven pre-repair CRLF values below are deterministic reconstructions from the canonical LF payloads because those raw snapshots were not hashed before restoration. The remaining fourteen CRLF values were observed before restoration. In every row, the canonical side equals the `HEAD` payload and the registered authority.

### First restoration set: reconstructed CRLF evidence

| Path | CRLF bytes | CRLF SHA-256 | Canonical LF bytes | Canonical LF SHA-256 |
|---|---:|---|---:|---|
| `docs/COMPOSITE_TRAINING_DESIGN_20260729.md` | 10,632 | `08a754140d9f5655c296032cb260a1f76f3b7bdf338eb307a6035d2692cb6c4e` | 10,570 | `0ae848f560dda18abc89deb7716b53b24f40b49f5a7d44a6d5f2e514c9d5ed7b` |
| `docs/STRATEGY_ADDENDUM_DC1_ROADMAP_20260729.md` | 3,137 | `876269201801bd26c4e8d094eab79215d4f968af0ff72b4b9f77dc741bfa5283` | 3,112 | `67a38f52529fadf79a9b229e8a88d045a645a1f36cdfc2be89a1effec953a78b` |
| `docs/PHASE_DC1_STAGE_A_PREREGISTRATION_DRAFT1_20260730.md` | 14,455 | `83548fe45481b67fc5911066d9309d4aef4bd139822ab1ccea641ed276d0e007` | 14,333 | `bd834c42d92b559dabd638c326dd76724f24adba6ade27bcdd4adb32703dc581` |
| `docs/PAPER2_PHASE2_A2_STEP237_TRIPWIRE_AMENDMENT_20260806.md` | 3,648 | `6ac360acbbec5b7d927dbfc87cea0bdaa5aa0110b4c26cee19b94e90e2045850` | 3,577 | `75ec9fd107ae1cf74a9414fc7f46b7b03d066c4554b53d4cc7a1740fdb38c895` |
| `docs/PAPER2_PHASE2_OPTION_B_POST_GENERATION_HASH_AMENDMENT_20260807.md` | 3,406 | `c7da140537c0bc83df8f18586b498461a895692d4e68c358f1f7dc076760703a` | 3,342 | `3a345b1c31b7855e7753d48abded8f92dbb44e1fea9670ddfaa13f6bacfeec56` |
| `docs/STRATEGY_HANDOFF_P31_P33PREP_20260810.md` | 5,440 | `21ecfc3c60328be0d40a569988a7af078d3a97e348e91db53860a4db6eb6b618` | 5,401 | `1ef98934ec2ce5164536a245f1a7931699568f7488c80a403c820c0b30675910` |
| `docs/STRATEGY_P34_RESULT_RESPONSE_20260814.md` | 12,992 | `21ec4f6f6653934534ef889b82dee30c08ada0b891a672ac40a2030bfbb59fd8` | 12,943 | `76b4dd29024f86fb6b01c76ba747e7f60eea0280315b233ab51923f61308a761` |
| `docs/STRATEGY_2BS_FINAL_CELL_AUTHORIZATION_20260823.md` | 4,234 | `1b33c0d7dbc1b932f493f943f3f846206b92c7bf66cc2bc6624cdcae11486dd4` | 4,196 | `60b52390d2db1e898a88bffaba494211e700322154c08208edc462f684c20911` |
| `docs/STRATEGY_ADDENDUM_D0_FIGURE_REVIEW_20260727.md` | 2,813 | `5f825478aaa6e398977ec2f1fb8ef5d37b33bc75d0b1fd18a4d980cc51d142b9` | 2,795 | `ff93c5011872e91d64dcdc380169b93beedd3097b6cee2987722d28af06a36b8` |
| `docs/figures/composite_architecture_20260729.svg` | 18,709 | `869370a524ce45f078c4ebfa26a4ef4df37195bed124bd304ee3218ab5af4824` | 18,487 | `444aa15ae4210096a7082d23ec9ec88380f25b1c96624808fc3107ee7907cf9f` |
| `outputs/stage5/stage5_paper2_phase3_p34_sampled_depth_preflight_20260813/receipts/main_seed_0.json` | 13,728 | `f578537fe5f40c1082baeb45300b7c2c08e79cc046ed74630a026c801cc1d972` | 13,291 | `39d7c7e7cd7676508bc5df415a60286d9eb67ac27678a67b3d17a9b67e35e762` |

### Exhaustive manifest trace: observed CRLF evidence

| Path | CRLF bytes | CRLF SHA-256 | Canonical LF bytes | Canonical LF SHA-256 |
|---|---:|---|---:|---|
| `docs/STRATEGY_TO_CODING_AGENT_A2_TRIPWIRE_RESOLUTION_20260806_r3.md` | 9,630 | `9d5eef0601c34f6aa2794399a897e4dd7964d05a67551c877e02451aa436fbdf` | 9,603 | `6e485df0ac40db4fc07431b20e6890c2545fd6043539200dcd599e2e5b270a45` |
| `docs/STRATEGY_GUARDRAIL_DOCTRINE_20260806.md` | 4,893 | `89cf77c75209d70fba1eb5c57043ae930a8b2813a0bebfc88f2c6a126356d464` | 4,864 | `d8f38b21bb8ad23a03fa88b4c9a7f0282bd435116b792c5e36fbae17102acb12` |
| `docs/PAPER2_PHASE2_A2_STEP237_HASH_PORTABILITY_ERRATUM_20260806.md` | 1,846 | `352ee5d0a77bda687636f0a8f42fc64b86a5d3e6eecd6334c9aee6d22fca4767` | 1,818 | `aa8bde489f98d99eb4259edb922b894d10b68fd071560a328606375350f00d45` |
| `docs/PAPER2_PHASE2_A2_EXTENSION_RESUME_INCIDENT_20260806.md` | 2,319 | `12e2261344694d6fdc643a037bbcffdbdaa9dfdd7aac9395f94826dd018fae97` | 2,286 | `5e5179a0799826cede216e32aa9d086a5832398563e6dc66c56f2a2ff94e920c` |
| `docs/PAPER2_PHASE2_OPTION_B_ENDPOINT_RESERIALIZATION_ERRATUM_20260807.md` | 2,898 | `3907c3cb7c6ff5a3afc1a702ae423e90524e9775bdb2e2e4bba384d8a1ed358b` | 2,841 | `7949b6e94f0746ae3e8918afee54d99b89ec2ae7930b4044c1690a7fd2d392c3` |
| `docs/STRATEGY_P33_PROTOCOL_LOCK_20260811.md` | 8,644 | `38ac694691161a38835d40e312a09236c2ec6dab43567e305bfa560475034a26` | 8,588 | `45e2221bc94cf6c13df38c7d0bcdbb4075256792dc5968cb33b1076336455c8d` |
| `docs/PROGRAM_RECORD_P33_PROTOCOL_LOCK_RATIFIED_20260811.md` | 1,756 | `a3205f661a9487600b3039ef72da084e6b2c111dcb734f72c7b449dcf8499405` | 1,748 | `4f3a333dbdee1d9379fbca2711fefbadd5bda0c17717e90f4988cba2c6af68f2` |
| `docs/STRATEGY_P33_LOCK_ERRATUM_E1_20260811.md` | 5,215 | `8edd020372b512519bbed3ac7d9336910ba2a9cb4dbb2bc6fe506fb15049bec8` | 5,189 | `ff2d65e32495f2f15f7d05578a10b27eb15673424a26b13b72a004fd7aef0e51` |
| `docs/STRATEGY_P33_LOCK_ERRATUM_E2_20260811.md` | 4,460 | `83013746ac45a0ad7699ee72241297d9a032ce01017ec0f0cf8323434a775ee3` | 4,442 | `4efc06f8435adb1a80027d217f17ad0d4b95ce1494565f9d8c28d6bdfa356f33` |
| `outputs/stage5/stage5_paper2_phase3_retention_preflight_20260811/summary.json` | 21,449 | `6ce71422d06f47ff909e0cab0d4cbfdc671c4a2b3c17f7177cec6922cf10dcdb` | 20,913 | `9a71e3e59526383b3dd830a320a0e18ad3778571f67dac1e262ee2713ea0ffd0` |
| `outputs/stage5/stage5_paper2_phase3_p34_sampled_depth_preflight_20260813/receipts/main_seed_1.json` | 13,662 | `43e19a2ad39b5da2454ba9f0ee9bbbf609e584ebaa74f0f10515897ba9179a0a` | 13,227 | `c4d1f9d136abc2a64214d3502c3d985e5c3349f56851d8c37140e9454ab897d2` |
| `outputs/stage5/stage5_paper2_phase3_p34_sampled_depth_preflight_20260813/receipts/slot_seed_0.json` | 15,081 | `84ce43f0de474f72b6738fc99456b385e4a7c0ca135f4cac28d285c48cd9e37e` | 14,606 | `70562b1588e2cbee8130fb1e0a4897462a268c8519d7ce05bc0e03a3701d2df4` |
| `outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/receipts/p34_task_guardrail_calibration.json` | 94,465 | `eb50fcacf1197e2436c34cb2d40e0a8cafbdd0cb6ba8dea7211e738209bcc17a` | 92,355 | `2efad887a5beade3bd21fc4d10caecae3675d6f1e25129b23cb0d9596d191476` |
| `docs/PAPER2_PHASE3_P34_AMENDMENT_A2_DRAFT_20260814.md` | 11,192 | `6f33b903e8321cb0ff62dba20cc54c115afab9f0e04c63be4d5863a32a4473be` | 11,056 | `0d6a7a4d7b07c16ec6d790af1ab931e1c88a0f7f54902bc68814b6e654fc9320` |

One nearby artifact is deliberately excluded from LF pinning: `outputs/stage5/stage5_paper2_phase3_p34_a2_autopsy_20260814/summary.json`. Its registered authority is the raw CRLF payload, 129,060 bytes with SHA-256 `25836439b34bebd83fd63286a1876cf974305f324479a0687c75e15e5037b1d4`. Its LF/`HEAD` form differs. Converting it would create a real receipt failure.

## Exact engineering quarantine

The machine-readable authority is `training/ablation_lm_engineering_quarantine_20260826.json`. The runner is `scripts/run_ablation_lm_engineering_gate.py`; it runs the full suite and compares the observed `FAILED` node IDs to the receipt.

| Exact node ID | Disposition |
|---|---|
| `tests/test_paper2_claim_evidence_ledger.py::test_paper2_claim_evidence_paths_exist` | Legacy evidence availability debt. Paper/publication evidence audit remains red until the two paths are restored or governed ledger authority changes them. |
| `tests/test_stage5_notebooks.py::test_current_bootstrap_target_markers_exist_in_launcher_files` | Autopsy v2 registry versus v3 launcher drift. That target remains stopped fail-safe. |
| `tests/test_stage5_notebooks.py::test_current_a100_bootstrap_plain_cell_matches_markdown_code` | Python bootstrap is a strict superset of its Markdown mirror. The Markdown form is not current and must not be published or pasted. |

The runner's verified result was:

```text
ablation engineering gate PASS: all tests ran and exactly 3 quarantined legacy nodes failed
full repository suite remains RED: 3 failed, 3089 passed, 19 warnings in 59.82s
```

## Remaining decisions

1. Bind exact `d=1024` target topologies and unique-parameter denominators for both registered 4- and 6-core rungs.
2. Decide whether tokenizer candidate comparisons hold total capacity fixed or hold the non-vocabulary body fixed.
3. Ratify the two-rung tokenizer decision rule; a valid per-rung accounting column is not a freeze authorization.
4. Resolve or formally supersede each of the three legacy failures; any change makes the strict engineering receipt stale.
5. Write a separate optimizer/training preregistration before constructing either AdamW or Muon. The present work remains a build substrate only.
