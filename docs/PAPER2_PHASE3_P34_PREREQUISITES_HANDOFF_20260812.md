# P3.4 Prerequisites Handoff: Task Graph and A_r Pricing

Date: 2026-08-12. Status: two prerequisites complete; calibration surfaces built; training remains unauthorized. Governing charter: `docs/STRATEGY_P34_CHARTER_20260812.md`, Drive `1lh2Vf3VG8yPXUjLPfr7IJkPCjnf2mKE6`, SHA-256 `80cb1b13eb48ffff064ff7cc6c0d02de773dfec80924c1c50736115821c97ce4`.

## 1. Executive reading

The P3.4 build now has a real task-inference graph and a measured answer to the A_r pricing question. The task graph passed a score-blind two-seed GPU preflight. It recomputes the current prefix, initializes a fresh scratch state for each emitted token, runs exactly four flow loops, writes only at the current nonzero position, disables the draft head for task scoring, and carries no sidecar state across emitted tokens. Repeated calls produced bit-exact logits on all 16 seed-prompt cases.

The A_r audit found that the learned rank-128 readout span contains 25.55% and 26.31% of cached oracle-direction energy across the two seeds, versus 14.16% for a matched leading state-covariance subspace. The bridge has learned reproducible aim geometry. However, realized pi_dir remains 14.901%, most oracle energy lies outside the readout span, and the existing state-to-direction forecast is weak. The evidence provisionally favors slot supervision, but the charter did not bind a numerical A_r fork rule. Strategy must make the registered one-line selection.

No task accuracy, gap_closed, sealed-partition result, optimizer, or P3.4 training step was produced.

## 2. Task inference graph design

The implemented v1 graph follows the charter literally:

- Qwen2.5-0.5B-Instruct at the pinned revision and BF16 serving precision.
- Greedy decoding through the frozen tied output head.
- Fresh sidecar scratch state for every emitted token.
- Four recurrent flow loops per token.
- Current-position writeback only; position zero is closed.
- Draft head inactive for battery scoring.
- No cross-token scratch or control-state persistence.
- Exact full-prefix recomputation with `use_cache=False` in v1, avoiding an unregistered cache semantic.

The model API gained optional `write_position_mask` and `draft_active` arguments. Their defaults preserve all previous behavior. Unit tests cover the default compatibility path, padded-prefix current-position selection, position-zero closure, draft-head inactivity, repeatability, and file-style launcher imports.

## 3. Score-blind preflight

The L4 preflight reconstructed each endpoint from three pinned layers: the Phase-3 migration, the P3.3 bridge/control state, and the P3.3-i1 output-projection update. All six hashes matched before inference.

It then evaluated eight unsealed DEV prompts for each seed. This was a contract test, not a performance evaluation.

| Check | Result |
|---|---:|
| Seeds | 2 |
| Prompts per seed | 8 |
| Exact repeated-logit cases | 16/16 |
| One selected current-position write cell | 16/16 |
| Maximum repeated-logit absolute difference | 0.0 |
| Task scores computed | No |
| Correctness or gap_closed computed | No |
| Sealed partitions touched | No |
| Optimizer constructed / steps | No / 0 |

The first run exposed a launcher-only defect: direct file execution omitted the repository root from `sys.path`, causing `ModuleNotFoundError: eval` before `main()` and before any receipt or model work. Commit `824b09ec` fixes the import path and adds a regression test. The staged checkpoints were not changed or recopied for the successful rerun.

Receipt: Drive `1UgqoPaHepWz2WYxFTwvnT9MB0bJ28dof`, SHA-256 `849d4d146e272fb83c283130639db9f2c09f639a60d488a3ede77f8dfcd9f8cc` (8,046 bytes).

## 4. A_r pricing result

The CPU-only audit used all 43,204 strict concurrent oracle rows and compared two rank-128 output-space subspaces.

| Measure | Seed 0 | Seed 1 |
|---|---:|---:|
| Oracle energy in learned readout span | 25.546% | 26.311% |
| Oracle energy in matched state-covariance span | 14.164% | 14.164% |
| Oracle energy outside learned readout span | 74.454% | 73.689% |
| Readout energy / realized pi_dir | 1.714 | 1.766 |
| Top-128 state variance explained | 84.299% | 84.299% |

The readout span is stable across seeds and carries roughly 1.8 times the energy of the dimension-only isotropic reference. This supports a learned aiming subspace. It does not establish that widening the readout will improve answers: the current span already contains more compatible energy than the model converts, while the banked loop-4 linear forecast has low holdout cosine (0.0952 and 0.0874). The current evidence therefore leans to the slot-supervision arm, which directly trains future-token information into the scratch slots.

Receipt: Drive `1YkJYPo-jiVzkLEsWqXkhetEdgj9D3JwU`, SHA-256 `68c697812804b3e113fcb8cde8f1888821ac81199fcdbe0bad6e964ada7a7c8d` (5,139 bytes). Detailed interpretation: `docs/PAPER2_PHASE3_P34_AR_PRICING_HANDOFF_20260812.md`, Drive `1fCaWGKowv9WXSAk0nqvacOmkQfOrE6`.

## 5. Built but not yet executable as a registered calibration

The following mechanisms are implemented and tested:

- empirical task-panel discordance and adjacent-look autocorrelation extraction;
- task guardrail calibration at the registered familywise false-stop ceiling and power floor;
- Tier-W reporting at the negative-three-point class;
- chi_max construction from oracle collateral at all four gate ceilings using a Clopper-Pearson upper bound plus an explicit strategy margin;
- per-loss share enforcement and the rung controller; and
- a pending machine-readable lock that refuses training while any required field is null.

These are intentionally not run with invented constants. Four design bindings are still absent.

## 6. Decisions required from strategy

1. **A_r fork.** Ratify slot supervision or capacity expansion, and state the high/low rule used. The provisional recommendation is slot supervision.
2. **DEV task panel.** Bind the deterministic selection rule and seed for the 1,024-row DEV panel. The eight-row preflight sample is not the guardrail panel.
3. **Empirical task trajectory.** Specify the source of the repeated-look task scores used to estimate adjacent-look autocorrelation. The new estimator needs a fixed 1,024-row panel across the exact campaign look count. No existing receipt unambiguously supplies that trajectory under the v1 task graph.
4. **Campaign schedule.** Bind total steps, learning rate, and the exact look schedule before calibration. The calibration certificate is specific to its look count.
5. **chi margin.** Bind the explicit strategy margin added to the measured upper collateral bound.
6. **Loss-share calibration.** Authorize the score-blind/share-only calibration procedure that converts the registered per-loss floors into scalar weights, or provide those weights.

The central ambiguity is item 3. Possible sources include rescoring the saved i1 trajectory under the new task graph, a distinct unregistered pilot trajectory, or a pre-specified proxy noise model. They are not equivalent, so the coding lane should not select one silently.

## 7. Verification and lineage

- P3.4 build commit: `e4aa557a`.
- A_r bank commit: `3db69cc8`.
- task-runner repair commit: `824b09ec`.
- Focused post-receipt tests: 26 passed, 143 deselected.
- Colab account: `mshapiro@pharmainitiatives.com`.
- Task preflight hardware: NVIDIA L4.
- Colab state after receipt download: no active sessions.
- Pending lock remains `status=prerequisites_pending`, `locked_before_training=false`, and `training_authorized=false`.

## 8. Plain-language summary

The machinery needed to test whether recurrent corrections improve complete answers now exists, and its most important inference rules work exactly as specified. The bridge also demonstrably learned a useful direction space rather than a random projection. What remains is not ordinary coding cleanup: strategy must choose the fork and bind the statistical population, campaign timing, and safety margins that make the next measurements interpretable. Once those values are fixed, the existing calibration code can produce the executed lock. Training should not start before that lock is reviewed and approved.
