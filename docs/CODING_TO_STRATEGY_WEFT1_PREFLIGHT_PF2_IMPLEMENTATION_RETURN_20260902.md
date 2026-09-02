# CODING → STRATEGY — WEFT-1 PRE-FLIGHT PF-2 implementation return

**Date:** 2026-09-02

**Status:** BUILD-AXIS IMPLEMENTATION RECEIPT · C1 and C2 remain fail-closed on Catches #33 and #34 · C7 stage 1 emitted and verified · no training, optimizer construction, GPU use, checkpoint, evaluation-panel contact, or sealed-data contact

**Implementation commit:** `5ebbefea36c13a8f86c5e78ed074efc6c91db12f`

## 0. Outcome

PF-2 and the ratified §8.1 amendment are implemented without changing attention numerics. The existing inverse-square-root attention is now explicitly recognized as `sqrt(d_head,base)/d_head` at the ratified fixed `d_head,base = d_head = 64`; the fused, math and explicit-reference paths are asserted at that shape. A future WEFT shape with another head dimension must implement the base-shape formula explicitly. Ordinary non-WEFT toy shapes keep their existing behavior.

The Jacobian panel is registered at `n=520` and meets both literal power frontiers. C1 stops before model or optimizer construction because eight load-bearing μP literals are absent from §8 (Catch #33). C2's terminal K=8 values pass the PF-2 thresholds, but the complete receipt stops on three visit-1 zero-reference gradients for which PF-2 supplies no eligibility rule (Catch #34). C7 stage 1 now emits all four present G-TOK families through the production receipt builders, while the complete C7 gate remains open on the sidecar and Catch #26/C-JAC-1.

## 1. Authority identity

All three governing documents are stored locally. C1 byte-verifies all three before construction; C2 byte-verifies PF-2 alongside its program and ratification chain; C7 byte-verifies PF-2, PF-1, the program/ratification pair, and the complete G-TOK semantics chain before construction:

| authority | bytes | SHA-256 |
|---|---:|---|
| `STRATEGY_TO_CODING_AGENT_LOOM1_HANDOFF_20260826.md` | 61,329 | `498f34b5966f0879c7f0a15ca8be02a603558781c35f59f03fb29cc9edd3eb02` |
| `STRATEGY_PREFLIGHT_AMENDMENT_PF2_20260902.md` | 13,097 | `be11390c28ae36210a1571f7c6d358ee54e977d239f5344f7e6402212448eb05` |
| `STRATEGY_HANDOFF_S81_AMENDMENT_20260902.md` | 3,403 | `dd79aaa6fd9bab15bf02aaef28f99f47c745d63eed6db2c9b929b4bb1cfbb418` |

Receipt promotion for C1, C2 and C7 recomputes its governing invariants. Mutation tests prove that replacing a frozen dataclass's public pass/complete bit, catch fields, authority identity, summary, terminal decision, topology, or staged statuses cannot manufacture a green receipt.

## 2. §8.1, Catch #32, and C1

Catch #32 is closed and Catch #28 is disposed exactly as ratified. `MUP_D_HEAD_BASE = 64` is a named production constant. At the ratified shapes:

```text
sqrt(d_head,base) / d_head = sqrt(64) / 64 = 0.125 = 1 / sqrt(64)
```

No attention code path or numerical behavior changed.

PF-2.1's topology is bound at `d ∈ {128,256,512}`, fixed `d_head=64`, Q heads `d/64`, KV heads `d/128`, `d_ff=11d/4`, lanes `2 × d/4`, split `4/2/4`, `K=4`, and intended synthetic shape `B=2,S=64`. C1 returns **Catch #33** before model initialization and before optimizer construction because §8 does not supply:

1. numeric model base width `d_base`;
2. numeric internal base initialization `sigma_base`;
3. a complete per-tensor initialization map;
4. numeric internal base learning rate `eta_base`;
5. a complete per-tensor learning-rate map;
6. residual-branch `alpha`;
7. embedding multiplier;
8. residual multiplier.

No width-transfer result is claimed from an unbound optimizer or initialization.

## 3. C2 precision result and Catch #34

PF-2.2's vector-relative-L2 estimand is applied per tensor and per visit. Thresholds remain explicitly disclosed as bound after the original diagnostic: `1%` for state/logits and `5%` for lanes/gradients. The terminal K=8 decision values are:

| terminal quantity | relative L2 | threshold | numerical decision |
|---|---:|---:|---|
| hidden state | `0.002378677322798898` | `0.01` | pass |
| scratch lanes | `0.016478415427698328` | `0.05` | pass |
| logits | `0.003659068615870767` | `0.01` | pass |
| concatenated 488,859-element parameter-gradient vector | `0.0071749515521762446` | `0.05` | pass |
| worst terminal module tensor, `engram.raw_residual_scale` | `0.02583730846608379` | `0.05` | pass |

The required maxima are also reported. Hidden and lane maxima occur at visit 8 and equal the terminal values. Logits peak at `0.0037359662511258457` on visit 6. The full gradient peaks at `0.0071749515521762446` on visit 8. The per-module worst tensor peaks at `0.06243657691629686` on visit 4 (`engram.gate_bias`), above the terminal-decision threshold; PF-2 explicitly makes the terminal visit the decision value, so this is retained as a diagnostic rather than silently substituted into the decision.

The complete receipt remains fail-closed as **Catch #34**. At visit 1, `reentry_bridge.layer_scale`, `reentry_bridge.prelude_norm.weight`, and `reentry_bridge.projection.weight` are autograd-disconnected with exact zero FP32 and BF16 norms. Their relative errors are undefined, and PF-2 binds neither a zero-denominator rule nor an eligibility rule. The learned rotor-carrier accumulation decision and CUDA evaluator identity remain deferred.

## 4. C3/C6 and C7

PF-2.3's complete C3/C6 gate is preserved. The bounded CPU evidence remains bit-identical same-seed replay, ten-stream O-9 isolation, and all 168 dependency-valid structural assignments executing the `4/2/4` block schedule. CUDA replay, generator-aware dropout, STOCH-K, and absent module integrations remain typed non-passes; none is promoted.

C7 stage 1 is now emitted from deterministic synthetic source receipts through the actual production matrix validator, vocabulary-selection join, confirmation-budget join, base and confirmation training-plan constructors, run receipts, and byte-crossing calculator. Independent audit rejected three earlier drafts before accepting the final path.

| path | `n` | trained bytes (`B_total`) | first-crossing indices |
|---|---:|---:|---|
| byte-matched base | 400 | `3,999,998,659` | `(100, 200, 400)` |
| fresh confirmation, selected V=32,768 budget row | 399 | `3,989,999,589` | `(100, 200, 399)` |

The fresh plan is cross-joined to its exact `planned_optimizer_steps=399` budget row and cannot borrow the base history. Boundary token length is derived from a concrete `TrainingDocumentV2` token record bound to the run's document ID and consumed-token count. `rho` is computed from ordered, distinct per-seed binary64 BPBs and rounded half-even to six decimals. `F*` is the exact integer minimum of the two floor-mean per-seed FLOP totals. Malformed, swapped, cross-joined, shifted and forged-completion receipts fail.

Stage-1 receipt SHA-256:

```text
04b9c1515a3902c2963eb1e13e5bfa42ede144549f88a44366953b76a422abd6
```

C7 remains staged: gate rate by K is pending the sidecar; `Lambda_adapters` and `Lambda_hat_core` remain pending Catch #26/C-JAC-1. `realized eta-lambda` is struck. The complete C7 gate is therefore incomplete. No Catch #35 was required.

## 5. Requested one-line dispositions

- **Catch #26 — OPEN:** exact standalone adapter factors and an empirical core estimate exist, but there is no ratified joint-state metric or production nonlinear-visit certificate/alarm; C-JAC-1 still governs.
- **Catch #27 — OPEN:** exact matched-background comparison plumbing exists, but A7's comparison graph and K/module eligibility remain unbound; no registered OBS-INV matrix is minted.

## 6. Jacobian panel

`MAIN_PANEL_EXAMPLES` and the corrected power receipt are set to `520`. The registered values are:

| frontier | realized SE at n=520 | literal threshold | minimum integer n |
|---|---:|---:|---:|
| primary, `sigma_slope=1.15` | `0.05092447485214237` | `0.051` | 519 |
| secondary, `sigma_slope=0.80` | `0.03578829611711741` | `0.036` | 514 |

Both literal frontiers pass. The four planted B1 phases remain green at `n=520`; no panel run or GPU work occurred here.

## 7. Verification and quarantine

```text
PF-2/S81/G-TOK focused regression     183 passed, 18 warnings
Full raw repository suite             4012 passed, 1 failed, 19 warnings
Strict exact-node quarantine gate     PASS
git diff --check                      clean
```

The sole failure is unchanged and exact:

```text
tests/test_paper2_claim_evidence_ledger.py::test_paper2_claim_evidence_paths_exist
```

The repository is therefore not described as globally green. The successor quarantine is `training/ablation_lm_engineering_quarantine_20260902_pf2.json`, 3,039 bytes, SHA-256 `c2190084bbdfd4f8f5ba70f839a30a5c8dc98b904c86ed59930f0fe75f97fa3a`, with review due 2026-09-04. It authorizes no training or sealed-data contact.

The updated build-status matrix is `docs/CODING_TO_STRATEGY_WEFT1_BUILD_STATUS_MATRIX_20260902.md`, 16,774 bytes, SHA-256 `79189268ee689121129f880cfb6c56d6d0f5fd70f04cc0ee42e99528dd443c06`.

## 8. Preserved boundary

- P-A and its durable corpus artifacts were not read, changed, or advanced by this receipt.
- No tokenizer was fit or frozen.
- No optimizer was constructed.
- No training or checkpoint was created.
- No GPU or PRE-FLIGHT A100 budget was consumed.
- No evaluation panel or sealed partition was contacted.
- C1, C2, C3/C6 and the complete C7 gate remain fail-closed exactly where stated above.
