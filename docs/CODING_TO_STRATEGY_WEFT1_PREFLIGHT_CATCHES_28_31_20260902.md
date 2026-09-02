# CODING → STRATEGY — WEFT-1 PRE-FLIGHT catches #28–#31

**Date:** 2026-09-02  
**Status:** FAIL-CLOSED WEEK-1 CONTINUATION RECEIPT · CPU evidence only  
**Implementation base:** `f68be351af6e0a04ed4528158b8f838e5688cc81`  
**Program authority:** `STRATEGY_PREFLIGHT_PROGRAM_20260902.md`, 15,575 bytes, SHA-256 `ceaa5338830307d3783296b8a4aef7bb87962eb35535d392f4c6d217dff88a5b`  
**Ratification authority:** `STRATEGY_PREFLIGHT_RATIFICATION_20260902.md`, 2,233 bytes, SHA-256 `4a13054d38c68e5e9476330528649d445ff845e639e0a36bb01641b54ef66965`  
**PF-1 authority:** `STRATEGY_PREFLIGHT_AMENDMENT_PF1_20260902.md`, 12,285 bytes, SHA-256 `4e3186c432b57f71b9f32a444a269eec08557ca5181a6896b477078dbbb40861`
**Attention-scale authority:** `STRATEGY_TO_CODING_AGENT_LOOM1_HANDOFF_20260826.md` §8.1, 61,329 bytes, SHA-256 `498f34b5966f0879c7f0a15ca8be02a603558781c35f59f03fb29cc9edd3eb02`, Drive `1XaE81mfqTOYEYGFMa-ZJwpLW-KQtMMC_` (raw Drive bytes reverified for this receipt)
**G-TOK semantics authority chain:** parent 13,975 bytes, SHA-256 `2e42664d0062a119c9fadcb76bf227a91134914920116627f9244f650defe72d`; S1 12,411 bytes, SHA-256 `c37c4be064fe447e01182acc11b1713239c761ddd50583a8299972b4b340bd2a`; S2 6,638 bytes, SHA-256 `5420a4e57c080d09f5f924acc859a5579edd1ca1939c8bbdaf727e5afd55ac5e` (all reverified locally by C7)

## 0. Outcome

The independent Week-1 checks resumed after PF-1 and found four additional fail-closed boundaries. One is a definite implemented-versus-authoritative-spec mismatch: the integrated GQA and standalone bicameral primitive use ordinary `1/sqrt(d_head)` scaling instead of the full build handoff §8.1 μP `1/d_head` coordinate. The other three are specification/integration boundaries that make the requested complete receipts unmintable without inventing a metric, module, emitter, or schema line.

The bounded CPU evidence is useful and retained:

- C1 measured three widths through ten synthetic AdamW steps and directly identified the executed attention scale;
- C2 measured every visit prefix of one fixed K=8 trajectory under CPU bf16 autocast against FP32 masters;
- C3 replayed one bounded current-graph CPU training step bit-identically and proved direct registry-level draw isolation across all ten materialized O-9 attention streams;
- C6 executed all 168 dependency-valid structural-flag assignments on the current 4/2/4 graph and counted ten block calls each; and
- C7 mechanically verified field presence for four already-bound G-TOK receipt families, but found that no toy emitter yet proves populated semantic values.

None of those partial results is promoted to a full-production or GPU result. No model/config behavior changed, no A100 meter started, no checkpoint or sealed data was consumed, and P-A was untouched.

The numbering follows PF-1's accepted #22–#25 and coding's returned #26–#27. C1 is #28 as first discovered; C2, C3/C6 and C7 are normalized to #29–#31 irrespective of provisional parallel-agent numbering.

## 1. Catch #28 — the ratified μP attention coordinate is absent

### Exact mismatch

The ratified attention-logit coordinate is `1/d_head`. The integrated `GroupedQueryAttention` and standalone `BicameralTransformerBlock` instead use explicit `1/sqrt(d_head)` in their math paths and the equivalent default scale in their fused SDPA paths.

| `d_head` | implemented / ratified scale ratio | integrated GQA max error to ordinary | integrated GQA max error to ratified | bicameral max error to ordinary | bicameral max error to ratified |
|---:|---:|---:|---:|---:|---:|
| 8 | `2.828427` | `4.37e-11` | `5.56e-05` | `3.58e-07` | `0.810815` |
| 16 | `4.000000` | `8.73e-11` | `1.93e-04` | `4.77e-07` | `0.989166` |
| 32 | `5.656854` | `1.75e-10` | `4.63e-04` | `4.77e-07` | `1.365855` |

The evidence is independent of a task loss: both execution paths numerically match the ordinary reference and reject the ratified reference. Existing source and test behavior also pin the ordinary scale. Coding made no silent patch because the PRE-FLIGHT rule says a failure on a ratified operator returns through the amendment path.

### Width-coordinate diagnostic

The CPU diagnostic ran `d=(64,128,256)` with fixed 8Q/4KV heads, `d_ff=11d/4`, the 4/2/4 graph, K=4, lanes `2×d/8`, a fixed synthetic `B=2,S=8,V=128` batch, and ten FP32 AdamW steps at `3e-4`. These are disclosed diagnostic choices, not S2 calibration. Width 512 remains the registered GPU-deferred cell.

| width | unique parameters |
|---:|---:|
| 64 | 492,427 |
| 128 | 1,926,707 |
| 256 | 7,633,027 |

Under this explicit diagnostic, 25 initialization coordinates and 34 post-step coordinates exceed the registered 2× drift literal. The largest is `core.0.feed_forward`: `8.998×` at initialization and `9.735×` after ten steps. Because head topology, toy optimizer/data details, and the exact μP initialization/update parameterization were not otherwise bound for C1, these activation ratios are descriptive catch evidence; the direct attention-scale mismatch is the definitive failure.

**Requested ruling C28-R1:** authorize the ratified `1/d_head` implementation across fused and math GQA and bicameral paths, including the existing square-root reference test's disposition; bind the width-coordinate toy protocol needed to interpret the remaining activation drift rather than inheriting this diagnostic's discretionary choices.

## 2. Catch #29 — C2's `1e-2 relative` threshold lacks an estimand

C2 binds a `1e-2 relative` threshold but does not specify the metric, denominator, tensor population, gradient target, or per-visit aggregation. Reasonable interpretations disagree on the same run.

The CPU-valid audit uses the current d=64, 4/2/4, K=8 materialized graph with FP32 master weights and CPU bf16 autocast. It traces every recurrent prefix and verifies that the FP32 trace's final output is exactly the model's ordinary FP32 forward. It is not GPU evaluator identity, not a learned checkpoint, and not the full toy WEFT graph.

The replay identity is bound in the machine receipt: root seed `20260902`, config SHA-256 `57ef01bbaa4aa89ea18b098cbb51ece021fc0849f9b958435f7f534f2f6d8e28`, input-panel SHA-256 `1bce335742f00f14c7e9abd88396776dc4d473fb955f60a923868618bf1d15f6`, and initial-model-state SHA-256 `52b993089afe982dd4027eb16500f761b0ed8cfc49a938de7f00e2e102831aac`. The published precision values below are regression-pinned.

| measurement | maximum vector-relative L2 | visit | K=8 vector-relative L2 | K=8 scalar-norm drift |
|---|---:|---:|---:|---:|
| hidden state | `0.002379` | 8 | `0.002379` | `0.0109%` |
| scratch lanes | `0.016478` | 8 | `0.016478` | `0.7338%` |
| logits | `0.003736` | 6 | `0.003659` | `0.0354%` |
| core Q-projection gradient | `0.011458` | 7 | `0.010894` | `0.2617%` |

K=8 relative loss drift is `1.41899e-05`. Under vector-relative L2, scratch lanes cross 1% at visits 5–8 and the selected gradient crosses at every visit; under scalar norm drift, every cell remains below 1%. The receipt therefore preserves `threshold=1e-2` but records `threshold_applied=false` and `threshold_passed=null`.

Representative absent integrations include the learned rotor carrier, full-width bicameral recurrent block, per-band callosum, loop sidecar and final `bridge_out`; the accepted build-status matrix is the exhaustive inventory. The measured scratch carrier is Birkhoff, so this run cannot decide FP32 rotor-carrier accumulation.

**Requested ruling C29-R1:** bind the relative-error formula and denominator, the state/logit/gradient population, whether gradients are in scope, the visit aggregation, and the exact fail rule. The GPU cell and carrier-accumulation decision remain deferred until the full named step exists.

## 3. Catch #30 — complete C3/C6 receipts assume unmaterialized cells

### Green bounded CPU evidence

- Two independently constructed same-seed models produce bit-identical loss, logits and post-AdamW state after one CPU training step; ambient RNG state is unchanged.
- The current 4/2/4 graph owns exactly ten O-9 attention streams. Consuming any one stream leaves every other stream's coordinate-zero draw bit-identical to its untouched control.
- All 168 dependency-valid assignments of the nine current structural switches execute exactly four prelude, two core and four coda block calls at K=1. This tests executed calls, not `ModuleList` lengths.

### Unavailable cells

1. deterministic CUDA replay has not run under the PRE-FLIGHT meter;
2. nonzero attention dropout is rejected by configuration pending a generator-aware fused kernel, so the isolation proof is not a `p>0` forward receipt;
3. no WEFT-1 STOCH-K sampler or registered sampling stream exists; and
4. the learned rotor carrier, per-band callosum and sidecar are absent, so C6 cannot enumerate their OFF combinations.

The unavailable cells are typed as explicit non-passes in the regression surface. They are not `pytest.skip` nodes, because the repository's exact engineering gate forbids skipped outcomes.

**Requested ruling C30-R1:** confirm that C3/C6 remain partial until STOCH-K, the generator-aware dropout path, deterministic CUDA execution and the missing module integrations exist; or separately bind a reduced bring-up C3/C6 gate. Coding recommends preserving the complete gate and reporting this bounded CPU subset.

## 4. Catch #31 — schema presence is not toy-run emission

C7 asks one toy run to emit seven families. The machine inventory proves schema-field presence only; it does not construct the toy emitter or prove populated semantic values:

| C7 line | status | concrete source or blocker |
|---|---|---|
| four rounded `rho` values | schema present; emitter unverified | `ArmTerminalStatisticsV2.rho_bpb_micros` |
| nine consumption fields plus optional boundary document | schema present; emitter unverified | `GTokRunReceiptV2` |
| exact integer `F*` | schema present; emitter unverified | `ConfirmationBudgetReceiptV2.target_flops` |
| precomputed checkpoint indices | schema present; emitter unverified | base and confirmation `bpb_checkpoint_steps` |
| gate rate versus K | schema only | `CompositionReceipt.sidecar_firing_fraction_by_step`; sidecar absent and C-S6-1/2 open |
| realized `eta*lambda` | authority conflict pending ruling | PF-1.6 struck A8 and moved its certificate tests to MEM-SYN-FW, but did not literally dispose C7's independently listed receipt line |
| `Lambda_k` | standalone/blocking | catch #26 and C-JAC-1 prohibit a production nonlinear-visit certificate/alarm |

Thus C7 cannot truthfully report `schema complete`. Field membership is not emitted receipt evidence; the four G-TOK families still need a toy emitter with populated semantic validation, while gate-rate and `Lambda_k` wait on their already-recorded module and certificate blockers. Coding also declines to infer whether PF-1.6 automatically removes C7's independently listed `eta*lambda` receipt line.

**Requested ruling C31-R1:** resolve whether PF-1.6 removes C7's `eta*lambda` line; require the toy emitter to populate and semantically validate the four present G-TOK families; and mark gate-rate/`Lambda_k` as pending their already-named integration and catch resolutions, or bind a staged dry-run receipt. Coding recommends preserving the complete C7 gate while retaining this field-presence inventory as bounded evidence.

## 5. Implementation and verification

| surface | file |
|---|---|
| C1 width/attention diagnostic | `analysis/weft1_preflight_c1.py` |
| C2 precision trace | `analysis/weft1_preflight_c2.py` |
| C3/C6 executable regressions and typed deferrals | `tests/test_ablation_lm_preflight_c3_c6.py` |
| C7 schema inventory | `analysis/weft1_preflight_c7.py` |

```text
New C1/C2/C3/C6/C7 tests             11 passed
Broader focused regression            209 passed, 18 warnings
Full raw repository suite             3,997 passed, 1 failed, 19 warnings
Strict exact-node quarantine gate     PASS
```

The raw-suite failure is the unchanged governed
`tests/test_paper2_claim_evidence_ledger.py::test_paper2_claim_evidence_paths_exist`.
No failure was added, removed or renamed, and no skipped outcome was introduced.
The immutable successor is
`training/ablation_lm_engineering_quarantine_20260901_preflight_c1_c7.json`,
SHA-256 `804f00215ee2b338b1dd98c0dfc8442da7a41d55c03fb716fa809170f38d7da5`.
It retains the 2026-09-04 review date and forbids a repository-wide green claim.

No lint result is claimed because `ruff` is unavailable in the verified Python 3.11 environment.

## 6. Preserved posture

```text
CATCH_28_ATTENTION_SCALE_RETURNED_NO_PATCH
CATCH_29_PRECISION_ESTIMAND_RETURNED
CATCH_30_C3_C6_PARTIAL_ONLY
CATCH_31_C7_CONFLICT_RETURNED
NO_FULL_PREFLIGHT_PASS
ZERO_A100_SPEND
P_A_UNTOUCHED
NO_CHECKPOINT_OR_SEALED_DATA_CONTACT
```
