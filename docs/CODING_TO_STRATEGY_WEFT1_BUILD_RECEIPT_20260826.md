# WEFT-1 build receipt — Jacobian, static K/V, bicameral core, callosum and observatory

**Date:** 2026-08-26 · **Branch:** `codex/bicameral-stage0` · **Build base:** `a032809cce3c9c2a01c96ea9f48782918b5f97b8` · **Scope:** build and tests only · **Run-axis consumption:** none

## 0. Outcome

The vocabulary-independent build queue is complete through the unambiguous primitives. S4’s reference static-K/V path remains green; the full-width bicameral core, exact Birkhoff callosum prediction receipt, P-5 Jacobian machinery, and causality-first observatory event schema now exist. Four implementation defects found during independent review were corrected before handoff.

The integrated S5/S6 production graph is deliberately not claimed. The governing handoff does not bind the final hemisphere combiner, paired-versus-consensus K/V, the sidecar’s state width, or the hard invocation gate’s gradient estimator. The rank/norm Jacobian receipt topology and the variance-component equation also need strategy rulings. Those six questions are isolated in `docs/CODING_TO_STRATEGY_WEFT1_S5_S6_INTEGRATION_RULINGS_20260826.md`; coding did not choose them silently.

The raw repository run completed with exactly the three quarantined legacy failures and no new failure: **3 failed, 3300 passed, 19 warnings in 106.38 seconds**. The machine-readable quarantine now expires fail-closed on its review date instead of treating the date as prose.

## 1. Authority used

| Artifact | Binding used here |
|---|---|
| Master build handoff | 61,329 B, SHA `498f34b5…eb02`, Drive `1XaE81mfqTOYEYGFMa-ZJwpLW-KQtMMC_` |
| Ratification record | 13,908 B, SHA `c5df7429…6d3a`, Drive `1Wb_FfEb-Sl-TgL23hcFOy58QcSurfaF3` |
| Observatory audit | 55,778 B, SHA `16297ade…d48b`, Drive `1M1R4iYdDwA2MrVUzbqJp_VXxzrjSvEC6` |
| Jacobian-panel handoff | 20,935 B, SHA `da7b…9738`, Drive `1ZGtLqLk83PCUCXAMSR4Hrv4xpnl5o_mD` |
| Latest adjudication / W-1 and P-5 | 13,629 B, SHA `f38d77a9…e1e7`, Drive `1d9PdVAS-kiln-QaYMBVCWitqrUro6bGA` |

W-1 governed execution: build is ungated; a run is gated only when it consumes a frozen vocabulary, training compute, sealed data, or a checkpoint. No action in this receipt consumed any of the four.

## 2. Commit ledger

| Commit | Result |
|---|---|
| `5ecf58fb` | Isolated per-module stochastic streams and paired recurrent-visit alignment. |
| `dc63ee4e` | Made T14b receipts derive from exact autograd evidence and fail closed. |
| `0c95a7cc` | Added the standalone full-width bicameral Transformer block and paired static K/V primitive. |
| `9f647404` | Removed distributed-replica identity from parameter initialization; all ranks now start from the same weights. |
| `a1e5acda` | Added the paired-depth Jacobian panel primitives and PT1–PT6. |
| `f589e787` | Hardened observatory event governance, callosum receipts, RNG caveat, and quarantine expiry. |
| `e78692dd` | Mirrored finite-positive RoPE and norm contracts in the standalone bicameral block. |
| `a032809c` | Corrected Jacobian gain normalization and made all unresolved production receipts fail closed. |

## 3. What exists now

### 3.1 S4 static K/V

The existing `AblationLM` reference path computes K/V from fixed `h0` once per block and reuses it across recurrent visits. Fork B-prime midpoint refresh is explicit, reports requested versus executed refresh, and has packed/padded gradient regressions. This is the full-sequence/reference implementation.

Production autoregressive `use_cache=True` serving remains a separately named increment. It is not implemented or claimed by this receipt.

### 3.2 Full-width bicameral core primitive

`BicameralTransformerBlock` keeps `hA` and `hB` at full `d_model` width. Its Q/K/V/O/gate/up/down maps are seven `SwapLinear` projections with common `mu` and low-rank disagreement factors. It binds 2:1 GQA at both rungs, live-state queries, fixed-`h0` K/V, RoPE, RMSNorm, SwiGLU, causal packed-document masks, padding masks, and structurally zero attention dropout.

At `d=512`, rank 32, the disagreement factors total exactly **299,008 parameters**, matching the handoff arithmetic. Every `mu`, `dU`, and `dV` receives finite nonzero gradient in the focused test.

The primitive stores paired A/B K/V payloads. Whether production should retain that representation is ruling C-S5-2, not an implementation claim.

### 3.3 S5 callosum

`PerBandBirkhoffCallosum` retains the exact two-lane mixer

```text
A(rho) = (1-rho) I + rho P,    rho in [0, 1/2].
```

The new `DeltaModePredictionReceipt` reports both quantities that earlier prose conflated:

```text
amplitude retention = (1-2rho)^K
squared-energy retention = (1-2rho)^(2K)
```

Observed values are detached FP32 diagnostics with finiteness checks. The receipt’s scope is explicitly callosum-only: intervening core dynamics can create new disagreement and are not covered by the closed form.

### 3.4 Jacobian panel

The panel implements forward-mode JVP gains, VJP power iteration, Hutchinson participation ratio, per-example Theil–Sen slopes, example-cluster bootstrap, FP32/autocast guards, RNG snapshot/reset, routing-branch equality, and P-5’s same example-owned directions at every `T in {1,2,4,8}`.

Independent review caught a high-value numerical error before any run: the first implementation measured `log ||Jv||`, relying on an approximately unit FP32 vector. An exact identity map then produced tiny nonzero gains which became a spurious `p=1` law after division by `T`. The corrected estimator is exactly

```text
log ||Jv|| - log ||v||.
```

The identity golden now returns bit-exact zero gain and is rejected as zero-lambda / unconditioned rather than reported as a scaling law.

Production report minting is intentionally fail-closed:

- the pilot is explicitly diagnostic-only at `n=32`;
- the main shape is exactly `n=512`, four probes and 10,000 receipted bootstrap replicates;
- a main result remains non-admissible until branch evidence is linked to issued per-example measurements rather than supplied as a boolean;
- norm/rank report assembly and P-4 confirmation remain blocked until strategy resolves the shared-64-example schema.

No pilot, panel, model-data measurement, or GPU run occurred.

### 3.5 Observatory event schema

The schema binds exactly eight Tier-1 instruments, with RESP-LEAK in Tier-1 and `A_state` in Tier-2. The registry is immutable. Metric events can only be minted through a factory that first validates the exact graph/config-bound T14b receipt; direct construction is rejected. Nested measurement payloads are type-preserving and immutable, and every event carries a canonical SHA.

Tier-2/3 events require effective `n`, a named null and an interval. Tier-1 events require a pre-registration SHA and branch outcome. T14 itself remains owned by `T14bReceipt` and cannot be duplicated as an ordinary metric event.

## 4. T14b coverage — explicit, not implied

There is **no production integrated T14b receipt yet**. Current coverage consists of direct causal-gradient regressions plus tests of the receipt machinery. The distinction matters.

| Module/path | Direct gradient coverage now | Production T14b receipt |
|---|---|---|
| Dense prelude/coda and recurrent static-K/V core | Exact future-zero regression at `K={1,2,4,8}`; packed and padded paths; midpoint refresh OFF/ON | Not minted |
| Front Hadamard experts | Included in the active packed/padded K=8 visit-by-visit regression | Not minted |
| Re-entry bridge | Included in the active packed/padded K=8 visit-by-visit regression | Not minted |
| Position-aligned scratch and legacy narrow lane carrier | Included in the active packed/padded K=8 visit-by-visit regression | Not minted |
| Engram | Included in the active packed/padded K=8 visit-by-visit regression | Not minted |
| Frozen long-term memory read | Included in the active packed/padded K=8 visit-by-visit regression | Not minted |
| Full-width bicameral core primitive | One-block paired static-K/V direct gradient test across a packed boundary and padding | Deferred: not integrated across recurrent K |
| Per-band callosum | Standalone pointwise lane mixer; T16 and delta-law tests | Deferred: not in `AblationLM` graph |
| Loop sidecar | Module does not yet exist | Deferred pending C-S6 rulings |
| Final bicameral combiner | Module does not yet exist | Deferred pending C-S5-1 |

The T14b machinery itself proves that a receipt must contain the complete stage by K by packed/padded cross-product, exact zero forbidden gradients, non-vacuous allowed-gradient liveness, and graph/config fingerprints. Its unit tests use synthetic causal stages; they are not presented as production-graph evidence.

## 5. Dropout RNG statement

Every current attention path has dropout structurally fixed at zero. Per-module generators are isolated, and the stream-level regression proves that advancing one attention dropout generator cannot advance any other module’s stream. This is stronger than shared-global-RNG behavior but it is **not** a nonzero-dropout forward integration test: no generator-aware `p>0` fused attention kernel is enabled. That arm remains explicitly deferred, and its eventual first enablement requires its own O-9 and T14b regressions.

## 6. Repository gate and quarantine

Raw full-suite result on `a032809c`:

```text
3 failed, 3300 passed, 19 warnings in 106.38s
```

After refreshing the locked pass count, the strict quarantine runner repeated the entire suite and passed:

```text
ablation engineering gate PASS: all tests ran and exactly 3 quarantined legacy nodes failed
full repository suite remains RED: 3 failed, 3300 passed, 19 warnings in 72.03s
```

The failures are the same three exact legacy nodes:

1. `tests/test_paper2_claim_evidence_ledger.py::test_paper2_claim_evidence_paths_exist`
2. `tests/test_stage5_notebooks.py::test_current_bootstrap_target_markers_exist_in_launcher_files`
3. `tests/test_stage5_notebooks.py::test_current_a100_bootstrap_plain_cell_matches_markdown_code`

No failure was added, removed or renamed. The quarantine review is due **2026-09-02** and the runner now rejects the receipt on that date or later. The Markdown A100 bootstrap remains non-current and must not be published or pasted.

## 7. Open rulings and next executable step

The strategy response should answer the six items in `docs/CODING_TO_STRATEGY_WEFT1_S5_S6_INTEGRATION_RULINGS_20260826.md`:

1. final per-band bicameral combiner;
2. paired versus consensus static K/V;
3. sidecar expert width and corrected parameter accounting;
4. exact hard-forward invocation gradient estimator and initialization;
5. norm/rank Jacobian receipt topology and joint-state metric;
6. variance-component amendment after P-5.

After those rulings, coding can integrate S5/S6 behind structural-OFF flags and mint the first honest full T14b receipt. Until then, the narrower statement is the correct one: the primitives and safeguards exist; the integrated experimental graph does not.

## 8. Do-not-claim boundary

- No tokenizer was fit, screened, selected or frozen.
- No optimizer was constructed.
- No training compute was consumed.
- No checkpoint was loaded or used.
- No evaluation panel or sealed partition was accessed.
- No production T14b receipt was minted.
- No S5/S6 architecture result exists.
- No production serving cache is claimed.
