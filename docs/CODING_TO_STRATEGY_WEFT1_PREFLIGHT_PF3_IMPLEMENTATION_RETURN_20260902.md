# CODING TO STRATEGY — WEFT-1 PRE-FLIGHT PF-3 Implementation Return

**Date:** 2026-09-02

**Status:** bounded build-axis implementation return; **not** a production WEFT-1 receipt, a complete A7 receipt, a training result, or a run-axis authorization

## 0. Outcome in plain language

PF-3 and D-PF-4 were verified byte-for-byte before implementation. The governed CPU work now has three concrete outcomes:

1. **C2 is complete and passes on its governed CPU evaluator.** Structurally ineligible zero-reference rows are now excluded only when both fp32 and bf16 references are exactly zero and structural disconnection is proved; an eligible zero still fails.
2. **C-JAC-1 is executable on the full joint state that exists today.** It emits the authorized two-number result, `Lambda_adapters` and `Lambda_hat_core`, with a convergence receipt. This is a full-current-visit estimate, not a production certificate: the full bicameral lanes and learned carrier are still absent.
3. **A7 has a non-forgeable CPU partial matrix.** Every eligible, currently integrated CPU cell passes exact structural-OFF identity and active-ON nontriviality, but the registered matrix is deliberately incomplete because CUDA cells and absent production modules are not green.

The dependency-neutral bicameral recurrence seam also exists as a standalone component. It does not modify or claim integration into `AblationLM.forward`; it refuses to choose the still-open production K/V and final-recombination semantics.

C1 did not run past inventory. PF-3.1 exposed one tensor class that the authority does not bind: `front_hadamard.router.weight` with shape `E x d`. This is returned as **Catch #35**, with no local classification invented.

An adversarial receipt review also closed five false-green paths before publication: authority/protocol fields are derived and mutation-checked in C1, C2 and C-JAC; CPU tensors cannot be labeled deterministic-CUDA evidence; active A7 controls must change both logits and loss; Jacobian estimators reject a finite derivative paired with a non-finite primal output; and standalone sidecar/rotor certificates now require derived fields and an explicit operator witness.

No GPU, sealed data, frozen vocabulary, checkpoint, model training, or training-budget compute was used.

## 1. Authority verification and scope

The local project mirrors were verified directly:

| authority | bytes | SHA-256 | verification |
|---|---:|---|---|
| `docs/STRATEGY_PREFLIGHT_PROGRAM_20260902.md` | 15,575 | `ceaa5338830307d3783296b8a4aef7bb87962eb35535d392f4c6d217dff88a5b` | exact |
| `docs/STRATEGY_PREFLIGHT_RATIFICATION_20260902.md` | 2,233 | `4a13054d38c68e5e9476330528649d445ff845e639e0a36bb01641b54ef66965` | exact |
| `docs/STRATEGY_TO_CODING_AGENT_LOOM1_HANDOFF_20260826.md` | 61,329 | `498f34b5966f0879c7f0a15ca8be02a603558781c35f59f03fb29cc9edd3eb02` | exact |
| `docs/STRATEGY_PREFLIGHT_AMENDMENT_PF2_20260902.md` | 13,097 | `be11390c28ae36210a1571f7c6d358ee54e977d239f5344f7e6402212448eb05` | exact |
| `docs/STRATEGY_HANDOFF_S81_AMENDMENT_20260902.md` | 3,403 | `dd79aaa6fd9bab15bf02aaef28f99f47c745d63eed6db2c9b929b4bb1cfbb418` | exact |
| `docs/STRATEGY_PREFLIGHT_AMENDMENT_PF3_20260902.md` | 14,632 | `7f0081e504366ce98f8bf183b7e14c0bed47647aa381196a9d5e9540b5334cef` | exact |
| `docs/STRATEGY_PREFLIGHT_RATIFICATION_DPF4_20260902.md` | 2,336 | `b4e13508c4135837dd197f19f2711026a3df429df7a463386ed5ac1aa94dba31` | exact |

D-PF-4 ratifies the integration pivot on the build axis. It permits queue step 2 only where independent of C-S5-1/2, puts steps 3–4 after that, and holds steps 1 and 5 for strategy's four S5/S6 rulings. Its standing boundary remains in force: no frozen `V`, sealed data, training-budget compute, checkpoint, or P-A interference. The 5 A100-hour PRE-FLIGHT meter remains available only for the already named GPU cells; this return consumed **0 A100-hours**.

No implementation below treats an older standalone primitive, a synthetic receipt, or a bounded CPU cell as proof that the corresponding production graph is integrated.

## 2. Catch #35 — PF-3.1 has no class for the Hadamard router

The C1 inventory constructed the governed toy topology at `d = 128, 256, 512` and classified every unique trainable tensor before applying PF-3 initialization. At each width:

- there are 147 unique trainable tensors;
- 146 have a direct PF-3.1 assignment;
- the sole unclassified tensor is `front_hadamard.router.weight`, shaped `(8, d)`.

The router is `nn.Linear(d_model, experts=8)`. Its fan-in scales with width while its fan-out remains fixed. It is not directly covered by the current rules:

- it is not a hidden matrix, because both axes do not scale;
- it is not an input-class matrix, because its fan-in is not width-independent;
- it is not the tied output/readout;
- it is a rank-2 projection, not a scalar or one-dimensional vector tensor.

Treating its eight rows as eight independent gate vectors would itself be a new tensor-class assignment. PF-3.1's catch clause says coding must return such a tensor rather than assign it locally.

The sibling `front_hadamard.expert_gains` tensor also has stored shape `8 x d`, but it is not a linear map: each fixed expert row multiplies WHT coefficients elementwise and is therefore a design-initialized width-vector gain. Its PF-3 vector-class treatment is explicit in the map and executable tests; the router remains different because it computes an actual `d -> 8` projection.

The C1 receipt is therefore fixed at `blocked_before_pf3_initialization`, `catch_number = 35`, and `disposition = catch_35_hadamard_router_mup_class_unbound`. It records:

- PF-3 initialization not applied;
- optimizer not constructed;
- forward not executed;
- training not performed;
- activation RMS not measured;
- A100-hours `0.0`.

The classified-map SHA-256 values are `2f74f2ba89d2be53af54f77087c728778aefa5e4fe7baff7dc72b401a17e286d` at `d=128`, `f1096e5c6ba46458f0e4924dd11e8f9c272fd118fcd5548bfb0f1700bf053948` at `d=256`, and `91612cea49ea8c72586466c2d90be69cf2947409de1b198bb44db06d98ffe7bd` at `d=512`.

**Requested strategy disposition for Catch #35:** bind the initialization, optimizer learning-rate scaling, and decay class for a fixed-expert-count `E x d` router projection, or amend the router topology. Until that binding lands, C1 cannot proceed to initialization or the width-coordinate experiment.

## 3. PF-3.2 — C2 eligibility and terminal precision result

The C2 evaluator now makes the PF-3.2 distinction structurally, rather than by accepting arbitrary zeros:

- a zero fp32 reference is excludable only when the row is declared structurally ineligible, the graph proves that the parameter did not execute on that visit, and its bf16 reference is also exactly zero;
- an eligible zero reference fails;
- a claimed-ineligible row with nonzero fp32 or bf16 reference fails;
- a structural-connectivity mismatch fails;
- excluded rows remain enumerated in the receipt and are not silently dropped;
- the terminal visit remains the decision value, while earlier-visit diagnostics remain visible.

The three excluded visit-1 rows are exactly:

- `reentry_bridge.layer_scale`;
- `reentry_bridge.prelude_norm.weight`;
- `reentry_bridge.projection.weight`.

They are ineligible because visit zero does not execute recurrent re-entry. Each has exact-zero fp32 and bf16 reference values. The inverse condition is retained elsewhere: a parameter ineligible at `K=1` must become live at `K=4`, or it is a frozen parameter hiding behind eligibility.

The terminal `K=8` CPU decision values are:

| observable | terminal relative-L2 value | PF-3.2 band |
|---|---:|---:|
| hidden state | `0.002378677322798898` | 1% |
| position-aligned scratch state, retained under the legacy C2 label `lanes` | `0.016478415427698328` | 5% |
| logits | `0.003659068615870767` | 1% |
| full gradient vector | `0.0071749515521762446` | 5% |
| worst terminal module tensor | `0.02583730846608379` | 5% |

All terminal values pass. The earlier visit-4 `engram.gate_bias` value `0.06243657691629686` remains in the diagnostic history; it does not replace the registered terminal decision.

This closes Catch #34 for the bounded CPU C2 evaluator. The learned rotor-carrier precision decision and GPU evaluator identity remain deferred until those governed cells exist.

## 4. PF-3.3 — current-graph full-joint-state Jacobian estimate

The governed C-JAC harness measures the terminal `K=8` recurrent visit using JVP/VJP power iteration on the full joint state that exists in the current bring-up graph. It implements PF-3.3's plain Euclidean concatenation with no dimension reweighting.

The vocabulary mapping is explicit. Current `PositionAlignedScratch` state has shape `[B, S, 2, scratch_width]`. It is classified as PF-3's **`scratch`** component: the private model API calls this tensor `lanes` only because it has a two-slot lane-shaped axis. That private name does not turn the narrow scratch state into the absent pair of full-width bicameral **`lanes`**. Counting it as both would double-count one tensor; counting it only as full-width lanes would falsely claim that the ratified bicameral state is integrated.

Thus the current component set is `(h, scratch)`, while `(lanes, carrier)` is explicitly absent. As modules land, the governed definition remains:

```text
z = [h; lanes; scratch; carrier-when-integrated]
```

with ordinary Euclidean concatenation after invalid-token projection.

The deterministic current-graph result is:

| receipt field | value |
|---|---:|
| `Lambda_adapters` | `1.0` |
| `Lambda_hat_core` | `1.0420480479402405` |
| paired lower-bound diagnostic | `1.0002751429235557` |
| power iterations | `47` |
| last relative change | `0.0009388240523025795` |
| convergence tolerance | `0.001` |

`Lambda_hat_core` is the PF-3 authority's empirical **full-visit** estimate. The legacy `AblationLM._visit_jacobian_spectral_norm` probe dimension-reweights hidden and scratch coordinates; it is locked to the non-governing label `non_governing_pre_pf3_dimension_reweighted_hidden_scratch_probe`, is excluded from the two-number receipt, and has a mutation test preventing promotion.

Current factor treatments are recorded rather than implied: anchored re-entry, position-aligned scratch update/injection, loop embedding, and both shared core blocks are included in the full nonlinear visit; the narrow two-lane Birkhoff scratch carrier contributes its exact factor of one. Missing full-width bicameral lanes, learned rotor carrier, per-band callosum, sidecar, and post-loop bridge are not placeholders silently assigned value one.

The two-number current-graph receipt is fully executable, convergence-checked, and mutation-resistant. It is **not** a production certificate or alarm. PF-3.3 prohibits those claims until the complete production topology exists and every live factor is enumerated.

## 5. PF-3.4 — A7 structural-OFF matrix

A7 is implemented with factory-sealed receipts and derived status; callers cannot promote a cell or the full matrix by replacing a status field. The present matrix covers `K in {1,2,4,8}`, fp32 and bf16 CPU cells, and typed CUDA deferrals for these materialized optional modules:

- upfront modified Hadamard experts;
- current static-K/V core;
- recurrent re-entry bridge;
- position-aligned scratch;
- narrow lane carrier;
- causal engram;
- read-only long-term memory.

The matrix has 160 cells:

| status | cells | meaning |
|---|---:|---|
| passed | 54 | eligible integrated CPU cell has bit-identical OFF/background logits and loss, and active ON differs nontrivially |
| ineligible | 4 | structurally ineligible combination, including re-entry at `K=1` |
| pending | 54 | named deterministic CUDA cells not executed in this CPU return |
| absent | 48 | integrated rotor carrier, per-band callosum/final production path, and loop sidecar do not exist in the production graph |

The `K=1` all-optional-OFF dense anchor passes bit-identical logits and loss in both CPU precisions against the standalone 4/2/4 dense background. ON-then-OFF restoration is exact for every eligible currently integrated CPU module/K cell, and each active ON leg is required to change both logits and loss. Executed evidence is bound to its actual device; a CPU tensor cannot mint a deterministic-CUDA cell, and a CUDA cell additionally requires deterministic algorithms to be enabled. A mutation that changes an exact OFF output, invents ON activity, changes eligibility, or attempts to mark an absent/pending cell passed is rejected.

`require_cpu_passed()` succeeds for this bounded CPU subset. The full A7 receipt remains **incomplete by construction**; no integrated-plus-OBS-INV-tested claim is made for the absent modules, and no CPU result is promoted into a CUDA cell.

## 6. Dependency-neutral bicameral recurrence seam

`models/ablation_lm/bicameral_recurrent.py` provides a standalone seam for the portion of D-PF-4 step 2 that does not choose open strategy semantics. It:

- owns repeated execution only, for `K in {1,2,4,8}`;
- uses `alpha = c/K`;
- preserves separated `h_A` and `h_B` outputs;
- accepts one caller-owned opaque cache per unique block and reuses it across visits;
- preserves dense equivalence when disagreement is zero;
- demonstrates live gradients for `mu`, `dU`, and `dV` in Q/K/V/O/gate/up/down paths.

The seam records cache policy as `caller-owned (C-S5-2 unbound)` and final recombination as `not executed (C-S5-1 unbound)`. It rejects a receipt that pretends either policy has been selected. It has not been wired into `AblationLM.forward`, and neither `model.py` nor `config.py` was changed for this seam. This is standalone recurrence scheduling evidence, not an integrated full-width bicameral model.

## 7. Exact four open S5/S6 questions

These are carried verbatim from coding's current question document, so the next strategy ruling can answer the questions actually held by the implementation:

### C-S5-1 — final bicameral recombination

> **Requested ruling:** ratify this unit-circle combiner, or supply the exact alternative formula and initialization.

The proposed map is the sequency-band unit-circle read `y_b = cos(theta_b) * mu_b + sin(theta_b) * delta_b`, initialized with `theta_b = 0`.

### C-S5-2 — singular versus paired static K/V

> **Requested ruling:** ratify paired eigenmode K/V, or explicitly remove K/V from the seven-projection disagreement count and bind a shared consensus cache.

### C-S6-1 — sidecar expert domain and parameter arithmetic

> **Requested ruling:** ratify lane-native accounting, or specify the exact full-width lift/project maps and include them in `N_unique`, `N_recurrent`, optimizer ownership, and T14b.

The lane-native proposal preserves `L=2`, `w=d/4`, rank 4, and the `{4,16}` sweep, with 2.097 M total / 12.288 K active parameters at target.

### C-S6-2 — exact hard invocation and trainable gradients

> **Requested ruling:** ratify the straight-through estimator and bind `tau`, initial eligible firing rate, and calibration/freeze rule; or choose a different estimator explicitly.

No one of these four choices was made locally. In particular, the standalone recurrence seam does not convert its caller-owned cache into a selected K/V representation and does not recombine the two hemispheres.

## 8. Verification performed

The focused combined CPU command was:

```text
python -m pytest -q tests/test_ablation_lm_mup.py tests/test_weft1_preflight_c1.py tests/test_ablation_lm_optim.py tests/test_ablation_lm_rng.py tests/test_ablation_lm_certificates.py tests/test_weft1_preflight_c2.py tests/test_weft1_preflight_cjac1.py tests/test_weft1_preflight_c7.py tests/test_ablation_lm_observational_invariance.py tests/test_ablation_lm_model.py tests/test_ablation_lm_preflight_c3_c6.py tests/test_ablation_lm_bicameral_core.py tests/test_ablation_lm_bicameral_recurrent.py
```

Result:

```text
194 passed, 18 warnings in 19.03s
```

The 18 warnings are PyTorch's existing `torch.jit.script` deprecation warning emitted by certificate tests. No failure was quarantined or suppressed. This was a focused suite, not a claim that the repository-wide suite or any GPU cell passed.

The raw repository-wide suite then completed with:

```text
1 failed, 4067 passed, 19 warnings in 159.78s
```

The sole failure is the already-governed legacy node `tests/test_paper2_claim_evidence_ledger.py::test_paper2_claim_evidence_paths_exist`; it still reports the same two absent, gitignored Paper 2 evidence paths. No PF-3 node failed, and no test was skipped, deselected, xfailed, or suppressed. The successor exact-node quarantine is `training/ablation_lm_engineering_quarantine_20260902_pf3.json`, 3,027 bytes, SHA-256 `f7983fb7c256f388786b07809942787b7697742445e7c3efcd64e5e57a857a9c`, review due 2026-09-04. Its independent gate replay passed with the same exact failure and `4,067` passes (`153.09s`). The repository therefore remains explicitly **red**, not globally green.

Canonical compact/sorted JSON identities for the deterministic CPU receipts are:

| receipt | canonical bytes | SHA-256 / digest |
|---|---:|---|
| C1 stopped inventory | 134,726 | `2858465119a1a5fbb48d2c8ed4effd4d130bf1c62c9f37faf6e509b9c49cba06` |
| C2 complete CPU result | 651,973 | `4e06832ee455daf16503643c8409511da458fa1a53c12fa94bcf96d6a866372a` |
| C-JAC-1 current graph | 5,676 | `f06ea51cad55464277c569f4e3c0a838f80077f71e734fdf92c9e7db5b7845ec` |
| A7 matrix | 160 module cells + 4 anchors | `1a28aa6d6825cec8799237d332ed919657754cc5e51e6aa0695c37eb0e1d2420` |

The JSON identities use `json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":"), default=str)` on the recorded Python/PyTorch runtime. A7 uses its factory's own canonical digest.

Additional receipt invariants and mutation tests cover:

- C1 cannot be promoted past Catch #35;
- C1's six-file governing chain is read and hash-verified, and its protocol/provenance literals cannot be rewritten while retaining a valid receipt;
- C2 cannot exclude an eligible zero or a zero that differs by precision/connectivity, and cannot forge a full-WEFT, trained, checkpoint, GPU, topology or provenance claim;
- the C-JAC state definition, seed, runtime, component inventory, Euclidean metric, convergence fields, two-number semantics, and legacy non-governing label cannot be mutated into a valid receipt;
- A7 cannot forge a passed cell, complete matrix, integrated-module status, or CUDA backend;
- a NaN primal output with a finite identity Jacobian is rejected by both scalar and joint-state estimators;
- sidecar receipt fields are derived from their weights, gate and bounds, while an orthogonality-by-construction rotor certificate requires an explicit operator witness;
- the recurrence seam cannot forge a chosen cache or recombination policy.

`python -m compileall -q models/ablation_lm analysis` also completed successfully. A Ruff result is not claimed because Ruff is not installed in this local interpreter.

## 9. Current boundary and next actions

The safe state is narrow:

- C2 CPU precision calibration passes under PF-3.2.
- The current bring-up graph has a converged PF-3.3 full-current-visit estimate, not a production certificate.
- The currently integrated CPU A7 subset passes, while full A7 remains incomplete.
- A standalone bicameral recurrence scheduler exists, while the production recurrent integration does not.
- C1 remains stopped at Catch #35 before initialization.

Strategy now owes two independent inputs: the Catch #35 mixed-scaling router class and the four exact S5/S6 rulings above. Coding can preserve and test the dependency-neutral seam while those arrive, but it cannot select the missing semantics or call the seam an integrated WEFT-1 graph. P-A and every run-axis gate remain untouched.

Executable implementation commit: `0efb4cdf2fc15b058c1faf6328a6f383350606fe`.
