# CODING → STRATEGY — WEFT-1 architecture, math, and EG-1 return

**Date:** 2026-09-03  
**Status:** BUILD-AXIS IMPLEMENTATION RETURN · no GPU, corpus, frozen vocabulary, checkpoint, model training, evaluation-panel, or sealed-data contact  
**Code anchor:** `28935e5d6ce7fb1578a95ddfb49442177425cc34`

## 0. Outcome in plain language

The architecture reconciliation, independent math check, sampled-`L_stage` decision, and EG-1 correction are implemented to the extent permitted by the current graph.

The one-page SVG now separates three things that earlier diagrams blurred: the ratified WEFT-1 target, the primitives that exist only in isolation, and the smaller recurrent bring-up graph that actually runs today. The target has 22 unique decoder blocks. Rung A is **9 / 4 / 9**; Rung B is **8 / 6 / 8**. The full-width paired core, learned rotor carrier and write, integrated per-band callosum and combiner, per-lane sidecar, `bridge_out`, and objective/halting stack remain visibly pending rather than being implied by standalone tests.

EG-1 is now exact in code. Its query and key are formed in the 64-dimensional memory space, both sides have unit-initialized trainable RMSNorm gains, there is no learned key projection, and the dot is divided by `sqrt(64)`. A distinct gate-only key norm keeps `gamma_k` out of the value path. T2 proves step-1 liveness for `W_Q`, `gamma_q`, `gamma_k`, every addressed table, `W_V`, and the residual gate.

One provisional implementation was rejected before publication: making the existing memory normalization affine gave `gamma_k` an unintended direct path into `W_V`. The final implementation restores the old non-affine value normalization and adds a separate affine key norm used only by the gate. A forward-hook regression proves that changing `gamma_k` changes the gate but cannot change the tensor passed to `W_V`.

The math companion remains green and the deterministic C1, C2, and C-JAC receipts were replayed after the final topology change. The full repository suite has one unchanged, governed Paper Two evidence-ledger failure; the WEFT engineering gate passes only by matching that exact node against a dated quarantine. No repository-wide green claim is made.

## 1. Authorities verified before acting

| authority | bytes | SHA-256 | result |
|---|---:|---|---|
| `STRATEGY_ARCHITECTURE_RECONCILIATION_20260903.md` | 20,695 | `0d81e9ab63d21720fecfbfcb629aaa5eeae6693eabbd9682b82adc7e3792ea8e` | exact |
| `STRATEGY_MATH_CHECK_20260903.md` | 16,587 | `509cac8c7f5f82a6a70d0bcc8494b02967d3f545e4e875e9bbfcdc2b93dedcff` | exact |
| `math_check_20260903.py` | 7,586 | `9dbe3724345382d451fd03af7e57d9503a1fe4d626d0c4cba9a4acc80b08195b` | exact |
| `STRATEGY_MATH_CHECK_RATIFICATION_20260903.md` / D-MC-1 | 2,868 | `9c5822daef5dbb0609bc3e46019cc4b1e332991c30e8a42c1b4432800a747ab1` | exact |
| `STRATEGY_ENGRAM_GATE_RATIFICATION_20260903.md` / EG-1 | 4,398 | `36f0255c1cc0e61b2d9019ce86b3b1e7446b0a2c3445a42ce20334213deae780` | exact |

The architecture reconciliation supersedes the previous layout where the listed decisions conflict. EG-1 amends the engram formula in place and records Catch #39 against strategy drafting, not coding.

## 2. EG-1 implementation

The executable gate is

```text
q_t = RMSNorm_gamma_q(W_Q h_t)       W_Q: R^d -> R^64
k_t = RMSNorm_gamma_k(e_t)           e_t: retrieved 64-coordinate row
g_t = sigmoid(<q_t, k_t> / sqrt(64) + b_g)
v_t = W_V(RMSNorm_no_affine(e_t))
h_t <- h_t + gamma_m * cap(g_t * v_t)
```

This gives the key-side gain a temperature degree of freedom without adding `W_K` or changing the value input. The two new gain tensors are `(64,)`, initialized to one, classified as muP vectors, assigned the base learning rate, zero decay, and the auxiliary AdamW partition. The engram query remains the authorized fixed-output `HIDDEN` exception; its shape is `(64, d)`. The value projection remains fixed-fan-in `INPUT`, shape `(d, 64)`.

The current generic optimizer partition groups heterogeneous auxiliary roles. That is safe for this unconstructed build state, but production optimizer construction must preserve the normalization gains' zero-decay rule rather than applying one blanket auxiliary weight decay.

## 3. Math and ordering checks

The supplied CPU companion was run under Python 3.11 with UTF-8 output. Its deterministic replay remains green:

- symmetry: `delta = 0` leaves both low-rank disagreement-factor gradients exactly zero while `mu` remains live; the existing T2 assertion explicitly requires nonzero gradients on both factors under the registered nonzero initialization;
- callosum: disagreement contracts by the closed form `(1 - 2 rho)^K`; `rho = 1/2` annihilates state disagreement but does not alter or identify the two weight sets;
- WHT: normalized transform is an involution to floating-point tolerance, and the sequency order is exactly `bitrev(gray(k))`; T4 now checks this at the proxy and target widths, 512 and 1024;
- Jacobian panel: the corrected regressor uses `ln(T)` and the registered panel remains `n = 520`;
- `L_stage`: final visit plus one uniformly sampled earlier visit is the default; all-visit decoding remains a registered contrast.

For D-MC-1, the earlier visit is drawn once per micro-batch from its own O-9 stream, never from the final visit, and both states pass serially through the same coda. The receipt now carries `coda_decodes_per_step` and `lstage_sampled_visit`. At `K=1`, only the final state is decoded and the sampled-visit field is null.

Relative to the previously priced one-coda baseline, two coda decodes produce multipliers of `1.323077` at `K=2`, `1.238636` at `K=4`, and `1.189189` at `K=6`. The planning midpoint of 234 A100-hours therefore becomes 289.84 A100-hours at `K=4`, a 55.84 A100-hour delta. This is a planning re-derivation, not consumed compute; exact allocation must be reminted after the integrated five-paired and per-lane-sidecar graph exists. If the allowance is exceeded, the registered rung-B-first de-scope order applies.

## 4. Deterministic receipt replay

### C1 inventory

At each of `d = 128, 256, 512`, 147 of 148 unique trainable tensors are classified. The sole governed non-pass remains the production-retired legacy `front_hadamard.router.weight`.

| width | classified-map SHA-256 |
|---:|---|
| 128 | `a995377ed0566ad2cef6f16084eab733f563d83f7453cb4ecfee56213ee651d6` |
| 256 | `3f6b98f7a420235754d822818fc94f4bbb4e5ed5bc5f84ac1e37463ed6bbea78` |
| 512 | `3bd1ccc707a8f7e3f063b07afca7b9c3e9a681001724d133fb0601fda776c8b0` |

Canonical compact C1 receipt SHA-256: `ff1976fcb84eed77970f620363fac26a76630cbd522788bb9615030601f706e4`.

### C2 current-graph CPU precision

The two gains change the state identity and gradient population, so stale Catch-37 goldens were not carried forward.

| field | reminted value |
|---|---:|
| initial model-state SHA-256 | `8420c11aa7746c2191206cb061bd9bcbecf217fe264e0fc24cdb1761ee2dc2fb` |
| trainable elements / tensors | 484,795 / 144 |
| module maxima | 142 |
| terminal hidden relative L2 | `0.002343670477777934` |
| terminal scratch-lane relative L2 | `0.016916588480255675` |
| terminal logits relative L2 | `0.00366168723457278` |
| terminal full-gradient relative L2 | `0.007065333916557898` |
| terminal worst module tensor | `0.023292915877290394` |

The worst terminal tensor remains `core_blocks.0.attention.key_norm.weight`. All registered terminal bands pass. The three visit-1 recurrent-reentry rows remain structurally ineligible and exact zero. Canonical compact C2 receipt SHA-256: `e75d200e1c6af963fbc4085f4b804daca95198cb528ebc0b325c1d3a03515859`.

### C-JAC current-graph estimate

The unit-initialized gains leave the current forward function and input-state Jacobian unchanged, confirmed by bit-identical replay: `Lambda_adapters = 1.0`, `Lambda_hat_core = 1.041439431237759`, 47 iterations, final relative change `0.0009611520575406325`. Canonical compact receipt SHA-256 remains `319ad9ef3414b6856a85bac10b38adb5530ed29c04bbee8c671bc78346ee3e3d`.

This is still a current-graph estimate over `z = [h; scratch]`, not a production certificate. Full-width lanes and the learned carrier are absent.

## 5. Diagram and source artifacts

| artifact | bytes | SHA-256 |
|---|---:|---|
| `docs/figures/weft1_architecture_and_build_state_20260903_r2.svg` | 30,499 | `8b0742706fd45123ef655cf1b6cd565aca1c78e0ed21474140e71a809c5e3f58` |
| `docs/CODING_TO_STRATEGY_WEFT1_BUILD_STATUS_MATRIX_R2_20260903.md` | 22,215 | `e5a140e214b66829d4248318745995d9d51dfac685c5387132d7f5c2dc861c04` |
| `docs/WEFT1_STEP6_OBJECTIVE_AND_SAMPLED_DECODE_SPEC_20260903.md` | 5,391 | `6cc1dde1105fc049d1202da39bc120c6baf6adb2afd6ba08d9c019a79069d6ec` |

The SVG is the editable diagram source requested for strategy. It embeds the full authority hashes and code anchor in its metadata, has an accessible title/description, and was rendered at 1800 x 1600 for visual inspection. Status is encoded by label and stroke/fill treatment, not color alone.

## 6. Verification and quarantine

- EG-1, liveness, C1, C2, and C-JAC focused set: **60 passed**, 19 warnings.
- Full repository suite: **1 failed, 4,070 passed, 20 warnings** in 157.35 seconds.
- Engineering-quarantine schema tests: **6 passed**; the wrapper then reran the full suite and reported **PASS** only for the exact quarantined node, with the underlying repository explicitly still red (`1 failed, 4,070 passed, 20 warnings` in 154.09 seconds).
- Sole failure: `tests/test_paper2_claim_evidence_ledger.py::test_paper2_claim_evidence_paths_exist`.
- The failure still names the same two absent legacy Stage-2A T3 evidence paths. It is unrelated to WEFT-1 and is preserved as a governed non-pass, not ignored or relabeled green.
- `training/ablation_lm_engineering_quarantine_20260903_eg1.json` is v18 and supersedes the retained v17 math-reconciliation quarantine by exact SHA. Review remains due 2026-09-04.
- The engineering wrapper accepts only that exact node and exact aggregate counts. The underlying repository remains red.
- `git diff --check` passes except for Git's existing LF-to-CRLF notices. Ruff is not installed in the recorded local interpreter, so no lint pass is claimed.

## 7. Remaining boundary and next build work

No strategy design question remains open for EG-1 or the architecture reconciliation. The next build-axis sequence remains:

1. integrate the five-paired bicameral path with shared-consensus K/V, S-2 final combiner, legacy registry, and exact visit schedule;
2. integrate the learned `J=8` carrier, one rank-8 write per hemisphere, `bridge_out`, and retention gauge;
3. integrate the per-band callosum once per visit and retire the narrow mixer;
4. integrate the per-lane sidecar, shared bank/PQ, pooled-lane jets, learned probes, calibration/freeze gate, and invocation-agreement receipt;
5. land the sampled objective/halting/K stack and then complete the certificate, A7, and production T14b surfaces.

The run axis remains independently sequenced by the ratified corpus programme. Nothing in this return changes P-A, P-B, G-TOK, compute tripwires, decontamination, frozen-vocabulary, or sealed-data authority.
