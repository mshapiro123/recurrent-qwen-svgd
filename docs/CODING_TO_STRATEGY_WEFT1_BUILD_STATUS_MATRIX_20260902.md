# CODING → STRATEGY — WEFT-1 ratified-module build-status matrix

**Date:** 2026-09-02  
**Status:** BUILD-AXIS RECEIPT · no training or sealed-data contact  
**Snapshot base:** `926a77a2`

**PF-2 implementation commit:** `5ebbefea36c13a8f86c5e78ed074efc6c91db12f`

**Architecture authority:** full build handoff, 61,329 bytes, SHA-256 `498f34b5…eb02`, Drive `1XaE81mfqTOYEYGFMa-ZJwpLW-KQtMMC_`  
**Ratification authority:** `STRATEGY_RATIFICATION_RECORD_20260826.md`, 13,908 bytes, SHA-256 `c5df74297594e75697ffb71d8d05d75efcf94f7857d55ddd357043200efb6d3a`, Drive `1Wb_FfEb-Sl-TgL23hcFOy58QcSurfaF3`  
**Status authorities:** PF-2, 13,097 bytes, SHA-256 `be11390c28ae36210a1571f7c6d358ee54e977d239f5344f7e6402212448eb05`; §8.1 amendment, 3,403 bytes, SHA-256 `dd79aaa6fd9bab15bf02aaef28f99f47c745d63eed6db2c9b929b4bb1cfbb418`

## 0. Outcome

The current `AblationLM` is a causally tested recurrent bring-up graph, not the complete WEFT-1 production graph. It contains the dense substrate, shared recurrence, a single-stream static-K/V path, position-aligned scratch, a narrow two-lane carrier, an upfront dense-routing Hadamard challenger, token-ID engrams, read-only long-term memory, and hidden-stream trajectory diagnostics. Several ratified production surfaces are absent or only standalone: paired bicameral K/V semantics, the integrated full-width bicameral core, learned rotor carrier and write, per-band callosum and final combiner, conditional loop sidecar, occupancy router, pooled-lane jets and learned probes, and the full objective stack.

**PF-LANG applies throughout:** a standalone certificate never implies that the integrated production graph passed. A causality result, a zero-tolerance numerical-equivalence check, a bit-identical OBS-INV result, and an operator certificate are four different claims. This snapshot mints no complete OBS-INV matrix and no production certificate.

## 1. Status vocabulary

| status | exact meaning |
|---|---|
| `absent` | no executable implementation of the ratified surface in the WEFT-1 production model path |
| `standalone primitive` | executable and unit-tested in isolation, but not reachable from `AblationLM.forward` |
| `standalone harness` | executable measurement/receipt machinery outside `AblationLM.forward` |
| `integrated` | reachable from `AblationLM.forward` behind its stated switch; no registered per-module OBS-INV pass is implied |
| `integrated + OBS-INV-tested` | integrated and covered by a registered, bit-identical structural-OFF comparison at the named cell |

No row in this snapshot qualifies for `integrated + OBS-INV-tested`. The dense anchor's existing check uses `torch.testing.assert_close(..., rtol=0, atol=0)`; that is strong zero-tolerance numerical evidence, but it is not relabeled as the registered bit-identity matrix requested by A7.

## 2. Ratified-module matrix

| ratified surface | status | implementation and test evidence | introducing / governing commit | precise limitation |
|---|---|---|---|---|
| Dense causal substrate: tied embedding/readout, pre-RMSNorm, QK norm, RoPE, GQA, SwiGLU, prelude/core/coda | `integrated` | `models/ablation_lm/model.py`; `test_t1_disabled_graph_is_exactly_the_dense_transformer` | `85bb8eae` | Existing anchor is zero-tolerance `assert_close`, not a registered bit-identical OBS-INV matrix. |
| μP attention-logit base-shape scale `sqrt(d_head,base)/d_head`, `d_head,base=64` | `integrated` | existing inverse-square-root attention is the exact numerical realization at the ratified fixed `d_head=64`; named base constant and fused/math/reference assertions in `config.py`, layer tests and bicameral target-shape tests | PF-2 implementation return | No numerical behavior changed. A future WEFT configuration with `d_head != 64` requires the explicit base-shape implementation; ordinary non-WEFT toy configurations retain their prior inverse-square-root behavior. |
| Shared recurrent core, inference-controllable `K`, `alpha_T=c/T`, requested/executed accounting | `integrated` | `AblationLM._run_recurrent_core`; K override and accounting tests | `85bb8eae`, `5c835c3c`, `83157bfb` | Current recurrent core is single-stream dense, not the full bicameral recurrent core. |
| Fixed-anchor static K/V, live-state queries: current single-stream form | `integrated` | `models/ablation_lm/model.py`; T15 equivalence, projection-count, gradient and T14b regressions | `51c16443` | Full-sequence/reference path only; production autoregressive serving cache is not claimed. |
| Production bicameral static K/V representation | `absent` | a standalone paired-block candidate exists, but C-S5-2 remains unresolved between paired-eigenmode and shared-consensus K/V | `0c95a7cc`, `e78692dd` (candidate only) | The need for a production representation is ratified; the paired representation is not. No choice or integrated recurrent call site exists. |
| Fork B-prime midpoint K/V refresh | `integrated` | explicit refresh flag plus requested/executed refresh receipt tests | `51c16443` | Challenger on the current single-stream graph, not the ratified paired production graph. |
| Position-aligned causal scratch lanes | `integrated` | `PositionAlignedScratch`; packed/padded causality and gradient-liveness tests | `85bb8eae` | Two narrow lanes accompany one full-width stream; they are not two full-width hemispheres. |
| Narrow two-lane Birkhoff carrier | `integrated` | `TwoLaneBirkhoffMixer`; retention and mean/disagreement tests | `85bb8eae`, `1c033bae` | Not the ratified per-band full-width callosum. |
| `bridge_in`, in-loop scratch readout, and anchored recurrent re-entry | `integrated` | `PositionAlignedScratch`, `AnchoredReentryBridge`; liveness and causality tests | `85bb8eae` | Re-entry is intentionally ineligible on visit zero; current readout is not the required post-loop coda `bridge_out`. |
| Final post-loop `bridge_out` into coda | `absent` | no post-loop bicameral-to-coda bridge call site | — | Current in-loop scratch readout does not satisfy this surface. |
| Stored bicameral modes (`mu`, low-rank `delta`) via `SwapLinear` | `standalone primitive` | `models/ablation_lm/bicameral.py`; equivalence, symmetry-break and optimizer-cohesion tests | `51c16443` | Not used by `AblationLM.forward`. |
| Full-width paired bicameral Transformer block and seven disagreement projections | `standalone primitive` | `models/ablation_lm/bicameral_core.py`; one-block dense equivalence, causality and gradient tests | `0c95a7cc`, `e78692dd` | One-block candidate only; not recurrently integrated, and paired K/V semantics remain unbound. |
| Per-band Birkhoff corpus callosum | `standalone primitive` | `models/ablation_lm/callosum.py`; T16 and closed-form delta-mode tests | `51c16443`, `f589e787` | Callosum-only evidence excludes intervening core dynamics and production integration. |
| Final per-band bicameral combiner | `absent` | no model call site | — | `combine(h_A,h_B)` remains unimplemented. |
| Euclidean `Cl(2,0)` rotor operator | `standalone primitive` | `models/ablation_lm/geometry.py`; theta-zero bit identity and K=1…8 norm-isometry tests | `1c033bae`; PF-1 worktree test | Tensor primitive only; no learned production rotor carrier. |
| Learned carrier rotor bank (`J=8`) | `absent` | no model parameter or call site | — | Standalone `Cl20Rotor` is not this integrated module. |
| Single gated rank-8 lane-to-carrier write per hemisphere | `absent` | accounting placeholders only | `83157bfb` | No executable write path exists. |
| Upfront modified Hadamard expert challenger | `integrated` | `ModifiedHadamardExpertBank`; WHT, router, causality and liveness tests | `85bb8eae`, `51c16443` | Dense soft input routing only; no registered matched-dense T12 result. |
| T12 matched-dense Hadamard control | `absent` | no parameter/FLOP-matched dense control implementation and receipt | — | Existing dense-equivalence tests do not instantiate the registered control. |
| Fixed-measurement occupancy router and sparse core expert bank | `absent` | calibration-stability diagnostics exist, but no fixed router/model path | `1c033bae` (diagnostic only) | Current learned dense router must not be described as the occupancy router. |
| Conditional shared loop sidecar, rank-4 experts and hard invocation | `absent` | accounting/schema placeholders only | `83157bfb`, `f589e787` | C-S6-1 width/accounting and C-S6-2 hard-invocation estimator/init remain unresolved. Legacy `models/sidecar_v2.py` belongs to an older programme and is not WEFT-1 S6. |
| Hidden-stream second-order trajectory jet prototype (`v`, `a`, curvature/Gram diagnostics) | `integrated` | `trajectory_jet_metrics`; initial plus visit hidden states retained | `1c033bae` | Uses the full hidden stream and is diagnostic only. |
| Ratified pooled-lane jet over recurrent lanes | `absent` | no pooled-lane state pipeline | — | Current hidden-stream prototype is not this object. |
| Frozen random hidden-state plane probes | `integrated` | `plane_probe_features` and model diagnostics | `1c033bae` | Hidden-stream diagnostic only; fixed projection-basis identity is not yet recorded in the composition receipt. |
| Learned lane-plane probes | `absent` | no learned probe parameters or lane-state probe path | — | Must not be inferred from the frozen hidden-state probes. |
| Rotor-QK attention arm | `absent` | no module, config field, or call site | — | `J_att=0` is effectively the only current topology. |
| `M_lex` causal engram, token-ID n-gram bring-up | `integrated` | `CausalTokenEngram`; packing, padding, provenance and liveness tests | `85bb8eae` | Token-ID hashing only. |
| `M_lex` byte-span polynomial hashing / tokenizer-agnostic address | `absent` | no byte-span address path | — | Explicitly deferred ablation in the handoff. |
| Read-only long-term memory with leave-one-record-out | `integrated` | `ReadOnlyLatentMemory`; frozen-store and provenance-exclusion tests | `85bb8eae` | Read-only retrieval arm; not delta-rule or fast-weight memory. |
| Adaptive halting head | `absent` | no parameter, config switch, or call site | — | `K` is controllable but not learned per example. |
| Staged-state objective `L_stage` and trajectory curriculum | `absent` | no loss or data pipeline in the model build | — | Remains run-axis/curriculum work. |
| Other registered auxiliaries: `L_div`, `L_ret`, `L_halt`, `L_inv`, `L_conv`, `L_plane` | `absent` | no complete model-loss implementation | — | Diagnostics with similar names do not constitute these objectives. |
| Z-loss with shared valid-token mask | `integrated` | `_language_model_loss`; exact zero-coefficient structural path tests | `1c033bae` | Auxiliary loss only; default is structural zero. |
| Composition accounting and requested/executed visit receipt | `integrated` | `models/ablation_lm/accounting.py` | `83157bfb` | Sidecar/callosal fields remain placeholders; fixed projection-basis SHA is not recorded. |
| O-9 per-module RNG registry and paired recurrent-visit alignment | `integrated` | `models/ablation_lm/rng.py`; module-stream isolation and seed-receipt tests | `5ecf58fb` | Applies to materialized modules; absent modules have no live draw stream yet. |
| Fitted carrier-retention gauge `r` and `r>=0.9` tripwire | `absent` | no fitted production carrier or integrated retention receipt | — | Narrow carrier tests are not the registered production gauge. |
| T14b exact-autograd receipt machinery | `standalone harness` | `observatory.py`; exact evidence and fail-closed guards | `dc63ee4e` | Not reachable from `AblationLM.forward`; no complete production receipt for absent modules. |
| Observatory event schema, including RESP-LEAK binding | `standalone harness` | `observatory_events.py`; schema tests | `f589e787` | External receipt machinery, not an integrated model module. |

## 3. Exact causality and OBS-INV boundary

The current static/dense bring-up graph has independent direct future-gradient tests at `K=(1,2,4,8)`. The maximal optional-module construction is tested at `K=8` with visit-by-visit logits; it is not four independent all-optionals-active constructions at K=1, 2, 4 and 8. The standalone bicameral block has a separate one-block packed/padded causal test. These are causality results, not OBS-INV or production-integration certificates.

The worktree contains typed exact-comparison and deferred-cell plumbing for A7, plus concrete counterexamples to two ambiguous literal readings. No A7 matrix is minted while catch #27 is open. In particular, no row above is promoted to `integrated + OBS-INV-tested` from a standalone certificate, a causality test, or zero-tolerance numerical equivalence.

## 4. PF-2 PRE-FLIGHT status

- **Jacobian panel:** the registered main panel is now `n=520`. The corrected power calculation meets both literal frontiers (`SE=0.05092447485214237 <= 0.051` and `SE=0.03578829611711741 <= 0.036`), and all four previously planted B1 calibration phases remain green.
- **C1:** the PF-2 topology is bound and the amended attention coefficient is byte-verified. The width run remains fail-closed as **Catch #33** because §8 does not bind eight load-bearing μP literals: numeric `d_base`, numeric `sigma_base`, a complete per-tensor initialization map, numeric `eta_base`, a complete per-tensor learning-rate map, residual `alpha`, embedding multiplier, and residual multiplier. No model or optimizer was constructed.
- **C2:** the terminal K=8 CPU check numerically passes the post-data PF-2 thresholds: hidden `0.002378677322798898`, lanes `0.016478415427698328`, logits `0.003659068615870767`, full-gradient vector `0.0071749515521762446`, and terminal worst module tensor `0.02583730846608379`. The complete receipt remains fail-closed as **Catch #34** because three visit-1 re-entry parameters are structurally disconnected with zero reference norm and PF-2 supplies no zero-reference/eligibility rule. The learned rotor-carrier decision and GPU evaluator identity remain deferred.
- **C3/C6:** PF-2's typed complete-gate posture stands; no bounded CPU subset is promoted to the unavailable CUDA, dropout, STOCH-K, or absent-module cells.
- **C7:** stage 1 emits the four present G-TOK families through the production matrix, selection, confirmation-budget and checkpoint-accounting builders from deterministic synthetic source receipts. The byte-matched base path is independently bound at `n=400`; the fresh confirmation path is joined to its selected budget row at `n=399`, with separate byte totals and first-crossing indices. Receipt SHA-256 is `04b9c1515a3902c2963eb1e13e5bfa42ede144549f88a44366953b76a422abd6`. Stage 2 remains open on the sidecar and Catch #26/C-JAC-1; the complete C7 gate remains incomplete and non-forgeable.
- **Catch #26 disposition:** **OPEN.** Exact standalone adapter factors and an empirical core estimate exist, but no ratified joint-state metric or production nonlinear-visit certificate/alarm exists; C-JAC-1 still governs.
- **Catch #27 disposition:** **OPEN.** Exact matched-background comparison plumbing exists, but A7's comparison graph and K/module eligibility remain unbound; no registered matrix is minted.

## 5. Queue implied by the evidence

The evidence supports this dependency order without choosing an unresolved design locally:

1. bind C-S5-1 final recombination and C-S5-2 production K/V representation;
2. integrate the selected full-width bicameral block into the recurrent model path;
3. integrate the learned rotor carrier, single write, final `bridge_out`, and retention gauge;
4. integrate the per-band callosum;
5. bind C-S6-1 sidecar width/accounting and C-S6-2 hard-invocation estimator/init, then implement and integrate the conditional loop sidecar;
6. bind catch #26's nonlinear-visit measurement/certificate semantics and catch #27's A7 comparison graph;
7. only then mint the first full production T14b and OBS-INV receipts.

This matrix does not authorize coding through unresolved C-S5-1/2 or C-S6-1/2 semantics, and it does not convert an older-programme module into a WEFT-1 implementation by name similarity. Catch #26 also preserves the open C-JAC-1 joint-state metric rather than supplying one locally.

## 6. Do-not-claim boundary

- No integrated full-width bicameral recurrent result exists.
- No selected production bicameral static-K/V representation exists.
- No integrated rotor-carrier or fitted retention result exists.
- No integrated per-band callosum or final-combiner result exists.
- No loop-sidecar or occupancy-router result exists.
- No registered matched-dense Hadamard result exists.
- No complete A7, production-certificate, or production T14b receipt exists.
- No G-TOK, model-training, checkpoint, evaluation-panel, or sealed-data work occurred in producing this matrix.
