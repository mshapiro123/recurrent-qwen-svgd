# CODING → STRATEGY — WEFT-1 ratified-module build-status matrix

**Date:** 2026-09-02  
**Status:** BUILD-AXIS RECEIPT · no training or sealed-data contact  
**Snapshot base:** `0efb4cdf2fc15b058c1faf6328a6f383350606fe`

**PF-2 implementation commit:** `5ebbefea36c13a8f86c5e78ed074efc6c91db12f`

**Architecture authority:** full build handoff, 61,329 bytes, SHA-256 `498f34b5…eb02`, Drive `1XaE81mfqTOYEYGFMa-ZJwpLW-KQtMMC_`  
**Ratification authority:** `STRATEGY_RATIFICATION_RECORD_20260826.md`, 13,908 bytes, SHA-256 `c5df74297594e75697ffb71d8d05d75efcf94f7857d55ddd357043200efb6d3a`, Drive `1Wb_FfEb-Sl-TgL23hcFOy58QcSurfaF3`  
**Status authorities:** PF-2, 13,097 bytes, SHA-256 `be11390c28ae36210a1571f7c6d358ee54e977d239f5344f7e6402212448eb05`; §8.1 amendment, 3,403 bytes, SHA-256 `dd79aaa6fd9bab15bf02aaef28f99f47c745d63eed6db2c9b929b4bb1cfbb418`; PF-3, 14,632 bytes, SHA-256 `7f0081e504366ce98f8bf183b7e14c0bed47647aa381196a9d5e9540b5334cef`; D-PF-4 ratification, 2,336 bytes, SHA-256 `b4e13508c4135837dd197f19f2711026a3df429df7a463386ed5ac1aa94dba31`

## 0. Outcome

The current `AblationLM` is a causally tested recurrent bring-up graph, not the complete WEFT-1 production graph. It contains the dense substrate, shared recurrence, a single-stream static-K/V path, position-aligned scratch, a narrow two-lane carrier, an upfront dense-routing Hadamard challenger, token-ID engrams, read-only long-term memory, and hidden-stream trajectory diagnostics. A new standalone bicameral recurrence seam executes the paired block for registered `K` with caller-owned cache objects and `c/K`, while explicitly declining both terminal recombination and a production cache-policy claim. It is not called by `AblationLM.forward`. Several ratified production surfaces remain absent or only standalone: selected paired bicameral K/V semantics, the integrated full-width bicameral core, learned rotor carrier and write, per-band callosum and final combiner, conditional loop sidecar, occupancy router, pooled-lane jets and learned probes, and the full objective stack.

**PF-LANG applies throughout:** a standalone certificate never implies that the integrated production graph passed. A causality result, a zero-tolerance numerical-equivalence check, a bit-identical OBS-INV result, and an operator certificate are four different claims. PF-3 closes the semantic gaps in C2, C-JAC-1, and A7; the current CPU evidence is green. This snapshot still mints no complete all-backend OBS-INV matrix and no production certificate because deterministic CUDA cells and ratified topology are incomplete.

## 1. Status vocabulary

| status | exact meaning |
|---|---|
| `absent` | no executable implementation of the ratified surface in the WEFT-1 production model path |
| `standalone primitive` | executable and unit-tested in isolation, but not reachable from `AblationLM.forward` |
| `standalone harness` | executable measurement/receipt machinery outside `AblationLM.forward` |
| `integrated` | reachable from `AblationLM.forward` behind its stated switch; no registered per-module OBS-INV pass is implied |
| `integrated + OBS-INV-tested` | integrated and covered by a registered, bit-identical structural-OFF comparison at the named cell |

The registered CPU portion now supplies bit-identity evidence for the dense anchor and 54 eligible integrated-module cells in fp32 and bf16. No row receives an unqualified `integrated + OBS-INV-tested` promotion in this matrix: 54 deterministic-CUDA cells remain pending, absent integrations remain typed non-passes, and A7 may be minted only when every required cell for every integrated module is green.

## 2. Ratified-module matrix

| ratified surface | status | implementation and test evidence | introducing / governing commit | precise limitation |
|---|---|---|---|---|
| Dense causal substrate: tied embedding/readout, pre-RMSNorm, QK norm, RoPE, GQA, SwiGLU, prelude/core/coda | `integrated` | `models/ablation_lm/model.py`; the registered A7 `dense_4_2_4_k1` CPU anchor passes `torch.equal` for logits and loss in fp32 and bf16 | `85bb8eae`; `0efb4cdf` | Deterministic-CUDA anchor cells remain pending, so the complete A7 matrix is not minted. |
| μP attention-logit base-shape scale `sqrt(d_head,base)/d_head`, `d_head,base=64` | `integrated` | existing inverse-square-root attention is the exact numerical realization at the ratified fixed `d_head=64`; named base constant and fused/math/reference assertions in `config.py`, layer tests and bicameral target-shape tests | PF-2 implementation return | No numerical behavior changed. A future WEFT configuration with `d_head != 64` requires the explicit base-shape implementation; ordinary non-WEFT toy configurations retain their prior inverse-square-root behavior. |
| Shared recurrent core, inference-controllable `K`, `alpha_T=c/T`, requested/executed accounting | `integrated` | `AblationLM._run_recurrent_core`; K override and accounting tests | `85bb8eae`, `5c835c3c`, `83157bfb` | Current recurrent core is single-stream dense, not the full bicameral recurrent core. |
| Fixed-anchor static K/V, live-state queries: current single-stream form | `integrated` | `models/ablation_lm/model.py`; T15 equivalence, projection-count, gradient and T14b regressions | `51c16443` | Full-sequence/reference path only; production autoregressive serving cache is not claimed. |
| Production bicameral static K/V representation | `absent` | a standalone paired-block candidate and cache-policy-agnostic recurrent seam exist, but C-S5-2 remains unresolved between paired-eigenmode and shared-consensus K/V | `0c95a7cc`, `e78692dd`; `0efb4cdf` seam | The seam accepts only caller-owned block cache objects and records `caller-owned (C-S5-2 unbound)`; it does not select a representation or establish an integrated recurrent call site. |
| Fork B-prime midpoint K/V refresh | `integrated` | explicit refresh flag plus requested/executed refresh receipt tests | `51c16443` | Challenger on the current single-stream graph, not the ratified paired production graph. |
| Position-aligned causal scratch lanes | `integrated` | `PositionAlignedScratch`; packed/padded causality and gradient-liveness tests | `85bb8eae` | Two narrow lanes accompany one full-width stream; they are not two full-width hemispheres. |
| Narrow two-lane Birkhoff carrier | `integrated` | `TwoLaneBirkhoffMixer`; retention and mean/disagreement tests | `85bb8eae`, `1c033bae` | Not the ratified per-band full-width callosum. |
| `bridge_in`, in-loop scratch readout, and anchored recurrent re-entry | `integrated` | `PositionAlignedScratch`, `AnchoredReentryBridge`; liveness and causality tests | `85bb8eae` | Re-entry is intentionally ineligible on visit zero; current readout is not the required post-loop coda `bridge_out`. |
| Final post-loop `bridge_out` into coda | `absent` | no post-loop bicameral-to-coda bridge call site | — | Current in-loop scratch readout does not satisfy this surface. |
| Stored bicameral modes (`mu`, low-rank `delta`) via `SwapLinear` | `standalone primitive` | `models/ablation_lm/bicameral.py`; equivalence, symmetry-break and optimizer-cohesion tests | `51c16443` | Not used by `AblationLM.forward`. |
| Full-width paired bicameral Transformer block and seven disagreement projections | `standalone primitive` | `models/ablation_lm/bicameral_core.py`; dense equivalence, causality and gradient tests; `models/ablation_lm/bicameral_recurrent.py` executes registered `K`, reuses caller caches, applies `c/K`, and preserves dense equivalence at zero disagreement | `0c95a7cc`, `e78692dd`; `0efb4cdf` seam | The recurrent seam is standalone and is not called by `AblationLM.forward`; it deliberately executes no C-S5-1 terminal recombination and selects no C-S5-2 cache policy. |
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

PF-3.4 closes Catch #27's missing definition and the worktree materializes its registered comparison graph. The matrix has **160 module cells**: 54 CPU passes, 54 deterministic-CUDA pending cells, 48 absent cells, and four structurally ineligible cells. The separate dense K=1 4/2/4 CPU anchors pass `torch.equal` on both logits and loss in fp32 and bf16. Every eligible integrated CPU module cell passes OFF idempotence bit-identically and has a non-trivial ON positive control; re-entry at K=1 is typed ineligible, not averaged into a pass. The 54 deterministic-CUDA cells remain pending and block complete promotion. In particular, no row above is promoted from a standalone certificate or causality result, and no complete A7 receipt is claimed.

## 4. PF-3 / D-PF-4 PRE-FLIGHT status

- **Jacobian panel:** the registered main panel is now `n=520`. The corrected power calculation meets both literal frontiers (`SE=0.05092447485214237 <= 0.051` and `SE=0.03578829611711741 <= 0.036`), and all four previously planted B1 calibration phases remain green.
- **C1 / Catch #33:** PF-3.1 binds the base shape, provisional numeric base constants, per-tensor init/LR classes, tied-readout multiplier, constant decoupled weight decay, residual rules, and fail-closed classification; Catch #33 is therefore closed as a missing-specification catch. The inventory classifies **146 of 147** unique trainable tensors at each of `d=128,256,512`. It then correctly stops as **Catch #35** on the sole unclassifiable tensor, `front_hadamard.router.weight`, with shapes `8x128`, `8x256`, and `8x512`: fan-in scales with width while fan-out is fixed, so it is neither PF-3 hidden, input, vector/scalar, nor tied-readout class. The LTM and engram projections were classified from their actual dimensions rather than module labels. No PF-3 initialization, optimizer construction, forward pass, ten-step run, or RMS ratio was executed after the stop.
- **C2 / Catch #34:** PF-3.2 closes Catch #34 and the current integrated-composition CPU receipt is complete and **passes**. At terminal K=8 the vector-relative L2 values are hidden `0.002378677322798898`, scratch lanes `0.016478415427698328`, logits `0.003659068615870767`, full-gradient vector `0.0071749515521762446`, and worst module tensor `0.02583730846608379` (`engram.raw_residual_scale`). All three visit-1 re-entry zero-reference cells are structurally ineligible and exactly zero in both precisions. The retained non-terminal leading diagnostic is `engram.gate_bias=0.06243657691629686` at visit 4. This is the PF-3 current-graph CPU result; the learned full rotor carrier, complete WEFT-1 topology, and GPU evaluator remain deferred rather than silently included.
- **C3/C6:** PF-2's typed complete-gate posture stands; no bounded CPU subset is promoted to the unavailable CUDA, dropout, STOCH-K, or absent-module cells.
- **C7:** stage 1 emits the four present G-TOK families through the production matrix, selection, confirmation-budget and checkpoint-accounting builders from deterministic synthetic source receipts. The byte-matched base path is independently bound at `n=400`; the fresh confirmation path is joined to its selected budget row at `n=399`, with separate byte totals and first-crossing indices. Receipt SHA-256 is `04b9c1515a3902c2963eb1e13e5bfa42ede144549f88a44366953b76a422abd6`. Stage 2 remains open on the sidecar and Catch #26/C-JAC-1; the complete C7 gate remains incomplete and non-forgeable.
- **C-JAC-1 / Catch #26:** PF-3.3 closes the missing joint-state-metric definition and authorizes a current-graph measurement. At terminal visit 8, plain Euclidean `z=[h;scratch]` power iteration converges in 47 iterations with last relative change `0.0009388240523025795`, yielding `Lambda_adapters=1.0` and `Lambda_hat_core=1.0420480479402405`. This is explicitly a current-graph measurement: ratified full-width `lanes` and `carrier` are absent, the legacy dimension-reweighted diagnostic is non-governing, topology is incomplete, and no production certificate or alarm is authorized.
- **A7 / Catch #27:** PF-3.4 closes the missing comparison and eligibility definitions. The registered CPU matrix passes as detailed in §3. Deterministic CUDA remains **54 pending cells**, so complete A7 promotion is correctly blocked; absent integrations remain typed non-passes.

## 5. Queue implied by the evidence

Under D-PF-4, the remaining window is an integration window. The evidence supports this dependency order without choosing an unresolved design locally:

1. strategy binds C-S5-1 final recombination and C-S5-2 production K/V representation;
2. continue step 2 only where independent of those choices: the standalone recurrent seam is complete, but insertion into `AblationLM.forward`, selected cache construction, and terminal recombination remain held;
3. integrate the learned rotor carrier, single write, final `bridge_out`, and retention gauge;
4. integrate the per-band callosum and final combiner;
5. bind C-S6-1 sidecar width/accounting and C-S6-2 hard-invocation estimator/init, then implement and integrate the conditional loop sidecar;
6. complete Catch #26/#27 topology incrementally as each module lands, adding its eligibility cells, certificate factor, receipt lines, and Track-B positive/negative controls;
7. only then mint the first full production T14b, A7, and certificate receipts.

In parallel, C1 remains held at Catch #35 until strategy classifies the `E x d` Hadamard router or changes the registered C1 topology; the ten-step width check must not be manufactured around it. Track-B calibrations B2–B9 run as their actual modules land. Track A/C hunters remain commit-time work. The deterministic-CUDA A7 cells, bf16 loop cell, and widest C1 width remain the only named PRE-FLIGHT GPU uses.

### 5.1 Exact outstanding S5/S6 questions carried forward

- **C-S5-1:** ratify the proposed unit-circle final combiner, or supply the exact alternative formula and initialization.
- **C-S5-2:** ratify paired-eigenmode K/V, or explicitly remove K/V from the seven-projection disagreement count and bind a shared-consensus cache.
- **C-S6-1:** ratify lane-native sidecar accounting, or specify the exact full-width lift/project maps and include them in `N_unique`, `N_recurrent`, optimizer ownership, and T14b.
- **C-S6-2:** ratify the hard-forward straight-through estimator and bind `tau`, initial eligible-visit firing rate, and calibration/freeze rule; or choose a different estimator explicitly.

This matrix does not authorize coding through unresolved C-S5-1/2 or C-S6-1/2 semantics, and it does not convert an older-programme module into a WEFT-1 implementation by name similarity. The present C-JAC-1 line supplies the now-bound current-graph measurement, not a production certificate.

## 6. Do-not-claim boundary

- No integrated full-width bicameral recurrent result exists; the new recurrence seam is standalone and absent from `AblationLM.forward`.
- No selected production bicameral static-K/V representation exists.
- No integrated rotor-carrier or fitted retention result exists.
- No integrated per-band callosum or final-combiner result exists.
- No loop-sidecar or occupancy-router result exists.
- No registered matched-dense Hadamard result exists.
- No complete all-backend A7, production-certificate, or production T14b receipt exists; the CPU A7 slice is green and deterministic CUDA remains pending.
- No C1 RMS-width result exists; Catch #35 stopped execution before initialization and training.
- No G-TOK, model-training, checkpoint, evaluation-panel, or sealed-data work occurred in producing this matrix.
