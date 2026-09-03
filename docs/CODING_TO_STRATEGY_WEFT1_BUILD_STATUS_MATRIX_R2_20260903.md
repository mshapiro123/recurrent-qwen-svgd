# CODING → STRATEGY — WEFT-1 ratified-module build-status matrix

**Date:** 2026-09-03
**Status:** BUILD-AXIS RECEIPT · no training or sealed-data contact
**Snapshot base:** `9137ea7e2dc471e0af696ae0fc4de24eaeae97b1` plus the Catch-37 remediation described here

**PF-2 implementation commit:** `5ebbefea36c13a8f86c5e78ed074efc6c91db12f`

**Architecture authority:** full build handoff, 61,329 bytes, SHA-256 `498f34b5…eb02`, Drive `1XaE81mfqTOYEYGFMa-ZJwpLW-KQtMMC_`
**Ratification authority:** `STRATEGY_RATIFICATION_RECORD_20260826.md`, 13,908 bytes, SHA-256 `c5df74297594e75697ffb71d8d05d75efcf94f7857d55ddd357043200efb6d3a`, Drive `1Wb_FfEb-Sl-TgL23hcFOy58QcSurfaF3`
**Status authorities:** PF-2, 13,097 bytes, SHA-256 `be11390c28ae36210a1571f7c6d358ee54e977d239f5344f7e6402212448eb05`; §8.1 amendment, 3,403 bytes, SHA-256 `dd79aaa6fd9bab15bf02aaef28f99f47c745d63eed6db2c9b929b4bb1cfbb418`; PF-3, 14,632 bytes, SHA-256 `7f0081e504366ce98f8bf183b7e14c0bed47647aa381196a9d5e9540b5334cef`; D-PF-4 ratification, 2,336 bytes, SHA-256 `b4e13508c4135837dd197f19f2711026a3df429df7a463386ed5ac1aa94dba31`; architecture reconciliation, 20,695 bytes, SHA-256 `0d81e9ab63d21720fecfbfcb629aaa5eeae6693eabbd9682b82adc7e3792ea8e`; math check, 16,587 bytes, SHA-256 `509cac8c7f5f82a6a70d0bcc8494b02967d3f545e4e875e9bbfcdc2b93dedcff`; D-MC-1, 2,868 bytes, SHA-256 `9c5822daef5dbb0609bc3e46019cc4b1e332991c30e8a42c1b4432800a747ab1`

## 0. Outcome

The current `AblationLM` is a causally tested recurrent bring-up graph, not the complete WEFT-1 production graph. It contains the dense substrate, shared recurrence, a single-stream static-K/V path, position-aligned scratch, a narrow two-lane carrier, token-ID engrams, and hidden-stream trajectory diagnostics. FRONT-WHT and H0-REENTRY remain reachable legacy controls but are structural-OFF in the production design; the narrow mixer retires when the callosum lands; LTM-RO is retained as an OFF-by-default post-loop arm. A standalone bicameral recurrence seam executes a seven-paired candidate block for registered `K`, but it is not called by `AblationLM.forward` and must be recut to the ratified five-paired target with shared-consensus K/V. The full-width bicameral path, learned rotor/write, integrated callosum and combiner, sidecar, bridge-out, and objective/halting stack remain absent or standalone.

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
| Shared recurrent core, inference-controllable `K`, `alpha_T=c/T`, requested/executed accounting | `integrated` | `AblationLM._run_recurrent_visit`; K override and accounting tests | `85bb8eae`, `5c835c3c`, `83157bfb` | Current recurrent core is single-stream dense, not the full bicameral recurrent core. |
| Fixed-anchor static K/V, live-state queries: current single-stream form | `integrated` | `models/ablation_lm/model.py`; T15 equivalence, projection-count, gradient and T14b regressions | `51c16443` | Full-sequence/reference path only; production autoregressive serving cache is not claimed. |
| Production bicameral static K/V representation | `absent` | S-3 binds one shared-consensus cache from `h0`, reused across visits; the existing standalone seam remains cache-policy agnostic | `0c95a7cc`, `e78692dd`; S5/S6 A1 | Selection is closed, but construction and the integrated recurrent call site land in build Step 2. |
| Fork B-prime midpoint K/V refresh | `integrated` | explicit refresh flag plus requested/executed refresh receipt tests | `51c16443` | Challenger on the current single-stream graph, not the ratified paired production graph. |
| Position-aligned causal scratch lanes | `integrated` | `PositionAlignedScratch`; packed/padded causality and gradient-liveness tests | `85bb8eae` | Two narrow lanes accompany one full-width stream; they are not two full-width hemispheres. |
| Narrow two-lane Birkhoff carrier | `integrated` | `TwoLaneBirkhoffMixer`; retention and mean/disagreement tests | `85bb8eae`, `1c033bae` | Not the ratified per-band full-width callosum. |
| `bridge_in`, in-loop scratch readout, and anchored recurrent re-entry | `integrated bring-up` | `PositionAlignedScratch`, `AnchoredReentryBridge`; liveness and causality tests | `85bb8eae`; reconciliation R-2 | `bridge_in` and lane initialization survive. H0-REENTRY is a legacy arm, raw-default-OFF and production-retired in Step 2; current readout is not `bridge_out`. |
| Final post-loop `bridge_out` into coda | `absent` | no post-loop bicameral-to-coda bridge call site | — | Current in-loop scratch readout does not satisfy this surface. |
| Stored bicameral modes (`mu`, low-rank `delta`) via `SwapLinear` | `standalone primitive` | `models/ablation_lm/bicameral.py`; equivalence, symmetry-break and optimizer-cohesion tests | `51c16443` | Not used by `AblationLM.forward`. |
| Full-width paired bicameral Transformer block | `standalone primitive` | `models/ablation_lm/bicameral_core.py`; dense equivalence, causality and both-factor gradient tests; `models/ablation_lm/bicameral_recurrent.py` executes registered `K` | `0c95a7cc`, `e78692dd`; `0efb4cdf` seam | The executable candidate still pairs seven projections. Step 2 must pair only Q/O/gate/up/down, share consensus K/V, integrate the seam, and add the S-2 combiner. |
| Per-band Birkhoff corpus callosum | `standalone primitive` | `models/ablation_lm/callosum.py`; T16 and closed-form delta-mode tests | `51c16443`, `f589e787` | Callosum-only evidence excludes intervening core dynamics and production integration. |
| Final per-band bicameral combiner | `absent` | S-2 unit-circle rule is ratified; no model call site | S5/S6 ruling; reconciliation R-7 | Lands with the bicameral path in Step 2 so dense equivalence can be tested immediately. |
| Euclidean `Cl(2,0)` rotor operator | `standalone primitive` | `models/ablation_lm/geometry.py`; theta-zero bit identity and K=1…8 norm-isometry tests | `1c033bae`; PF-1 worktree test | Tensor primitive only; no learned production rotor carrier. |
| Learned carrier rotor bank (`J=8`) | `absent` | no model parameter or call site | — | Standalone `Cl20Rotor` is not this integrated module. |
| Single gated rank-8 lane-to-carrier write per hemisphere | `absent` | accounting placeholders only | `83157bfb` | No executable write path exists. |
| Upfront modified Hadamard expert challenger | `integrated legacy arm` | `ModifiedHadamardExpertBank`; WHT, router, causality and liveness tests | `85bb8eae`, `51c16443`; reconciliation R-1 | FRONT-WHT is not on the production path and is structural-OFF there. The target Hadamard challenger belongs in the core FFN slot as a post-production arm. |
| T12 matched-dense Hadamard control | `absent` | no parameter/FLOP-matched dense control implementation and receipt | — | Existing dense-equivalence tests do not instantiate the registered control. |
| Fixed-measurement occupancy router and sparse core expert bank | `absent` | calibration-stability diagnostics exist, but no fixed router/model path | `1c033bae` (diagnostic only) | Current learned dense router must not be described as the occupancy router. |
| Per-lane conditional loop sidecar, shared bank/PQ and hard invocation | `absent` | accounting/schema placeholders only | `83157bfb`, `f589e787`; S-4-prime/S-5 | Semantics are bound: lane-native private invocation/write, shared bank and PQ, rank-4 default/rank-16 arm, freeze at the occupancy-router stability gate. Legacy `models/sidecar_v2.py` is not this module. |
| Hidden-stream second-order trajectory jet prototype (`v`, `a`, curvature/Gram diagnostics) | `integrated` | `trajectory_jet_metrics`; initial plus visit hidden states retained | `1c033bae` | Uses the full hidden stream and is diagnostic only. |
| Ratified pooled-lane jet over recurrent lanes | `absent` | no pooled-lane state pipeline | — | Current hidden-stream prototype is not this object. |
| Frozen random hidden-state plane probes | `integrated` | `plane_probe_features` and model diagnostics | `1c033bae` | Hidden-stream diagnostic only; fixed projection-basis identity is not yet recorded in the composition receipt. |
| Learned lane-plane probes | `absent` | no learned probe parameters or lane-state probe path | — | Must not be inferred from the frozen hidden-state probes. |
| Rotor-QK attention arm | `absent` | no module, config field, or call site | — | `J_att=0` is effectively the only current topology. |
| `M_lex` causal engram, token-ID n-gram bring-up | `integrated` | `CausalTokenEngram`; packing, padding, provenance, exact memory-space-gate and liveness tests | `85bb8eae`; Catch #37 remediation | Query maps `d→64`; the normalized retrieved row is the key; the gate dot is in memory space and divided by `sqrt(64)`. Token-ID hashing remains the only address form. |
| `M_lex` byte-span polynomial hashing / tokenizer-agnostic address | `absent` | no byte-span address path | — | Explicitly deferred ablation in the handoff. |
| Read-only long-term memory with leave-one-record-out | `integrated registered arm` | `ReadOnlyLatentMemory`; frozen-store and provenance-exclusion tests | `85bb8eae`; reconciliation R-3 | LTM-RO is structural-OFF by default and permitted only post-loop/pre-coda; it is not a production-default or in-loop write. |
| Adaptive halting head | `absent` | no parameter, config switch, or call site | — | `K` is controllable but not learned per example. |
| Staged-state objective `L_stage` and trajectory curriculum | `spec bound; runtime absent` | `docs/WEFT1_STEP6_OBJECTIVE_AND_SAMPLED_DECODE_SPEC_20260903.md` | D-MC-1 | Final visit plus one uniform earlier visit is bound; O-9 sampler, losses and step logits land in Step 6. The wider curriculum remains separately governed. |
| Other registered auxiliaries: `L_div`, `L_ret`, `L_halt`, `L_inv`, `L_conv`, `L_plane` | `absent` | no complete model-loss implementation | — | Diagnostics with similar names do not constitute these objectives. |
| Z-loss with shared valid-token mask | `integrated` | `_language_model_loss`; exact zero-coefficient structural path tests | `1c033bae` | Auxiliary loss only; default is structural zero. |
| Composition accounting and requested/executed visit receipt | `integrated` | `models/ablation_lm/accounting.py` | `83157bfb`; D-MC-1 remediation | Now emits `coda_decodes_per_step` and per-forward `lstage_sampled_visit`; sidecar/callosal fields and fixed projection-basis SHA remain pending. |
| O-9 per-module RNG registry and paired recurrent-visit alignment | `integrated` | `models/ablation_lm/rng.py`; module-stream isolation and seed-receipt tests | `5ecf58fb` | Applies to materialized modules; absent modules have no live draw stream yet. |
| Fitted carrier-retention gauge `r` and `r>=0.9` tripwire | `absent` | no fitted production carrier or integrated retention receipt | — | Narrow carrier tests are not the registered production gauge. |
| T14b exact-autograd receipt machinery | `standalone harness` | `observatory.py`; exact evidence and fail-closed guards | `dc63ee4e` | Not reachable from `AblationLM.forward`; no complete production receipt for absent modules. |
| Observatory event schema, including RESP-LEAK binding | `standalone harness` | `observatory_events.py`; schema tests | `f589e787` | External receipt machinery, not an integrated model module. |

## 3. Exact causality and OBS-INV boundary

The current static/dense bring-up graph has independent direct future-gradient tests at `K=(1,2,4,8)`. The maximal optional-module construction is tested at `K=8` with visit-by-visit logits; it is not four independent all-optionals-active constructions at K=1, 2, 4 and 8. The standalone bicameral block has a separate one-block packed/padded causal test. These are causality results, not OBS-INV or production-integration certificates.

PF-3.4 closes Catch #27's missing definition and the worktree materializes its registered comparison graph. The matrix has **160 module cells**: 54 CPU passes, 54 deterministic-CUDA pending cells, 48 absent cells, and four structurally ineligible cells. The separate dense K=1 4/2/4 CPU anchors pass `torch.equal` on both logits and loss in fp32 and bf16. Every eligible integrated CPU module cell passes OFF idempotence bit-identically and has a non-trivial ON positive control; re-entry at K=1 is typed ineligible, not averaged into a pass. The 54 deterministic-CUDA cells remain pending and block complete promotion. In particular, no row above is promoted from a standalone certificate or causality result, and no complete A7 receipt is claimed.

## 4. PF-3 / D-PF-4 PRE-FLIGHT status

- **Jacobian panel:** the registered main panel is now `n=520`. The corrected power calculation meets both literal frontiers (`SE=0.05092447485214237 <= 0.051` and `SE=0.03578829611711741 <= 0.036`), and all four previously planted B1 calibration phases remain green.
- **C1 / Catch #33:** PF-3.1 binds the base shape, provisional numeric base constants, per-tensor init/LR classes, tied-readout multiplier, constant decoupled weight decay, residual rules, and fail-closed classification; Catch #33 is closed. After the authorized Catch-37 removal of the learned engram key projection, the inventory classifies **145 of 146** unique trainable tensors at each of `d=128,256,512`. It still correctly stops on the sole legacy FRONT-WHT router tensor; reconciliation R-1 closes that tensor for production by retiring the front module, while the current bring-up C1 receipt remains typed incomplete rather than being rewritten around it.
- **C2 / Catch #34:** The Catch-37 topology change invalidated the old current-graph state hash and numerics, so C2 was replayed and reminted under its existing PF-3 measurement definition. The initial-state SHA is now `699fd7d782b7d8bd652b8ddfe552a1fb89e61ff02b901d737d54f19d1e7e6a73`, with 484,763 trainable elements. At terminal K=8 the vector-relative L2 values are hidden `0.002343670477777934`, scratch lanes `0.016916588480255675`, logits `0.00366168723457278`, full-gradient vector `0.007065333907931817`, and worst module tensor `0.023292915877290394` (`core_blocks.0.attention.key_norm.weight`). All three visit-1 re-entry zero-reference cells remain structurally ineligible and exactly zero. The terminal gate remains **PASS**; no production or GPU result is implied. Canonical compact receipt SHA-256: `5c5441a09f53651e26410ce49bd660f2f11de781c3f2bea3ececa07e2911438b`.
- **C3/C6:** PF-2's typed complete-gate posture stands; no bounded CPU subset is promoted to the unavailable CUDA, dropout, STOCH-K, or absent-module cells.
- **C7:** stage 1 emits the four present G-TOK families through the production matrix, selection, confirmation-budget and checkpoint-accounting builders from deterministic synthetic source receipts. The byte-matched base path is independently bound at `n=400`; the fresh confirmation path is joined to its selected budget row at `n=399`, with separate byte totals and first-crossing indices. Receipt SHA-256 is `04b9c1515a3902c2963eb1e13e5bfa42ede144549f88a44366953b76a422abd6`. Stage 2 remains open on absent sidecar/topology cells; the complete C7 gate remains incomplete and non-forgeable.
- **C-JAC-1 / Catch #26:** PF-3.3 closes the missing joint-state-metric definition and authorizes a current-graph measurement. Replayed after Catch #37, terminal-visit-8 plain Euclidean `z=[h;scratch]` power iteration converges in 47 iterations with last relative change `0.0009611520575406325`, yielding `Lambda_adapters=1.0` and `Lambda_hat_core=1.041439431237759`. Canonical compact receipt SHA-256: `319ad9ef3414b6856a85bac10b38adb5530ed29c04bbee8c671bc78346ee3e3d`. This remains a current-graph estimate: full-width bicameral lanes and carrier are absent, and no production certificate or alarm is authorized.
- **A7 / Catch #27:** PF-3.4 closes the missing comparison and eligibility definitions. The registered CPU matrix passes as detailed in §3. Deterministic CUDA remains **54 pending cells**, so complete A7 promotion is correctly blocked; absent integrations remain typed non-passes.

## 5. Queue implied by the evidence

Under D-PF-4 and the reconciliation, the remaining window is an integration window with no open design blocker on Steps 2–5:

1. integrate the five-paired bicameral path, shared-consensus K/V, S-2 combiner, production legacy-arm registry, A7/eligibility/certificate cells, and exact `visit_schedule` receipt;
2. integrate the learned J=8 rotor carrier, one gated rank-8 write per hemisphere, final `bridge_out`, and retention gauge;
3. integrate the per-band callosum once per visit and retire the narrow two-lane mixer;
4. integrate the per-lane sidecar, shared bank/PQ, per-lane jets, learned probes, calibration/freeze gate, and invocation-agreement receipt;
5. implement Step 6: D-MC-1 sampled `L_stage`, `L_div` and registered auxiliaries, halting, K curriculum, and the STOCH-K O-9 sampler;
6. complete the certificate topology and full A7 matrix, then mint the first production T14b/OBS-INV receipts.

The Hadamard core-expert arm, LTM-RO, FRONT-WHT, H0-REENTRY, KV-PAIR and the other named challengers remain in the experiment queue rather than the production critical path. Track-B calibrations run as their actual modules land. Deterministic-CUDA A7 cells and other registered PRE-FLIGHT GPU cells retain their prior authority boundaries.

### 5.1 Closed architecture choices now governing the queue

- paired set: Q, O, gate, up and down; K/V are shared consensus;
- combiner: S-2 unit-circle per-band rule, implemented with Step 2;
- visit nesting: attention, FFN, lane update, sidecar per block; rotor/write then callosum once per visit;
- sidecar: per-lane invocation and private writes with a shared expert bank/PQ;
- L-stage: final visit plus one uniformly sampled earlier visit through the same coda.

The present C-JAC-1 line is a reminted current-graph estimate, not a production certificate.

## 6. Do-not-claim boundary

- No integrated full-width bicameral recurrent result exists; the new recurrence seam is standalone and absent from `AblationLM.forward`.
- The production K/V representation is selected, but its integrated cache does not exist yet.
- No integrated rotor-carrier or fitted retention result exists.
- No integrated per-band callosum or final-combiner result exists.
- No loop-sidecar or occupancy-router result exists.
- No registered matched-dense Hadamard result exists.
- No complete all-backend A7, production-certificate, or production T14b receipt exists; the CPU A7 slice is green and deterministic CUDA remains pending.
- No C1 RMS-width result exists on the bring-up graph; its sole unclassified FRONT-WHT router is a production-retired legacy tensor.
- No G-TOK, model-training, checkpoint, evaluation-panel, or sealed-data work occurred in producing this matrix.
