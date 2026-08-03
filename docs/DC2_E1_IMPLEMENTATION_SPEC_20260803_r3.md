# DC2 Implementation Specification — LMB/CFH-PLRR Version One Build (Phase 0 through E1)

Date: 2026-08-03. Strategy lane. Companion to: `LATENT_MICROSTEP_FLOW_DISTILLATION_STRATEGY_HANDOFF_V0.6_20260802.md` (Drive `1NIYcukkR3XFI-x6RQUH_j5bRNXltpK1b`, 88,353 bytes; SHA recorded at fetch) and STRATEGY_TO_CODING_AGENT_V1C_BANK_V05_20260803.md. Purpose: resolve every open constant and integration point in v0.5 into a buildable version-one specification, with defaults marked **[default — v0.5.1 may override]** where the governing document left a choice open. v0.6 governs on any point this document does not address; if they conflict, stop and report.

**Revision r2 (2026-08-03), supersedes Drive `1FjgAoQZkHTilTZNGc9yn-TDnuGbUqHYY`:** incorporates Mark's geometry-contract addendum, strategy-verified (numerically: the metric identity, the alpha = 0 isometry within the retained subspace, the normalization/interpolation non-commutation, the tangent-projection first-order characterization of renormalized dynamics, and the double-epsilon scale discrepancy concentrating in low-eigenvalue directions). Changes: whitener storage contract and eps_abs = 1e-6 (section 3), direction/magnitude factorized refiner update (section 4.2), trust-region norm control and explicit loss forms (section 5), condition-number and floored-fraction logging (sections 3, 7).

**Revision r3 (2026-08-03), supersedes Drive `15PXun3d-Zth6J-paSIH37XNQAEwG3dhI`:** the three open reconciliations resolved with Mark — magnitude nonlinearity is **softplus** on geometric grounds (section 4.2; supersedes r2's sigmoid default), eps_abs = 1e-6 retained and made inert-by-assertion via the new fit-health check (section 3), per-slot innovation normalization ratified.

## 1. Build order (revised per coding-agent review, 2026-08-03 — alpha selection moved after module build)

1. V1d capped-radius diagnostic — its result sets the writeback clamp constants in section 4.4.
2. Stage 0A data job (lattice + teacher-state collection, section 2).
3. Experiment 0A: canonicalizer and probe audit **across the alpha grid** (screening role: eliminate clearly unsuitable alpha values; never final selection).
4. Experiment 0B: geometry/interpolation audit **across the surviving alpha values** (same screening role).
5. Student modules (section 4) with the identity/assertion battery (section 8) green before any loss is attached.
6. **Matched DEV-only alpha pilots** on the built module (identical data, budget, seeds per arm).
7. **Alpha selection** from the pilots, by flow convergence, upper-model quality, and verified acceptance — the criteria require a trained module, which is why selection cannot precede step 5.
8. E1 lock (alpha and geometry contracts locked first), then Phases A–E per v0.6 §15.1 with the loss activation table in section 6.

## 2. Stage 0A data job — full deliverable

Two products from one teacher pass over dev material, document-isolated packing throughout:

**(a) Sparse logit lattice** (as previously specified): union top-K candidate ids, K = 128; log-probs per model (0.5B zero-loop, 7B, 14B, 32B-where-cascaded) in bf16; one tail-mass value per model; entropies; JS agreement normalized by log(teacher count queried); student gap; teachability T(K); scale-coherence cosine; verifier label where available; stratum, position, and bucket metadata; ~1 percent full-logit audit subset.

**(b) Teacher-state collection for the canonicalizer.** Teacher: **Qwen2.5-14B [default — 7B acceptable for a first pipeline shakeout; v0.5.1 names the final choice]**, hidden width d_T = 5120. Selected layers: **{16, 32, 44} of 48 [default: one-third, two-thirds, late]**. Boundaries: the J = 4 future-token positions plus up to 4 span boundaries where trace data exists (N ≤ 8). Per boundary and layer, store the post-block hidden state in bf16 (≈ 245 KB/example raw; the canonical endpoint cache after section 3 is ≈ 2 KB/example — raw states may be discarded after canonicalizer fitting except for the audit subset). Collection contexts: teacher-forced on dev text; privileged future spans visible to the teacher only, per the v0.5 §11.13 hitch.

## 3. Canonicalizer and probe (Experiment 0A) — concrete v1

Pipeline exactly per v0.5 §11.10: per-layer RMSNorm → layer mixture (**fixed uniform 1/3 weights initially [default]**, learned simplex later) → positional resampler to M = 8 slots → predictive factorized projection d_T → r_c = 256 → M × r with r = 128 → fixed truncated whitening.

- RRR fit: whitened cross-covariance SVD, closed form, on ≥ 200k boundary samples.
- Target Y **[default]**: concat of (i) R_logit — the top-K = 128 sparse future teacher log-probs for j = 1..4, compressed by a fixed rank-64 random projection per horizon; (ii) R_state — the Δh to the +J boundary at the middle selected layer, compressed rank-64; (iii) verified answer label where present. Broad multi-target per v0.5 §11.11.
- **Whitening per the v0.6 geometry section, with the accepted contract corrections (2026-08-03)**: fixed population statistics estimated on a calibration split, evaluated on a document-disjoint holdout, frozen thereafter. **One effective-eigenvalue rule, applied once**: `lambda_eff = maximum(lambda_raw, tau * lambda_raw.max(), eps_abs)` with **tau = 1e-4 and eps_abs = 1e-6 [defaults, to be stated in v0.6.1 — 1e-6 supersedes this spec's earlier 1e-8, per the geometry addendum]**; the fitted whitener persists exactly **{mu, U, lambda_raw, lambda_eff, alpha}** and the forward transform is exactly `z_alpha = (z - mu) @ U * lambda_eff.pow(-0.5 * alpha)` — **no second epsilon anywhere in the forward** (fitting and inference share one metric by construction; a forward-pass epsilon would perturb precisely the low-eigenvalue directions the floor exists to control). **Fit-health assertion (r3)**: covariance statistics are accumulated in **fp64** (offline, cheap; buffers stored fp32) and the fit asserts `tau * lambda_raw.max() > eps_abs` — equivalently lambda_max > 1e-2. If the assertion fails, the canonical space's scale is pathological and the fit fails loudly instead of letting the absolute floor silently reshape the metric. Under the assertion, eps_abs **never binds** in a healthy fit: the metric is purely relative-floored, scale-equivariant, with effective condition number capped at 1/tau = 1e4 (max cross-direction amplification 100× at alpha = 1) — which is what makes the 1e-6-versus-1e-8 choice immaterial by construction; 1e-6 is kept as the guard because it sits ~10× above fp32 machine epsilon and bounds worst-case pathological amplification at 1e3. Logged per fit: `kappa_raw` = lambda_raw_max over the smallest retained *positive* raw eigenvalue; `kappa_eff` = lambda_eff_max / lambda_eff_min; both restricted to the retained subspace when a rank cutoff is applied (the addendum's reference fitter omits the cutoff — the prose rule governs); and the **floored fraction** n_floor / r (count of raw eigenvalues below the floor) — a high floored fraction is evidence the proposed canonical rank exceeds the statistically supported rank and feeds the 0A screening verdict. **PCA coordinates, one frozen basis shared by every alpha arm**: all arms use the same frozen PCA basis; alpha = 0 applies no variance equalization and retains only the common orthogonal PCA rotation — so the A35 ablation isolates scaling, not orientation (Huber is per-coordinate and not rotation-invariant; the shared basis removes that confound). The induced original-coordinate metric is M_alpha = U_r * lambda_eff^(-alpha) * U_r^T — Euclidean in the retained subspace at alpha = 0 (M_0 = U_r U_r^T), fractional Mahalanobis at alpha = 0.5, regularized Mahalanobis at alpha = 1. **Partial whitening alpha = 0.5 default, A35 grid {0, 0.5, 1.0}**; 0A/0B screen the grid, final selection per build-order step 7 — never from covariance or probe geometry alone. Per-coordinate flow-gradient audit with and without equalization; probe-fidelity audit; per-workload statistics audit before any mixture decision.
- Mandatory baseline: whitened PCA at matched rank. Pass criterion per v0.5 §22 Exp 0A.
- **Functional probe**: shared low-rank decoder with horizon embeddings, **parameter budget ≤ 1.0M [default]** — deliberately below the refiner's budget (section 4.2) per the anti-compensation rule. Trained on frozen Z_T only, then frozen with hash. Probe targets: the sparse future teacher distributions (top-K + tail).
- Freeze set after 0A: teacher, canonicalizer (mixture weights included), whitening statistics, probe. All hashed; hashes asserted at every later use.

## 4. Student modules — shapes, inits, integration of the V-series results

Total new trainable parameters ≈ 2.5–3.5M (refiner ≈ 0.6M, bridge ≈ 1.2M, heads ≈ 0.5M, initializer/control ≈ 0.3M) — report exact counts in the build receipt.

**4.1 ScratchpadInitializer.** Learned anchors A ∈ R^{8×128}, init N(0, 0.02²); one cross-attention read (queries = anchors, keys/values = RMSNorm(h0) with low-rank projections 896→128); output projection **init N(0, 1e-3²)** so S_0 starts anchor-dominated.

**4.2 SharedResidualFlow (refiner).** Per v0.6 §12.9 **amended by the geometry-contract addendum (section 5)** — the update is factorized into direction and magnitude: input concat [RMSNorm(z), context_proj(h̄0), step_embedding], hidden = 4 × 128 = 512, SiLU. Raw innovation u_k from the vector field (**final layer N(0, 1e-3²), bias zero**); **direction ū_k = per-slot RMSNorm(u_k) with learned gain, init 1 [ratified by Mark, 2026-08-03]**; **scalar magnitude per (example, loop): m_k = softplus(head(features) pooled over slots), head weights zero, bias −4.0** (softplus(−4) ≈ 0.0181 ≈ sigmoid(−4) ≈ 0.0180 — init value and init gradient are unchanged from r2). **Softplus [resolved with Mark, 2026-08-03; supersedes r2's sigmoid default]**: the magnitude is a step *length* in the M_alpha metric space, and lengths are unbounded — at the β = 1 rung the demanded per-loop step equals the full remaining displacement, whose whitened-population RMS is O(√2) (measured on correlated toy populations: at student/teacher correlation 0.5, ~48 percent of β = 1 steps exceed one canonical RMS unit — sigmoid's ceiling). A sigmoid can only represent those steps by leaking magnitude into the direction gain, which entangles exactly what the factorization separates; softplus is linear for large arguments (no saturation), and the **trust-region penalty is the single magnitude authority** — geometry says lengths are unbounded, safety says excess pays rent, one mechanism each. **State update `z <- z + m_k * ū_k` with no outer normalization** — norm applies to input features and the innovation only, never the persistent state. Initialization note: with innovation normalization, the update scale at init is set by the magnitude head alone — the 1e-3 final-layer init now fixes only the initial *direction* — so the double-zero rule is satisfied with exactly one near-zero factor (the magnitude), and gradients flow through the normalized direction from step one; the realized init update ratio r_0 ≈ 0.018 / RMS(z_0) is a required line in the first build receipt. Full BPTT through K ≤ 4; loop cap asserted.

**4.3 AnchoredBridge.** Cross-attention Q = RMSNorm(h0), K/V = RMSNorm(S_k), low-rank 896↔128 projections; **P_out init N(0, 1e-3²) — never zero** (carry-forward; the prose gap in v0.5 §4.4 is closed here); ρ_k = 0.95, g_k = σ(−4) ≈ 0.018, both learnable scalars per loop.

**4.4 Writeback normalization with the measured safety envelope built in.** The update-normalization rule integrates the V1c/RMS results directly:

```
rms_ref  = min(RMS(h0_pos), 0.550893)        # p99 cap — audit rule, per position
delta_h  = P_out @ CrossAttn(...)
delta_h  = delta_h / (RMS(delta_h) + eps) * stopgrad(rms_ref)
h_k      = h0 + rho*(h_{k-1} - h0) + g_k * delta_h
```

The cap value `0.550893` and the tube constant c = 0.15 are **provisional pending V1d** — the launcher reads both from a single constants file whose hash appears in every receipt, so V1d's outcome updates one file, not scattered literals. **Position-zero handling [default]: the writeback gate is forced closed at sequence position 0** (the audit's highest-risk bucket); position 0 still contributes context through attention. Logged as a masked case; revisit after E1 telemetry.

**4.5 ResidualDraftHead.** Tied vocabulary embedding; per-horizon low-rank adapters W_{Δ,j}: 128 → rank 64 → 896, J = 4; cumulative logits ℓ_{k,j} = ℓ_{k−1,j} + b_{k,j}·Δℓ_{k,j}; scalar write gates **b init σ(−3.5) ≈ 0.03 [default]**, one scalar per (example, loop, horizon) computed from the control state — no per-token or per-channel gating in v1.

**4.6 Control state.** c_k ∈ R^{32}, unnormalized, GRU-style update from [Pool(S_k), innovation norm, student entropy, top-2 margin, position bucket]. **Gate features in v1: position bucket (0, 1–3, early, mid, late), student entropy, top-2 margin, innovation norm, c_k. The oracle first-order distance is never an inference input** — training-time curriculum weight and evaluation stratification only.

## 5. Supervision surfaces — what trains what (binding)

- **z-space (flow hitch), geometry-preserving contract (2026-08-03 correction):** endpoints, interpolation targets, and flow losses live in the **unnormalized partially-whitened coordinates** — targets are Z̃_{k+1} = (1−β_k)·sg[Z_{S,k}] + β_k·sg[Z_T] with **no per-example renormalization** (renormalizing after interpolation bends the straight path onto a sphere and voids the Mahalanobis interpretation the whitening establishes). RMSNorm is applied only to recurrent-module **inputs and innovations**, never to the persistent canonical state after residual updates: the refiner update is `z <- z + m_k * ū_k` per section 4.2, with `norm(z)` used solely in feature construction (renormalized persistent dynamics are *projected* dynamics — first order, `(I − zz^T/‖z‖²)·Δz`, the radial component removed — a different geometry that cannot be described as rectified flow under M_alpha; if spherical dynamics are ever preferred, that is a separate contract, never mixed with this one). **Losses, both computed directly in canonical coordinates with no per-example normalization on either side**: L_state,k = Huber(z_{k+1}, Z̃_{k+1}) and L_Δ,k = Huber(z_{k+1} − z_k, Z̃_{k+1} − sg[Z_{S,k}]). **State-scale control is by penalty and telemetry, never projection**: the per-loop update ratio r_k = RMS(Δz_k)/(RMS(z_k)+eps) carries a trust-region penalty L_trust = Σ_k max(0, r_k − r_max)² with **r_max = 0.5, weight 0.01, active from Phase A [default]** (zero inside the region — a pure rail; watch item: r_max = 0.5 can rent-charge geometry-demanded β = 1 steps for distant pairs — the penalty is soft by design, and the Phase-A r_k distribution decides whether r_max moves at E1 lock); the endpoint-relative norm penalty Σ_k [log(RMS(z_k)/RMS(z_T))]² is **telemetry-only in v1**, promoted to a weighted loss only on drift evidence (it pulls state scale toward the endpoint's — a modeling choice, not a rail); radial drift ΔR_k = RMS(z_{k+1}) − RMS(z_k) is logged per loop and stays diagnostically meaningful precisely because the state is never renormalized. t-grid [0, 0.5, 0.8, 1.0] → β = [0.5, 0.6, 1.0] [default]; Huber + 0.1·cosine; functional-probe KL on student states. No V-series constraint binds here (new parameters, no pretrained manifold).
- **h-space (bridge writeback):** governed by section 4.4's clamp; preservation per the two-tier margin; results read against the V1d envelope.
- **Logit space (drafter):** cumulative KL primary, geometric bridge target and progress loss per v0.5 §5, LK→EAL staging per §8. Margin-unbounded; judged on acceptance currency.

Every E-stage report attributes results to their surface.

## 6. Loss activation table (per v0.5 phases; weights = v0.5 §12.6/§15 values)

| Phase | Active losses (weights) | Loops | Horizons |
|---|---|---|---|
| A | final CE (1.0), preserve KL (0.1), flow (1.0), func (0.5), cum KL (1.0), trust (0.01) | 1 | TF, J=4 |
| B | + local CE (0.5), LK (per §8 schedule); rel (0.01) | 1–2 | TF |
| C | + bridge target (β table), progress loss, consensus no-op (0.01) | 1–4 | mixed rollout |
| D | + EAL primary, KL anchor lowered | 1–4 | student rollout |
| E | + span/state targets; span removal curriculum | 1–4 | student rollout |

No loss activates ahead of its phase; the deltaDiag cosine is logged from Phase A but never weighted before an ablation justifies it.

## 7. Precision, clipping, telemetry

Full fp32 on every path carrying recurrent gradients (refiner, bridge, flow losses; RG-11 binds). Per-module clipping: refiner 1.0, bridge 0.5, heads 1.0. Gradient atlas from step zero of every run: per-(module × loss × loop) norms, conflict cosines, JVP gains (target distribution near 1), clip fractions, gate open-rates, scratch effective rank, realized writeback ratio RMS(g·Δh)/RMS(h0) (target band 0.01–0.05), tube-radius equivalent, per-loop canonical update ratio r_k and radial drift ΔR_k (section 5), and the auxiliary-versus-full-tail audit every 8–16 batches (A20). Canonicalizer-fit receipts carry kappa_raw, kappa_eff, and the floored fraction per section 3; the A35 arms share {mu, U_r, retained rank, lambda_raw, lambda_eff, recurrent architecture, optimization schedule} with **only** lambda_eff^(−alpha/2) varying — asserted, not assumed.

## 8. Assertion battery before first loss

Zero-loop bit-identity to pretrained Qwen (fp32 exact); K ≤ 4 asserted; teacher/canonicalizer/probe under no_grad with hash checks; no gradient through sampled tokens or target construction; document isolation on every packed batch; constants-file hash in every receipt; frozen-set parameter hashes before/after every run.

## 9. Explicitly OUT of version one

VAE/CVAE and attention-pooling canonicalizers (0A ablations only, on evidence); deterministic AE; DeltaNet matrix state; adaptive halting and learned routing (Experiment 8, pinned); FFT/oscillator components; per-token/per-channel write gates; upper-layer unfreezing or LoRA (E4 — trigger-held on V1d/E1 evidence); persistent cross-token scratchpad; any 32B teacher use outside the cascade rule; any frozen-slice contact.

## 10. Open items this spec does not decide

v0.5.1's target-assignment table (bucket × loop → p*); the final canonicalizer teacher choice if not 14B; E1 gate thresholds (set from V1d + 0A/0B at lock); AngelSpec adopt/adapt/reimplement (assessment pending). Defaults above hold until those land.
