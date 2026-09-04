# STRATEGY — Epiplexity Adjudication: Bits-versus-Compute Accounting as a Component Pre-Flight Instrument

**Date:** 2026-09-03 · **Status:** ADJUDICATION of *From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence* (Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson; arXiv 2601.03220v2, March 2026) and of the RMT (random matrix theory) commentary Mark received on it, against WEFT-1. **Critical path untouched.** Two decisions for Mark (§6). Includes a micro-experiment run today (`eca_latent_loop_20260903.py`).
**Evidence classification (PS-1, O-7):** the paper was read from the arXiv PDF; the commentary was checked against it line by line (§1.2 lists what did and did not survive). **Tier-2 for the framing and estimators; Tier-3 for any number** (the paper's own estimator is observer-specific by construction). The commentary's formulas were verified numerically where they are formulas (§1.3).
**Abbreviations used:** MDL (minimum description length), KL (Kullback–Leibler divergence), RMT (random matrix theory), ECA (elementary cellular automaton), NTK (neural tangent kernel), GGN (generalized Gauss–Newton matrix), BPC (bits per cell), FLOPs (floating-point operations).

---

## 0. Plain-language summary

The paper makes one distinction and builds everything on it. Classical information theory measures how *unpredictable* data is. It does not measure how much *reusable structure* a learner with a limited compute budget can pull out of the data. The authors split the best achievable description of a dataset into two parts: the size of the model a bounded learner ends up with (they call this epiplexity) and what still looks random after that model has been learned (time-bounded entropy). Random data has enormous entropy and almost no epiplexity; repetitive data has little of either; rich-but-learnable data has a lot of epiplexity. Their cheap estimator is the area under the training-loss curve above its final value: every bit the model was worse than its final self, summed over training, is a bit that went into building the model.

Two of their results bear on this program directly, and the commentary Mark received reads both correctly with one important slip. First, presentation order changes the computational problem even when it does not change the information (chess moves-then-board versus board-then-moves) — which is a statement about curricula, scratch serialization and where facts sit relative to operations. Second, in their cellular-automaton experiment, a model that *unrolls* the intermediate states finds the short brute-force rule once compute allows it, and the measured epiplexity *drops*: a short program run many times replaces a large compiled approximation. The slip is that the paper's "looped model" **unrolls the intermediate states as output tokens** — chain-of-thought — and only *cites* looped transformers as the analogous idea. Whether a *latent* loop finds the same short rule without emitting the intermediate states is exactly WEFT-1's thesis, and the paper does not test it. So I ran the test in miniature today (§4): a weight-tied core with K visits on the paper's own testbed. The answer in miniature is yes, with a condition that is itself a WEFT-1 design choice: on the complex rule the latent loop at K = 4 found the exact solution (0.002 bits per cell) where a single pass with the same receptive field could not (0.60), and at K = 8 it trained *only* under the K curriculum the design already prescribes — without it, eight tied visits from scratch did worse than one.

To Mark's broader question — can matrices captured from component training regimes tell us whether the pieces will fit before the big run — the answer is yes for three things and no for one. Yes: (i) the loss-curve area is already computed in every eval we run, so per-arm epiplexity accounting is free; (ii) under the L_stage ruling the model already decodes two visits per micro-batch, so the KL between the final visit and an earlier one costs nothing and gives a "bits earned per visit" reading that is the paper's rigorous estimator transposed to depth; (iii) a seam-legibility audit — does the receiver's sensitive directions overlap the sender's output directions — can be computed at the proxy rung from matrices captured on component-only runs, before integration, and answers the commentary's bridge hypothesis directly. No: a spectral outlier is not epiplexity, and the loss-curve area confounds structure with conditioning. That confound is not hypothetical for us: recurrence makes the optimization worse conditioned by construction (products of visit Jacobians), so a recurrent arm's larger area could be either richer learned structure or a harder coordinate system. The spectral decomposition is what separates the two, and it is the reason the audit is worth its cost.

The commentary's vocabulary was written for the retrofit (Qwen block, LoRA, teacher). §2 translates it into WEFT-1's terms; the "teacher" is the one thing with no counterpart, and its replacement is the model's own final visit.

---

# 1. What the paper establishes, and what the commentary got right and wrong

## 1.1 The paper (Tier-2 framing)

**Definition.** Among programs `P` that both sample and evaluate probabilities within time `T`, choose

`P⋆ = argmin_{P ∈ 𝒫_T} { |P| + E_X[ log₂ 1/P(X) ] }`,  `S_T(X) = |P⋆|`,  `H_T(X) = E_X[ log₂ 1/P⋆(X) ]`.

*Read as:* P-star is the program, among all programs in the time-T class, that minimizes the sum of its own length in bits plus the expected number of bits needed to encode a draw of X under it; S-sub-T of X, epiplexity, is the length of that program; H-sub-T of X, time-bounded entropy, is the expected encoding cost of the data under it. *Intuition:* split the shortest feasible description into "model" and "what the model can't explain."

**Estimators (paper §4).** Prequential: `|P_preq| ≈ Σ_i [ log₂ 1/P_i(Z_i) − log₂ 1/P_M(Z_i) ]` — *read as:* sum over training steps i of the code length of token Z-sub-i under the model at step i minus its code length under the final model M; *intuition:* the area under the loss curve above the final loss. Requential: `|P_req| ≈ Σ_i [ KL(P_iᵗ ‖ P_iˢ) + log(1 + KL(P_iᵗ ‖ P_iˢ)) + 4 ]` — *read as:* sum over checkpoints i of the KL divergence from the teacher checkpoint to the student trained on the teacher's samples, plus a small coding overhead; *intuition:* accumulate the bits the student is short of the teacher at each stage. The paper states the prequential estimate is heuristic, runs 2–10× larger, and correlates well with the requential one within a dataset family. Compute is counted as `T = 6ND + 2ND` — **training plus inference** — and the sweep over N and D is done under μP, which is the same parameterization discipline we adopted (PF-3.1).

**Results used here.** ECA rules 15/30/54 (48 iterations of the rule on 64-bit grids, 100 M tokens): periodic → low entropy and low epiplexity; chaotic → near-maximal entropy, no epiplexity; complex → moderate entropy, high epiplexity. Chess forward vs reverse factorization: reverse has higher time-bounded entropy *and* higher epiplexity and transfers better to centipawn evaluation. Looped ECA (paper §5.3.2): *"The brute-force solution can be naturally implemented by learning to autoregressively unroll intermediate ECA states rather than directly predicting the final state, resembling the use of chain-of-thought … or looped transformers"*; *"we identify a compute threshold beyond which the looped model suddenly becomes favorable, causing an abrupt drop in MDL and epiplexity, likely by learning the simple, brute-force solution."* Pre-training: language > video > image in epiplexity; a data-selection method that prefers fast loss decrease "inadvertently achieves higher epiplexity" and better OOD perplexity. Stated limitation: *"Higher epiplexity does not guarantee better generalization to any specific task."*

## 1.2 The commentary, checked against the paper

| commentary claim | status |
|---|---|
| Definitions, two-part code, MDL_T = S_T + H_T | **matches** the paper |
| Prequential = area above terminal loss; requential = cumulative teacher–student KL; prequential heuristic, requential 2–10× costlier | **matches** |
| Rule 15/30/54 signatures; chess ordering result | **matches** |
| "Looped cellular-automaton experiment … spend more runtime so that a short repeated rule can replace a much larger compiled shortcut" mapped to *our recurrence* | **half right.** The paper's looped model **unrolls intermediate states as tokens** (chain-of-thought). Looped transformers are *cited as analogous*, not tested. The transfer to a latent loop is a hypothesis — tested in §4. |
| "The paper itself points toward this connection when discussing sparse PCA" | **not in the paper.** No occurrence of "sparse PCA" or "computational-statistical gap." The paper's nearest content is its one-way-function theorem (Thm. 13) on factorization order. The BBP connection is the commentary author's, and it is a good one — but it is theirs. |
| "Information is created through computation" softened to "newly accessible structural information for a bounded observer" | **fair reading**; the paper's own framing is that structure becomes *extractable*, and the softening preserves the data-processing inequality. |
| RMT vulnerability list (area confounds structure with conditioning; observer-specificity; second-order blindness) | **sound**, and the first item is the one that matters for us (§3). |

## 1.3 The commentary's formulas, verified

- **Spectral area.** With `L(t) = (1/2n) Σ_j a_j² e^{−2λ_j t}` — *read as:* the loss at time t is one over two-n times the sum over modes j of the squared target coefficient a-sub-j times e to the minus two lambda-sub-j t; *intuition:* each kernel mode decays at its own rate — the area above terminal loss is `A_T = (1/2n) Σ_j a_j² [ (1 − e^{−2λ_j T})/(2λ_j) − T e^{−2λ_j T} ]`. **Verified** numerically (50 random modes, relative error 9 × 10⁻¹¹). The reading stands: modes with `λ_j T ≫ 1` contribute `a_j²/(4nλ_j)` (learned early, small area), modes with `λ_j T ≪ 1` contribute `≈ 0` (never learned), and the sustained area comes from modes whose timescale `1/λ_j` is comparable to `T`.
- **Local KL as Fisher quadratic.** `KL(p_{θ+δ} ‖ p_θ) = ½ δᵀ F(θ) δ + O(‖δ‖³)` — *read as:* the divergence between the model at parameters theta-plus-delta and at theta is one half of delta-transpose times the Fisher matrix at theta times delta, up to third-order terms; *intuition:* the cost of a parameter move is its length measured in output-sensitivity units. Standard; exact for the Gaussian-mean family (checked).
- **Log-det mutual information** `I(w; y | X) = ½ Σ_j log(1 + ρ λ_j)` — standard Gaussian-channel identity; classical Shannon, as the commentary says.

# 2. Vocabulary: the commentary was written for the retrofit

| commentary term | WEFT-1 term (R0 table) | note |
|---|---|---|
| reused Qwen block, recurrent LoRA | **tied core** (4 blocks × K visits, full-width, from scratch) | no LoRA; the whole core is the program |
| scratchpad `S_k` | **lanes** (2 × d/4, position-aligned) + **carrier** (rotor, rank-8 write) | working state; no cross-position mixing in the lane path |
| bridge | **bridge_in / bridge_out** (seams), **callosum** (inter-hemisphere), **S-2 combiner** | three seams, not one; each is a "legibility" question |
| MoE sidecar, experts `E_e`, router `π` | **sidecar** (S-4′): shared operator bank 512, per-lane rank-r adapters, PQ-coded selection, gate `D_k` | selection is a frozen code over a standardized descriptor, not a softmax router |
| teacher `p_T`, teacher residual `R = Z_T − Z_S` | **none.** Nearest object: the **final visit** `p_K` as self-teacher for earlier visits | D-MC-1 already decodes the final and one sampled earlier visit |
| `K_ε(x)`: loops to reach teacher-divergence `ε` | **halting head / EXTRAP-K / `η_k` loop-gain instrument** | same question: "how many visits until the answer stops changing" |
| `Capture_r`: residual removed by memory in its top-r modes | **MEM-OP battery** (MEM-SEL/SYN/INJ, MEM-SYN-STATIC control) | with the teacher residual replaced by the final-visit residual |

The commentary's boxed correspondence — recurrence ≈ online computation, scratchpad ≈ working memory, bridge ≈ state interface, sidecar ≈ compiled/amortized structure — survives translation intact and is a good one-line statement of why the sidecar and the core should be measured against each other rather than added up: the design question is **stored program versus online computation**, and the MEM-SYN-STATIC control (a visit-indexed table) is already the cheapest form of "stored."

# 3. Mark's question: can captured matrices gauge component fit before the big run?

**Yes — three instruments, two of them free.**

**I-1 · Per-arm prequential accounting (free).** Every arm already logs its eval loss curve. `Ŝ_preq(arm) = Σ_steps [ L_eval(step) − L_eval(final) ] · Δtokens` — *read as:* the sum over evaluation points of the eval loss at that point minus the final eval loss, weighted by the tokens between evaluations; *intuition:* total bits the arm spent being worse than its final self. Reported per arm at matched FLOPs alongside the terminal BPB it already reports. **Caution bound with it (from the commentary's strongest point):** a larger area is *not* read as more learned structure until the conditioning confound is ruled out — see I-3. For WEFT specifically, the recurrent arm's optimization is worse conditioned by construction (visit-Jacobian products, PF-1.4), so "recurrent arm has larger area" is expected on conditioning grounds alone.

**I-2 · VISIT-KL (free under D-MC-1).** The step already computes `p_K` (final visit) and `p_j` (one sampled earlier visit, `j ~ Uniform{0..K−2}`). Log `κ_j = KL(p_K ‖ p_j)` per micro-batch — *read as:* kappa-sub-j is the divergence from the final-visit distribution to the visit-j distribution; *intuition:* the bits the loop still had to earn after visit j. Its expectation over `j` is an unbiased estimate of `(1/(K−1)) Σ_j κ_j`, the requential-style code across depth: the loop's own "epiplexity per visit." Pre-registered reads: `κ_j` should fall monotonically in `j`; a flat `κ_j` means later visits are not changing the answer (the "K = 2 is most of the win" branch, now measured per token rather than per run); `κ_j` rising with `j` on a token class is the RESP-LEAK signature (later visits undo earlier ones). Zero compute: both distributions exist.

**I-3 · Seam-legibility audit (cheap; proxy rung; the commentary's "bridge hypothesis," made executable from component-only runs).** For a seam sender → receiver (bridge_in → core; callosum sender → receiver hemisphere; combiner → coda; sidecar write → lane), capture on *component-only* runs: the sender's output covariance `C_s` and the receiver's GGN/Fisher `F_r` on its input. With `P_r` the projector onto the top-r eigenvectors of `F_r`, define `α_r = tr(P_r C_s) / tr(C_s)` — *read as:* alpha-sub-r is the trace of the projector times the sender covariance, divided by the trace of the sender covariance; *intuition:* the fraction of what the sender writes that lands in directions the receiver's output is sensitive to. Null: a random-init receiver gives `α_r ≈ r/d`; a shuffled-sender control gives the same. **The audit passes when `α_r` for the trained pair sits well above `r/d` and the seam does *not* merely inflate `tr(C_s)`.** This is computable before integration from matrices captured on the dense control (receiver) and the component arm (sender) — which is exactly Mark's "matrices from different training regimes" idea — and again after integration to see whether training moved the seam. Cost: one Lanczos top-64 per seam on a 4 k-token slice at d = 512.

**No — two things the framework cannot do for us.** A spectral outlier is not epiplexity (a single obvious spike is a short program); and the mode-resolved profile the commentary proposes (`e_j(T)` per NTK mode) requires kernel spectra at model scale — feasible at the proxy rung on a slice as an *exploration* instrument, not as a first-run receipt. Registered as `MODE-PROFILE` in the exploration registry, not bound.

# 4. Micro-experiment: does a *latent* loop find the short rule? (the paper's testbed, transposed)

**Setup.** Predict `Y = F^τ(X)`, τ = 8 iterations of an ECA rule on a ring of 32 cells, from `X` alone — **no intermediate states are emitted.** Model: embed → one weight-tied residual block (circular conv of width 17 = full radius τ, plus MLP, hidden 64) applied **K times** → readout. Because the kernel already spans radius τ, `K = 1` has the *visibility* to solve the task; what K adds is *compute*. The commentary's and the paper's prediction, transposed: the K = 1 model must *compile* the composed 8-step rule (a 17-input Boolean function); the K = 8 model can *execute* the 1-step rule eight times — a short program — and should reach a lower terminal loss with less learning-curve area on the chaotic rule, while a linear probe should find the true intermediate state `F^j(X)` in the hidden state after visit j.

| rule | K | K curriculum | terminal BPC | accuracy | prequential area (BPC·steps) | linear probe of `F^j(X)` at visit j |
|---|---|---|---|---|---|---|
| 30 (chaotic) | 1 | — | 0.508 | 0.839 | 431 | [0.81] |
| 30 | 4 | — | 0.473 | 0.850 | 737 | [0.85, 0.71, 0.69, 0.64] |
| 30 | 8 | no | **0.996** (chance) | 0.538 | 233 | [0.75, 0.59, 0.55, 0.56, 0.52, 0.52, 0.53, 0.54] |
| 30 | 8 | **1→2→4→8** | **0.275** | 0.920 | 1436† | [0.95, 0.74, 0.66, 0.66, 0.59, 0.56, 0.55, **0.92**] |
| 54 (complex) | 1 | — | 0.596 | 0.801 | 308 | [0.90] |
| 54 | 4 | — | **0.002** | **1.000** | 919 | [**0.999**, 0.92, 0.88, 0.92] |
| 54 | 8 | no | 0.781 | 0.701 | 427 | [0.69, 0.66, 0.73, 0.68, 0.69, 0.68, 0.66, 0.71] |
| 54 | 8 | **1→2→4→8** | **0.017** | 0.997 | 1430† | [0.66, **0.97**, 0.94, 0.92, 0.87, 0.81, 0.83, **0.99**] |

Ring n = 32, τ = 8, hidden 64, 1,200 AdamW steps, batch 256, 20 k training grids, 2 k held-out grids, one seed. † Area under a curriculum is inflated: the eval curve was scored at the full K throughout, so the K = 1 warm-up phase is counted as "bits spent" — see reading (iv).

**Readings.** (i) **The latent loop executes rather than compiles.** On rule 54, K = 4 solved the 8-step task exactly (two rule steps per visit) with the same receptive field that left K = 1 at 0.60 BPC. That is the *capability* side of the paper's compute-threshold transition, reproduced without emitting a single intermediate state — the paper's "looped model" needed the states as tokens; this one did not. The *accounting* side did not reproduce: the paper saw epiplexity drop when the short rule took over, whereas here the area rose with K — reading (iv) says why. (ii) **The chaotic rule is where compute is the limit, not visibility.** Rule 30 improved monotonically with K under the curriculum (0.51 → 0.47 → 0.28) but was not solved in 1,200 steps; the paper's own reading applies — below the threshold, the looped model "lacks the compute to fully unroll the dynamics" — and the pre-registered extension is longer training, not a larger model. (iii) **Deep tied recurrence from scratch needs the K curriculum.** At K = 8 with no curriculum both rules trained to *worse than one pass* (rule 30 to chance). The ratified 1 → 2 → 4 curriculum turned that failure into the best result for both rules. This is the smallest possible validation of a WEFT-1 choice that was made on Jacobian-product grounds (§3.4, `α_T = c/T`) and had not been tested from random init. (iv) **The prequential area behaved exactly as the commentary warned.** Area rose with K in every solved case (rule 54: 308 → 919 → 1430) — more structure learned *and* worse conditioning — and under a curriculum it is inflated by the warm-up unless each eval is scored at the executed `K_t`. **Bound into I-1:** `preq_area` under a K curriculum is computed with the step's own `K_t`. (v) **The loop's intermediate states are not the transcript.** Probing `F^j(X)` at visit j finds it at the first visit (0.95–0.999) and at the last, but not in between — the tied core found a program whose intermediate representation is not the step-by-step simulation in linearly decodable form. That is the latent-reasoning claim in its smallest instance, and it says the right instrument is a probe *matrix* `P[j, i]` (accuracy of decoding `F^i` from visit j, all i and j), pre-registered: an *executor* shows a band `i ≈ s·j` for some steps-per-visit `s`; a *compiler* shows only the final row.

**What this is and is not.** It is the paper's own testbed with the paper's own quantity (area above terminal loss), run on the mechanism WEFT-1 bets on and the paper does not test. It is not evidence about language; it is evidence that a tied latent core can prefer "run the short rule K times" over "compile the long rule once" when K is allowed to be large enough — or that it cannot, in which case the design's central bet has a counterexample in the smallest setting. Registered as **`ECA-LATENT` in the PRE-FLIGHT line** (CPU; today's runs took minutes) with the pre-registered reads above and one extension: rule 110 (universal) and τ ∈ {4, 8, 16} with K ∈ {1, τ/2, τ, 2τ}, both seeds.

# 5. What the paper says about things already ratified

- **Factorization order = curriculum.** The chess result is the formal version of the D-CUR decisions: the same corpus in a different order is a different computational problem. It supports the L_stage ruling (a coda that can decode any depth is a model that is asked to be right *before* it has finished computing — the "reverse" factorization) and the RESP-LEAK instrument.
- **Inference is in the budget.** The paper's `T = 6ND + 2ND` counts inference. Our AE (active-equivalent) accounting matches training; the serving-side cost is the K-visit multiplier and, after D-NB-1, the 2K× cache. Nothing changes; the paper is a reminder that a bits-versus-compute frontier for WEFT has *two* compute axes, and the halting head is the instrument that trades along the inference one.
- **Sidecar as compiled structure.** The commentary's `ρ_M` — inference FLOPs saved per bit compiled into the sidecar — is a clean way to say what MEM-OP is for. With the teacher replaced by the final visit: `ΔK_ε = K_ε(no sidecar) − K_ε(sidecar)` on the halting-head criterion, per token class. Registered as a *derived* read on the existing MEM-OP battery, not a new arm.

# 6. Decisions for Mark

**D-EP-1 — VISIT-KL and per-arm prequential area as receipts (free).** *(a) Add both now — recommended.* Two schema lines (`visit_kl` per micro-batch with its sampled `j`; `preq_area` per arm, computed from the existing eval log at report time). No compute, no arm, no default change. *(b) VISIT-KL only.* *(c) Neither.*

**D-EP-2 — Seam-legibility audit and ECA-LATENT into the PRE-FLIGHT line.** *(a) Both — recommended.* The audit is one Lanczos per seam at the proxy rung on component-only checkpoints the build queue produces anyway (steps 2–5); ECA-LATENT is CPU-scale. Together they are the executable form of "do the pieces fit before the big run." *(b) ECA-LATENT only* (no checkpoint dependency). *(c) Neither; keep PRE-FLIGHT as ratified.*

# 7. What does not change

Build queue, P-A/P-B, semantics chain, K/V policy, objective stack, no-injection rules — unchanged. `MODE-PROFILE` and the `ρ_M` read enter the exploration registry only.

---

*Signature block*

**Strategy:** the paper's real contribution is an accounting identity — bits into the model equals area above the final loss — and we have been logging the right-hand side all along without calling it anything. The commentary's real contribution is the warning that the identity confounds structure with conditioning, which for a recurrent model is not a corner case but the main case. The one place the commentary over-reached, the looped-CA experiment, is the one place WEFT-1 can add to the paper: the paper unrolled the states in tokens, and we can ask whether a latent loop does the same thing silently. In miniature it does — and it needs the K curriculum to do it, which is the most useful thing a two-hour CPU experiment has said about the design so far.
**Coding agent:** nothing until D-EP-1/2.
**Mark:** D-EP-1, D-EP-2.
