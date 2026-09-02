# STRATEGY → CODING AGENT — `LOOM-1` FULL BUILD HANDOFF

**Date:** 2026-08-26 · **Supersedes:** all prior partial specifications. This is the single authoritative build document.
**Baseline:** commit `1c033bae` on `codex/bicameral-stage0` (audited build-only substrate). §2 is the exact delta from that commit.
**Governing artifacts:** charter r2 (108,821 B, SHA `36f9d0ed…298a`) · substrate rulings (18,430 B, SHA `29ba4acc…381e`) · topology rulings (23,926 B, SHA `162af016…3956`) · loop-sidecar amendment (12,910 B, SHA `8e301a9d…2158`). Where this document and those disagree, **this document governs** and the disagreement is a defect to report.
**Authorization:** nothing here authorizes a training run. The gates in §13 do that, in order.

---

# 0. Plain-language summary

We are building a small language model — about 300 million parameters — that thinks by going around a loop instead of by being deep. The same block of weights is applied four to six times in a row, and between passes the model keeps notes on a narrow scratchpad. That is the whole idea: a model whose *parameter count* stays small while its *amount of computation* is two or three times larger, because the expensive part is reused rather than duplicated.

The reason we are building it at all is a failure we understand well. The previous program spent months trying to bolt reasoning machinery onto a frozen off-the-shelf model, and nothing worked. The explanation turned out to be a single piece of mathematics: whether a learning signal can be found depends on how strong it is, how many directions you have to search, and — the killer — how much the training examples resemble one another. When every example pulls the same way, which is what a big text corpus does, there is a hard ceiling on finding the rare specific thing, and **more data does not lift it**. Every design choice in this document is aimed at that formula: make the channel a signal must travel through as narrow as possible, subtract the shared tendency before supervising, and reuse weights so each direction gets seen many times.

The model has five parts worth naming. A **protected text stream** that the loop can only modify through a deliberately narrow, gated channel, mixed by a rotation so that information cannot be destroyed. **Position-aligned scratch lanes** that carry the model's working state from one pass to the next, arranged so no position can ever see a later one. **Two hemispheres** stored not as "left" and "right" but as "what they agree on" and "how they differ," which changes what the optimizer is able to notice, with a bounded exchange between them. A set of **small specialist blocks** where the choice of which one runs is made by a cheap fixed measurement rather than a learned dispatcher. And a **sidecar library** beside the loop that the loop can consult *when it needs to* — not on every step, which is what an ordinary mixture-of-experts does, but only when the geometry of its own trajectory says it has just changed what kind of work it is doing.

Two results come out of this that are worth having whether or not the architecture wins. The first is a scaling rule for reuse — how to set step sizes and learning rates when the same weights are applied over and over, which the standard rules do not cover and nobody has characterised. The second is a curve showing when, during training, a model becomes too saturated to steer. We have exactly one point on that curve today. Both fall out of runs we want anyway, and that is what makes this a research programme rather than a bet.

The build is staged so the cheap questions run first. A small proxy model — about 50 million parameters, costing tens of GPU-hours in total — settles every tuning constant and every either-or question. Only then does the real run proceed, at roughly 200 to 300 GPU-hours. Along the way there are fourteen tests that must pass, a list of tripwires that stop the line, and a set of decision rules whose outcomes are written down *before* the data arrives, so that we cannot rationalise a result afterwards. Four design errors have already been caught by writing things down rather than building them, including one where a construction was proposed two pages after explaining why that construction cannot work. That is what this document is for.

---

# 1. How to read this, and the one standing rule

> **Surface ambiguity. Do not guess.**

This programme keeps a ledger of errors caught before they contaminated a result; it now stands at fourteen, and the large majority were caught because an implementer refused to resolve an ambiguity silently. If a symbol is bound twice, a shape does not compose, a threshold arrives without its branch outcomes, or a stated identity fails when you check it — **stop and report.** A defect found at the desk costs nothing. §12 lists the traps we already know about; assume it is incomplete.

Two of the fourteen came from your side of this handoff. "Split-Clifford" was ambiguous in a way that could have silently broken a Tier-1 guarantee, and answering a question with *"neither, currently"* rather than picking an option exposed a confound sitting inside the committed proxy sweep. Both were the right behaviour.

Code below is PyTorch-flavoured: correct in shape and intent, not necessarily runnable verbatim.

---

# 2. Delta from commit `1c033bae`

**Retained unchanged** — pre-RMSNorm with FP32 reductions, QK-RMSNorm, token-only RoPE, SwiGLU, tied embeddings, depth-scaled init, causal SDPA, zero dropout; structural switchability with the locked `α_T = c/T`, `p = 1` rule; requested-versus-executed visit accounting; shared causal segmentation across packed-document boundaries, padding, engram n-grams and shifted loss; Euclidean `Cl(2,0)` rotor with norm-preservation tests; coupled scratch matrices on AdamW with a wrapper-aware Muon allowlist; read-only long-term memory with leave-one-record-out; z-loss; retention and router-calibration diagnostics; the machine-readable quarantine authority.

**To change or add:**

| # | item | status | §  |
|---|---|---|---|
| 1 | **Core is full-width**, not lane-only. Lanes are persistent state alongside it. | correction | 5.4 |
| 2 | **KV discipline**: queries from `h_k`, keys/values computed once from `h_0` and cached | new | 5.3 |
| 3 | **Target topology** 9/4/9 and 8/6/8 at `d = 1024` | not encoded | 4 |
| 4 | **Proxy sweep** → reallocation at constant 10 blocks: 4/2/4, 3/4/3, 2/6/2 | replaces 2/2/2 sweep | 4 |
| 5 | **GQA ratio constant at 2:1 across rungs** (8Q/4KV proxy, **16Q/8KV** target) | correction | 4 |
| 6 | **Second jet** — retain `z_{k−2}`, assemble `(v, a)` | first-order only today | 5.9 |
| 7 | **Loop sidecar** with two-stage conditional invocation | missing | 5.8 |
| 8 | **Birkhoff bound on the callosum** — per-band gain reparameterized as `ρ_b ∈ [0, ½]` | new | 5.6 |
| 9 | **Composition receipt** on every run | new | 9 |
| 10 | **Byte-span polynomial hashing** for `M_lex` (token-ID keys acceptable for bring-up) | deferred ablation | 5.11 |

---

# 3. The design mathematics — why every choice

## 3.1 One inequality governs the architecture

Recovering a direction of strength `θ` in ambient dimension `d` from `n` samples with inter-sample correlation `ρ` (sim-verified, D-M4):

```
cos²(estimate, truth)  =  θ² / ( θ² + d·( ρ + (1−ρ)/n ) )
```

```
ρ = 0 , n → ∞ :   cos² → 1              data buys detection
ρ > 0 , n → ∞ :   cos² → θ²/(θ² + dρ)   ← HARD FLOOR. No quantity of data lifts it.
```

A training corpus **is** correlated samples. The floor is why the previous programme failed: cross-entropy could only see the corpus-average descent direction, and the corpus-average direction does not fix individual wrong answers. The design consequence is that **more data and better optimizers are the two least valuable interventions available**, and narrowing `d` and rejecting `ρ` are the two most valuable.

**Order of magnitude.** *(Interpretation, labelled: the measured common-mode fraction of the correction field, 0.713, stands in for `ρ`. These are analogous, not identical, and the true `ρ` for the gradient problem is a Tier-2 instrument — §10.)*

| | `d` | `ρ` | `θ²` needed for `cos² = 0.5` |
|---|---|---|---|
| frozen retrofit as built | 896 | ≈ 0.71 | ≈ 636 |
| `LOOM-1` rank-8 write, common-mode rejected | 8 | ≈ 0.01 | ≈ 0.08 |

## 3.2 Four levers, and which module pulls each

| lever | mechanism | § |
|---|---|---|
| **↓ d** | low-rank disagreement `r = 32`; rank-8 carrier write; rank-4 sidecar experts; 128-wide sequency bands | 5.5, 5.4, 5.8, 5.7 |
| **↓ ρ** | batch-centred supervision targets; the `δ` channel as a structural common-mode rejector; mode-wise Adam's separate normalizer | 6, 5.5, 7 |
| **↑ θ** | staged-state supervision against real intermediate content, not a margin proxy | 6.2 |
| **↑ n** | **weight tying multiplies `n` by `K`**; the shared sidecar multiplies it by `C·K` | 5.4, 5.8 |

**Weight tying is a variance-reduction device for the detection problem**, not primarily a parameter economy. A `K = 4` tied core sees each of its directions four times per sequence. The shared sidecar is the extreme case: its parameters receive gradient from every core block at every executed visit, `n → n·C·K`, making it the most heavily sampled component in the model.

## 3.3 The optimizer is the same argument

With `g_A = g_μ + g_δ`, `g_B = g_μ − g_δ`, variances `σ_μ²`, `σ_δ²`, independent:

```
in (W_A, W_B) coords :  δ-component of update = g_δ / √(σ_μ² + σ_δ²)
in (μ, δ)     coords :  δ update              = g_δ / σ_δ

gain on the specialization axis = √(1 + σ_μ²/σ_δ²)
```

V9 measured ≈ 5×, implying `σ_μ²/σ_δ² ≈ 24` — about 96 % of gradient variance is common mode. In mixed coordinates the common mode's variance sits in the *denominator* of the disagreement mode's update: the `dρ` floor of §3.1 wearing a different hat. **This is why `(μ, δ)` are the stored tensors.**

## 3.4 Why `α_T = c/T` and not `c/√T`

```
J_total = ∏_{k=1}^{T} (I + α_T J_k) ,  ‖J_k‖₂ ≤ L

α_T = c/T   ⇒  ‖J_total‖₂ ≤ (1 + cL/T)^T ≤ e^{cL}        bounded independently of T
α_T = c/√T  ⇒  ‖J_total‖₂ ≈ e^{cL√T}                     grows with depth
```

For correlated tied visits, `1/T` is the safe anchor. **`p = 1` is the default; the μR sweep measures departure from it** (§8.3). Do not lock the sweep out — the exponent is a headline deliverable.

---

# 4. Configuration

## 4.1 Global

```
d_model            proxy 512  |  target 1024        power of two — required for the WHT
head_dim           64                               both rungs
query heads        8 | 16                           = d / head_dim
KV heads           4 | 8                            GQA ratio held at 2:1 ACROSS RUNGS  ← §4.3
vocabulary         32,768 tied, PROVISIONAL         pending the §13 G-TOK screen
d_ff (dense)       1408 | 2816                      2.75 × d, SwiGLU, bias-free
lane stack L·w     2 × 128 | 2 × 256
disagreement rank  32
carrier rotors J   8            Euclidean Cl(2,0), theta zero-init
carrier write      rank 8, scalar gate
sequency bands E   8            sweep {8, 16, 32}
sidecar            512 experts, rank 4 (sweep {4, 16}), top-k 3, shared across the loop
bivector probes    32 learned   (frozen-random = registered null)
rotor-QK           J_att = 2    (J_att = 0 = control)
K                  curriculum 1 → 2 → 4, K_max 8 with halting
context            2048, then 4096 after promotion
norm               pre-RMSNorm eps 1e-5 + final RMSNorm; no biases anywhere
RoPE               theta 500,000, token axis ONLY
precision          BF16 matmuls; FP32 for norms, loss, WHT, jets, occupancy, all diagnostics
```

## 4.2 Per-block costs and rung tables

```
proxy  d=512 : attn 0.786M  ffn 2.163M  delta 0.299M | dense 2.949M  core 3.459M  sidecar 2.097M  emb 16.78M
target d=1024: attn 3.146M  ffn 8.651M  delta 0.598M | dense 11.796M core 13.210M sidecar 4.194M  emb 33.55M
```

**Proxy sweep — reallocation at constant 10 blocks.** Total blocks are held fixed and outer blocks move into the core, which is *the same operation the target performs*. The superseded 2/2/2→2/4/2→2/6/2 sweep grew total capacity alongside core depth, so `p` would have absorbed a capacity effect; it also swung the vocabulary fraction 13.3 pp against 0.1 pp here.

| arm | blocks | body | total | vocab % | recurrent % | fixed:rec | AE(K=4) |
|---|---|---|---|---|---|---|---|
| **4/2/4** | 10 | 32.6 M | 49.4 M | 33.97 | 27.6 | 2.62 | 60 M |
| **3/4/3** | 10 | 33.6 M | 50.4 M | 33.28 | 47.4 | 1.11 | 81 M |
| **2/6/2** | 10 | 34.6 M | 51.4 M | 32.62 | 66.0 | 0.52 | 103 M |

**Target rungs — 22 unique blocks, two outer blocks traded into the core.**

| rung | P/C/D | body | total | vocab % | recurrent % | fixed:rec | AE(K=4) | AE(K=6) |
|---|---|---|---|---|---|---|---|---|
| **A** | 9/4/9 | 269.4 M | **302.9 M** | 11.08 | 21.2 | 3.72 | 440 M | 555 M |
| **B** | 8/6/8 | 272.2 M | **305.8 M** | 10.97 | 30.7 | 2.26 | 523 M | 689 M |

`N_active-eval` at `K = 6` lands at 555 M and 689 M — inside the 500–700 M-equivalent target from ~305 M unique. That is the operating point the whole architecture exists to reach.

**Compute at `D = 40 B` tokens**, `C ≈ 6·N_ae·D`, A100 at ~1.5 × 10¹⁴ FLOP/s (≈ 40 % MFU, BF16):

```
Rung A, K=4 :  1.06e20 FLOP  ≈ 196 A100-hr
Rung B, K=4 :  1.25e20 FLOP  ≈ 232 A100-hr
Rung B, K=6 :  1.65e20 FLOP  ≈ 306 A100-hr
```

Proxy arms are 10–20 A100-hr each. **All arms belong at the proxy.**

## 4.3 Ruling: the GQA ratio is constant across rungs

Earlier drafts specified 16Q/4KV at the target against 8Q/4KV at the proxy — a 2:1 ratio becoming 4:1. **That is a topology change, not a width change**, and μP transfers across width only. Fixed: `n_kv` scales with width, ratio held at **2:1**. Cost is +0.52 M per target block and 2× KV cache, both accepted. This is more generous than Granite's 5:1 or MiniMax's 6:1 and is a deliberate trade of memory for transfer validity; if KV memory becomes binding at serving time, that is a **post-hoc** ratio change on a trained model, not a training-time deviation.

---

# 5. Architecture

## 5.1 Forward pass

```python
def forward(tokens, mods, K_req, train=True):
    h  = mods.prelude(mods.embed(tokens))         # (B,T,d) — M_lex fires inside prelude block 1
    h0 = h                                        # retained for retention gauge AND KV cache
    kv_cache = [blk.project_kv(h0) for blk in mods.core]   # computed ONCE. §5.3
    lanes = mods.bridge_in(h0)                    # (B,T,L,w) position-aligned. §5.4
    hA = hB = h
    traj, step_logits, executed = [], [], 0

    for k in range(K_req):
        hA, hB, lanes = core_pass(k, hA, hB, lanes, kv_cache, mods)
        executed += 1
        traj.append(pooled_valid(lanes))                       # (B,d) for jets, valid tokens only
        if train or mods.halting_enabled:
            hk = mods.combine(hA, hB)                          # per-band (a_b, b_b), swap basis
            step_logits.append(mods.readout(mods.coda(hk)))    # SHARED coda — §6.2
        if mods.halting_enabled and k >= 2 and not train:
            g = jet(traj[k-2], traj[k-1], traj[k], mods.P, mods.Q)
            if mods.halt_head(halt_features(g, k, K_req)).sigmoid().mean() > 0.5:
                break

    out = mods.readout(mods.coda(mods.combine(hA, hB)))
    return out, step_logits, traj, (h0, hA, hB), executed   # executed != requested. §9
```

```python
def core_pass(k, hA, hB, lanes, kv_cache, mods):
    for i, blk in enumerate(mods.core):
        # 1. FULL-WIDTH attention: queries from current h, K/V from cached h0
        hA = hA + blk.attn(q_src=hA, kv=kv_cache[i], hemi=+1)
        hB = hB + blk.attn(q_src=hB, kv=kv_cache[i], hemi=-1)
        # 2. FULL-WIDTH FFN or parameter-matched Hadamard expert bank
        hA = hA + blk.ffn(hA, hemi=+1); hB = hB + blk.ffn(hB, hemi=-1)
        # 3. lanes: causal, position-aligned, carry state across k
        lanes = blk.lane_update(lanes, hA, hB, mods.engram_values)
        # 4. sidecar: CONDITIONALLY invoked, writes to lanes only
        lanes = mods.sidecar(lanes, jet_state=mods.jet_cache, occ_delta=mods.occ_delta)
    # 5. carrier: orthogonal rotor + single gated rank-8 write
    hA = mods.rotorA(hA) + mods.alpha_T * mods.gamma_A * mods.write_A(lanes)
    hB = mods.rotorB(hB) + mods.alpha_T * mods.gamma_B * mods.write_B(lanes)
    hA, hB = mods.callosum(hA, hB)
    return hA, hB, lanes
```

## 5.2 Carrier rotor — and the trap it replaces

**Do not use a zero-initialized Householder reflection.** `H(v) = I − 2vvᵀ/(vᵀv + ε)` looks correct because `H(0) = I` exactly, but the update term is `O(‖v‖²)` near the origin, so `∇_v = 0` at `v = 0` and **the module never wakes.** That is the zero-matrix trap (N-6) in a new costume, and it was in an early draft of this design.

Use a **rotor** — a Givens rotation in a learned plane, with the angle as the gate. For orthonormal `â, b̂`, with `A = âb̂ᵀ − b̂âᵀ` (the matrix form of the bivector `â ∧ b̂`), `A² = −(ââᵀ + b̂b̂ᵀ)`, so:

```
exp(θA) = I + sin θ · A + (cos θ − 1)(ââᵀ + b̂b̂ᵀ)

R x = x + sin θ ( â(b̂·x) − b̂(â·x) ) + (cos θ − 1)( â(â·x) + b̂(b̂·x) )
```

Exactly orthogonal for every `θ`; exactly the identity at `θ = 0`; `∂R/∂θ|₀ = A ≠ 0`, so it wakes in one step. `O(d)` per rotor.

**Signature is load-bearing.** Under a *split* signature (`Cl(p,q)`, indefinite metric) with `â` and `b̂` in opposite subspaces, `A² = +(ââᵀ + b̂b̂ᵀ)` and

```
exp(θA) = I + sinh θ · A + (cosh θ − 1)(...)      singular values e^{±θ} ,  ‖R‖₂ = e^θ
```

— a **boost**, growing as `e^{Kθ}` over `K` passes, which would destroy carrier information-preservation, the retention tripwire, and the Jacobian bound of §3.4. **Euclidean `Cl(2,0)` only in the carrier.** Split signature is permitted in diagnostics if labelled. T5 is the arbiter.

```python
class RotorCarrier(nn.Module):
    def __init__(self, d, J=8, eps=1e-8):
        super().__init__()
        self.a = nn.Parameter(torch.randn(J, d) / d**0.5)   # NOT zero
        self.b = nn.Parameter(torch.randn(J, d) / d**0.5)   # NOT zero
        self.theta = nn.Parameter(torch.zeros(J))           # zero: identity + LIVE gradient
        self.eps = eps
    def forward(self, x):                                   # (B,T,d)
        for j in range(self.theta.shape[0]):
            a = self.a[j]; a = a / (a.dot(a) + self.eps).sqrt()      # eps INSIDE the sqrt
            b = self.b[j]; b = b - b.dot(a) * a
            b = b / (b.dot(b) + self.eps).sqrt()
            xa, xb = x @ a, x @ b
            s, c = torch.sin(self.theta[j]), torch.cos(self.theta[j])
            x = (x + s * (xb.unsqueeze(-1)*a - xa.unsqueeze(-1)*b)
                   + (c - 1) * (xa.unsqueeze(-1)*a + xb.unsqueeze(-1)*b))
        return x
```

**Second function, and it is the stronger justification.** With `K` successive rank-8 writes into a carrier that nothing reads until the coda, an orthogonal rotation between steps is **what stops all `K` writes landing in the same rank-8 subspace.** The rotor rotates the write subspace so successive conclusions occupy different directions.

## 5.3 KV cache discipline

A tied core applied `K` times sees different input each pass, so naively `KV = n_core × K × T × 2 × d_kv` — `K×` a normal transformer. Resolution:

> **Queries come from the current `h_k`. Keys and values are computed once from `h_0` at the prelude output, cached per core block, and reused for every pass.** Causally masked as normal.

```
KV cache = n_core_blocks × T × 2 × d_kv          identical to a standard transformer, NO K multiplier
```

The reading is also the design intent: **the loop re-queries a fixed context with an updated question.**

**Cost:** the loop cannot re-read the text in light of what it now knows. **Registered arm Fork B′:** recompute token-stream KV once at the midpoint pass — one genuine re-read for `2×` cache instead of `K×`. Implement behind a flag from the start; it is a two-line change later and a refactor if deferred.

**Training:** cache disabled under multi-branch execution. The above is a serving concern.

## 5.4 Lanes, causality, and the full-width ruling

**Correction to an earlier specification:** the core is **full-width**. An earlier draft had the loop operating only on narrow lanes, which prices out at ~1.2 M per core block and ~7 M for `N_recurrent` — against a target of ~85 M. That would have made the model a 270 M standard transformer with a small attachment, and both headline results would have been measuring a marginal module. **Lanes are persistent state carried alongside a full-width core, not a replacement for it.**

**Causality is the hard constraint.** Global scratchpad slots that cross-attend to the whole sequence and write back to all positions **leak the future** under teacher forcing — position `t` becomes influenced by slots that read positions `> t`. It shows up as suspiciously good training loss and broken generation, which is the worst failure mode there is because it looks like success.

```
lanes s[t,k] ∈ R^{L×w}   position-aligned
  inputs at position t:  already-causal h_t  |  causal engram values at t  |  s[t,k−1]
  NO cross-position mixing in the lane path
  where the loop must relate positions, it does so through the core's own causally-masked attention
```

The shared causal segmentation already in `1c033bae` — one segmentation across packed-document boundaries, padding, engram n-grams and shifted loss — is the right implementation and makes this bug class structurally unavailable. Keep it.

**Two lanes of width `w` is structurally the bicameral split** (§5.5); read lane A and lane B as the hemispheres.

## 5.5 Bicameral pair — `SwapLinear`

Store `(μ, δ)`, **never** `(W_A, W_B)`. `W_A = μ + δ`, `W_B = μ − δ`.

```python
class SwapLinear(nn.Module):
    def __init__(self, d_in, d_out, rank=32, sigma_delta0=0.02):
        super().__init__()
        self.mu = nn.Parameter(torch.empty(d_out, d_in)); nn.init.normal_(self.mu, std=d_in**-0.5)
        self.dU = nn.Parameter(torch.randn(d_out, rank) * sigma_delta0)   # NEITHER may be zero
        self.dV = nn.Parameter(torch.randn(d_in,  rank) * sigma_delta0)
    def forward(self, x, hemi):                      # hemi = +1 (A) or -1 (B)
        return F.linear(x, self.mu) + hemi * ((x @ self.dV) @ self.dU.T)
```

Four non-negotiables:

1. **Both `dU` and `dV` nonzero at init.** Symmetric start ⇒ identical hemispheres forever under any swap-symmetric gradient. Symmetry is broken by construction at a registered magnitude or never.
2. **Optimize the stored tensors.** Do not reconstruct `W_A`/`W_B`. Adam's per-element second moment on `(μ, δ)` *is* mode-wise Adam — §3.3's ~5×.
3. **`μ` and `δ` share one optimizer partition** (§7). A semantic Muon rule that captures `μ` as "an ordinary dense hidden matrix" while `δ = UVᵀ` falls to AdamW would set the μ/δ *ratio* by two independently-tuned learning rates under different scaling laws — and the ensemble law and the common-mode rejector both depend on that ratio.
4. **Separate weight decay** `λ_μ`, `λ_δ`. This is how diversity is priced.

**Diversity budget — an interior target, not a maximization.** The ensemble law prices the gain at `(1−ρ)/2` (12.77 % at the frozen 0.7446), but maximal diversity has been measured to *lose*:

```
L_div = λ_div · (ρ̂(A,B) − ρ*)²        ρ* ∈ {0.3, 0.5, 0.7} swept, plus λ_div = 0
```

## 5.6 Callosum — WHT band exchange with a Birkhoff bound

The only inter-hemisphere channel, applied once per core pass:

```
x_A ← x_A + Ŵ⁻¹( M_A ⊙ Ŵ( sg?(x_B) ) )  reparameterized per §5.6.1
```

### 5.6.1 The Birkhoff reparameterization — why the gains are bounded

Learned per-band gains have no norm constraint, so their contribution can grow, and over `K` applications that compounds. The fix comes from the doubly-stochastic lane-mixing formalism, and for **two lanes it is exactly our own swap eigenbasis**:

```
A(ρ) = (1−ρ)I + ρP ,  P = swap permutation,  ρ ∈ [0, ½]

in the swap eigenbasis A(ρ) is DIAGONAL:
    eigenvalue 1        on the consensus mode μ
    eigenvalue (1−2ρ)   on the disagreement mode δ
```

So the doubly-stochastic mixer, written in our coordinates, **is a scalar decay on the disagreement channel**. Consequences: spectral norm ≤ 1 by construction; exact identity at `ρ = 0`; and `A(ρ)^K` damps `δ` by `(1−2ρ)^K` while leaving `μ` untouched — a closed form for how much disagreement survives `K` passes. It also explains symmetric collapse structurally: `ρ = ½` annihilates `δ`.

**Ruling: reparameterize each per-band gain as `ρ_b ∈ [0, ½]`** (sigmoid × ½, zero-init in the pre-activation so `ρ_b = 0` at init is not required — a small nonzero start is preferred per §5.13). Keep the WHT band structure; add the Birkhoff constraint on top. Keep `A` **static** initially: an input-dependent `A` gains a `(∂A/∂X)X` Jacobian term and double stochasticity no longer bounds the network Jacobian.

### 5.6.2 The WHT, and the scaling trap

`W_d W_d = d·I`, so `W_d/√d` is orthogonal and involutive. The fast butterfly is `O(d log d)` — at `d = 1024` that is ~10 k ops per token, free next to a `d² ≈ 10⁶` matmul.

```python
def wht(x):                       # unnormalized: returns W_d @ x
    d = x.shape[-1]; orig = x.shape
    x = x.reshape(-1, d).float(); h = 1
    while h < d:
        x = x.view(-1, d // (2*h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.stack((a + b, a - b), dim=2).reshape(-1, d)
        h *= 2
    return x.reshape(orig)
```

```python
x_rt = wht(wht(x)) * (2.0 ** -p)          # CORRECT — 2^-p is exact in binary FP
x_rt = wht(wht(x) / d**0.5) / d**0.5      # WRONG — 1/sqrt(d) is not, and rounds twice
```

**Sequency (Walsh) ordering is required** — bands are meaningless in natural Hadamard order. The permutation is Gray-code-of-bit-reversal, but **do not trust that description**: T4 is the authority. Build `W_d`, permute rows, assert row `k` has exactly `k` sign changes. If it fails, one of the four obvious variants passes — find it and **report which**, because band semantics depend on it.

### 5.6.3 Gradient coupling — three registered arms

**(A, default)** stop-grad on the sender both directions — information crosses, gradient does not; hemispheres specialize because each must be *useful* to the other through the shared loss. **(B)** alternating parity by step. **(C)** full coupling. **Tripwire on all three:** log `cos(∇_A L, ∇_B L)`; a monotone rise toward 1 is the co-adaptation signature and stops the line.

## 5.7 Hadamard experts and the routerless occupancy router

Experts act block-diagonal in sequency, dense within band. At `d = 1024`, `E = 8` gives 128-wide bands.

```python
class OccupancyRouter(nn.Module):
    """Parameter-free. (m, s) are FROZEN buffers, never parameters."""
    def occupancy(self, x):
        with torch.autocast('cuda', enabled=False):
            c = wht(x.float())[..., self.perm]
            e = torch.stack([(c[..., sl]**2).sum(-1) for sl in bands(self.E)], -1)
        return e / (e.sum(-1, keepdim=True) + 1e-8)                  # (B,T,E)
    def forward(self, x, prev_sel=None):
        o = (self.occupancy(x) - self.m) / (self.s + 1e-8)           # WHITENED
        sel = o.topk(self.topk, dim=-1).indices
        if prev_sel is not None:                                     # hysteresis
            margin = o.gather(-1, sel).min(-1).values - o.gather(-1, prev_sel).min(-1).values
            sel = torch.where((margin < self.tau).unsqueeze(-1), prev_sel, sel)
        return sel, o
```

**Whitening is what makes load balance structural.** Raw sequency energy is overwhelmingly low-frequency, so unwhitened top-k picks the same bands every time and experts never specialize. Whitening by each band's corpus mean and std selects bands that are *unusually* energetic **for this input** — the informative signal, balanced by construction. **No load-balancing loss, no router z-loss, no expert-choice trick.** If a balancing term seems necessary, the calibration is wrong — report it.

**`(m, s)` freeze timing is a defect if unspecified.** Calibrating at step 0 whitens against a *randomly initialized* band distribution and freezes it forever. **Calibrate after a short dense warmup and freeze at a step chosen by a measured calibration-stability gate** (§10, §13). Log the occupancy histogram from step 0 and find where `(m, s)` stops moving.

**Two conditions on the modified experts** (fixed orthogonal transforms, per-expert permutations/signs, learned diagonals): permutations and signs must be **fixed** — what makes the basis load-bearing is that the model trains through a fixed chart; and **occupancy must be computed on pre-diagonal coefficients**, or a learned diagonal becomes a back-door learned router.

**Parameter-matched dense-FFN control is mandatory** (T12). Sweep `E ∈ {8, 16, 32}`: a controlled ablation elsewhere found 128-experts-choose-8 beating 32-choose-2 at matched compute (MATH 19.6 → 24.1, HumanEval 29.7 → 32.5), so `E = 8` is coarse by current practice.

## 5.8 The loop sidecar — conditional invocation

This is the mechanism that distinguishes the design from an ordinary MoE, and it is **new**. An ordinary MoE fires top-k on every token at every layer. Here the loop consults a shared library **only when the geometry of its own trajectory says it has changed what kind of work it is doing.**

```
Stage 1 — INVOCATION   D_k = σ(a·κ̂_k + b·‖v̂_k‖ + c·Δo_k + d)      FOUR parameters
                       fire iff D_k > τ ; else the sidecar contributes exactly 0
Stage 2 — SELECTION    quantized jet-descriptor code indexes the bank (parameter-free)
```

| signal | reading |
|---|---|
| `κ_k` | the trajectory is turning — a pivot — the operation is changing |
| `‖v_k‖` | near zero means converged; nothing left to do |
| `‖o'_k − o'_{k−1}‖` | **the band profile moved between steps: the computation entered a different regime** |

Selection uses the 44-dim descriptor `[o' (8), g_t (4), b (32)]` product-quantized to a code that **indexes** the bank — no learned descriptor→expert matrix, so the routerless prohibition is not tripped. Codebook ≈ 11 K parameters, frozen on the same schedule as `(m, s)`.

**Placement rules, both binding:**

- **The sidecar writes to the lanes only, never the carrier.** The carrier must have exactly one write path or the retention gauge becomes unattributable and the instrument that would have caught the previous programme's `r = 0.44` loses its diagnostic power. Conceptually right too: an expert engaged mid-loop changes *how the model is thinking*, not *what the text says*.
- **One shared bank across the loop**, not one per core block. Matches intent, and maximises the `↑n` lever — parameters receive gradient from every block at every executed visit, `n → n·C·K`.

**Sizing:** 512 experts at rank 4 = 4.19 M at target (5.5 % of `N_recurrent`), top-3 active 24.6 K. Sweep rank ∈ {4, 16}; 24.6 K active against a 13.2 M core block is either exactly right by §3.1's detectability argument or too weak to matter, and the sweep settles it.

**Unification worth knowing while you build it:** the halting head and the invocation gate read the same jet state. *"Am I done?"* and *"do I need a tool?"* are two readings of one measurement — share the feature computation and train them jointly.

| jet signature | halting | invocation |
|---|---|---|
| `‖v‖ → 0, ‖a‖ → 0` converging | **halt** | no |
| `‖v‖ > 0, κ ≈ 0` stable refinement | continue | **no** — the case ordinary MoE wastes compute on |
| `κ ↑` pivot | continue | **yes** |
| persistent `‖v‖`, alternating κ — confused | halt + flag | no, and log |

## 5.9 Jets, Clifford, and the probe identity

**Three facts of geometric algebra suffice.** `uv = u·v + u∧v`. The wedge *magnitude* is redundant — `‖u∧v‖² = ‖u‖²‖v‖² − (u·v)² = det G` — so **never materialize a bivector for a size**; build it only for **orientation**. Orientation compares in `O(Kd)` via frozen or learned plane probes:

```
b_k = (p_k·u)(q_k·v) − (q_k·u)(p_k·v) = ⟨ u∧v , p_k∧q_k ⟩
E[ ⟨b, b'⟩ ] ∝ ⟨B, B'⟩_F        unbiased JL estimator, variance ∝ 1/K
```

**The 2-jet is required — first-order is insufficient.** `κ`, `det G`, the Gram eigenratio, plane consistency and the pivot signature all need `v` and `a` *simultaneously*. `a` is the difference of consecutive `v`, so first-order primitives compose into it at the cost of retaining `z_{k−2}`. **Retain it.**

```python
def jet(w2, w1, w0, P, Q, eps=1e-8):
    """w2,w1,w0 = z_{k-2}, z_{k-1}, z_k, each (B,d). P,Q: (Kp,d) probes."""
    v = w0 - w1
    a = w0 - 2*w1 + w2
    vv, aa, va = (v*v).sum(-1), (a*a).sum(-1), (v*a).sum(-1)
    detG = (vv*aa - va*va).clamp_min(0.0)          # REQUIRED: goes numerically negative
    kappa = detG.sqrt() / (vv.sqrt().pow(3) + eps) #   for near-collinear v,a — the common case
    eigratio = detG / ((vv + aa).pow(2) + eps)
    vh = v / (vv.sqrt().unsqueeze(-1) + eps)
    ap = a - (va / (vv + eps)).unsqueeze(-1) * v   # component of a orthogonal to v
    ah = ap / (ap.norm(dim=-1, keepdim=True) + eps)
    b  = (vh @ P.T)*(ah @ Q.T) - (vh @ Q.T)*(ah @ P.T)      # (B,Kp)
    return dict(nv=vv.sqrt(), na=aa.sqrt(), va=va, kappa=kappa, eigratio=eigratio, b=b)
```

**Probes are learned, with frozen-random as the registered null.** All jet arithmetic in FP32 regardless of autocast.

**Standing caution — program law.** Raw `κ` and angle values in `d = 1024` concentrate and are meaningless in absolute terms. **Only contrasts against a matched smooth-noise null are quotable** (Gaussian random walk with per-step norms matched to the data). Never report "κ = 0.4, therefore the trajectory turns."

**Registered exploratory arms** (`λ` may be 0): plane consistency `L_plane = −Σ_k cos(b_k, b_{k+1})` — the hypothesis being that a loop turning through a consistent plane has a reusable direction of refinement, which is what additive recurrence requires. And late-step deceleration `L_conv = Σ_{k≥K/2} softplus( (v·a)/(‖v‖‖a‖+ε) )`.

## 5.10 Rotor-QK attention

Rotate the query in a learned plane before the dot product — relational rather than similarity matching. `R` orthogonal ⇒ `‖Rq‖ = ‖q‖`, so score scale and temperature are untouched; `θ = 0` ⇒ identity bit-for-bit; `∂score/∂θ|₀ = (Aq)·k ≠ 0`. Applied to `q` before the standard kernel — **no attention-kernel change, FlashAttention included.** ~4 K parameters per layer. `J_att = 0` is the registered control and must be a config flag, not a deleted branch.

## 5.11 `M_lex` — the lexical engram

**Framing:** *32 K is the model's **output** vocabulary; the engram is a much larger sparse **input-side** vocabulary.* This separates the cost that must stay small (the softmax and output matrix) from the capability that can be supplied sparsely (input-side pattern reconstruction).

**Honest scope, because an earlier draft overclaimed:** the engram supplies content without spending context tokens, but it does **not** shorten the token sequence, does **not** improve fertility, and does **not** recover text truncated outside the window. It compensates on **one** axis; the fertility and truncation costs are simply paid.

Placement after prelude block 1 — contextual query, still early. Orders `{2,3}`, four independent hashes per order, 65,521–131,071 prime rows per table, 8-dim rows, concatenated memory width 64, ≈ 4.2–8.4 M sparse parameters. Contextual gate:

```
g_t = σ( RMSNorm(h¹_t)ᵀ RMSNorm(W_K e_t) / √d_m )
h¹_t ← h¹_t + γ_m · cap( W_V (g_t e_t) )
```

**Token-ID keys are correct for bring-up** — the vocabulary is frozen before the engram is built (§13 ordering), so the confound that would have made byte-addressing mandatory does not arise. **Byte-span polynomial hashing is the registered ablation**, and it buys two things: portability if `V` is ever revisited, and freedom to use BPE dropout (which must otherwise be off while token-ID keys are live).

**Do not** lowercase or NFKC-normalize code, identifiers, or formulas. Sparse tables use Adam/SparseAdam, zero weight decay, separate LR sweep — **and are excluded from `N_body` and from μP width scaling** (§9).

## 5.12 Bridges

`bridge_in` produces `lanes` from `h_0`, position-aligned. `bridge_out` — the coda reads the recombined carrier plus lane state. Both gated per §5.13. **With every gate at its OFF configuration the model is bit-for-bit a plain prelude+coda transformer**, which makes Stage 0 a known-good baseline checkable against any reference implementation, and every later stage a controlled comparison against a *live* baseline.

## 5.13 The gating law — corrected

The naive claim is that a scalar zero-gate wakes in one step because `∂L/∂g = ⟨∂L/∂out, f(x)⟩ ≠ 0`. **True for `g`, false for everything inside `f`**, whose gradient is `g · (…) = 0` while `g = 0`. Harmless for a small block; a genuine cold-start trap for a sparse table with millions of rows, where the table cannot become useful while the gate suppresses its gradient and the gate cannot grow until the table is useful.

> **Exact OFF is structural — the control arm skips the module entirely.**
> **The ACTIVE arm uses a small nonzero LayerScale (≈ 1e-3 … 1e-2), never exact zero.**
> **Never zero both a branch output and its gate.**

Test consequences: **T1 tests the structurally-skipped OFF configuration; T2 tests that in the ON configuration every parameter — including inside `f` — has nonzero gradient at step 1.**

---

# 6. Objective

```
L = L_CE
  + α_s · L_stage      staged solution-state alignment   (primary)
  + β   · L_div        diversity budget, interior target
  + γ   · L_ret        retention
  + ζ   · L_halt       halting supervision
  + ι   · L_inv        invocation-gate supervision
  + δ   · L_conv       late-step deceleration     (exploratory, may be 0)
  + η   · L_plane      plane consistency          (exploratory, may be 0)
  + z-loss on output logits
```

Every coefficient and schedule registered before the run. **Every auxiliary term gets an ablation arm at the proxy.** We do not ship an eight-term loss without knowing which terms do anything.

## 6.1 Why not a margin objective

An earlier design supervised **per-step margin improvement**. It was withdrawn. On the previous substrate, margin-only selection crowned two arms that failed completely at the answer level, and shuffled targets moved margins by +0.47/+0.98 while moving **zero** answers. A per-step margin objective could optimize a margin channel carrying no answer information — reproducing that failure inside the loss function.

## 6.2 `L_stage` — staged solution-state alignment

For appropriate problems, generate `x, z₁, …, z_K, y` where `z_i` is intermediate computational state. The model need not *emit* the `z_i`; the target encourages its recurrent trajectory to *represent* progressive solution state.

```
L_stage = Σ_k [ 1 − cos( φ(h_k) , ψ(z_k) ) ]        φ, ψ small learned readers
```

**Direction matching, not coordinate matching**: do not force the model to occupy the same coordinates at every step; teach it to move in the same direction through representation space. Decode through the **shared coda** — that is what forces every step into the same output geometry and makes tying's `n → nK` real rather than nominal.

**The `z_k` generation pipeline is not yet specified and is the subject of the next document.** Build `L_stage` behind a flag; the curriculum handoff supplies the data.

**Live monitor regardless:** running correlation between per-step margin gain and generative accuracy gain. If they dissociate, we are in the §6.1 failure and need to know immediately.

---

# 7. Optimizer

**AdamW is the bring-up optimizer** until identity, causality, gradient-liveness and Jacobian gates pass.

**Muon eligibility is semantic, and the allowlist must exclude paired-mode tensors.** Eligible: ordinary dense hidden matrices — attention Q/K/V/O and SwiGLU gate/up/down in the *unique* dense path. **Not eligible:** tied embedding/head; norm gains; scalar gates; step/role/lane embeddings; engram tables and retrieval parameters; Hadamard gains and sparse routers; sidecar experts; **and `μ` together with `δ`** (§5.5 rule 3). Add an explicit assertion that no paired-mode tensor is captured.

**Two interactions that general Muon results do not cover:**

```
polar(cG) = polar(G)  for c > 0
```

**Scaling a recurrent loss or gradient by `1/T` does NOT produce a `1/T` Muon learning rate.** Forward residual scaling and update-size control decouple completely under orthogonalized updates and need separate handles: forward `α_T` for activations and backprop, a **separate post-Muon parameter-group multiplier** for recurrent update size, and direct measurement of realized update RMS.

```
tied recurrent matrix receives  G = Σ_{k=1}^{T} G_k   BEFORE orthogonalization
```

**Muon cannot resolve conflicting loop gradients** — it orthogonalizes their sum. Hence the per-loop gradient-cosine telemetry of §10.

**Honest accounting:** reported ~½-the-FLOPs Muon results omit Newton–Schulz optimizer FLOPs, and that overhead is noticeable at 50 M with small token batches. **Complete measured FLOPs, including optimizer, in every comparison.**

**Mode-wise Muon** — applying polar orthogonalization separately to `μ` and `δ` — is the natural resolution of the collision with mode-wise Adam and **remains unsupported pending its own derivation**: `δ = UVᵀ` is low-rank, and orthogonalizing `U` and `V` separately is not orthogonalizing `δ`. Do not implement it on intuition.

**Gate:** crossed experiment `{AdamW, hybrid Muon} × {T = 1, T = 4}`, compared on bits-per-byte, tokens-to-loss, complete measured FLOPs, wall-clock-to-loss, peak memory, failure rate, and per-loop help/harm.

---

# 8. Scaling programme

## 8.1 μP width transfer

```
m_width = d_model / d_base
σ_internal = σ_base / √m_width
η_internal = η_base / m_width
attention logits ÷ d_head            (NOT ÷√d_head)
h ← h + α · F_θ(h)                   α a first-class parameter, found on the proxy
embedding and residual multipliers   first-class parameters — do NOT copy others' constants
```

**Verify, do not assume,** that μP's input/output-weight taxonomy handles the vocabulary-fraction change between rungs (33.97 % → 11.08 %). Registered check on the transfer.

## 8.2 Power-LR

`η_opt/β ∝ T^{−0.51}` and `η(n) = min[η_max, β·a·n^b]` with `a ≈ 4, b ≈ −0.51, η_max = 0.02` as **priors, not values**. Proxy factorial: `a ∈ {3,4,5} × b ∈ {−0.45, −0.50, −0.55}`, plus 2–3 recurrence-scaling values.

## 8.3 μR — the recurrence scaling law

μP handles width because independent layers have independent gradients. A tied core does not: `g_θ = Σ_{k=1}^{T} g_{θ,k}`, correlated by construction.

| | `α(T) = c·T^{−p}` | `η_R(T) = η₀·T^{−q}` |
|---|---|---|
| updates independent | variance preservation ⇒ `p = ½` | `‖g‖ ~ √T` ⇒ `q = ½` |
| updates aligned | total-displacement control ⇒ `p = 1` | `‖g‖ ~ T` ⇒ `q = 1` |
| **Jacobian boundedness (§3.4)** | **⇒ `p = 1` is the anchor** | — |

Sweep `p ∈ {0, 0.25, 0.5, 0.75, 1}` and measure `‖Δh_t‖`, `cos(Δh_t, Δh_{t+1})`, `‖h_T − h_0‖`, loss.

**The jets are the mechanism estimator, and this is why they matter beyond halting.** Since `Δh_t = v_t` and `Δh_{t+1} − Δh_t = a_t`:

```
cos(Δh_t, Δh_{t+1}) = (v·v + v·a) / ( ‖v‖ · ‖v + a‖ )
```

— a function of the same three dot products that give `κ` and `det G`. **The jet controller is the live estimator of where the model sits between the independent and aligned limits, and therefore of `p` and `q` themselves.** Plane consistency extends it: alignment says *how much* successive updates agree, orientation says *whether they agree by turning through a common plane*.

**Registered structural prediction, numeric class not permitted to carry a decision:** `p` agrees within ±0.15 across rungs; `q` is noisier.

## 8.4 What transfers proxy → target, and what does not

| quantity | transfers? | why |
|---|---|---|
| `p`, `q` | **yes** | the Jacobian bound concerns the **core's own** Jacobian composed `T` times; it does not reference the surrounding model |
| μP width constants | **yes** | provided embeddings and sparse memory stay in their own groups |
| **Power-LR constants** | **NO — re-verify at target** | global LR depends on the whole loss landscape, which composition changes |
| **`K > 1` beats `K = 1` at matched compute** | **NO — re-test at target** | maximally sensitive to recurrent fraction: proxy spans 27.6–66.0 %, target 21.2–30.7 % |
| recurrence-count curriculum | partially | re-verify cheaply |

A short confirmation run at the target is far cheaper than a wrong LR across ~200–300 A100-hours.

---

# 9. Capacity accounting

| term | definition | used for |
|---|---|---|
| `N_unique` | all distinct trainable parameters, embeddings included | headline size, vocabulary fraction |
| `N_body` | `N_unique` − embeddings − sparse-addressed memory | **the quantity μP width-scales** |
| `N_fixed` | body parameters executed once per forward | — |
| `N_recurrent` | body parameters in the tied core, **including the shared sidecar** | — |
| `N_active-eval(K)` | `N_fixed + K·N_recurrent` | **FLOPs `C ≈ 6·D·N_ae`**; comparison to dense baselines |
| `N_active-token(K)` | as above, counting only firing experts under top-k | inference cost |
| `N_sparse-addressed` | engram tables and codebooks | **reported separately; never in `N_body`; never μP-scaled** |

**Four binding rules.** (1) Sparse-addressed memory is excluded from `N_body` and from μP scaling. (2) The vocabulary-fraction denominator is `embeddings + N_body`, engram excluded — the engram is a separately-gated arm and including it would make the vocabulary decision depend on an ungated construct. (3) **`N_active-eval` uses *executed* visits, not requested** — under halting and conditional invocation, executed `K` and firing fraction `φ` vary per example; report distributions and use means. (4) Every figure carries its rung and its `K`.

**Composition receipt — print on every run:** `N_unique`, `N_body`, `N_fixed`, `N_recurrent`, `N_sparse-addressed`, vocabulary fraction, recurrent fraction, `fixed:rec`, `N_active-eval(K)` at executed `K`, and `φ` per step index. Composition drift is invisible unless printed, and two of the fourteen catches were exactly that.

---

# 10. Instruments

**Tier 1 — tripwires; a breach stops the line.**

| instrument | tripwire |
|---|---|
| retention `r` (fitted contraction of `h_k` on `Q h_0`, frozen random `Q ∈ ℝ^{64×d}`) | **`r < 0.9`** |
| `‖δ‖/‖μ‖` per band | → 0 means hemispheres merged |
| `ρ̂` branch correlation | → 1 means merged |
| rotor orthogonality `abs(‖Rx‖/‖x‖ − 1)` | `> 1e-6` |
| finiteness on every ε-guarded path | any NaN/Inf |
| routing flip rate | sustained high = unstable router |
| expert occupancy histogram (**per byte covered**, not per token) | collapse to one band |
| cross-hemisphere gradient alignment `cos(∇_A L, ∇_B L)` | monotone rise → 1 |
| margin-vs-generative correlation | dissociation (§6.1) |
| estimated Jacobian spectral norm | departure from the `e^{cL}` bound of §3.4 |
| per-loop gradient cosines | also the μR mechanism estimator (§8.3) |
| dual write magnitude — **accumulated AND deployed** | aggregation arithmetic masquerading as inactivity |
| invocation rate `φ` per step index | uniform ⇒ §13 G-INV |

**Tier 2 — the science, at every checkpoint.** Correction-field decomposition on a held-out panel: common-mode fraction, Marchenko–Pastur edge outlier count, cross-fitted leave-half-out `ρ_res` with MP-edge deflation, conditional cosine with nuisance deflation — **all under nested cross-fitting; no gate statistic may reuse a hyperparameter-selection fold.** Jet profile of the lane trajectory against the matched smooth-noise null. Halting statistics (mean executed `K`, accuracy vs `K`). SDPA forward and backward against a small FP32 reference.

**Two measurements nobody has, and both are by-products of runs we want:**

1. **The developmental window.** Plot the *reachable row-specific residue* against training tokens at every checkpoint. The previous programme has exactly one point on this curve, at ~1,800× compute-optimal, and found 0.01–0.02. **The curve is answerable whether or not the architecture wins.**
2. **`ρ` for the gradient problem, measured directly** — the inter-sample correlation of the per-row supervision targets. §3.1's justifying arithmetic currently rests on an analogy in its place. Cheap; compute it early.

**Test the instruments, not only the model** (T11): inject a deliberate 0.5× contraction into the carrier and assert the retention tripwire fires. A tripwire that cannot fire reads as a pass.

---

# 11. Test manifest — all must pass before any training run

| # | test | assertion |
|---|---|---|
| T1 | init bit-identity | all modules structurally OFF ⇒ `torch.equal(model(x).logits, baseline_transformer(x).logits)` |
| T2 | cold-start gradient | in the ON configuration, **every** parameter incl. inside gated modules has nonzero grad after one backward |
| T3 | WHT round-trip | `max\|wht(wht(x))·2⁻ᵖ − x\| / \|x\| < 1e-5` (fp32); two-sided `√d` variant demonstrably worse |
| T4 | sequency order | row `k` of the permuted Hadamard matrix has exactly `k` sign changes, ∀k |
| T5 | rotor | `torch.equal(R(θ=0)x, x)`; `abs(‖R(θ)x‖/‖x‖ − 1) < 1e-6` for random θ — **the Euclidean/split arbiter** |
| T6 | degenerate inputs | zeros / constant / single-token / identical lanes ⇒ no NaN or Inf anywhere |
| T7 | symmetry break | `‖δ‖ > 0` at init; symmetric-init run provably diverges from asymmetric |
| T8 | zero-init audit | for every zero-initialized parameter, grad ≠ 0 after one backward |
| T9 | stop-grad | under arm A, no gradient reaches the sender through the callosum |
| T10 | router determinism | same input twice ⇒ same `sel`; `(m,s)` absent from all optimizer groups |
| T11 | instrument sanity | injected 0.5× carrier contraction **trips** the `r < 0.9` tripwire |
| T12 | matched control | dense-FFN control within 1 % params and 5 % FLOPs of the expert bank |
| T13 | shape/dtype contract | every module matches §4; WHT/jet/occupancy regions run FP32 |
| **T14** | **causality** | **gradient of loss at position `t` w.r.t. inputs at `t' > t` is exactly zero, for every `k`, with lanes, engram and sidecar all ON** |
| T15 | KV equivalence | cached-KV forward matches recomputed-KV forward to fp32 tolerance at `K = 1` |
| T16 | Birkhoff bound | callosum `‖A(ρ)‖₂ ≤ 1` for all reachable `ρ_b`; `δ` damping matches `(1−2ρ)^K` |

**T14 is the one that matters most.** A causality leak looks like success — better training loss, broken generation. Test it directly by gradient, with every optional module enabled, at every `k`.

---

# 12. Foot-guns

**F1 — Zero-initialized matrices in multiplicative paths never wake.** Gates are scalars or angles. T8 checks every zero-init parameter automatically.
**F2 — The Householder trap.** §5.2. `O(‖v‖²)` near the origin ⇒ `∇ = 0` at zero. Use rotors.
**F3 — `√0` and `x/0`.** Every `sqrt` takes `+eps` **inside**: `(x.dot(x) + eps).sqrt()`, never bare and never `+eps` outside.
**F4 — WHT round-trip scaling.** One `2⁻ᵖ`, not two `d^{−0.5}`.
**F5 — Sequency vs natural order.** Bands are meaningless in natural order. Trust T4.
**F6 — `0 × NaN = NaN`.** A zero gate does not protect against an upstream NaN. Assert finiteness *before* the gate multiply.
**F7 — Symmetric init kills the bicameral split permanently.** `dU`, `dV` both nonzero.
**F8 — Do not reconstruct `W_A`/`W_B` for the optimizer**, and do not let a semantic Muon rule capture `μ` while `δ` falls elsewhere.
**F9 — Batch-concat non-identity.** In BF16, concatenating different batches changes last-bit results. **Execution schedule is part of evaluator identity**: any two numbers being compared must share batching, ordering and kernel path, and the harness must record them. This produced a celebrated false positive in the previous programme before it was retracted.
**F10 — Frozen calibration constants must be frozen.** `(m, s)`, probe seeds, quantizer codebooks are buffers. Assert absence from every optimizer group and invariance across a run.
**F11 — `det G` goes numerically negative.** `clamp_min(0.0)` before `sqrt`. Near-collinear `v, a` is the common case.
**F12 — Isotropic random is not a neutral control.** In an anisotropic state space isotropic noise concentrates where the Jacobian barely looks. Covariance-matched controls, and a **no-injection baseline row in every generative table**.
**F13 — Two seeds producing identical target outputs are not two replications.**
**F14 — Do not add a load-balancing loss.** If the router looks unbalanced the whitening calibration is wrong. Report it.
**F15 — A permanently red test baseline is how a real failure hides.** Quarantine with a machine-readable authority and a per-node reason, or fix. Byte-lock failures on *sealed* artifacts are stop-the-line.
**F16 — The carrier has exactly one write path.** Two would make the retention gauge unattributable.

---

# 13. Gates, with branch outcomes registered before data

| gate | question | branches |
|---|---|---|
| **G-TOK** | vocabulary at 16 K / 24 K / 32 K / 48 K | Decide on the **rung-B target column**, **byte-matched budgets**, **bits-per-byte** (never cross-tokenizer perplexity). Verify rung A within 0.5 pp and no reversal of arm ordering. Proxy runs detect anomalies only. *48 K wins* ⇒ compute-optimal-compression dominates and the 32 K default was wrong; *≤ 32 K wins* ⇒ the parameter-allocation argument holds and 32 K freezes. |
| **G-CAL** | when to freeze router `(m, s)` | Log occupancy from step 0; freeze at measured stability. **Never step 0.** |
| **G-K** | does `K > 1` beat `K = 1` at **matched compute**? | *yes* ⇒ proceed; *no* ⇒ recurrence is inert in a substrate we control — a stronger negative than the frozen one. Bank and stop the loop line. **Must be re-tested at target composition** (§8.4). |
| **G-EXP** | expert bank vs parameter-matched dense FFN | dense control mandatory (T12); `E ∈ {8,16,32}` |
| **G-INV** | is invocation genuinely conditional? | Firing rate must be **non-uniform across `k` and κ-aligned**, held-out, both seeds. *uniform* ⇒ the gate is decorative; **report an ordinary MoE rather than claiming a conditional one**. *near 0 %* ⇒ inert; fix or remove — **a bank that is safe and idle is a failure, not a null.** |
| **G-STEP** | does `L_stage` improve **generative** accuracy at matched compute? | *margins only* ⇒ remove the term and bank the replication |
| **G-OPT** | AdamW vs hybrid Muon | crossed with `T ∈ {1,4}`; complete measured FLOPs including optimizer |
| **G-MUR** | do `p`, `q` agree across rungs? | agreement ⇒ a transferable scaling law; disagreement ⇒ the law needs a depth term — **either is a result** |

**Kill criteria.** Stage 0 fails to match a parameter-matched reference ⇒ implementation defect, no science. `ρ̂ → 1` or `‖δ‖ → 0` ⇒ hemispheres merged. Retention `r < 0.9` ⇒ rotor or gate schedule wrong. Expert occupancy collapses ⇒ calibration wrong. **And the deep one:** if the correction-field decomposition at 10× compute-optimal already shows the frozen signature (~71 % common mode, `ρ_res ≈ 0.015`), the intervention window closes early and this architecture's motivation is gone — **that is still the developmental-window measurement, and still the most valuable thing in the programme.**

---

# 14. Build order

Each stage ends with a **pre-run receipt** (seeds, config hash, dry-run cost projection, branch map with **both** outcomes written before data), **one byte-verified handoff**, and blind predictions registered in **two explicit classes** — structural/directional and numeric — with **no decision permitted to ride on the numeric class**.

| stage | content | gates |
|---|---|---|
| **S0** | embed + prelude + coda + readout, everything else structurally OFF. Match a parameter-matched reference. | T1, T3, T4, T13, T14 |
| **S1** | **Tokenizer screen** on frozen audit corpus. Freeze `V`. | **G-TOK** |
| **S2** | μProxy calibration: μP width, Power-LR factorial, **μR sweep of `p`, `q`** with jet instrumentation, on the 4/2/4 → 3/4/3 → 2/6/2 reallocation sweep | G-CAL, G-MUR |
| **S3** | Hadamard experts + rotor-QK against parameter-matched controls; `M_lex` | G-EXP |
| **S4** | Carrier + lanes + bridges + loop; retention gauge live | **G-K**, T5, T6, T8, T11, T15 |
| **S5** | Bicameral split + callosum; `r`, `ρ*` sweeps; coupling arms | T7, T9, T16 |
| **S6** | Loop sidecar + conditional invocation | **G-INV** |
| **S7** | `L_stage` at full weight (**requires the curriculum handoff**) | **G-STEP** |
| **S8** | Optimizer crossed experiment | **G-OPT** |
| **S9** | Transfer to target rung; re-verify Power-LR and G-K at target composition; four-phase curriculum; developmental-window curve | — |

**Cost anchors:** proxy arms 10–20 A100-hr each; target 196–306 A100-hr depending on rung and `K`. Under the budget doctrine, **caps are anomaly tripwires at 3–5× projection, not budgets.** A projection merely over a cap raises the cap with a note; a projection of the wrong *shape* or order of magnitude stops the line and returns to strategy. **Registered science is never trimmed to fit a cap.**

---

# 15. References

**Program-internal** (byte-verified, in the project): charter r2 `36f9d0ed…298a` · substrate rulings `29ba4acc…381e` · topology rulings `162af016…3956` · loop-sidecar amendment `8e301a9d…2158` · program retrospective `d0e5b281…9163a0`.

**External, cited where used:**
- Tokenizer-Agnostic Engram Module — byte-span polynomial hashing, hash equivalence across tokenizers — https://arxiv.org/abs/2607.29065
- Compute Optimal Tokenization — match training **bytes** not tokens across tokenizers; `α ≈ 0.465`, `β ≈ 0.471` — https://arxiv.org/html/2605.01188v1
- EntropyMoE — scalar-signal routing beats hidden-state routing, 1024× fewer router parameters — https://arxiv.org/html/2608.06398
- Scaling Laws with Vocabulary — larger models deserve larger vocabularies — https://arxiv.org/abs/2407.13623
- DeepSeek Engram — conditional memory as a second sparsity axis — https://arxiv.org/html/2601.07372v1
- Tensorizing Engram — CP decomposition sharing latents across n-gram orders — https://arxiv.org/html/2606.08347v1
- Byte Latent Transformer — entropy patching (registered scale-up arm, not now) — https://arxiv.org/abs/2412.09871
- Kimi K3 — hybrid attention, auxiliary-loss-free balancing (declined; see §7 and charter §7.4) — https://arxiv.org/pdf/2607.24653
- MiniMax M2 — full attention over sliding-window, QK-Norm, fine-grained expert ablation — https://sebastianraschka.com/blog/2026/minimax-m2-technical-report.html
- Granite 4.1 — small dense reference topology — https://huggingface.co/blog/ibm-granite/granite-4-1

---

*Signature block*

**Strategy:** this supersedes every prior partial specification. Four design errors are documented in place rather than silently corrected — the Householder freeze (§5.2), the global-scratchpad causality leak (§5.4), the lane-only core (§5.4), and the engram context overclaim (§5.11) — because each is a trap an implementer could re-enter. The arithmetic in §4 is derived and checkable; where it reproduces a figure you computed independently, that agreement is the provenance signal and is noted.
**Coding agent:** build in the order of §14. Every optional module structurally OFF for its control arm and small-nonzero for its active arm. The test manifest is not advisory, and **T14 outranks everything else** — a causality leak is the only failure here that improves your loss while destroying the result. **Surface ambiguity; do not guess.** Two of the fourteen catches in this programme's ledger are yours, both from declining to resolve something silently.
**Mark:** the curriculum handoff is next and `L_stage` (§6.2) blocks on it — the `z_k` staged-state pipeline, verifiable-trajectory sources, the decontamination boundary against the sealed batteries, and phase-transition criteria are all still unspecified.
