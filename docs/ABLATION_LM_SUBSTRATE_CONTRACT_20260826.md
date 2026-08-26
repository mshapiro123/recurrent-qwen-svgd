# Ablation-First Small Recurrent LM Substrate Contract

**Date:** 2026-08-26
**Status:** design and engineering-preflight contract only. This file does not authorize corpus contact, tokenizer fitting, optimizer construction, training, evaluation-panel scoring, or sealed-partition access.
**Scope:** a new from-scratch PyTorch decoder LM. It is informed by the banked recurrent, scratchpad, bridge, Hadamard, bicameral, working-memory, Engram-control, and jet work, but it is a new lineage. It does not revise or reopen any result in `CODEX_NEW_CHAT_MASTER_HANDOFF_20260825.md`.

## 1. Governing objective

Build one conservative language-model substrate on which each proposed innovation has:

1. a causal definition;
2. an exact or explicitly tolerance-bounded null;
3. a parameter- and compute-aware comparator;
4. a gradient-path test;
5. a separately registered scientific arm; and
6. a composition rule that prevents a kitchen-sink result from obscuring which mechanism mattered.

The intended endpoint may combine regular and latent Engrams, upfront modified-Hadamard experts, recurrent reasoning, a causal scratch lane and bridge, a dense token lane, a bicameral specialization with a Corpus Callosum, long-term memory, low-dimensional Clifford primitives, and jet diagnostics. That endpoint is not the first build. Each pillar must earn composition under the gates below.

## 2. Fixed substrate versus staged research

The following distinction is binding.

| Class | Included before any architecture ablation | Not included until its own lock |
|---|---|---|
| Token path | decoder-only causal transformer; tied token embeddings and unembedding; full causal attention | sliding/local/linear attention; untied vocabulary; sequence-level bidirectional scratch tokens |
| Block | Pre-RMSNorm; QK-RMSNorm; GQA; RoPE; bias-free SwiGLU; identity residual | DeepNorm, NormFormer, post-norm, state-dependent hyper-connections, learned residual topology |
| Numerics | BF16 matrix operations; FP32 norm reductions, attention normalization, logits/loss reductions; PyTorch SDPA with an FP32 reference | quantization, sparse kernels, custom fused research kernels |
| Optimization | AdamW as the bring-up reference; optimizer fields locked only in a later training preregistration | Muon or mode-wise Muon; optimizer selected from the architecture result |
| Research pillars | none | Engram, Hadamard experts, recurrence, scratch/bridge, bicameral/callosum, long-term memory, Clifford computation, jet-conditioned control |

No research pillar may be smuggled into the dense baseline as a “minor implementation detail.”

## 3. Dense decoder substrate

### 3.1 Initial target geometry

The first target geometry is deliberately small and transparent:

| Field | Contract value |
|---|---:|
| Model width | `d_model = 512` |
| Query heads | 8 |
| KV heads | 4 |
| Head width | 64 |
| MLP width | 1,408, rounded from the SwiGLU `8d/3` rule |
| Shared recurrent-core blocks | 4 and 6 are the registered scale candidates |
| Engineering bring-up | 2 shared core blocks only; never a scientific winner |
| Prelude / Coda | 2 / 2 unique blocks in the initial allocation |
| Initial context | 4,096 tokens at target geometry; shorter sequences are legal only for unit/smoke tests |
| RoPE | full-head rotary embedding, base `theta = 500,000`; token coordinate only |
| Attention | full causal GQA through `scaled_dot_product_attention` |
| Dropout | 0 in the first controlled comparison |
| Linear biases | absent |
| Embedding policy | input/output weights tied |

“Two,” “four,” and “six” in this contract refer to the number of shared recurrent-core blocks, matching the proposed `T=4` arithmetic: the two-block bring-up executes eight core block-passes. With the initial two-block Prelude and two-block Coda, total unique decoder blocks are respectively 6, 8, and 10. Every receipt reports both unique blocks and executed block-passes; neither may be shortened to an ambiguous “depth.” The two-core-block model exists to make failures cheap: forward/reference equality, masks, cache behavior, gradients, checkpoint replay, and module nulls. Neither tokenizer selection nor architectural promotion may use its task loss as evidence. The first registered core-depth screen is the 4-versus-6 comparison, with its training budget, seeds, selection tolerance, and compute accounting frozen before either result is read.

### 3.2 Block equations

For token state `x_l`, the fixed block is

```text
u_l       = x_l + Attn_l(RMSNorm(x_l))
x_{l+1}   = u_l + SwiGLU_l(RMSNorm(u_l)).
```

Attention uses separately instantiated `q_proj`, `k_proj`, `v_proj`, and `o_proj` parameters. Packing them is prohibited in the reference implementation because it obscures parameter ownership, ablation boundaries, and any later optimizer grouping. The same separation applies to SwiGLU gate, up, and down projections.

All RMSNorm gains start at one and use FP32 reduction with `eps = 1e-5`. A final RMSNorm precedes tied unembedding. No logit soft cap or auxiliary z-loss is active in the reference; either requires a separate numerical contract.

### 3.3 Initialization and residual contract

Embeddings and ordinary linear matrices use the same seeded normal initializer with standard deviation `0.02`. Attention output and MLP down projections receive the additional factor `(2L)^(-1/2)`, where `L` is the total number of unique Prelude + Core + Coda decoder blocks in that model. RMSNorm gains start at one. Optional branch gates obey Section 7 rather than this initializer.

This rule is fixed for bring-up but must be rechecked at both target depths. It is not an assertion that initialization theories compose automatically with recurrent scaling, Muon, or hyper-connections.

## 4. Tokenizer and vocabulary ruling

### 4.1 Candidate family

Fit one byte-level BPE recipe at four candidate vocabulary sizes: **16K, 24K, 32K, and 48K**. The provisional configuration is **32K**, not a declared winner. The recipe, normalization, pre-tokenization, byte fallback, special-token inventory, training text manifest, seed, library version, and serialization hash must be identical except for vocabulary size.

Required properties are:

- every byte sequence round-trips without `<unk>`;
- normalization is explicitly bound and reversible effects are reported;
- explicitly textual control, padding, BOS, EOS, chat, and fill-in-the-middle symbols occupy a frozen reserved range; scratch, recurrent, callosal, and memory state remain latent and never consume token IDs;
- no evaluation or sealed text participates in fitting or in the selection screen; and
- whitespace, code, mathematics, non-Latin text, and pathological byte strings receive separate fertility reports.

### 4.2 Target-ratio screen

Vocabulary is selected on a frozen raw-text audit corpus by a predeclared Pareto/utility rule, not by familiarity with 32K and not by one proxy parameter calculation. For every candidate `V`, record at both the intended 4-core-block and 6-core-block geometries:

```text
N_V       = total encoded token count on the frozen raw-text corpus
BPB_V     = validation negative log-likelihood in bits per raw byte
C_V       = fraction of raw-text windows fitting in the 4,096-token context
P_vocab,V = tied embedding/unembedding parameters and share of total unique parameters
M_V       = measured peak training memory
L_V       = measured training and decode latency at the intended batch/sequence regime.
```

The utility must compare raw-text BPB and context coverage against vocabulary parameter share, memory, and latency. The quantitative indifference tolerances and weighting rule are a **gate to lock before the screen**; this contract does not invent them. Select the smallest Pareto-admissible vocabulary within those locked tolerances. If the Pareto set or the 4-core-block and 6-core-block rulings disagree, return the complete table for a design ruling; do not select the most flattering row.

It is explicitly invalid to choose a vocabulary solely because `V * 512` looks like an acceptable embedding share. That is a useful accounting term, not the target-ratio decision. Report token count and BPB in raw-text units so vocabulary candidates remain comparable.

### 4.3 What Engram may and may not repair

A regular Engram may reduce pressure on the dense network to relearn frequent local patterns. It does not retroactively make an over-fragmenting tokenizer efficient: `N_V`, context coverage, attention cost, and latency are measured before Engram is promoted. Tokenizer selection therefore precedes the Engram result.

## 5. Correct causal lane contract

The scratch path is a **causal lane indexed at every real token position**, not a single global scratch token that can read an entire teacher-forced sequence.

For observed prefix tokens `x_1,...,x_t`, the state used to predict `x_{t+1}` may depend only on that prefix and on retrieval records independently legal at time `t`:

```text
h_t^(k), s_t^(k) = Phi_k(x_<=t, h_<=t^(<k), s_<=t^(<k), memory_legal_at_t)
p(x_{t+1})        = Softmax(Unembed(FinalNorm(h_t^(T)))).
```

During parallel teacher forcing, both token and scratch attention masks are lower triangular. A scratch state at position `t` may read token/scratch states at positions `j <= t` at an already-computed substep; it may never read `j > t`. A cross-token persistent state, if later admitted, must be a prefix scan `m_t = G(m_{t-1}, h_t)` and may first affect the prediction of `x_{t+1}`. Appending a sequence-global scratch slot that reads answer tokens and then broadcasts backward is prohibited.

RoPE always receives the real token coordinate `t`. Recurrent visit `k`, lane identity, memory age, and expert identity are carried by separate embeddings or control features and never alter the RoPE index or KV-cache position.

The mandatory falsification test perturbs every suffix token after position `t` while holding the prefix fixed. All logits through `t` and all retrieval addresses used through `t` must remain exactly equal in FP32 reference mode, or tolerance-equal only where a documented nondeterministic kernel precludes bit equality. A failed suffix test is a stop-the-line result, not a training bug to work around.

## 6. Recurrent scaling and gradient propagation

The recurrent reasoning core is a weight-tied residual vector field. For a requested `T` visits,

```text
z^(k+1) = z^(k) + alpha_T F_theta(RMSNorm(z^(k)), s^(k), q^(k)),
alpha_T = c / T,                 k = 0,...,T-1.
```

`c` is a fixed horizon constant bound before training; the reference value is `c = 1`. The `T = 1` graph is the one-visit reference, not an empty bypass. Dense/no-recurrence identity is obtained by the explicit recurrence-off graph, not by redefining `T`.

If `||J_F||_2 <= L_F`, then

```text
||dz^(T)/dz^(0)||_2 <= product_k (1 + c L_F / T) <= exp(c L_F).
```

When `c L_F / T < 1`, the corresponding lower singular-value bound is `product_k(1 - c L_F/T)`. Because the same parameters are visited `T` times, their worst-case aligned first-order contributions scale as `T * alpha_T = c`, rather than linearly in `T`. These are local sufficient bounds, not guarantees for the full model: RMSNorm derivatives, attention, callosal writes, dynamic routers, and non-normal Jacobians remain empirical obligations.

The `c/T` law fixes total update horizon while refining the recurrent discretization. It may therefore limit test-time-compute gains if useful reasoning requires a longer horizon rather than a finer one. `1/sqrt(T)`, DeepLoop-style depth scaling, learned step size, or adaptive halting are successor arms only; they are not silently substituted after a result.

Before any optimizer is constructed, use JVP/VJP power iteration to estimate the largest singular value of each visit and the total unrolled map; also log per-visit update RMS, gradient RMS, gradient cosine/alignment, attention entropy, and finite-horizon gain. Spectral radius alone is insufficient because a non-normal Jacobian can exhibit transient amplification.

## 7. Optional-branch and zero-gate law

Every optional writer attaches as

```text
y = x + g F_theta(x).
```

Two distinct configurations are required and may not be conflated:

1. **Identity attachment:** `g = 0` exactly. It must reproduce the parent graph and proves inertness. At this point `dL/dtheta = 0`; that is expected.
2. **Trainable positive control:** the branch has nonzero output and a separately bound small positive gate, or the gate alone is first trained while the branch is frozen. It must produce finite nonzero gradients to every intended parameter group.

Initializing both `g = 0` and `F_theta(x) = 0` creates a dead branch: neither the gate nor branch weights receive a useful first update. That initialization is prohibited for a training arm. The receipt records gate values, branch-output RMS, parameter-gradient RMS, and optimizer membership before step one.

An off arm is the structural off graph whenever practical. A zero-gated executed graph is an engineering identity control, not automatically a compute-matched scientific control.

## 8. Two-lane bicameral carry and Corpus Callosum

### 8.1 Exact two-lane Birkhoff form

Let the dense/solver lane and scratch/critic lane have equal-width states `h` and `s`. Every two-by-two doubly stochastic nonnegative carrier has the exact form

```text
P(rho) = [[1-rho, rho],
          [rho, 1-rho]],          0 <= rho <= 1.
```

In the mean/difference eigenbasis

```text
m = (h + s) / sqrt(2),            eigenvalue lambda_+ = 1
d = (h - s) / sqrt(2),            eigenvalue lambda_- = 1 - 2 rho in [-1, 1].
```

Equivalently, `P = Q diag(1, lambda_-) Q^T`. This exact eigenbasis parameterization replaces iterative Sinkhorn for two lanes. It preserves the common mode, contracts or sign-flips only the difference mode, is closed under products, and satisfies `||P||_2 <= 1`. Identity is `rho = 0` / `lambda_- = 1`; averaging is `rho = 1/2`; swapping is `rho = 1`.

The full mathematical family permits `0 <= rho <= 1`. The first trainable carrier is restricted to `0 < rho < 1/2`: it is static, identity-favoring, and damps disagreement without a sign flip. Averaging and swapping remain exact intervention endpoints, not initial trainable states. A state-dependent `rho(X)` is a later arm because its Jacobian contains the extra term `(dP/dX)X`; Birkhoff membership alone then does not bound the full recurrence.

### 8.2 Directionality is not Birkhoff mixing

Two-lane Birkhoff transport is necessarily symmetric. It cannot represent a directional solver-to-critic or critic-to-solver Corpus Callosum. Directional messages therefore use separately named, low-rank residual writers:

```text
h' = P_h([h,s]) + g_{s->h} U_{s->h}(s)
s' = P_s([h,s]) + g_{h->s} U_{h->s}(h).
```

Each writer has its own zero-gate identity test, trainable positive control, parameter inventory, operator-norm telemetry, and ablation. Carry stability must never be claimed as stability of these innovation terms. “Corpus Callosum” refers to the directional writers; `P` is the bounded carrier. The first code substrate implements `P` only. It does not yet claim to implement the directional Corpus Callosum.

Lane semantics are hypotheses, not labels guaranteed by construction. Cross-seed lane analysis is performed in the mean/difference basis or with gauge-invariant measures; branch swapping is a required intervention.

## 9. Staged research pillars

Pillars are introduced in the following order. Passing a mechanical gate permits an experiment to be registered; it does not establish usefulness.

### P1. Regular Engram, before latent Engram

The first Engram is a causal lexical lookup keyed by already-observed token n-grams. For the logit at position `t`, its key may include tokens through `x_t`, never `x_{t+1}`. Values are fused immediately after the first causal contextual Prelude block through a gated residual; they are not queried from raw embeddings and do not move to another layer without a registered placement arm. The index, collision policy, n-gram orders, hash count, value dimension, self-entry exclusion, and OOV behavior are locked before training.

Required controls are: Engram off; address-shuffled values with identical table size and lookup count; parameter-matched dense input adapter; and latency/memory accounting. Latent learned keys, semantic retrieval, and cross-example writable memory wait until the regular lookup clears its own gate.

### P2. Upfront modified-Hadamard experts

For `d_model = 512`, use the normalized Walsh-Hadamard transform `Q = H/sqrt(512)` as a fixed orthogonal basis. A modified expert consists of learned diagonal modulation and permutation around `Q`/`Q^T`, with a causal per-token router and an output gate before block one. The first research lock must bind expert count, active experts, router regularizer, and exact formula; no MoE choice is implied by this architecture contract.

Required tests are dense-matrix forward and backward equivalence, `Q^TQ = I`, norm preservation, deterministic routing, load telemetry, and exact zero-gate identity. The scientific controls are a parameter-matched dense upfront adapter and a compute-matched dense FFN. The banked result that a Hadamard diagonal bank captured only a minority of a prior correction field is a boundary, not a prohibition: this pillar is an efficient lexical/feature expert hypothesis, not a claim that it spans oracle corrections.

### P3. Recurrent reasoning plus the causal scratch lane and bridge

Activate the Section 6 recurrent core and Section 5 lane together only after their individual identity and causality tests pass. The bridge is a named directional writer with exact off behavior. Register `T` values and the `c/T` law before training. Compare against both a same-parameter dense model and a compute-matched weight-tied repeated-dense control; tokens, active FLOPs, and wall time are all reported.

### P4. Bicameral specialization and Corpus Callosum

Fork the causal state into two equal-width lanes, apply the exact Section 8 carrier, and add directional callosal writers one direction at a time before the bidirectional composition. Required interventions are lane swap, one-lane lesion, each callosal direction off, difference-mode clamping, and a parameter-matched wider single lane. A bicameral win requires benefit over the wider single-lane and compute-matched repeated-dense controls, not merely over the smallest dense model.

The first code substrate stops before that promotion boundary: its two position-aligned scratch sublanes are a P3 causal-state mechanism, not evidence of P4 solver/critic specialization. It implements the bounded symmetric carrier as an independently switchable primitive, but it does not yet implement or name any directional writer as the Corpus Callosum.

### P5. Long-term memory and latent Engram

Start with a frozen, content-addressed read-only store. Queries at token `t` are derived only from prefix-safe states; retrieved records carry provenance and cannot include the held-out answer, the current row's own target, or later tokens. Training-time leave-one-out/self-entry exclusion is mandatory. Index construction, refresh schedule, write permissions, document split, and cache invalidation are receipted.

Only after the regular Engram and read-only memory pass may latent keys/values or writable episodic memory be registered. Controls include empty retrieval, shuffled values, wrong-key retrieval, matched random values, and a same-parameter dense adapter. Retrieval gains are reported with index memory, lookup latency, and context tokens displaced.

### P6. Clifford primitives

Clifford algebra enters as a low-dimensional, typed computation primitive, not as a replacement for the 512-dimensional hidden space. The first candidate uses a fixed small algebra and an explicit multiplication table, projects selected channels into multivector coefficients, applies the geometric product and named grade projections, then maps back through a gated residual. Full `Cl(512)` construction is prohibited because coefficient count is exponential.

The design lock must state signature, grades retained, coefficient dimension, multiplication convention, and parameter-matched bilinear/dense controls. The two-lane code uses the split-Clifford idempotent convention `mu=(h+s)/2`, `delta=(h-s)/2`, so `h=mu+delta` and `s=mu-delta`; Section 8's `/sqrt(2)` coordinates remain the orthonormal spectral convention. Consequently `||h||^2+||s||^2 = 2(||mu||^2+||delta||^2)`, which is tested rather than conflated with Parseval equality. FP64 tests must verify basis squares, anticommutation, multiplication-table signs, grade projection, associativity where the chosen implementation should preserve it, and autograd against finite differences. A shuffled-sign multiplication table is a useful structural null. No optimization benefit may be claimed from the algebra without measured conditioning, gradient, and efficiency evidence.

### P7. Jets: diagnostics first, controller later

Jets first describe recurrent trajectories:

```text
v_k = z_k - z_{k-1}
a_k = v_k - v_{k-1}
||v_k wedge a_k||^2 = ||v_k||^2 ||a_k||^2 - <v_k,a_k>^2.
```

Record step norm, turning cosine, wedge/Gram invariant, local JVP gain, and full-horizon gain by layer, token type, and outcome. Use basis-invariant quantities for cross-seed comparisons. The banked TM-0 active-token-mean jet estimator returned `ROTATION-ABSENT`; therefore no turning-plane router or jet-conditioned gate is in the initial architecture. Such a controller requires a fresh preregistered estimator, independent population, smooth-noise and no-rotation nulls, and an ordinary MLP controller matched for parameters and inputs.

## 10. Connection and layer-structure formalism

Represent the model as a typed, gated graph rather than an unconstrained list of skips. For lane-state matrix `X` at physical block `l` and recurrent visit `k`,

```text
X_{l,k+1} = P_l X_{l,k}
            + alpha_T sum_r b_{l,r} F_{l,r}(C_{l,r} X_{l,k})
            + sum_q g_{l,q} M_{l,q}(X_{l,k}).
```

`P_l` is the static two-lane carrier; `b` is a preregistered binary topology mask; `C` is a fixed typed read map; and `g M` is an explicitly gated directional message. Every edge therefore has an owner, direction, causal mask, scale, parameter count, and null.

Topology selection is a constrained registered screen, not free-form differentiable NAS:

```text
min_a  E_seed[BPB_dev(a)]
       + lambda_F log(active_FLOPs(a))
       + lambda_K log(KV_bytes(a))
       + lambda_L log(measured_latency(a))
```

subject to causal correctness, fixed unique-parameter bands, stability tripwires, and predeclared indifference tolerances. The lambdas/tolerances and candidate masks are locked before the screen. Only the 4/6 recurrent-core-depth candidates and a small, named Prelude/Core/Coda allocation grid are admissible initially. Search may prune candidates; it may not invent a new edge after seeing validation results.

## 11. Ablation and null contracts

Every research result must include the following hierarchy:

1. **Structural null:** module absent; parent graph unchanged.
2. **Attachment null:** module executed at exact zero gate; logits and parent gradients checked against the structural null.
3. **Positive control:** nonzero gate and nonzero branch output; every intended gradient path live.
4. **Parameter-matched control:** ordinary dense module with the same unique trainable-parameter band.
5. **Compute-matched control:** comparable active FLOPs/KV traffic and repeated computation where relevant.
6. **Content null:** shuffled address/value, lane pairing, multiplication table, or trajectory feature, chosen before results.

The first scientific sequence is `D`, `D+P1`, `D+P2`, `D+P3`, and only then registered pairwise interactions among pillars that independently pass. Bicameralism, memory, Clifford, and jet control enter after the causal recurrent/scratch foundation. The full “best of” model is evaluated only after every included pillar has a banked single-pillar result and its necessary pairwise interaction has been tested.

All arms use the same frozen tokenizer, raw-text split, data order, seed set, context policy, loss definition, and checkpoint schedule unless the differing field is the registered factor. Report unique/trainable parameters, tokens, active FLOPs, peak memory, KV bytes, and wall-clock time to the same target loss. A step-count comparison alone is invalid.

## 12. Muon boundary and open desk item

Muon is **not an architecture arm and not yet the reference optimizer**. AdamW remains the bring-up reference so architecture gates do not become optimizer-interaction experiments.

The open desk item is **mode-wise Muon in the two-lane eigenbasis**. It must answer, without model training:

- whether common-mode and difference-mode matrices are physically separate parameters or slices of one packed tensor;
- whether Newton-Schulz orthogonalization is applied separately per mode, per projection, or to a packed matrix;
- how post-normalization update multipliers implement any intended mode-specific learning rate;
- whether `alpha_T = c/T` is a forward scale only or is incorrectly expected to shrink a Muon-normalized update;
- update RMS, orthogonalization residual, transpose consistency, memory, and latency for square, tall, and wide matrices; and
- which tensors remain with AdamW: embeddings/unembedding, norm gains, scalars, gates, temperatures, and other non-matrix parameters.

Under an exact polar map, `polar(cG) = polar(G)` for positive scalar `c`; practical Newton-Schulz is only an approximation. Scaling a gradient slice inside a packed matrix is therefore not a reliable slice-local Muon learning rate. Mode-specific updates require separate tensors/groups or explicit multipliers **after** Muon normalization. The same caution applies, with different mechanics, to persistent positive gradient scaling under Adam-family moment normalization.

Only after the desk math, tensor-layout audit, FP64 SVD-polar reference tests, and microbenchmarks pass may Muon be proposed as a separately registered optimizer experiment. It is not added to the 4/6 architecture sweep and may not be selected because it wins one architecture's pilot.

## 13. Hard pre-optimizer gates

Do not construct an optimizer until every applicable gate is green:

1. tokenizer determinism, reserved IDs, byte round-trip, and frozen-manifest receipts;
2. dense FP32 forward/backward reference and SDPA parity;
3. suffix-perturbation causal-lane falsification;
4. RoPE relative-position identity and proof that visit/lane indices never enter token positions;
5. structural-off versus zero-gate attachment identity;
6. positive-control gradients for every intended module and boundary activation;
7. Engram next-token shift and self-entry exclusion;
8. Hadamard dense equivalence, norm, and gradient checks;
9. two-lane Birkhoff row/column sums, eigenvalues, product closure, and spectral norm;
10. recurrence requested/executed visit counts and `alpha_T = c/T` receipt;
11. per-visit and unrolled JVP/VJP finiteness and finite-horizon gain below a preregistered ceiling;
12. cache/recompute equality at every legal cache regime;
13. mixed-precision finiteness and BF16/FP32 gradient-cosine floor;
14. exact parameter inventory, optimizer allowlist, and tied-embedding pointer identity; and
15. deterministic checkpoint/RNG replay before any long run.

Any failed identity, causality, nonfinite, parameter-ownership, or replay check is a red engineering result. It cannot be averaged away across seeds.

## 14. Build and evidence order

1. Freeze the tokenizer audit corpus and lock the target-ratio utility tolerances.
2. Fit and screen 16K/24K/32K/48K tokenizers; freeze one serialized tokenizer.
3. Implement the two-block dense bring-up and pass all fixed-substrate gates.
4. Materialize both 4-core-block and 6-core-block target models, reporting 8 and 10 total unique blocks under the initial 2/Core/2 allocation; register their training comparison before any result.
5. Bring up P1 and P2 independently behind exact gates.
6. Bring up recurrent scaling, then the corrected causal scratch lane and bridge.
7. Add the exact two-lane carrier; then directional callosal writers one at a time.
8. Add read-only long-term memory; latent Engram follows only on evidence.
9. Run Clifford primitives as a typed low-dimensional arm and jets as diagnostics.
10. Compose only banked pillars under the ablation sequence in Section 11.
11. Keep the mode-wise Muon question on the desk until its separate mechanical gate is complete.

## 15. Research anchors and inherited boundaries

The conservative substrate follows the convergent choices documented in [Qwen3](https://arxiv.org/abs/2505.09388), [OLMo 2](https://arxiv.org/abs/2501.00656), [Grouped-Query Attention](https://arxiv.org/abs/2305.13245), and [RoPE](https://arxiv.org/abs/2104.09864). The two-lane carrier specializes the constrained carry principle of [mHC](https://arxiv.org/abs/2512.24880). The recurrent scale is a deliberately simple bounded-horizon starting point; [DeepLoop](https://arxiv.org/abs/2607.13491) is a successor scaling reference, not a compositional guarantee. The Muon desk item is anchored to the [large-scale Muon report](https://arxiv.org/abs/2502.16982).

Local empirical boundaries remain binding: regular n-gram lookup is only an Engram-style control primitive so far; Hadamard reachability was limited for the prior correction-field estimand; prior recurrent loops could move while output idled; and TM-0's primary jet estimator did not find reproducible turning planes. The new build may test new estimands, but it must not restate those boundaries as positive evidence.

---

**Contract line:** 32K is provisional; 16K/24K/32K/48K are all screened under a target-geometry utility locked before results. Two shared core blocks are engineering only; four and six shared core blocks are the first registered core-depth candidates. The dense causal substrate is fixed, every innovation enters behind a named null and positive control, two-lane carry uses its exact Birkhoff eigenbasis, recurrence uses `alpha_T = c/T`, and mode-wise Muon remains an open desk item rather than an arm.
