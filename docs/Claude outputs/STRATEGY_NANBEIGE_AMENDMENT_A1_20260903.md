# STRATEGY — Nanbeige Adjudication, Amendment A1: The Model Card and Code Read; §5 Completed

**Date:** 2026-09-03 · **Status:** AMENDMENT A1 to the Nanbeige4.2-3B adjudication (12,518 B, SHA-256 `eb8ab8b1…93aad2d`). Completes its §5 from primary sources after Mark supplied the card URL; **corrects two rows of its §1 table**; records the released model's exact configuration; registers nothing on the critical path. Read with the D-NB-1 ratification record (live K/V default), which this amendment corroborates from the shipped code.
**Sources read (PS-1):** `README.md`, `config.json` (1.02 kB, quoted in full below), `configuration_nanbeige.py` (defaults, docstrings, validators), `modeling_nanbeige.py` (122 kB; the loop, LoopSplit, hyper-connection, depth-attention and n-gram classes quoted verbatim) — all at `huggingface.co/Nanbeige/Nanbeige4.2-3B`, revision `main` as of today. Corroborating third-party readings: the llama.cpp support PR (#25994, "`num_loops=2` on 4.2-3B"), the MIT-RLX Rust port README, the mlx-community 4-bit card. **Tier-1 for what the code does; the card says nothing about training.** Numbers below were computed before writing (`nanbeige_check_20260903.py`, 4 checks).

---

## 0. Plain-language summary

The card answered the open question, and the answer changes how the earlier adjudication should be read. The three features it names — LoopSplit, mHC with depth attention, concatenated n-gram embeddings — **are in the shipped code but are switched off in the released 3B model.** The `config.json` has none of their keys, so they take their defaults, and every default is "disabled." The released Nanbeige4.2-3B is exactly what the arXiv report describes and nothing more: a 22-layer stack run twice with shared weights, each pass keeping its own keys and values. The MIT-RLX port calls the three features "Nanbeige4.5 features," which is a third party's label, but it fits the evidence: this is the *next* model's architecture, published ahead of the model.

That makes the code a different kind of evidence than I treated it as. It is not a description of a trained system; it is a **design disclosure**. And the design it discloses is, piece for piece, the shape of WEFT-1. LoopSplit keeps the outer layers unlooped and repeats a contiguous middle block — a prelude, a recurrent core, a coda. Its hyper-connections can *double the number of residual streams inside the looped block only* — extra lanes that exist in the core and are contracted away at the exit, which is what our lanes are. Its mixing matrices are projected onto the doubly-stochastic set — the constraint family our callosum's §5.6.1 reparameterization is the two-lane closed form of. Its "depth attention" lets the looped block read value states cached from earlier, un-looped depths — a fixed anchor set read by a live query, which is the *midpoint* between the static and live K/V policies we just argued about. Its n-gram module is a **gated add at chosen layers, gated by a normalized query–key dot product** — the engram gate we ratified this morning, with one twist in the temperature. And it carries an intermediate-loop loss whose weights must sum to at most one, the same object as our `L_stage`. None of this was copied in either direction; the two designs were made independently, from the same literature, and arrived at the same skeleton. That is worth saying plainly: **convergent design is evidence the skeleton is sane. It is not evidence it works** — Nanbeige has not published a model that uses any of it, and neither have we.

Two corrections to my own record follow from the read. The §1 table implied that n-gram embeddings and mHC are *in* Nanbeige4.2-3B; they are not — the third-external-adoption note for n-gram memory is downgraded to "third external *implementation*." And the K/V finding is now anchored to code rather than prose: the shipped model's cache index is `layer_idx + loop_idx × num_hidden_layers` — separate slots per pass, i.e. **live K/V — the policy Mark chose this morning is the one Nanbeige ships.** The ablated "shared" arm exists in the code too, and it is *not* our static arm: it reuses the K/V from the looped block's *first execution*, not from a pre-loop state. That is closer to our Fork B′ family than to KV-STATIC, and I note the cheap way to make the S2 contrast directly comparable.

Three small things were checked numerically and are recorded as findings, none of which changes WEFT-1: LoopSplit always produces effective depth exactly 2L whatever the split, so it is a compute-matched ladder over *where* the repeats go, not a way to add depth; their Sinkhorn projection at twenty iterations is doubly stochastic only approximately (spectral norm up to 1.006 on random logits), whereas our two-lane form is exact by construction — a point in favor of §5.6.1 and a caution for the WEFT-2 multi-lane seed; and their gate's signed-square-root compression cuts saturation at a badly scaled logit from 46 % to 3 %, but has an unbounded derivative at zero — an alternative to EG-1, not a replacement.

---

# 1. The released model, from `config.json`

```json
{ "architectures": ["NanbeigeForCausalLM"], "head_dim": 128, "hidden_act": "silu",
  "hidden_size": 3072, "intermediate_size": 10752, "kv_channels": 128,
  "loop_loss_weights": [], "max_position_embeddings": 262144,
  "num_attention_heads": 48, "num_hidden_layers": 22, "num_key_value_heads": 8,
  "num_loops": 2, "rms_norm_eps": 1e-05, "rope_theta": 70000000,
  "skip_loop_final_norm": false, "tie_word_embeddings": false,
  "torch_dtype": "bfloat16", "vocab_size": 166144, "initializer_range": 0.02 }
```
(Non-architectural keys omitted; nothing else architectural is present. In particular **absent:** `enable_double_loop_split`, `loop_middle_layers`, `loop_share_kv`, `enable_hyper_connection`, `enable_mhc`, `num_residual_streams`, `enable_depth_attention`, `emb_neighbor_num`, `emb_split_num`, `ngram_vocab_size_ratio`, `insert_ngram_layer_idx` — all of which default to disabled/`None` in `configuration_nanbeige.py`.)

| | Nanbeige4.2-3B (shipped) | WEFT-1 rung A |
|---|---|---|
| unique layers / executed depth | 22 / 44 (`num_loops: 2`, whole stack) | 22 / 34 at K=4 (9 + 4·4 + 9) |
| d, d_ff, d_ff/d | 3072, 10752, **3.5** | 1024, 2816, 2.75 |
| heads Q/KV, d_head | 48 / 8, 128 | 16 / 8, 64 |
| vocabulary, readout | 166,144, **untied** | 32,768 (provisional), tied |
| RoPE θ, context | 7e7, 256 K | 5e5, — |
| K/V across passes | **separate slots per pass** (`cache_layer_idx = layer_idx + loop_idx·num_hidden_layers`) | `kv_policy = live` (D-NB-1) |
| intermediate-loop loss | implemented (`loop_loss_weights`, `sum ≤ 1.0`); **empty in the released config** | `L_stage`, sampled decode (D-MC-1) |
| LoopSplit / mHC / depth attention / n-gram | **implemented, disabled** | prelude-core-coda / lanes + callosum / — / engram |

Docstring of record for the loop: *"Number of times the complete decoder-layer stack is executed with shared parameters. Increasing this value increases the model's effective depth and FLOPs without adding a separate set of decoder-layer weights."* And for the loss: *"Weights associated with intermediate loop outputs during multi-loop training. When non-empty, the model executes `len(loop_loss_weights) + 1` loops instead of using `num_loops`."* The validator `sum(loop_loss_weights) must be <= 1.0` is the same normalization D-MC-1 imposes (`Σ w_k = 1` in expectation).

# 2. §5 completed — the four disclosed components, verbatim, and their WEFT-1 twins

## 2.1 LoopSplit — a prelude/core/coda decomposition with a compute invariant

Docstrings: *"Whether to enable LoopSplit. LoopSplit keeps the outer decoder layers unlooped and repeatedly executes a contiguous middle block."* — *"Number of contiguous middle decoder layers repeatedly executed by LoopSplit. Must be a positive factor of `num_hidden_layers`."*

```python
first_unlooped_layers = (num_hidden_layers - loop_middle_layers) // 2
middle_start = first_unlooped_layers
middle_end = middle_start + loop_middle_layers
middle_repeats = (num_hidden_layers + loop_middle_layers) // loop_middle_layers
return ([(idx, None) for idx in range(0, middle_start)]
      + [(idx, repeat_idx) for repeat_idx in range(middle_repeats) for idx in range(middle_start, middle_end)]
      + [(idx, None) for idx in range(middle_end, num_hidden_layers)])
```

**Finding (verified).** Executed depth is `(L − M) + (L/M + 1)·M = 2L` **for every admissible M**: at L = 22, M = 11 gives prelude 5 / core 11 × 3 / coda 6; M = 2 gives 10 / 2 × 12 / 10; M = 22 is the shipped whole-stack ×2. LoopSplit is therefore a **compute-matched ladder over where the repeats go**, at fixed 2L, not an instrument for adding depth. That is the axis our rungs A (9/4/9, K=4) and B (8/6/8) sample coarsely, and the axis the `η_k`-vs-reallocation instrument measures. **No change**; recorded as external convergence on the split, and as a cleaner way to phrase the rung comparison ("at matched executed depth, how narrow may the looped block be?").

**Shared K/V, precisely.** *"Whether repeated executions of a LoopSplit middle layer reuse the key and value states produced by that layer's first execution."* (`loop_share_kv`, requires LoopSplit.) In code: `skip_cache_update = use_loop_shared_kv and loop_share_kv_repeat_idx > 0`. So Nanbeige's ablated "shared" arm freezes K/V **after the looped block's first visit**, from the loop's own state — not from a pre-loop state as our KV-STATIC does. Our static arm is *more* static than theirs; the closest analog of theirs in our vocabulary is a `first` policy (compute at visit 1, reuse thereafter).

> **Recommendation, non-binding, zero cost if trivial:** allow `kv_policy = first` as a fourth value of the ratified switch (`{live, static, midpoint, first}`), default unchanged (`live`). It is one branch in the K/V source selection. It is **not** an S2 arm; it exists so that, if the live-vs-static contrast is inconclusive, the exact Nanbeige axis can be run without a spec change. The agent may decline if it is not a one-line addition.

## 2.2 Hyper-connections / mHC — multi-stream residual with Sinkhorn mixing, doubled in the loop

Docstrings: *"Whether to replace the standard single residual path with multiple residual streams connected around each attention and MLP sublayer."* — *"Whether to use manifold-constrained hyper-connections (mHC), which constrain the learned residual-stream mixing matrices with Sinkhorn normalization."* Defaults when enabled: `num_residual_streams=4`, `mhc_sinkhorn_iterations=20`, `mhc_init_gating_factor=0.01`.

The mechanism, from `NanbeigeHyperConnectionModule`: per sublayer and per token, a zero-initialized linear map from the concatenated streams produces logits for (i) `h_pre` — sigmoid read-weights selecting which streams feed the sublayer, initialized one-hot on `layer_idx % n` (**round-robin: each layer reads a different stream by default**); (ii) `h_post` — `2·sigmoid` write-weights to every stream, initialized 1; (iii) `h_res` — an `n × n` residual mixing matrix, `SinkhornKnopp.apply(h_res_logits, 20)`, initialized to the identity via bias `+20` on the diagonal and `−20` off it. Logits are multiplied by `r = 1/(RMS(h) + ε)` — scale-invariant — and by `alpha = 0.01`. The Sinkhorn backward is a true differentiation through the normalization (custom `autograd.Function`).

Two loop-specific switches are the interesting part. `mhc_double_stream_position_for_loop ∈ {"mid", "edge"}`: *the looped middle layers (or, alternatively, the outer layers) run with `2 × num_residual_streams`* — the looped block carries twice the residual streams of the outer layers, with a stream-count conversion at the boundary (the conversion code was not quoted verbatim by the fetch; the per-layer stream-count rule above was). And `mhc_diff_for_loop`: the middle layers hold a **separate hyper-connection module per loop index** (`self_attn_mhc_loop_hcs[loop_idx − 1]`, `mlp_mhc_loop_hcs[…]`) while attention and MLP weights stay shared — **per-visit untied mixing, tied compute.**

**Mapping to WEFT-1, with the differences stated.** (a) Streams doubled *inside the loop only* = our lanes (2 × d/4 alongside h_A/h_B in the core, `bridge_in`/`bridge_out` at the seams). (b) Sinkhorn-projected `h_res` = the Birkhoff constraint of §5.6.1; for two streams it collapses to our `A(ρ) = (1−ρ)I + ρP` with eigenvalues `{1, 1−2ρ}` (verified). (c) **Where they differ:** Nanbeige's mixing is **input-conditioned per token**; §5.6.1 rules ours **static** for exactly the reason the handoff gives — an input-dependent `A` adds a `(∂A/∂X)X` Jacobian term and double stochasticity no longer bounds the network Jacobian. Nanbeige accepts that; we deferred it. (d) **Finding (verified):** at 20 Sinkhorn iterations on random logits of std 3, the worst row/column-sum error is 4.4 × 10⁻² and the spectral norm reaches **1.006** — their Birkhoff bound is *approximate*; our two-lane form has `‖A‖₂ ≤ 1` exactly. Recorded in favor of §5.6.1 and **carried to the WEFT-2 seed register (item 8, next seed revision):** a multi-lane callosum needs either an exact parameterization (a convex combination of a fixed permutation set, which is exact) or a certified iteration count, before its bound is cited in any certificate. (e) `mhc_diff_for_loop` — per-visit mixing parameters on a tied core — is **named `VISIT-MIX`**: per-visit tables for `ρ_b` (8 × K params) and the combiner's `θ_b`. Not registered; it is the residual-mixer analogue of MEM-SYN-STATIC (a visit-indexed table) and, if that control ever wins, this is the cheapest place to spend the win. First run unchanged.

## 2.3 Depth attention — a live query over anchored earlier-depth values

Docstring: *"Whether to enable depth attention. At each decoder layer, the current query selects and mixes value states from cached anchor depths."* Defaults: `depth_attention_stride=None`, `depth_attention_recent_window=0`, `depth_attention_static_anchor_once=True`; validator: *LoopSplit with depth attention requires `static_anchor_once=True`.*

```python
keys   = [key for key, _ in source_kv] + [current_key]
values = [value for _, value in source_kv] + [current_value]
logits = (query_for_kv.unsqueeze(0).float() * key_stack.float()).sum(dim=-1)
depth_probs = torch.softmax(logits * softmax_scale, dim=0)
return (depth_probs.unsqueeze(-1) * value_stack).sum(dim=0)
# … if layer_idx % config.depth_attention_stride == 0: depth_attention_kv_cache.append((layer_idx, key_states, value_states))
```

So: per token and per head, a softmax **over depth** (anchor layers at a stride, plus the current layer) using the current query against each depth's key at the *same position*, producing a mixed value that then enters ordinary sequence attention. Under LoopSplit the anchors are cached **once** — the loop reads a fixed set of earlier-depth values with a live query. That is precisely a **hybrid of KV-STATIC and KV-LIVE**: live keys/values for the sequence, plus a static, prelude-derived value bank selected by the current query. In our terms it is a `midpoint`-family policy with the mix learned per token rather than fixed by visit. **Named `DEPTH-ANCHOR`** as a value of `kv_policy` for the record; not registered — the S2 live-vs-static contrast decides first, and if static ≈ live this arm has nothing to add, while if live ≫ static it is the natural cache-economy candidate alongside Fork B′. No-injection status: the anchors are the hemisphere's *own* prelude/early-visit states — the same source KV-STATIC already reads — so it is admissible under R-3 without a new ruling; it must never read the other hemisphere's anchors (§5.6, one channel).

## 2.4 N-gram embeddings — a gated engram add at chosen layers

Docstrings: *"Maximum N-gram length for N-gram embeddings"* (`emb_neighbor_num` = n), `emb_split_num` = k tables per order, `ngram_vocab_size_ratio` × `vocab_size` = base table size m, tables of size `_ngram_embedding_vocab_sizes(m, k(n−1), force_prime)` (consecutive primes above m if `ngram_mod_force_prime`), hash `ngram_ids = input_ids + Σ_{j=2..n} shifted_ids[j] · vocab_mods[j−2]` (polynomial rolling hash, per-order modulus). Two fusion modes at the input: `"average"` — per-table projections summed and averaged with the token embedding, `x = (x + ngram·k(n−1)) / (1 + k(n−1))`; `"concat"` — raw table embeddings concatenated and passed through one `concat_proj` to `hidden_size`, then **added** to the token embedding (`x = x + ngram_embeddings`). Either may be skipped at the input (`skip_ngram_for_input`) in favor of layer insertion.

The layer-insertion module is the one that matters to us:

```python
class NanbeigeNgramLayerFusion(nn.Module):           # at insert_ngram_layer_idx (or all layers)
    def forward(self, hidden_states, ngram_embeddings):
        key = self.key_proj(ngram_embeddings); normed_key = self.ngram_norm(key)
        normed_hidden = self.hidden_norm(hidden_for_gate)          # optional down-projection to fusion_size
        gate = (normed_hidden * normed_key).sum(dim=-1, keepdim=True) / math.sqrt(self.fusion_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid()
        fused = gate * self.value_proj(ngram_embeddings)             # optional output_proj back to hidden_size
        return hidden_states + fused
```

**Read against EG-1.** Same object: `h ← h + g · W_V e`, with `g = σ(⟨RMSNorm(h′), RMSNorm(W_K e)⟩ / √d_f)` — a normalized query–key dot over a fusion width `d_f` (optionally reduced by `ngram_layer_downproject_size`), at a chosen layer. This is our **form B/C** (learned key map, normalized both sides) at a reduced width — i.e. within the rank-bounded family the engram-gate ratification showed to be equivalent; their `d_f` plays our `d_m = 64` when the down-projection is set. Two differences. (i) **Site:** they allow the add at any set of layers (`insert_ngram_layer_idx`) or all; ours is bound to prelude block 1 (M_lex) and to post-loop LTM, and the no-injection rule forbids it inside the loop — their design would permit an in-loop n-gram write, which ours deliberately does not. (ii) **Temperature:** the signed square root `sign(z)·√|z|` before the sigmoid. **Finding (verified):** at a well-scaled logit (std 1) it changes nothing (saturation 0.3 % → 0.0 %, gate std 0.208 → 0.203); at a catch-#37-scaled logit (std 4) it cuts saturation from **46 % to 3 %**. It is a robustness device against exactly the defect catch #37 named. Its cost: the derivative `∂g/∂z = σ′ / (2√|z|)` is unbounded at `z → 0` (12.5 at `|z| = 10⁻⁴`; their `clamp_min(1e-6)` caps it at 125). **Recorded as an EG-1 alternative (`EG-1-SQRT`), not adopted:** EG-1's trainable RMSNorm gains already own the temperature, and a bounded-derivative gate is preferable in a module that sits on the certified prelude path. If the engram sweep's gate-selectivity diagnostic ever shows saturation, this is the first alternative to try.

**Correction to the parent §1 table.** Rows 7 and 8 ("n-gram embeddings concatenated at input (card)", "mHC with depth attention (card)") are amended to **"implemented in `modeling_nanbeige.py`; disabled in the released 4.2-3B config."** The "third external adoption of n-gram memory at scale" line becomes "third external *implementation*; not in a released model" — Qwen 3.8's remains the scale adoption of record.

# 3. What this does to the earlier adjudication

| item | before A1 | after A1 |
|---|---|---|
| K/V finding (§3, D-NB-1) | report prose: shared "consistently lower" | **anchored in shipped code:** separate K/V slots per pass; `loop_share_kv` = freeze after *first execution of the looped block* (≈ our `first`, not our `static`) |
| from-scratch, two-pass priors | Tier-2 | unchanged |
| n-gram / mHC rows | implied present | present in code, **disabled** in 4.2-3B |
| §5 | pending | complete; three named alternatives (`VISIT-MIX`, `DEPTH-ANCHOR`, `EG-1-SQRT`), one non-binding switch value (`first`), one WEFT-2 seed item (exact multi-lane Birkhoff) |
| the design-convergence observation | — | **new:** the disclosed next-generation Nanbeige is a prelude/core/coda loop with core-only extra streams, Birkhoff-constrained mixing, per-visit mixing tables, anchored depth reads and a gated n-gram add — WEFT-1's skeleton, arrived at independently. Evidence of sanity, not of success. |

# 4. What does not change

Build queue (step 2 with `kv_policy = live`), P-A/P-B, semantics chain, EG-1, D-MC-1, the no-injection rules — unchanged. Nothing enters S2 from this amendment. Nothing for Mark to decide.

---

*Signature block*

**Strategy:** the card was worth the fetch for one sentence's worth of fact — the three features are off in the shipped model — and for what the code disclosed instead: a second team, independently, drawing our diagram. I have recorded that as encouragement and refused to record it as evidence. The concrete yields are small and exact: the K/V flip now rests on code, the Nanbeige "shared" arm is `first` not `static`, LoopSplit is a 2L-invariant ladder, Sinkhorn-20 is not exactly doubly stochastic, and the signed-sqrt gate is a good trick with a bad derivative.
**Coding agent:** nothing required. Optional, zero-cost if it is one line: `kv_policy = first`. Note `EG-1-SQRT` in the engram-sweep alternatives list; note `VISIT-MIX` and `DEPTH-ANCHOR` in the MEM-OP / K/V alternatives lists as named, unregistered.
**Mark:** nothing to decide. The card is now read; Nanbeige is closed unless 4.5 ships.
