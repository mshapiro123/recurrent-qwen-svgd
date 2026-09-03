# STRATEGY — Nanbeige4.2-3B Adjudication: A From-Scratch Looped Transformer at 28 T Tokens, and a Negative Result Against Our Static K/V

**Date:** 2026-09-03 · **Status:** ADJUDICATION of the Nanbeige4.2-3B technical report (arXiv 2607.22083), located and read at Mark's request. **Critical path untouched**; one decision raised for Mark (D-NB-1, §4). Three architecture components named on the model card (LoopSplit, mHC with depth attention, concatenated n-gram embeddings) are **not described in the arXiv report** and the Hugging Face card could not be fetched without Mark's approval — §5 lists what is pending.
**Evidence classification (PS-1, O-7):** 3 B non-embedding / 4 B total, **28 T pretraining tokens**, trained from scratch as a looped model, weights released, Apple-Silicon deployment paper and independent blog coverage exist. The report is a *product* report: qualitative ablation statements, few numbers, no seeds. **Tier-2 for design priors, Tier-3 for any quantitative transfer.** Sources: the arXiv HTML (read in full by fetch), the deployment paper (arXiv 2608.13987), Raschka's commentary; the model card is pending.

---

## 0. Plain-language summary

Mark is right that this one matters, and right that we should have had it in the register. Nanbeige4.2-3B is the largest from-scratch looped transformer with a public report: the whole 22-layer stack is executed twice, so 3 billion non-embedding parameters do the work of a 44-layer network, trained on 28 trillion tokens. Three of its statements bear directly on WEFT-1.

The first is support. They trained the loop from scratch and report that upcycling a trained model into a loop performed "significantly better" the other way — from-scratch wins, because the representations adapt to layer reuse throughout pretraining. That is the argument this program made when it stopped retrofitting Qwen and built WEFT-1 from random initialization. It is also worth noting for the compute plan that they measured the loop as retaining "approximately 75% of the token efficiency" of a standard transformer while gaining capacity — a quantitative prior for what our matched-compute dense control should show.

The second is a caution we already had from GRT. Two passes was their best trade-off; more passes gave "marginal additional improvement" and slowed training substantially. GRT found 82 percent of the recurrent gain in the first two visits. Two independent programs at very different scales now say the same thing: the second visit is where the money is. Our K curriculum goes to four, and the loop-gain instrument will tell us whether visits three and four earn their cost — but the prior has firmed, and the honest branch "K = 2 is most of the win" should be treated as likely, not remote.

The third is the important one, and it cuts against a ratified choice. Nanbeige tested sharing the KV cache across the two passes — computing keys and values once and reusing them on the second pass — which is exactly WEFT-1's static K/V design, where keys and values are computed once from the prelude output and re-queried at every visit. They found the shared configuration's gains "consistently lower" than recomputing K/V on each pass, and kept the full loop despite the doubled cache. Our design chose static K/V for two reasons: cache economy at serving, and a design intent ("re-query a fixed context with an updated question"). Nanbeige is Tier-2 evidence, at 3 B and 28 T tokens, that the intent costs quality. It is not decisive for us — their loop is two passes over the whole stack, ours is four visits over a four-block core with lanes and a carrier doing work their model does not have — but it is the first external test of this exact axis and it went the other way. In training, the difference between the two forms is cheap (training recomputes regardless; only the *source* of K/V changes), so the right response is not to argue but to measure: a paired static-versus-live contrast at the proxy rung, first in S2, before the target run commits. Whether the first run's *default* should flip in the meantime is Mark's call, and it is put to him below.

---

# 1. What the report establishes

| Nanbeige4.2-3B | WEFT-1 | reading |
|---|---|---|
| Looped Transformer: the **whole 22-layer stack executed twice** (44 effective layers, 3 B non-emb / 4 B total) | prelude 9 / **core 4 tied, executed K** / coda 9; AE(K=4) = 440 M on 303 M unique | Different design point: Nanbeige doubles per-token compute for a full second read; WEFT adds ≈ 18 % per visit and keeps prelude/coda fixed. Both are "capacity without parameters"; WEFT's bet is that the *middle* is where recurrence pays. |
| **Trained from scratch as a loop; upcycling performed significantly worse** | from-scratch (Papers 1–3 were upcycles) | **External support** for the from-scratch decision, at 3 B / 28 T. Recorded. |
| **Two passes optimal; more passes marginal and slow to train** | K curriculum 1 → 2 → 4, K_max 8; `η_k` instrument | Agrees with GRT (82 % of gain in visits 1–2). Prior firmed: **expect most of the loop gain at K = 2.** The `η_k`-vs-reallocation instrument is the arbiter; the "K = 2 is most of the win" branch is now the base case, not the surprise. |
| **≈ 75 % of a standard transformer's token efficiency**, with "a significant capacity gain" | matched-compute dense control gets 64 % more data (D-CUR-4) | A quantitative prior for the control comparison: per-token learning is slower in the loop; the loop must win on capacity to win at all. Our control is well placed to measure exactly this. |
| **KV-cache sharing across passes: gains "consistently lower" than recomputing per pass**; full loop retained despite 2× cache | **§5.3 static K/V from h₀, reused every visit** (S-3/A1: shared consensus, static) | **Tier-2 negative against our ratified axis.** See §3. |
| Fixed two passes at inference; no adaptive depth | inference-controllable K, halting head | Consistent with GRT's beyond-horizon caution; Nanbeige simply does not go there. |
| n-gram embeddings concatenated at input (card) | M_lex engram, gated add at prelude block 1 | Third external adoption of n-gram memory at scale (Qwen 3.8, now Nanbeige). Site and combination differ (concat at embedding vs gated add at block 1) — an engram-sweep factor, pending the card's details. |
| mHC with depth attention (card) | lanes + Birkhoff callosum (§5.6.1 cites the doubly-stochastic mixing formalism) | If mHC is manifold-constrained hyper-connections (multi-stream residual with doubly-stochastic mixing), Nanbeige's residual is a multi-lane Birkhoff mixer — **the same family as our callosum.** Pending the card; see §5. |
| GSM8K 92.7 vs Qwen3.5-4B 84.4; GPQA 53.3 vs 43.1 (base) | — | Product results on 28 T tokens with a math/code-heavy mix; **not attributable to the loop** and not used as evidence for it (C-4/R-4 discipline). |

# 2. What it does *not* establish

No numbers for the loop ablations (loop count, KV sharing) — qualitative only; no seeds; no per-loop gain curve; no dims, heads, or vocabulary in the report; no stability recipe (init, scaling, LR) for the loop. The 75 % figure is unaccompanied by its comparison protocol. Everything quantitative here is a prior, not a measurement we can lean on.

# 3. The K/V finding against §5.3, and what to do about it

**The axis, precisely.** Nanbeige "sharing" = K/V computed on pass 1 and reused on pass 2. WEFT-1 static = K/V computed from h₀ (pre-loop) and reused on every visit. Same axis; ours is the *more* static of the two (Nanbeige's shared cache at least came from the loop's own first pass). Nanbeige "full loop" = K/V recomputed from the current pass's hidden states = the **KV-LIVE** form I recorded in A1 as "a named alternative, not registered."

**Why ours was chosen.** (i) Serving cache: static = 1×, live = K×. (ii) Design intent: the loop re-queries a fixed reading with an updated question; the lanes and carrier, not attention, carry the evolving state. Nanbeige has neither lanes nor carrier — in their model, live K/V is the *only* way pass 2 can see pass 1's work. In WEFT-1 the lanes and carrier are meant to do that job, so their finding may not transfer. That is a hypothesis, not a defense.

**Cost of finding out.** In *training* there is no cache (§5.3): static and live differ only in what feeds the K/V projections — h₀ or the current hemisphere state — so the compute difference is the K/V projections per visit (small). The serving-cache cost is real only at inference and only if live wins. **A paired static-vs-live contrast at the proxy rung is therefore cheap, and it belongs first in S2**, before the target run's K/V representation is frozen.

> **Registered: KV-LIVE (promoted from named alternative to pre-registered S2 contrast, first in order).** Arms: static K/V from h₀ (ratified) vs K/V recomputed per visit from each hemisphere's current state (paired-projection question inherits S-3: μ-only K/V from the *current* state in the live arm). Proxy rung, both seeds, matched tokens; primary read `η_k` vs reallocation and pooled BPB; secondary reads: retention gauge `r`, callosum `ρ̂(A,B)`, RESP-LEAK (does the loop's gain migrate into attention when K/V go live?). **Branches fixed now:** *live wins by > 2ŝ* ⇒ the target run adopts live K/V and the serving-cache cost is paid knowingly (Fork B′ midpoint refresh becomes the cache-economy arm); *no difference* ⇒ static stands and the "lanes and carrier carry the evolving state" hypothesis is supported — a differentiating finding vs Nanbeige; *static wins* ⇒ recorded as a scale- or topology-dependent reversal of Nanbeige's result. **KV-PAIR** (A1) is unchanged and orthogonal.

# 4. D-NB-1 — decision for Mark: the first run's default while the contrast is pending

*(a) Keep static K/V as the default; run KV-LIVE first in S2 (recommended).* Honors the ratified design and its intent; the contrast decides before the target run; no churn in the integration queue (step 2 integrates the static cache as ruled). *(b) Flip the default to live K/V now.* Follows the only external evidence on the axis; costs the serving cache at K×, changes step 2's cache semantics before the seam is integrated, and abandons the design intent untested. *(c) Fork B′ midpoint refresh as the default.* A compromise the handoff already registered (2× cache); still untested either way.

# 5. Pending the model card (fetch needs Mark's approval)

The card names **LoopSplit**, **mHC with depth attention**, and **concatenated n-gram embeddings**; none is described in the arXiv report, and the Hugging Face pages require approval to fetch. If mHC is DeepSeek's manifold-constrained hyper-connections — a multi-stream residual whose mixing matrices are projected onto the Birkhoff polytope — then Nanbeige's residual stream is the *multi-lane* generalization of our callosum's §5.6.1 reparameterization, and its depth attention is a read over earlier depths that our jets partially resemble. LoopSplit may be a partial-stack loop (a split point between looped and un-looped layers) — i.e., a prelude/core/coda decomposition like ours rather than the whole-stack loop the report describes. Both would matter; both are unverified until the card is read. **Mark: paste or approve `https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/main/README.md` and I will complete §5 with quotes.**

# 6. What does not change

Build queue, P-A/P-B, semantics chain, no-injection rules — unchanged. New: KV-LIVE promoted to first S2 contrast; two priors firmed (from-scratch; K = 2 base case); the 75 % token-efficiency prior recorded for the dense-control comparison.

---

*Signature block*

**Strategy:** the report is thin on numbers and thick on one thing we needed to hear: at 3 B and 28 T tokens, a loop that re-reads its own work beat a loop that re-queries a frozen reading. Our design bet that lanes and a carrier make the frozen reading sufficient — and that bet is now the first thing S2 tests, cheaply, before anything is frozen at target. The from-scratch and two-pass findings are welcome corroboration of choices already made.
**Coding agent:** nothing to build now; KV-LIVE enters the S2 registry as the first contrast with its branches; step 2 proceeds with the static cache as ruled unless D-NB-1 changes the default.
**Mark:** one decision, D-NB-1. And the card URL, so §5 can be finished from the source rather than inferred.
