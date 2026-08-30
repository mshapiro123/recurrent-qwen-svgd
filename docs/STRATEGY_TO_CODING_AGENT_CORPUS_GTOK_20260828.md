# STRATEGY → CODING AGENT — Corpus Pipeline, Freeze, and G-TOK Execution

**Date:** 2026-08-28 · **Programme: WEFT-1**
**Governing:** build handoff (61,329 B, SHA `498f34b5…eb02`) as amended by the ratification record (13,908 B, SHA `c5df7429…6d3a`), the G-TOK rulings (19,442 B, SHA `167fc17d…f0d2`), the English-scope decision (11,075 B, SHA `19399342…02d3`), the engram/tokenizer note (25,351 B, SHA `0221545d…65b5`), curriculum r2 (27,495 B, SHA `14f0ba5d…ea22`), the Qwen adjudication r2 (15,798 B, SHA `6c2568d5…0a1f`), and the curriculum decision record (8,354 B, SHA `61fc7727…8b6d`). Where this handoff consolidates and any of those disagree with it, **report the disagreement as a defect; do not reconcile silently.**
**Authority granted here:** this is the first **run-axis** authorization since S0. It covers exactly two things — the corpus pipeline runs (materialization, dedup, decontamination), and the **G-TOK training exception** as bounded in §6. Nothing else on the run axis is authorized: no optimizer partitions beyond flat AdamW, no checkpoint retention, no sealed-battery access outside the hermetic screen, no target-run training.
**Verify before building:** the bytes and SHA of this document against the delivery receipt. If either disagrees, stop and report.

---

## 0. Plain-language summary

All curriculum decisions are ratified and your queue is now full through the tokenizer decision. The work is one pipeline with three checkpoints: build the corpus, freeze it behind gates, run the tokenizer screen.

The corpus is about forty gigabytes of text drawn from two sources — Ai2's Dolma 3 for everything, plus HuggingFace's FineWeb-Edu for part of the web slice — in four fixed proportions: 45% general text, 25% code, 15% mathematics, 15% science. Both sources are under the same license. Because both web slices descend from Common Crawl, they will overlap, and you must deduplicate between them before the freeze and report how much you removed. Everything is pinned to exact upstream commits, materialized to local shards, hashed shard by shard, and described in a manifest whose own hash becomes the corpus's identity. After materialization the pipeline goes offline and never touches the network again.

Three rules protect the corpus from plausible-looking mistakes. English-only filtering happens at the document level and only on the general stratum — never on code, math, or science, and never by stripping bytes, because a math corpus with the Greek deleted is not a math corpus; a gate at the freeze fails if the math or code strata contain zero non-ASCII bytes, which would prove a filter did exactly that. Decontamination inherited from upstream counts for nothing — our own hermetic screen runs against our sealed batteries on the exact shards we hashed. And no length bucketing anywhere, because sorting documents by length silently sorts them by kind.

The tokenizer screen then trains eight small throwaway models — four vocabulary sizes, two seeds each — on identical byte streams, and picks the vocabulary by bits-per-byte with a pre-registered decision rule that leans toward smaller vocabularies. Every hyperparameter, the tie rule, and what each outcome means are already fixed, so the screen cannot be argued with after the numbers arrive. It costs about six GPU-hours with a hard stop at twelve. When it finishes, the vocabulary freezes, and the program moves to the calibration stage.

One piece is deliberately not in this handoff: the reasoning-trace tail that enters late in training. Its sources need a license review that Mark owns, so it gets its own freeze later. Do not block the main freeze on it.

---

# 1. What this handoff covers

| phase | what | axis | gate to pass |
|---|---|---|---|
| **P-A** | corpus materialization: pin → select → dedup → sample → shard → hash → manifest | run (pipeline only) | manifest complete, reproducibility test D1 |
| **P-B** | freeze: C1–C3 gates + hermetic decontamination | run (pipeline only) | all gates green; a decontamination hit **stops the line** |
| **P-C** | G-TOK: 4 arms × 2 seeds, decision, freeze `V` | run (G-TOK exception) | decision rule §6.7; tripwire 12 A100-hr |

Out of scope, explicitly: the P4 reasoning tail (separate freeze, gated on Mark's license review); all S2+ run-axis work (μP calibration, Power-LR, recurrence sweep — staged after `V` freezes); anything consuming the frozen vocabulary before it exists.

# 2. The corpus specification

**Total: 10¹⁰ tokens ≈ 3.8×10¹⁰ bytes** (planning fertility 3.8 B/tok; the manifest records bytes — bytes are the ground truth, token counts are estimates until `V` freezes).

| stratum | share | byte target | source(s) | selection |
|---|---|---|---|---|
| general | 45 % | 1.71×10¹⁰ | **~22 % Wikipedia/Wikibooks (Dolma 3)** + **~39 % Dolma 3 web top-quality bucket** + **~39 % FineWeb-Edu** | engram-dense: entity-rich, reference-style; language-ID ≥ 0.9 English at document level |
| code | 25 % | 0.95×10¹⁰ | StackEdu (Dolma 3) | top of quality distribution; **no language filtering** |
| mathematics | 15 % | 0.57×10¹⁰ | FineMath 3+ (Dolma 3) | highest tier; **no language filtering** |
| science/technical | 15 % | 0.57×10¹⁰ | arXiv (Dolma 3) first, olmOCR PDFs to fill | **no language filtering** |

The general-stratum split (1.0 B / 1.75 B / 1.75 B in token terms) is byte-proportional within the stratum. Every stratum is oversupplied 8×–1,800×: take the top of each quality distribution and stop. **No document repeats. No epoch logic anywhere.**

**Upstreams and pinning.** `allenai/dolma3` (subsets: web, Wikipedia/Wikibooks, StackEdu, FineMath 3+, arXiv, olmOCR) and `HuggingFaceFW/fineweb-edu`. Pin each with `load_dataset(..., revision="<commit-sha>")` — never a branch name — and record both resolved SHAs in the manifest. **Hash what you materialize, not what upstream claims**: large corpora reference external files not covered by the repo SHA. After materialization the pipeline is offline.

# 3. Filtering rules — binding, from D-G-6

1. Language selection is **document-level, general stratum only**: language-ID with ~0.9 English retention threshold; a document is kept or dropped, never edited.
2. Code, mathematics, and science strata are **not language-filtered at all**.
3. **No byte-class filtering anywhere, at any stage.** No ASCII restriction, no script stripping, no transliteration, no lowercasing, no NFKC. Incidental non-English inside retained documents stays verbatim.
4. **No length bucketing, sorting, or ordering by document length** (C-1). Shard-level shuffle with a recorded seed from the data pipeline's own `torch.Generator` (O-9 registry).

# 4. Cross-source deduplication — new obligation from D-CUR-2

Dolma 3 web and FineWeb-Edu both descend from Common Crawl; overlap is expected, and duplicated documents would double-count exactly the highest-quality web text.

**Procedure, in order:** (1) exact-match pass — SHA-1 over whitespace-normalized document bytes, both slices; (2) near-duplicate pass — MinHash/LSH over byte-level 13-gram shingles, Jaccard ≥ 0.8 (reuse the decontamination machinery; no tokenizer exists yet, so shingles are bytes, not tokens); (3) on any match, **drop the FineWeb-Edu copy** — Dolma 3 is canonical, so the dedup direction is deterministic; (4) top up the FineWeb-Edu slice from its quality-ranked remainder to restore the byte target; (5) **report in the manifest**: exact-dup rate, near-dup rate, bytes dropped, bytes topped up. Deduplication must be deterministic under the recorded seed (test D2).

# 5. Manifest, gates, and decontamination

## 5.1 Manifest (the corpus identity)

Per shard: path, byte count, SHA-256, stratum, source. Per source: upstream ID, resolved commit SHA, selection criteria, byte total. Per stratum: byte total, document count, **non-ASCII byte fraction** (C3), language-ID threshold applied or "none". Global: dedup report (§4), shuffle seed, pipeline code version, creation timestamp. Then **SHA-256 over the manifest itself** — this hash is the corpus identity and travels in every composition receipt from now on.

## 5.2 Freeze gates — all must pass, in this order

| gate | assertion | on failure |
|---|---|---|
| **C1** | non-zero non-ASCII byte count in **mathematics** and **code** strata | **freeze fails** — a zero proves a byte filter ran somewhere |
| **C2** | exact byte round-trip on the fixture set: Greek, accented Latin, CJK, RTL text, typographic punctuation, tabs, mixed indentation | freeze fails |
| **C3** | per-stratum byte counts and non-ASCII fractions recorded in manifest | freeze fails |
| **DECON** | hermetic screen (salted hashes + MinHash/LSH, no plaintext leaving the screen) against **every sealed battery**, run **on the materialized shards** — inherited decontamination credited for nothing | **a hit stops the line** and returns to strategy |

## 5.3 Reproducibility

**Test D1:** re-running selection from the pinned revisions with the recorded seeds reproduces the shard hashes exactly. This is the receipt that the manifest describes a *procedure*, not just an artifact.

# 6. G-TOK execution — the protocol as amended, consolidated

Everything below is already ratified; this is the single consolidated statement to run from.

**6.1 Authorization bounds (the training exception).** Training solely for tokenizer selection: S0 architecture only (every WEFT module structurally OFF), `d = 512`, **≤ 4×10⁹ raw bytes per arm**, no sealed-battery data, no checkpoint retained beyond the screen's own evaluation, no optimizer partition beyond flat AdamW. **Muon is prohibited in G-TOK.** Any run outside these bounds requires separate authorization.

**6.2 Screen corpus.** A designated, stratified **4×10⁹-byte subset of the frozen corpus** (same 45/25/15/15, drawn under the recorded seed, listed in its own sub-manifest with a SHA), plus a disjoint **2 % held-out slice**, stratified identically, frozen with the manifest, used for every arm's BPB. Identical byte stream across arms — token counts will differ; that is the phenomenon under measurement.

**6.3 Arms.** `V ∈ {16,384 · 24,576 · 32,768 · 49,152}`, exactly `V` IDs including ~64 reserved protocol tokens.

**6.4 Model.** Registered proxy `4/2/4` with the recurrent core structurally OFF: eight dense blocks, `d = 512`, 8Q/4KV, `d_ff = 1408`, pre-RMSNorm, QK-RMSNorm, SwiGLU, tied embeddings, RoPE θ = 500,000, context 2048. All WEFT modules OFF.

**6.5 Optimizer and seeds.** AdamW, **identical hyperparameters across all arms — no per-arm learning-rate correction for embedding size** (catch #16: μP does not correct for `V`; the undertrained-row cost is part of what the screen measures). 2 seeds per arm, varying init and data order. **A seed split — the seeds disagreeing on arm ordering — escalates to strategy and is never averaged away.** No absolute BPB number from the screen is quotable outside it; only orderings and gaps.

**6.6 BPE recipe (R-G4f) + VOCAB-EXT.** Deterministic byte-level BPE; all 256 byte values present; no reachable `<unk>`; distinct BOS/EOS/PAD/doc-boundary/FIM/chat-role tokens; no irreversible normalization; single-digit number tokens; explicit multi-space and indentation tokens; regex pre-tokenization splitting on script and Unicode-category boundaries; trained on the screen corpus; BPE dropout and stochastic segmentation OFF; deterministic merge order with recorded seed and **SHA over the merge table**. **VOCAB-EXT is binding:** merge table, its SHA, corpus manifest, and pre-tokenizer regex are preserved as the extension basis; any future extension is append-only continuation; **existing token IDs are never renumbered.**

**6.7 Measurement and decision.**

```
BPB = ( Σ NLL in nats over the held-out slice ) / ( ln 2 × raw bytes of that slice )
```

Pooled and per-stratum (general/code/math/science), both seeds, evaluated at **0.25×, 0.5×, and 1.0× of the byte budget** (catch #17's three-point curve — three evaluations, no extra training, no checkpoints). Never compare cross-tokenizer perplexity.

Decision rule, pre-registered: **minimum pooled BPB at byte-matched budget, agreed by both seeds.** Tie if `|ΔBPB| < 2ŝ` (pooled within-arm SD across seeds) → **prefer the smaller `V`**. **Asymmetric band (D-C-2): a larger `V` displaces a smaller one only by winning by more than `3ŝ`** — pricing the screen's structural engram-off bias toward large vocabularies. Admissibility guard: no arm whose rung-B vocabulary fraction exceeds 20 % (none of the four does). Then the **compute-matched confirmation on the top two arms only** (equal measured FLOPs, unequal bytes); a reversal is a finding that returns to strategy.

Branch outcomes, fixed before data: *48 K wins* ⇒ compression pressure dominates and the 32 K default was wrong; *32 K/24 K wins* ⇒ the allocation argument holds, freeze it; *16 K wins* ⇒ the engram carries more load than assumed and `M_lex`'s gate rises in priority; *seed split* ⇒ escalate. (Standing prior, stated before data and not part of the rule: post-English-only, expect 24 K/32 K.)

**6.8 Reported per arm (R-G4g).** Pooled + per-stratum BPB (both seeds, three budget points) · bytes/token by stratum · P50/P95 raw bytes covered by 2,048 tokens · undertrained-row count · exact byte round-trip assertion · full-softmax throughput and decode latency · complete measured FLOPs · target vocabulary fraction at both rungs.

**6.9 Cost.** ~6.2 A100-hr projected (8 runs at 4×10⁹ B + compute-matched confirmation). **Tripwire 12 A100-hr** — a projection materially above it means the harness is wrong, not the budget.

# 7. Pipeline tests — D1–D6, gating P-B and P-C

| # | assertion |
|---|---|
| **D1** | reproducibility: re-selection from pins + seeds reproduces every shard hash |
| **D2** | dedup determinism: identical dedup decisions and rates on re-run |
| **D3** | stratum byte totals within 1 % of targets; general-stratum split within 2 % of 22/39/39 |
| **D4** | language filter scope: zero language-ID rejections logged in code/math/science strata (proves rule 2 held) |
| **D5** | C2 fixture round-trip through the full pipeline (materialize → shard → read back), byte-exact |
| **D6** | G-TOK stream: the 4×10⁹-byte screen subset and 2 % held-out slice are disjoint (document-level), stratified within 1 %, and byte-identical across all four arms |

No G-TOK run is admissible until D1–D6 pass — same ordering discipline as T14b and PT1–PT6.

# 8. What happens after `V` freezes

Not authorized here — listed so the road is visible: S2 run-axis work (μP width calibration, Power-LR factorial, recurrence sweep with the Jacobian panel per its handoff + rulings P-1…P-5, router `(m,s)` calibration after dense warmup per catch #12), each under its own authorization. The build-axis queue (S4–S6, observatory schema, WEFT-2-seed diagnostics) continues in parallel throughout, unaffected by anything in this document.

**De-scoping order, pre-registered (D-CUR-4):** if tripwires fire or budget lands near the floor — rung B first (reverting to scenario B), then one C-deep checkpoint, then back to Mark. The dense control and rung A are never dropped unilaterally.

---

*Signature block*

**Strategy:** first run-axis authorization since S0, deliberately narrow: the pipeline and the bounded screen, nothing else. The three things most worth your suspicion are the ones that fail silently — a byte filter masquerading as English selection (C1 catches it), Common Crawl overlap double-counting the best web text (§4 catches it), and a screen corpus that drifted from the training distribution (D6 catches it). The P4 tail is deliberately absent; its freeze waits on Mark's license review and must not block this one.
**Coding agent:** build order is P-A → P-B → P-C with D1–D6 green before any training run. The manifest SHA becomes the corpus identity in every receipt. A decontamination hit or a seed split comes back to strategy — neither is yours to resolve locally. Everything else you need is in the governing chain; where documents disagree, this one governs, and a disagreement is a defect to report.
**Mark:** your one open action is the license review (Dolma 3 + FineWeb-Edu ODC-By terms; P4 generator terms) before the corpus freeze.
