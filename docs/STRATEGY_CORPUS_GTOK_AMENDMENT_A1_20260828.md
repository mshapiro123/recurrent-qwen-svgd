# STRATEGY — Amendment A1 to the Corpus/G-TOK Handoff: Execution Unlock

**Date:** 2026-08-28 · **Status:** AMENDMENT to the corpus/G-TOK handoff (16,431 B, SHA `2aecb647…f74c`), answering the coding agent's preflight receipt at `5b39d8a1` (receipt 13,192 B, SHA `00def1bf…476b`). **All four blockers are resolved below. Execution of P-A → P-B → P-C is unlocked upon the agent's verification of this amendment's bytes and hash.**
**Precedence:** the handoff as amended by this document governs; where the two disagree, **this amendment governs**. The banked receipt-hash chain is unchanged — this document amends forward, it does not restate.
**Verdict on the preflight:** correct on all four blockers, and the fail-closed posture — draft everything, mint nothing, start no run — is exactly right. One blocker is my arithmetic error with a traceable origin two documents upstream; it is catch #20.

---

## 0. Plain-language summary

The agent refused to execute an ambiguous handoff and built the safe scaffolding instead. That refusal was correct on every point, and this amendment supplies what was missing.

The embarrassing one first. I described the tokenizer screen's model as eight blocks when it executes ten, and the error is older than the handoff — it has sat in the G-TOK rulings since they were written. The confusion was about what "structurally OFF" means: it disables the WEFT machinery inside the blocks, it does not delete the blocks. The proxy is four prelude plus two core plus four coda — ten blocks, visited once each — exactly as the target's twenty-two-block dense baseline is nine plus four plus nine. The agent's graph was right and my prose was wrong. The corrected cost estimate moves from 6.2 to about 6.5 GPU-hours, still half the tripwire.

The dataset routing is resolved by putting the authority where the ground truth is: the agent, at the machine that resolves the repos, binds each source family to an exact repository, configuration, and commit, under acceptance criteria this amendment fixes — right content family, verified license, enough bytes, pinned revision. The agent's receipt already drafts these bindings, and they are adopted on condition they meet the criteria, with every bound value echoed in the execution receipt so the choice is auditable afterward.

The byte arithmetic gets the missing rounding rule: targets are defined in whole bytes, documents are never split, both streams round down to document boundaries, shortfalls are reported, and the holdout is defined as two percent of the training target rather than of the combined total, which removes the circularity that had no integer solution.

Everything else the agent listed as unbound is now bound to specific values: the full AdamW configuration, the concrete tokenizer-training recipe including the pre-tokenization pattern, the deduplication parameters, and the seed-naming scheme that ties every random choice in the pipeline to the per-module generator registry. The agent's drafted D1–D6 schemas are adopted as authoritative on the same echo-in-receipt condition. Nothing remains for Mark; his license review stays the only open action, and it gates the freeze, not the build.

---

# 1. Catch #20 (mine) — the proxy executes ten blocks, and the error's origin

**The agent's graph is correct.** The registered proxy `4/2/4` is 4 prelude + 2 core + 4 coda = **ten blocks**. The "eight dense blocks" phrase originated in **R-G4c of the G-TOK rulings** and was copied into the corpus handoff §6.4 — it survived two documents because it was never recomputed, only quoted.

**The semantic ruling that prevents recurrence of the error:**

> **Structural-OFF disables the WEFT machinery inside and around blocks (lanes, carrier, rotors, callosum, sidecar, engram); it never removes blocks from the graph.** Core blocks remain as ordinary dense transformer blocks, visited once at `K = 1`. This is the same semantics OBS-INV already asserts at target — `9/4/9` at `K = 1`, modules OFF, ≡ the dense **22**-block baseline — and the proxy is its exact analogue at **10** blocks. Any future description of a structural-OFF configuration states the executing block count computed as prelude + core + coda.

**Corrected cost:** per-arm 0.61–0.72 A100-hr (N = 37.9–54.7 M at ten blocks), 8 runs ≈ 5.2, plus the compute-matched confirmation ≈ 1.3 → **≈ 6.5 A100-hr projected** (was 6.2). **Tripwire 12 unchanged.** R-G4c and handoff §6.4 are amended to "ten dense blocks (4 prelude + 2 core + 4 coda, core as plain blocks, all WEFT modules OFF)".

# 2. Source-route bindings — authority delegated to the resolver, criteria fixed here

I named `allenai/dolma3` subsets from a summarized card; the agent, resolving the actual repos, reports the pool carries Common Crawl and olmOCR and that the other families need their own exact routes. **The resolver is the party with ground truth, so the binding authority moves to the agent under fixed acceptance criteria:**

> **Ruling A1-R2.** Each source family (Dolma 3 web · Wikipedia/Wikibooks · StackEdu · FineMath 3+ · arXiv · olmOCR · FineWeb-Edu) is bound to an exact **(repository, configuration, revision-SHA)** triple by the agent. A binding is admissible iff:
> 1. the content matches the family named in curriculum r2 §4 (same corpus lineage, not a look-alike);
> 2. the license is **ODC-By or a no-more-restrictive open license, verified from that repository's own card** and recorded in the manifest per source;
> 3. available bytes ≥ the stratum target with margin;
> 4. the revision is a commit SHA, never a branch;
> 5. **route bindings drafted in the preflight receipt's §6 are adopted as written iff they satisfy 1–4.** Any family where §6's draft fails a criterion, or where no admissible route exists, comes back to strategy **by family** — the rest proceed.
>
> **Every bound triple is echoed in the execution receipt**, making the delegation auditable after the fact. The manifest remains the corpus identity; the bindings live in its per-source section.

# 3. Byte accounting — the integer rule

> **Ruling A1-R3.** All targets are integer bytes; **documents are never split**. Per arm:
> - **Train stream `T`:** target 4,000,000,000 bytes, composed per stratum (45/25/15/15) by **document-aligned floor** — the largest document-boundary-aligned total not exceeding each stratum's byte target.
> - **Held-out stream `H`:** target **`0.02 × T` = 80,000,000 bytes** — defined against the *train target*, not the combined total, which removes the circular definition — same stratification, same document-aligned floor, **document-disjoint from `T`** (D6 unchanged).
> - **Tolerance:** each stratum within **±0.5 %** of its target after flooring; shortfalls reported in the sub-manifest with the exact byte deficit. A stratum outside tolerance fails the sub-manifest, not silently.
> - The identical `T` and `H` byte streams serve **all four arms** (D6's byte-identity assertion unchanged).

# 4. Remaining bindings — every value the preflight listed as unbound

**4.1 AdamW (screen-only; single parameter group per the flat-AdamW bound).**
`lr = 3×10⁻⁴` · `betas = (0.9, 0.95)` · `eps = 1×10⁻⁸` · `weight_decay = 0.1` (decoupled, single group — no per-tensor exemptions, which would be a partition) · cosine decay to 10 % of peak · warmup 1 % of steps · batch **256 sequences × 2048 tokens ≈ 0.52 M tokens** · gradient clip 1.0 · bf16 compute with FP32 master weights and FP32 loss reduction. Identical across arms and seeds; the standing limitation (uncalibrated LR, within-screen comparisons only) is unchanged.

**4.2 Tokenizer training.** HuggingFace `tokenizers` byte-level BPE, trained on the **full train stream `T`** (one recipe, no subsampling decision to drift). `min_frequency = 2`. Pre-tokenization: ByteLevel with the split pattern

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

— note `\p{N}` **single-digit** number tokens per R-G4f, splits on Unicode letter/number/category boundaries, no lowercasing, no NFKC, whitespace and indentation preserved. Deterministic merge order; seed from the registry (§4.4); SHA over the merge table; VOCAB-EXT retention obligations unchanged. If the library's determinism across platforms is in doubt, the merge-table SHA across two independent fits is the test — a mismatch is a defect report.

**4.3 Deduplication (amending handoff §4 with the missing parameters).** Matching canonicalization — **for match computation only, stored bytes untouched**: NFC + whitespace collapse. Exact pass: SHA-1 over canonicalized bytes. Near-dup pass: MinHash **128 permutations**, LSH **16 bands × 8 rows**, byte-level **13-gram** shingles on the canonicalized text, Jaccard **≥ 0.8**. Drop direction FineWeb-Edu, top-up from its quality-ranked remainder, rates in the manifest — all as in the handoff.

**4.4 RNG naming (extending the O-9 registry to the pipeline).** Every stochastic choice draws from a generator seeded by the established SHA-256 derivation `module_seed(run_seed, name)` with names: `corpus.shuffle` · `corpus.split` · `corpus.dedup` · `corpus.topup` · `gtok.bpe` · `gtok.init.{arm}.{seed}` · `gtok.data.{arm}.{seed}`. No stream is shared across purposes; ablating or re-running one stage advances no other stage's stream — the same invariant, same mechanism, as the model-side registry.

**4.5 Revisions.** "Latest at pin time" is acceptable; the **resolved SHA recorded in the manifest is what freezes**, and D1 reproduces from the recorded SHA, never from "latest".

# 5. D1–D6 schemas and the drafted infrastructure

> **Ruling A1-R5.** The agent's drafted source/shard/dedup/split/stream schemas and D1–D6 diagnostics are **adopted as authoritative**, on two conditions: the schemas are echoed (or SHA-referenced) in the execution receipt, and any point where a drafted schema and this amendment disagree is **reported as a defect, not reconciled silently** — the standing rule, now applied in the agent's favour as much as mine. Unconditionally-fail-closed gate minting is ratified as the permanent posture: gates are minted only by the authoritative path, never by drafts.

The preflight's operational choices are all accepted: the separate execution-authority chain preserving banked receipt hashes; the provenance, cluster-disjointness, codec, multilingual-fixture and LF transport guards; preserving `.runlogs/` untouched with a narrow ignore; and declining a repo-wide green claim while the quarantine review is open — that last one is the same discipline as withholding the T14b production receipt, and it continues to be right. **The quarantine review date requested in the foundations adjudication §4 is now due with the next receipt.**

# 6. Execution unlock

With this amendment verified (bytes + SHA against the delivery receipt), **P-A → P-B → P-C are unlocked as specified in the handoff as amended.** Sequence and gates unchanged: D1–D6 green before any G-TOK run; C1–C3 + DECON before the freeze; a decontamination hit or a seed split returns to strategy. The execution receipt carries: the bound route triples (§2), the realized `T`/`H` byte totals and shortfalls (§3), the schema SHAs (§5), the corrected ten-block config identity (§1), and the quarantine review date.

Nothing new for Mark. **The license review remains his one open action and gates the freeze** — with one addition from §2: the per-source license verification the agent records satisfies criterion 2 mechanically, but the *program-level* review (attribution obligations, responsible-use terms, P4 generator terms) is still the human step.

---

*Signature block*

**Strategy:** four blockers, four resolutions, one catch. #20 is mine and its origin is instructive — "eight dense blocks" was written once in R-G4c, never recomputed, and quoted forward through two documents; the fix is semantic (structural-OFF never deletes blocks) rather than numeric, so it cannot recur by miscounting. The route-binding delegation is the right shape for a fact only the resolver can see: criteria fixed here, values bound there, everything echoed back for audit. The agent's preflight — build the scaffolding, mint nothing, refuse the ambiguity — is the fifth time its fail-closed judgement has been exactly right, and the drafted-schema adoption in §5 reflects that earned standing.
**Coding agent:** verify this amendment's bytes and hash, then execute. Your §6 draft bindings are adopted where they meet A1-R2's criteria; echo the bound triples, realized byte totals, schema SHAs, and the quarantine review date in the execution receipt. The ten-block identity replaces "eight" everywhere it appears; the corrected projection is ~6.5 A100-hr against the unchanged tripwire of 12.
**Mark:** nothing to decide; the license review before the freeze remains yours.
