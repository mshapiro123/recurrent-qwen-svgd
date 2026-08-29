# STRATEGY — Amendment A2: Literal Bindings for Exact Replay

**Date:** 2026-08-28 · **Status:** AMENDMENT to the corpus/G-TOK handoff (16,431 B, SHA `2aecb647…f74c`) as amended by A1 (12,997 B, SHA `e996f89f…3267`), answering the A1 integration receipt at `d7ec9856`/`af965921` (receipt 10,109 B, SHA `82b18d76…93dd`), whose §4 enumerates literal choices still incompatible with exact replay.
**Precedence:** handoff → A1 → **A2**; later governs on disagreement. Receipt-hash chain unchanged.
**Structure:** §1–§6 bind the six named categories. §7 is the closure rule for anything in receipt §4 that these bindings do not reach, so no further round-trip is required to start materialization.

---

## 0. Plain-language summary

The integration is accepted — routes bound and checked against the pinned upstream cards, the ten-block correction encoded, the v1 receipt chain preserved under a separate authority domain, and the quarantine actually improved rather than merely reviewed. The agent still has not started materialization because a handful of choices remained that two honest re-runs could make differently: where exactly a stream stops filling, which language classifier, which hashing library, how shards are framed on disk, how the cost tripwire is enforced. Each is now bound to one literal answer.

The unifying principle, stated once so the individual rules read as instances of it: **wherever a specification says "deterministic," the authority is a replay test, not a library's reputation.** Implementations are pinned by version and hash, and D1/D2 re-runs are what prove determinism — if the pinned implementation fails its replay test, that is a defect report, not a silent substitution.

The closure rule at the end handles anything §4 enumerates that these six bindings miss: the agent binds it under the same pattern established in A1 — version-pinned, replay-tested, echoed in the execution receipt — and returns only the items it judges to genuinely need a strategy choice. Materialization starts without another round-trip.

---

# 1. Stream termination — greedy fill, first-fit continuation

The "largest document-aligned total not exceeding the target" phrasing in A1-R3 is, read literally, a knapsack problem — unreplayable and not intended. The literal rule:

> **A2-R1.** Per stratum, documents are taken in **shuffled-stream order** (seed `corpus.shuffle`). **Greedy fill:** append each document whole; when the next document would exceed the stratum byte target, **skip it and continue scanning** for the next document that fits (**first-fit continuation**); stop when the deficit is within tolerance (±0.5 %) or the stream is exhausted. Skipped documents are not consumed — they remain eligible for the held-out stream. Deficit and skip count reported per stratum in the sub-manifest. Fully deterministic given the seed and the pinned sources.

The same rule builds `H` (target 8×10⁷ bytes) from the stream *after* `T`'s consumption point, preserving document-disjointness by construction.

# 2. Tokenizer serialization and determinism

> **A2-R2.** Canonical artifact: the single `tokenizer.json` produced by HuggingFace `tokenizers` — **the SHA-256 over this file is the tokenizer identity**; the merge table is its `merges` section (VOCAB-EXT obligations attach to it). Library version **pinned exactly** (record `tokenizers==X.Y.Z` + Python version in the manifest). Single-process training; any stochastic trainer option disabled; the `gtok.bpe` registry seed is supplied where the API accepts one and otherwise recorded as unused. **Determinism is proven, not assumed: two independent fits on the identical stream must produce identical `tokenizer.json` SHAs** (the double-fit test from A1 §4.2, now mandatory before the screen). A mismatch is a defect report naming the library version.

# 3. Language identification

> **A2-R3.** Classifier: **fastText `lid.176.bin`**, identified by the SHA-256 of the model file, recorded in the manifest. Input: the document's first **65,536 bytes** (bounded, deterministic; scoring input only — stored bytes untouched). Keep iff top label is `en` with probability ≥ 0.9. **General stratum only**, per D-G-6; D4 continues to assert zero language-ID rejections logged in the other strata.

# 4. Dedup implementation

> **A2-R4.** The MinHash/LSH parameters of A1 §4.3 stand. The implementation (library or in-repo) is the **agent's binding**, pinned by version and recorded in the manifest, with the permutation seed drawn from `corpus.dedup`. **D2 — bit-identical dedup decisions and rates on re-run — is the authority**, not the library. The exact-pass SHA-1 and the canonicalization-for-matching-only rule are unchanged.

# 5. Shard framing

> **A2-R5.** Shard = **zstd-compressed JSONL**, one object per line: `{"id", "source", "stratum", "text"}`, LF line endings, UTF-8 throughout. `id` is a stable content-derived identifier (SHA-1 of raw document bytes). **The shard's identity hash is SHA-256 over the *uncompressed* JSONL bytes** (compression-level-independent); the compressed file's SHA is recorded separately for transport integrity. Target ~512 MB uncompressed per shard. Documents must be **valid UTF-8 at ingest**; an invalid document is dropped whole and counted per source in the manifest — this is document-level validity screening, not byte filtering, and the count makes it auditable. C2's fixture round-trip runs through this exact framing (D5 unchanged).

# 6. Tripwire enforcement

> **A2-R6.** The 12 A100-hr G-TOK tripwire is enforced by **two mechanisms**, both mandatory:
> 1. **Pre-flight projection:** before the arms launch, a ≤100-step calibration burst per arm measures realized throughput; the projected total is computed from measured, not assumed, tokens/sec. **A projection above 12 A100-hr halts before any full run starts** and returns to strategy — per BD-1, this means the harness is wrong, not the budget.
> 2. **Runtime meter:** a cumulative GPU-hours counter across all G-TOK runs; crossing 12 aborts pending and running work and returns to strategy with the meter's log. Additionally, any single run exceeding **2× its own per-arm projection** aborts that run alone.
> Enforcement is hard abort + report, never a warning that scrolls by. Calibration-burst hours count toward the meter.

# 7. Closure rule — no further round-trip to start

> **A2-R7.** Any item enumerated in receipt §4 that §§1–6 do not reach is **delegated under the A1-R2 pattern**: the agent binds it — version-pinned, replay-tested under D1/D2, echoed in the execution receipt — and materialization proceeds. Only items the agent judges to require a genuine strategy choice (a trade-off, not a literal) come back, **by item**, while the rest execute. The execution receipt lists every binding made under this rule, each marked `A2-R7`.

This is deliberate: the preflight and integration receipts have both shown the agent's literal-binding judgement to be sound, and the replay tests — not my enumeration — are what actually guarantee replayability. The loop should close at the machine that runs the tests.

# 8. Accepted from the integration receipt

The seven route bindings checked against the pinned upstream cards; the v2 authority domain preserving all v1 receipt hashes; the encoded floors, optimizer headline, regex, dedup dimensions, and RNG namespaces; and the **quarantine refresh — two failures resolved, one governed Paper-2 evidence-ledger failure remaining, next review 2026-09-04** — which is what a quarantine with an expiry is supposed to look like. Ruff's absence from the active interpreter is noted and optional; the strict engineering gate and compilation passes stand as the receipt's verification.

---

*Signature block*

**Strategy:** six literal bindings and a closure rule. The principle worth keeping from this exchange: determinism claims are proven by replay tests, never by library reputation — pin the version, run the double-fit, and treat a mismatch as a defect with a name attached. A2-R7 exists because two consecutive receipts have shown the agent binds literals well and the tests are the real authority; the loop should close where the tests run, not here.
**Coding agent:** verify A2's bytes and hash, bind any §4 remainder under A2-R7, and begin P-A. The execution receipt carries the A2-R7 binding list alongside everything A1 §6 already requires. Build-axis work continues in parallel, unaffected.
**Mark:** nothing to decide. The license review before the freeze remains your one open action.
