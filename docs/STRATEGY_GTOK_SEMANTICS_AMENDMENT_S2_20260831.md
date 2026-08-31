# STRATEGY — G-TOK Semantics Amendment S2: Five Residual Literals, Closed

**Date:** 2026-08-31 · **Status:** AMENDMENT closing the five residual questions in `CODING_TO_STRATEGY_WEFT1_GTOK_S1_RESIDUAL_QUESTIONS_20260831.md` (3,976 B, SHA-256 `13cd3532…6d0fa`, commit `350c2cbf`). S1's verbatim-return clause fired as designed; these are the promised answers, kept as close to one line each as executability allows.
**Precedence:** handoff → A1 → A2 → A3 → semantics ruling → S1 → **S2 (this document)**. Where S2 and S1 conflict, S2 governs — specifically Q4 (burst length) and Q5 (when `n` is computed) supersede the corresponding S1-L5/L4 clauses. Reverting the partial code change before encoding assumptions was the right call; no catch is charged to anyone this round — these are genuine residual degrees of freedom, now bound.
**Nothing here requires a decision from Mark.**

---

> **Q1 — raw-ρ precision and rounding.**
> Pooled BPB per run is accumulated and computed in **float64** (NLL sums and byte counts in float64 throughout); `ρ(arm)` is the float64 mean of the two run values. For **all reporting and all comparisons** (L1 ranking, tie test), ρ is rounded **half-even to 6 decimal places** of BPB. "Exact tie" = equality after that rounding, broken toward smaller `V`. Rationale: 1e-6 BPB is ~3 orders below any plausible ŝ, so the rounding can never affect a real decision, and it makes tie semantics platform-independent.

> **Q2 — exact stream / trained / dropped accounting.**
> Per run, the receipt carries, computed from the run's **realized data order**:
> `stream_bytes` — total UTF-8 bytes of the arm's declared stream (sum of post-A2-R5 document byte lengths; identical across arms by construction);
> `stream_docs` — document count of that stream;
> `stream_tokens` — token count of the full stream under the arm's tokenizer;
> `trained_tokens = n × 524,288` with `n = floor(stream_tokens / 524,288)`;
> `dropped_tokens = stream_tokens − trained_tokens`;
> `trained_bytes` — the exact bytes of the first `trained_tokens` tokens in the run's data order (well-defined because the tokenizer is byte-level and losslessly invertible); `dropped_bytes = stream_bytes − trained_bytes`;
> `trained_docs_full`, `boundary_doc_id` + its consumed-token count (the **at most one** document split by the terminal cut), and `dropped_docs`.
> Governing distinction: the **document-aligned floor governs stream construction only** (A1-R3/A2); batch consumption may split a document at the token level, and the boundary fields make that split auditable. Cross-seed differences in these fields (from data order) are expected and reported, never reconciled.

> **Q3 — confirmation RNG root, role mapping, and data order.**
> Root: `run_seed = int.from_bytes(sha256(b"gtok.confirm.{V}.{s}").digest()[:8], "big")` — the O-9 derivation applied to the registry string, no new mechanism. Role mapping: this `run_seed` drops into the **existing run-harness seed slot**, which derives every per-role stream (model init, data order, any stochastic op) exactly as the byte-matched harness does — one root, existing derivation tree, nothing bespoke. Data order: **redrawn** from the confirm seed, never replayed from the byte-matched run — the stream *contents* are identical by construction; the *order* belongs to the seed, which is what makes the confirmation an independent draw rather than a correlated echo. Model init likewise fresh from the confirm seed.

> **Q4 — calibration length and the stability formula.**
> **The governed 100-step implementation stands** — S1-L5's "8" is superseded by its own deference clause, now made unconditional: `B = 100`. `f_step = (measured FLOPs over steps 1..100) / 100`. Stability statistic: with `f_i` the measured FLOPs of step `i`, `S = (max f_i − min f_i) / median(f_i)`; **stop-and-report iff `S > 0.01`**. All burst steps are in-run, counted toward the realized total, the tripwire meter, and `n`, per S1-L5's binding semantics (in-run, counted, halt-checked).

> **Q5 — the `n` ↔ schedule/checkpoint circularity.**
> Dissolved by moving `n` fully pre-launch; **the burst verifies, it never sets.**
> For **byte-matched runs** there was never a circle: `n` comes from `stream_tokens`, known before launch.
> For **fresh confirmation runs**: `f_step` is taken **pre-launch from the same arm's completed byte-matched runs** — `f_step := floor((F_seed0 + F_seed1) / 2) / n_bytematched`, same hardware, same batch shape, same model config, so per-step FLOPs are the same quantity already measured. Then `n = floor(F* / f_step)` **before launch**, the cosine horizon is fixed to `n`, and the L6 checkpoint indices are precomputed from the stream and `n` — everything the schedule needs exists before step 1, and S1-L4's "computed once after the calibration burst" is superseded accordingly. The in-run 100-step burst retains exactly one job: the A2-R6 projection-halt check, which now includes verifying the burst-measured per-step FLOPs against the pre-launch `f_step` **within 1 %** — halt on violation (it would signal a hardware or config drift between byte-matched and confirmation runs, which must surface, not be absorbed). The end-of-run ±1 % validity band on `F_realized` vs `F*` is unchanged.

---

**Closure check against the memo's five titles:** ρ precision — Q1. Accounting — Q2. RNG root / roles / replay-vs-redraw — Q3. Calibration length + stability formula — Q4. Circularity — Q5. If the memo's text holds a facet these still miss, the verbatim-return clause remains in force — but the intent is that this page plus S1 plus the parent ruling now pin every free variable between the current commit and the sequenced launch.

*Signature block*

**Strategy:** the substantive ruling here is Q5 — the circle existed because S1 gave the burst two jobs (set the budget, check the projection) when the byte-matched runs already contain the budget-setting measurement for free. One job per mechanism, and the circle vanishes. Q4 is the deference clause doing what it was written to do: a governed implementation beats a number I chose while not looking at the code.
**Coding agent:** verify bytes and SHA-256, then bind Q1–Q5 and resume implementation and CPU regression. The pre-launch `f_step` derivation (Q5) is the one code-path change of substance; the rest are field definitions and constants. Sequencing per S1-R1 is untouched: everything holds behind P-B.
**Mark:** nothing to decide. Three documents now specify G-TOK end to end; the next stop on this path should be a gate, not a question.
