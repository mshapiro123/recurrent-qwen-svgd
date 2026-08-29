# STRATEGY — Amendment A3: Corpus-Declaration Divergences, Ruled by Family

**Date:** 2026-08-29 · **Status:** AMENDMENT to the corpus/G-TOK handoff as amended by A1 (12,997 B, SHA `e996f89f…3267`) and A2 (8,615 B, SHA `f7a2655b…ce02`), ruling on the corpus-declaration gate stop at `86d13d9f`. **Precedence:** handoff → A1 → A2 → **A3**.
**Verdict on the stop:** correct, and the gate earned its existence on its first firing. Neither blanket answer — accept observed, narrow selectors — is granted. The ruling is a decision *procedure* per family, §2–§3, executable without a further round-trip in the expected cases.

---

## 0. Plain-language summary

The gate that stopped the pipeline compares what we *declared* each upstream source contains against what the pinned repository *actually* enumerates, and both web sources diverged — in opposite directions. That stop was right: the numbers we need are tiny against either pool, so nothing here is about running out of data. It is entirely about identity — whether the selector is matching exactly the content family we named, no more and no less. A mystery superset accepted blindly could pull lower-quality tiers into the top-bucket slice; a selector trimmed blindly to match a wrong declaration could silently skew which slice of the web we sample.

The ratios in the numbers already say most of what happened. For Dolma web, the observed pool is 2.5× the declaration in files *and* bytes together — the signature of the selector matching two or three sibling groups of similar size beyond the intended one, which is exactly what extra quality tiers look like. For FineWeb-Edu, files are down a third but bytes only a fifth — the missing files are small ones, which is what it looks like when a declaration was computed from card totals that included auxiliary sample configurations the selector correctly excludes.

So the ruling is the same procedure for both, with opposite expected outcomes: produce a one-page enumeration breakdown grouping the observed files by path pattern, classify each group in-family or out against the family definition the curriculum already states, bind the selector to exactly the in-family groups, and re-mint the declaration from that explained enumeration. On the evidence of the ratios, Dolma web will narrow and FineWeb-Edu's observed totals will stand — but the breakdown, not the presumption, is what decides. The one case that must come back to strategy is if Dolma 3 turns out not to expose quality tiers in a selectable form at all, because then the phrase "top quality bucket" in the corpus specification needs a different mechanism, and that is a design choice rather than a literal.

The principle this hardens into a standing rule: a declaration is a hypothesis about upstream; the enumeration at the pin is ground truth; and when they disagree, the resolution is always an *explained re-derivation* — never acceptance of what we don't understand, never trimming reality to match a guess.

---

# 1. Standing rule, and what the gate proved

> **A3-R0.** A source declaration is a **hypothesis about upstream**, recorded at binding time from cards and metadata. The enumeration at the pinned revision is **ground truth**. On divergence, the resolution is **explained re-derivation**: break the observed enumeration into path-pattern groups, classify each group against the family definition in curriculum r2 §4, bind the selector to exactly the in-family groups, and **re-mint the declaration from the explained enumeration**. Blind acceptance of a superset and blind narrowing to match a declaration are both prohibited. The breakdown is a durable artifact (hashed, referenced in the manifest).

Availability is settled and out of the question for both families: the web-slice need is ~6.7×10⁹ B, against which dolma_web offers 67× (declared) to 168× (observed) and fineweb_edu 554× (observed). **Identity is the only live question.** The gate fired on its first real encounter with upstream and caught a divergence in each direction — it stays, unchanged, permanent.

# 2. `dolma_web` — observed 2.47× files, 2.51× bytes: presumptively narrow

Files and bytes scale **together** at ~2.5×, the signature of the selector matching **two to three sibling groups of similar size** beyond the intended one — which is what additional quality tiers (or parallel subsets) look like, not stray metadata or indices.

> **A3-R1.** Produce the enumeration breakdown. The family definition is **"Dolma 3 web, top quality bucket"** (curriculum r2 §4). Expected resolution: the observed pool contains multiple quality tiers or parallel web subsets; **narrow the selector to the top-bucket group alone**, re-mint the declaration from it, and proceed — no further strategy input needed. If the declared 6,009/448 GB turns out to correspond exactly to the top-bucket group inside the observed pool, that is confirmation the original declaration was right and the selector was loose.
>
> **Return-to-strategy branch, the one real design case:** if the breakdown shows Dolma 3 web exposes **no selectable quality structure** — one undifferentiated pool — then "top quality bucket" as written has no referent, and the selection criterion needs a different mechanism (per-document quality scores as metadata filters, if present; or a different definition of the general-web slice). That is a trade-off, not a literal, and comes back with the breakdown attached.

# 3. `fineweb_edu` — observed 0.67× files, 0.82× bytes: presumptively accept

The missing files are **smaller than average** (files down 33 %, bytes down only 18 %) — consistent with a declaration computed from card totals that included **auxiliary configurations** (`sample-10BT`/`sample-100BT`/`sample-350BT`, or the `score-2` variant) which the selector correctly excludes.

> **A3-R2.** Produce the enumeration breakdown. The family definition is **FineWeb-Edu, main data, all CC dumps** — explicitly excluding `sample-*` and score-variant configs. Expected resolution: the selector already matches exactly the main-data dumps, the shortfall is attributable to the declaration having counted out-of-family configs, and the **observed totals stand** — re-mint the declaration to them and proceed.
>
> **The check that must not be skipped:** confirm the observed group covers **all** main-data CC dumps at the pin, not a truncated subset. A selector silently missing dumps would skew the temporal distribution of the web sample — a quiet cousin of the length-bucketing trap. If dumps are missing, **widen to all in-family dumps**, re-mint, and proceed. Either direction resolves locally; only an ambiguous breakdown returns.

# 4. Scope, and what does not change

Both re-minted declarations flow into the manifest per A1-R2 (echoed in the execution receipt, with the breakdown artifacts' hashes). Stratum byte targets, the 22/39/39 general-stratum split, sampling rules (A2-R1), dedup, and every downstream gate are unchanged — this amendment touches only *which upstream files are eligible*, before sampling begins. The pinned revisions do not change; nothing here re-opens the pins.

Accepted from the report without comment beyond credit: the redirect-failure fix at `86d13d9f`; stopping at the gate rather than pressing through it; minting no source quartet or authoritative transport artifact while the question was open; the V4 transport receipts on the manifest and Wikipedia assets; and not modifying the exact runtime to add pytest — the runtime's identity outranks test convenience, which is the right instinct, and running the suite in the working environment while keeping the exact runtime pristine is the correct division.

---

*Signature block*

**Strategy:** the gate fired on its first contact with upstream reality and caught divergences in both directions — that is the system working. The ruling is a procedure, not a verdict, because the breakdown is ground truth I do not have: classify groups against the family definitions already in the curriculum, bind to the in-family groups, re-mint declarations from explained enumerations. The ratios make the expected outcomes fairly clear — narrow Dolma web, accept FineWeb-Edu observed — but the one branch that must come back is Dolma exposing no quality structure, because then "top quality bucket" needs a redesign, not a selector fix.
**Coding agent:** verify A3's bytes and hash, produce the two breakdowns, resolve per §2–§3, attach the breakdown hashes to the re-minted declarations, and relaunch P-A. Only two things return to strategy: a Dolma pool with no selectable quality structure, or a FineWeb-Edu breakdown that is genuinely ambiguous about dump coverage. Everything else proceeds under this amendment.
**Mark:** nothing to decide; the license review before the freeze remains your one open action.
