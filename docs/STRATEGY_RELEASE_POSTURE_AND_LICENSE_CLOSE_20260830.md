# STRATEGY — Release Posture Ratified · License Review Closed · Freeze Unblocked

**Date:** 2026-08-30 · **Status:** RATIFICATION RECORD. Mark ratified the public-release posture and all five license findings on 2026-08-30. **The license review (items 1–7) is closed. P-B — the corpus freeze — is unblocked** upon P-A completion and the mechanical attribution drafting below. Item 8 alone survives, elevated, gating only the later P4 freeze.
**Precedence:** handoff → A1 → A2 → A3 → this record. Amends the license brief (6,358 B, SHA `29969786…656a`) §2 by closing it.

---

## Plain-language summary

The program is public by design: models and research released on HuggingFace with whatever flow-down terms the datasets require. Mark confirmed this and ratified the five findings that follow from it. That intent dissolves most of the license checklist — attribution obligations are trivially met by a public model card, and the research-use framing on the corpora is exactly what we are doing. What remains is one hard item, three release-mechanics rules, and one closed question.

The hard item is the reasoning-trace data for the final training phase. Provenance of who generated those traces is not disclosed at the model-card level upstream, so it must be verified per dataset subset — and any subset that does not disclose its generator, or whose generator's terms restrict training on outputs, is excluded rather than assumed safe. Public release makes any mistake there visible and permanent, which is why this is now the single item of real diligence left, and why it gates only the late-phase freeze, not the one in front of us.

The release-mechanics rules: we publish the manifest and pipeline, never the raw text shards — the replay test makes the manifest a stronger reproducibility claim than a copy, without becoming a distributor of scraped text; the code-corpus snapshot date goes in the model card so the opt-out posture is inherited honestly; and the provenance sentence and the no-named-comparisons rule ship verbatim, because for public research the real exposure is an overclaim on a community tab, not a courtroom.

Weights release under Apache 2.0. FastText stays. The agent drafts three attribution lines from the dataset cards, and the freeze gate is then free to mint the moment the pipeline reaches it.

---

# 1. The ratified dispositions

| # | item | disposition |
|---|---|---|
| **R-1** | **P4 generator ToS** (was item 8) | **Elevated; the one remaining diligence item.** Per-subset generator-provenance check before the P4 freeze. **Rule: a subset with undisclosed generator provenance, or a generator whose terms restrict output-training, is excluded — never assumed.** Safe pool: open-weights generator lineages. Gates the P4 freeze only. |
| **R-2** | **Release the manifest, not the shards** | Public research artifact = pipeline code + manifest (pins, SHAs, seeds, dedup rates) + D1 replay instructions. **Raw text shards are never published.** The manifest SHA is the public corpus identity; replay is the reproducibility claim. |
| **R-3** | **StackEdu snapshot disclosure** | The pinned revision and its date go in the model card; upstream opt-out/takedown posture is inherited and stated. |
| **R-4** | **Provenance and claims language ships verbatim** | The model card carries, unedited: *"from-scratch in weights, not in data provenance — trained from random initialization on an open corpus including model-generated reasoning traces in a declared final phase."* C-4 stands in public: comparative claims ride on the matched-compute control only; **no sentence of the form "WEFT-1 outperforms [named public model]" is ever written.** |
| **R-5** | **Weights license: Apache 2.0.** Attribution-only upstream terms (ODC-By ×2, StackEdu notice) satisfied by model-card attribution. **Item 7 closed: fastText stays** — tool-use posture, industry-standard, no swap. |

# 2. What this unblocks, and the one mechanical task

**License items 1–7: closed.** The sole pre-freeze remainder is **drafting the attribution text** for Dolma 3, FineWeb-Edu, and StackEdu from their dataset cards — delegated to the coding agent as mechanical work, stored with the manifest so release documentation inherits it (license brief §2.4(b) discharged by the agent rather than Mark).

**Sequencing now:** P-A completes (shards, dedup, draft manifest) → attribution text lands in the manifest → **P-B mints** (C1–C3 + DECON) → G-TOK runs → `V` freezes → S2. No human gate remains on this path. The P4 freeze, later and separate, waits on R-1's per-subset check, which the agent can execute against the exclusion rule without further strategy input — only a subset that *fails* the rule but seems worth fighting for comes back.

---

*Signature block*

**Strategy:** the public-release intent turned a seven-item review into one rule with teeth (R-1's exclude-if-undisclosed), three mechanics rules, and a closed question. The manifest-not-shards posture deserves the emphasis: the receipts culture built the ideal public artifact by accident — a corpus identity that anyone can verify and replay without us hosting a byte of scraped text.
**Coding agent:** draft the three attribution lines from the pinned dataset cards into the manifest; then nothing stands between P-A completion and minting P-B. Apply R-1's exclusion rule mechanically when the P4 tail is assembled; return only a subset that fails the rule but has a case. R-4's two sentences go into the model-card template now, verbatim, so they cannot be paraphrased later.
**Mark:** the license review is off your plate. Your next decision arrives with the G-TOK result — or sooner only if a gate fires.
