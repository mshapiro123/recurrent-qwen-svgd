# Strategy Response — E1 Draft Banked; EVAL-D Population Ratified at 8,000 Balanced, with the Comparability Estimator and Score-Blind Conditions

Date: 2026-08-08. Responds to: PAPER2_PHASE2_E1_PREREGISTRATION_DRAFT_HANDOFF_20260808.md (repo `main`, commit `077fa31b`; repo byte-lock governs). Governing: E1 charter (Drive `1HhPKvOd3w3hiL6pbzCvDU4MJsLNtpdNf`) and its ratification record (Drive `19q4UxbpOxLL7qYZzjY7puI2Kywp1n8-x`).

## 1. The draft is banked and the incompatibility finding is ratified as handled

The E1 preregistration package is accepted as drafted: checkpoint hashes matching all four Option B endpoints, the ratified bands and quality margin encoded exactly, machine-readable preregistration with rule inventory, the score-blind readiness checker, and verbatim mirrors of the charter and ratification closing the document chain. The EVAL-D finding is the lock discipline working: the legacy cache (7B token cache, base features) cannot feed the Option B evaluator (four-horizon 14B lattice, probe targets, student states, canonicalizer coordinates), and discovering that *before* the lock rather than mid-pass is the difference between a regeneration task and a spent evaluation. The central fact is confirmed and stated for the record: **rebuilding evaluation infrastructure over the frozen partition does not spend the read-once evaluation — computing scores does.** The cache generation is precompute over registered rows; the evaluation is spent at first score, and only then.

## 2. Population ratified: 8,000 anchors, 4,000 general / 4,000 code — with one estimator addition

The size is right (8,000 matches the DEV analysis scale at which the Option B CIs excluded zero for 0.35–0.50 percent effects, so power is known adequate by construction) and the balanced split is right for the stratum secondaries (equal per-stratum power, clean code/general reporting). One consequence is handled now rather than argued later: DEV's natural mixture was not 50/50, and the measured effect was larger on code — so a balanced population shifts the pooled expectation slightly relative to the exploration numbers. The preregistration therefore states the primary and adds one secondary: **the primary band applies to the pooled balanced-population estimate, with the 50/50 weighting stated plainly; a preregistered secondary reweights the pooled estimate to the DEV mixture** for direct comparability with Option B's 0.351/0.496 percent. Both numbers are registered before contact; neither can be chosen after. Per-stratum estimates remain registered secondaries as drafted.

## 3. Conditions on the cache-generation cell (the score-blind contract, made explicit)

1. **Anchor selection is deterministic and registered**: the selection rule and seed are in the preregistration before the cell runs, anchors drawn only from the frozen EVAL-D document partition, zero overlap with every training and DEV document set — asserted against the partition hashes, not assumed.
2. **Score-blind means no model-quality signal of any kind leaves the cell**: no student scoring, no EAL, no retention, no acceptance, no aggregate that could leak an outcome. The cell emits caches, hashes, counts, and integrity receipts only; the existing readiness checker verifies the cache without scoring, as built.
3. **Label and state generation match the Option B teacher pass exactly**: same pinned revisions, same cascade policy (fraction reported, compared against the 16.748 percent fresh-document rate as a population note), 14B states for all anchors per the standing all-anchor decision, admission ledger recorded.
4. The cell's rule inventory carries the evaluation-mode cliffs — partition integrity and lineage — per the doctrine.

## 4. Sequence confirmed

A100 cache-generation cell (score-blind, conditions above) → EVAL-D freeze receipt with all hashes → E1 lock commit (preregistration + rule inventory + freeze receipt) → the one-shot pass → results to strategy, then Paper Two. Nothing else stands in the queue; this is the last build before the exam.

## 5. Plain-language summary

The final-exam paperwork is done and correct, and the pre-flight check caught something valuable: the exam hall was built for an older version of the test and has to be refitted before anyone sits down. Crucially, refitting the hall doesn't use up the exam — only grading answers does, and no answers get graded during the refit; the builder is explicitly forbidden from computing anything that even resembles a grade. The class list is approved: eight thousand questions, half general text and half code, the same size at which our practice measurements were statistically solid — with one bookkeeping addition so the practice-versus-exam comparison stays apples-to-apples, since the exam has more code questions than the practice set did and code is where the system shines. Refit the hall, seal the receipts, lock the plan, sit the exam once.
