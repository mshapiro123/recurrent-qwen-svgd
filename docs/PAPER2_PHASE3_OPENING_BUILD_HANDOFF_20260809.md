# Paper Two Phase 3 Opening Build Handoff

Date: 2026-08-09. Scope: authorized build work only. No Phase 3 optimizer step has run, no DEV score has been computed, and no CONFIRM partition has been touched.

## 1. Governing record

The repository now mirrors the complete eight-file authority chain from Drive. `training/paper2_phase3_opening_contract.json` records every Drive ID, byte count, and SHA-256. The raw markdown and SVG files are marked `-text` in `.gitattributes`, preventing Windows line-ending conversion from invalidating their Drive hashes.

The technical amendment and reasoning-scope addendum are implemented as binding constraints. GSM8K and MBPP are primary target batteries, ARC-Challenge is secondary, and ARC-Easy, MMLU, and Tier-1 are floor/retention batteries only. Floor-only rows cannot enter the verified training objective.

## 2. Per-position gate build

Phase 2 remains a separate historical class and still produces its registered 1,184,917-parameter build receipt. Phase 3 adds `Phase3PerPositionAnchoredBridge` and `Phase3StudentModules`; no existing Phase 2 checkpoint schema was changed.

The new gate logit is the migrated Phase 2 per-loop scalar bias plus three zero-initialized projections: per-position hidden state (896 weights), attended scratch state (128), and recurrent control state (32). This adds 1,056 parameters, establishing the Phase 3 sidecar count at 1,185,973.

The CPU build receipt establishes:

- scalar gate-bank preservation;
- zero initialization of all three new projections;
- exact position-uniform gate values at migration;
- exact migrated writeback under the scalar reference equation;
- position-zero closure;
- bit-exact zero-loop hidden-state and logit identity;
- no optimizer and zero training steps.

`training/paper2_phase3_migration.py` implements a one-way, source-hash-checked migration that writes a fresh checkpoint with no inherited optimizer state. A checkpoint-integrated migration against both E1-confirmation endpoints remains required before P3.3.

## 3. P3.1 infrastructure

`training/paper2_phase3_p31.py` and `eval/prepare_paper2_phase3_p31.py` implement:

- deterministic document-disjoint verified-train, DEV, and CONFIRM assignment with seed 20260809;
- exact dataset-revision, reader-version, item-content, partition, and complete-ledger hashes;
- fixed target/floor roles and equal per-battery macro weights;
- a score-blind ledger with explicit contamination limitations;
- atomic partition leases, including a permanent CONFIRM-spent state once scoring starts;
- paired one-sided Student-t upper confidence bounds;
- the registered two-consecutive-look stop rule;
- a seeded Gaussian-copula campaign simulator with exact binomial upper confidence reporting (via the beta quantile) for familywise false-stop calibration.

Calibration smoke results are not a lock. Under the provisional design of 20 looks, paired discordance 0.20, and adjacent-look correlation 0.80, 256 rows at one-sided alpha 0.0005 failed clearly. In 100,000 campaigns, 256 rows at alpha 0.0001 produced 5 false stops, with a conservative upper probability of 1.051e-4, narrowly missing the target. A 512-row, alpha 0.00005 candidate produced 2 false stops, with a conservative upper probability of 6.296e-5. The final design must replace the provisional discordance and correlation with estimates from the frozen paired row history and should report detection power below the -3-point boundary before lock.

## 4. P3.2 infrastructure

`training/paper2_phase3_p32.py` and `eval/eval_paper2_phase3_p32_preflight.py` implement:

- separate agreement and verified cache schemas;
- mandatory 14B/32B concurrence for admitted agreement rows;
- a hard prohibition on correctness labels in the unverified agreement stratum;
- verified teacher-right/student-wrong labels only from programmatic verifiers;
- positive, negative, and ignored gate labels, with ignored rows excluded from gate loss;
- batched agreement-direction gradients from the wrong-versus-teacher-token margin;
- a batched-versus-single-row equivalence check.

The synthetic preflight is finite and exact for directions and gradient norms; its maximum margin difference is 5.96e-8. This proves the batching implementation on an independent linear callback. It does not replace one-batch equivalence through the real frozen upper model, actual cache generation, or the document-disjoint ridge forecast.

## 5. Verification

The focused suite passes 63 tests, including all new Phase 3 contracts plus the Phase 2 student, matched-alpha, Option B training, and Option B lock regressions. The historical Phase 2 build independently reran at 1,184,917 parameters with all assertions green. Source compilation and `git diff --check` pass.

## 6. Remaining decisions and next work

1. Pin exact dataset revisions, reader versions, and content manifests for all six batteries. The assembler intentionally rejects placeholders at lock time.
2. Estimate paired discordance and adjacent-look dependence from the fixed DEV checkpoint series, then rerun the false-stop and power simulations to choose the final floor-battery size and one-sided alpha.
3. Decide whether the 32B-concurrence rule admits only cascade-covered rows (the current strict implementation) or permits a separately named unavailable-32B class. No silent fallback exists.
4. Run checkpoint-integrated scalar migration on both E1-confirmation endpoints and record source and destination hashes.
5. Generate the real P3.2 cache, run one-batch equivalence through the frozen upper model, and fit the document-disjoint linear-decodability forecast.
6. Draft and lock P3.3 only after those receipts. P3.3 training remains unauthorized in the current machine-readable contract.
