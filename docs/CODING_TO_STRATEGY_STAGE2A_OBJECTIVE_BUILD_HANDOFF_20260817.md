# Coding Handoff to Strategy — Stage 2A Objective Bound, Population Amendment Required

**Date:** 2026-08-17  
**Repo commit:** `f558eb136a50f138dcd89efdbe78c53f58eab7ed`  
**Branch:** `codex/phase3-opening-build`  
**State:** loss-free build complete; training structurally disabled; no optimizer constructed

## 1. Bottom line

The exact T3a objective binding is verified and implemented. The executable estimator is `0.5 * L_CE + 0.5 * L_KL`, with forward teacher-to-student KL at temperature 1.0 on the renormalized cached top-128 teacher lattice. Both terms use only KP-1R answer-bearing positions, exclude prompt, formatting-only, and position-zero tokens, and reduce within example before averaging the batch.

Fingerprint retrieval now supports mandatory train-time leave-one-out exclusion before top-k. The literal n-gram arm has no row-owned memory entry, so the self-entry exclusion is vacuous for that arm and is stated that way in the lock. The post-initializer scratch write, fixed fingerprint geometry, V(x) rule, controls, and zero-gate identity path are also implemented under tests.

The lock cannot yet proceed to signature because the registered 8,192 one-row-per-slot population is impossible under the combined data contracts. This is a population-arithmetic issue, not an implementation failure.

## 2. Objective authority and implementation

The raw Drive authority was fetched and byte-verified:

- Drive ID: `1-2iiv8aaTrBvUR2Zxs4V6BW1P8OLotb_`
- Bytes: `4,821`
- SHA-256: `78cbf2fb397cf2c6319636523a7feea44b1e21e8941ee32e898323e697f18a22`
- Repo receipt: `docs/STRATEGY_T3A_OBJECTIVE_BINDING_20260817.receipt.json`

Implemented contracts:

1. Exact 0.5/0.5 CE and KL weighting; alternative weights are rejected.
2. Forward `D_KL(p_T || p_S)` over exactly 128 teacher-lattice tokens, with both distributions renormalized on that lattice.
3. Teacher CE target at the same answer-bearing positions.
4. Per-example answer-position mean followed by batch mean.
5. Position-zero, prompt, and formatting exclusion assertions.
6. Fingerprint self-slot exclusion before top-k retrieval.
7. A 512-row admitted non-DEV validation split that must remain outside memory as well as training.

## 3. New population arithmetic

The exact KP-1R reference table contains 10,231 rows:

- 8,712 non-DEV rows;
- 1,519 DEV rows, all prohibited from memory;
- 5,844 of the non-DEV rows are 14B-correct.

V(x) then requires 14B/32B family concurrence, which can only reduce the 5,844. The objective binding additionally requires 512 admitted non-DEV rows held out for validation. Those rows must be excluded from memory; otherwise validation rows could retrieve themselves and would not measure deployment-like generalization.

Therefore the absolute one-row-per-slot ceiling is `5,844 - 512 = 5,332` before the 32B concurrence filter. The registered 8,192 slots cannot be materialized. Even 4,096 cannot be guaranteed until the concurrence pass reports the final admitted count.

## 4. Requested prospective amendment

Recommended binding language:

1. **Validation split:** after V(x) admission, select 512 rows by battery-stratified `SHA256(seed:battery:item_id:content_sha256)` using seed `20260817`. Freeze and hash this manifest first. Validation has zero overlap with memory, training batches, DEV, CONFIRM, and EVAL-E.
2. **Memory slots:** from the remaining admitted non-DEV rows, set the slot count to the largest power of two not exceeding the available count, capped at 4,096. Select slots under the existing memory seed `20260816`. Parameter-match T3b to the resulting count. The final count and all battery proportions are disclosed before any model sees DEV.

This rule is deterministic, prospective, and does not select on task effects. It avoids guessing 4,096 before the required 32B pass establishes feasibility. If strategy instead requires exactly 4,096, the build must stop when fewer than 4,608 rows survive V(x), because 512 validation rows must still be reserved.

## 5. Verification

Focused Stage 2A and adjacent Phase 3 suite: **69 passed**.

Full repository suite: **2,807 passed, 4 failed, 1 warning**. The four failures are the same unchanged Windows CRLF byte checks on historical Drive-locked markdown files. No new regression appeared.

The two existing Drive build artifacts were replaced in place and byte-verified:

| Artifact | Drive ID | Bytes | SHA-256 |
|---|---|---:|---|
| Stage 2A charter draft | `1WKFJx8j049bye_PQsoR6Y39E3RvyPtx6` | 13,596 | `04d2d892b58ecf824884b34e4883065fdc0d04f1a9124aace3a4b80d855cebe0` |
| Draft build receipt | `1JA8iBFT-MKYoxSnNr2dyAZe314Hp-m0z` | 3,241 | `dbf5348434ca60631ace56937344872d06970cb17453626bcd6fe17f46b7b2e5` |

CONFIRM and EVAL-E remain sealed. DEV scores were not computed. Optimizer steps remain zero.

## 6. Decision requested

Please ratify or amend the deterministic validation and dynamic slot-count rule in section 4. Once bound, coding can run the score-blind 14B/32B admission and content-build pass, freeze both manifests, fit the non-DEV geometry, and assemble the final executed lock for Mark's signature.

