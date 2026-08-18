# Paper Two Stage 2A CV-1 and D5 Diagnostic Specification

**Date:** 2026-08-18
**Status:** locked before scoring
**Authority:** `STRATEGY_T3_SCREEN_RESPONSE_20260818.md`, Drive `1g6Uh1869pNT3v8RtUzFmdPfOuFRYLz4S`, 9,978 bytes, SHA-256 `f8b5d2b45a7c6f855943ea88ab3efadfb872eafd945bfaca6aa396bff1f4086a`

## Purpose

CV-1 separates memory-content value from pathway cost after the Stage 2A screen returned `SCREEN_BELOW_PROCEED_THRESHOLD`. D5 asks whether fingerprint retrieval relevance predicts row-level benefit on MMLU and GSM8K. Both are diagnostic DEV reuse. They cannot revise the registered T3 verdict, authorize T3-full, or spend CONFIRM or EVAL-E.

## Frozen inputs

- Host checkpoints: seed-0 T3a EMA step 1,200 and seed-0 T3b EMA step 1,200.
- Frozen substrate lineage: the seed-0 P3.5 Arm-S EMA step-4,400 chain already bound by the Stage 2A lock.
- Panel: the same frozen 1,024-row DEV panel.
- Read: K=4, bridge ceiling 0.02, registered Stage 2A amplitude 0.05.
- No optimizer, backward pass, parameter update, training row, CONFIRM row, or EVAL-E row is permitted.

## Crossed-value cells

Each host retains its trained gate, injection projection, slot weights, addressing rule, and all frozen upstream state. Only its learned value bank is varied:

- `correct`: the host checkpoint's EMA value bank unchanged.
- `shuffled`: a complete-row permutation of that same learned bank using seed `20260818`.
- `random`: deterministic per-coordinate moment-matched Gaussian values using seed `20260819`; each coordinate matches the correct bank's population mean and standard deviation.

For T3a, the bank is `reader.values`. For T3b, the four literal-table tensors are concatenated, transformed as one 4,096-row bank, and restored to their original table shapes. This makes the comparison host-local and leaves the addressing mechanism unchanged.

Each value condition is read at gate-dose multipliers `{0, 0.5, 1.0}`. The multiplier scales the retrieved value immediately before the existing linear injection projection. Dose zero must reproduce the initialization output exactly. It is physically scored once per host and reused for all three logical zero-dose cells. The two host zero-dose outputs must also match row for row.

Every cell reports pooled and per-battery accuracy, fixes, regressions, and paired sign tests against both:

1. the frozen 502/1,024 base reader; and
2. the dose-zero initialization read from the same seed-0 lineage.

## D5 targeted read

D5 uses T3a's dose-1 correct, shuffled, and random row receipts. For pooled MMLU+GSM8K and for each battery separately, define:

`content_advantage = 2 * I(correct-value answer correct) - I(shuffled answer correct) - I(random answer correct)`.

The primary relevance variable is the correct-value cell's mean top-1 retrieval score. Secondary variables are compatibility-gate mean and retrieval-entropy mean. Report Spearman correlation with `content_advantage`, a deterministic 10,000-permutation two-sided p-value, and Holm-adjusted primary p-values across MMLU and GSM8K. Also report strict content wins (correct succeeds, both controls fail), strict content losses (correct fails, both controls succeed), and metric means for each group. D5 is descriptive and has no pass/fail gate.

## Integrity and exit contract

- Assert source checkpoint hashes and the frozen Stage 2A panel/geometry hashes.
- Assert value transformations do not alter any non-value tensor.
- Assert dose-zero identity before reading any nonzero cell.
- Write completed cells and status receipts incrementally; blocked outcomes retain tables and exit 2.
- Every receipt states `confirm_scored=false`, `eval_e_scored=false`, and `optimizer_constructed=false`.
