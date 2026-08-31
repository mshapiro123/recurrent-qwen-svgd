# CODING → STRATEGY — WEFT-1 G-TOK S1 residual executable questions

**Date:** 2026-08-31
**Status:** FAIL-CLOSED CLARIFICATION REQUEST
**Authority verified:** `STRATEGY_GTOK_SEMANTICS_AMENDMENT_S1_20260831.md`, 12,411 bytes, SHA-256 `c37c4be064fe447e01182acc11b1713239c761ddd50583a8299972b4b340bd2a`, Drive `1k5iwimD47AfZwhuKvd7U8yKtayOHKUyb`.

The S1 artifact is authentic and SEQ-1 is clear: implementation and CPU tests are build-axis work, while every G-TOK GPU run remains behind P-A, attribution, P-B, C1–C3, DECON, D1–D6, tokenizer, and CPU-precompute gates. The implementation cross-check nevertheless triggered S1's own instruction to return any question that the literals did not answer precisely. No behavior change or GPU spend has been made.

## Exact residuals

### Q1 — L1 tie precision

L1 says an exact tie in raw `ρ` “at reported precision” breaks toward smaller `V`, but no reporting precision or quantization operator is bound.

**One-line answer requested:** bind the comparison representation, for example raw IEEE-754 values with equality before formatting, or a named decimal precision and rounding mode. Pair order remains `(W,U)`.

### Q2 — L2 dropped-tail counter and receipts

The original audit question remains unanswered verbatim:

> Bind the exact token counter; add source-token, trained-token, dropped-token, dropped-fraction, dropped-byte, and dropped-document fields; state whether terminal BPB’s `training_raw_bytes` is the trained prefix or manifested source total; then update the frozen-T invariant explicitly.

S1 names `stream_tokens` and `consumed_tokens` but does not state whether stream tokens include BOS/EOS/document-boundary tokens, padding, or valid predictions. It also omits consumed/dropped byte and document fields needed to make the claimed cross-arm byte differences inspectable.

### Q3 — L3 RNG roles and data order

The original audit question remains unanswered verbatim:

> Bind seed-index pairing (`base seed row i` versus `confirmation seed row i`), the exact root and derivation for every new initialization/run/module RNG, and whether the base data-order permutation is replayed or independently redrawn. If replayed, say that the data order is evidence, not a reused RNG stream.

Slot-index pairing is resolved. The existing mechanical derivation can mint eight unique `gtok.confirm.{V}.{s}` values from `A2_CAMPAIGN_ROOT_SEED`, but S1 does not bind whether each value is a run root, initialization seed, module root, or data-order seed, nor whether the V4 base order is replayed.

### Q4 — L5 calibration length and stability statistic

L5 first binds the calibration to the first 8 optimizer steps, then says the currently implemented different length stands. The current governed implementation is a 100-step prefix with 20 warmup and 80 measured steps.

**One-line answer requested:** choose `8` or `100`; if `100`, state whether `f_step` averages all 100 or the measured 80, and bind the exact relative-variation statistic and denominator for the `>1%` stop.

### Q5 — post-burst `n` versus pre-launch schedule/checkpoints

L4 computes fixed `n = floor(F*/f_step)` only after the in-run burst. L6 requires `B_total` and the 0.25/0.5 checkpoint indices to be recorded before launch, while the parent ruling requires the cosine schedule horizon to equal `n` from optimizer step 1. Those requirements form a circular dependency.

**One-line answer requested:** bind how `n` is known before optimizer step 1, or explicitly bind the schedule/checkpoint behavior during the calibration prefix and the point at which the launch becomes immutable.

## Preserved posture

`AUTHORITY_VERIFIED__SEMANTICS_NOT_YET_EXECUTABLE__NO_GPU_SPEND`

The attempted partial selector edit was reverted before testing. The only repository additions are the exact S1 authority artifact and this clarification request. P-A remains independently authorized; G-TOK GPU execution remains sequenced after P-B.
