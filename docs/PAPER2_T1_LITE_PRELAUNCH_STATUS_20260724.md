# Paper Two T1-lite Prelaunch Status

**Date:** 2026-07-24

## Registered run

The full-block fresh-base T1-lite experiment is locked before training at
commit `44459f30edb1a3c0e83b95479385925f5e4d30a8`. The immutable evaluation
manifests were finalized before launcher creation at commit
`8ea5ce64`. The launcher implements the 10,500-step staged curriculum,
stage-boundary liveness receipts, final-step EMA primary evaluation, raw
secondary evaluation, and all four registered gates. D0 and the C track remain
unauthorized.

The causal gate intervenes on the two continue/stop logits themselves. The
registered sweep contains exactly 4,608 forced-stop executions and 1,024
forced-continue executions. Training checkpoints are resumable from Drive,
and completed stage-boundary receipts are included in the checkpoint written
after each boundary evaluation.

## S1 tokenizer and ladder audit

The official tokenizer artifacts for Qwen2.5-0.5B and Qwen3-0.6B have the same
151,643-entry model vocabulary with identical IDs. Qwen3 has four additional
added tokens. The relevant pair is `<think>` at ID 151667 and `</think>` at ID
151668; neither literal token exists in the Qwen2.5 tokenizer.

Therefore, a Qwen3 teacher is compatible with a Qwen2.5 drafter at the text
trace level after retokenization, but not at the raw token-ID or added-token
embedding-row level. Receipt:
`outputs/stage5/stage5_paper2_s1_tokenizer_audit_20260724/summary.json`.

## WP0 status

Complete. `PROJECT_STATUS_PAPER.md`, `EXPERIMENT_LOG.md`, and
`paper2_claim_evidence_ledger.json` record Phase G as closed on the tested
frozen high-level re-entry interface under the registered `NO-CHANNEL` and
`BOTH_FAIL` verdicts. No Phase G successor is active.

## WP1 status

Complete. The read-only oracle train-row diagnostic evaluated both all 1,899
training variants and the seeded, depth-stratified matched 106-variant cohort.
On all variants, non-default control was 820/4,064 (20.18%) for additive and
947/4,064 (23.30%) for FiLM. Both fall in the filed at-or-below-25-percent
`did_not_fit_command_mapping` band. On the matched cohort, the corresponding
rates were 48/225 (21.33%) and 59/225 (26.22%). The result does not alter the
registered held-out `BOTH_FAIL` verdict. Receipt:
`outputs/stage5/stage5_phase_g_oracle_train_readout_20260722/summary.json`.

## Authorization boundary

Only T1-lite is authorized. No D0, C-track, intra-block oracle, width, or
natural-trace experiment is launched by this implementation.
