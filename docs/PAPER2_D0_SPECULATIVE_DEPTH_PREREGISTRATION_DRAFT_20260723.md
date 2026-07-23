# D0 Speculative-Decoding Depth Recoverability - Preregistration Scaffold

**Date:** 2026-07-23
**Status:** drafting only; no training authorized

## Question

Can behavior from a same-tokenizer Qwen speculative-decoding teacher ladder
produce depth labels that are recoverable by the recurrent student and useful
for allocation, without degrading natural-surface capability?

## Required Decisions Before Lock

1. Teacher models, revisions, hashes, tokenizer identity, and teacher role.
2. Corpus, license, leakage policy, split construction, and manifest hashes.
3. Candidate generation, independent answer verification, and acceptance rule.
4. A disagreement-to-depth mapping fixed before outcome analysis.
5. Distillation objective, student initialization, trainable set, budget, and seeds.
6. Exact definition and threshold for depth-recoverable fraction.
7. Exact acceptance-rate-uplift baseline, denominator, and statistical test.
8. Natural-surface non-degradation battery and stopping rule.
9. Pre-written positive, null, harmful, and uninterpretable readings.

The machine-readable mirror is
`training/speculative_depth_d0_spec.py`. It deliberately contains unresolved
sentinels, rejects lock while any remain, and has no launcher.

## Dependencies

D0 requires a banked T1-lite verdict and its own locked preregistration. It is
not launched automatically from T1-lite. Paper packaging and the natural-trace
composite track remain deferred until D0 reads out.
