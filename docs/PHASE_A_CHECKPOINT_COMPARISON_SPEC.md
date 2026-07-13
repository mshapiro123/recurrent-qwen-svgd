# Phase A Checkpoint Comparison

## Question

Did fixed 4,000-step dense training improve, merely waste compute, or damage
held-out performance after the training losses had already saturated?

## Frozen Inputs

- The exact 1,792-row depth-1-to-14 Phase-A evaluation set.
- Arms B, C, and D at saved steps 2,000 and 4,000.
- Checkpoint step, model revision, and supervision surface are asserted from
  `stage5_dense_full_metadata.json` before every evaluation.

## Readouts

- Shared deterministic full-symbol reader for all six settings.
- Overall and depth-stratified accuracy.
- Paired helped, harmed, tied, net-correct, and two-sided sign-test results for
  step 4,000 minus step 2,000 within each arm.
- Paired B/C/D comparisons at step 4,000.
- C depth-2 errors classified as correct, one-step early, earlier-orbit,
  unrelated, or parse failure.
- Compressed complete continuations and a compact paired-row table retained in
  GitHub so the result is auditable.

This is eval-only. It trains no parameters and cannot open Phase G-alpha.
