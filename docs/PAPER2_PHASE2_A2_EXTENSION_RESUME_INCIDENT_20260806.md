# Phase-2 A2 Extension Resume Incident

**Date:** 2026-08-06

## Scope

Implementation correction only. The registered A2 data, objective, four-arm matrix, step-237 ancestry, attempt-238 batch, guardrails, step-1,000 extension rule, and endpoint verdicts are unchanged.

## Observed Run

The repaired step-237 launch passed all 32 preflight tests and staged the four exact locked checkpoints. `seed_0_full_a2` then:

- resumed at step 237;
- selected and applied the locked attempt-238 batch;
- trained through step 1,000 without a relative-gradient-explosion event;
- passed the step-1,000 directional audit with aggregate primary share `0.999455`;
- recorded mean accepted length `2.126053`, relative quality-safe oracle headroom `0.013848`, and point quality `0.995552`.

The registered extension rule therefore requested the single continuation to step 2,000. Before that extension could take an optimizer step, the runner stopped with `registered step-237 source SHA mismatch for seed_0_full_a2`.

## Cause

The step-1,000 checkpoint correctly stored `resume_lineage`, including the immutable step-237 source SHA and attempt-238 assertion. On reopening the same arm for the registered extension, the loader restored model, optimizer, schedule, random-state, and guardrail telemetry but did not assign the stored lineage back to the in-memory `resume_lineage` variable. It consequently treated the updated step-1,000 checkpoint as a pristine step-237 source and compared its current file SHA to the old source SHA.

## Correction

The loader now restores and type-checks persisted `resume_lineage` before either resume-mode validator runs. A continuation checkpoint validates its immutable ancestry from that lineage; a pristine checkpoint continues to validate its raw file SHA. Regression tests cover valid and malformed persisted lineage.

The existing Drive checkpoint is retained. A rerun resumes `seed_0_full_a2` at step 1,000 and does not repeat its completed updates. The other three arms remain at their previously staged step-237 states until reached by the matrix runner.

## Scientific Effect

None. The failure occurred between the step-1,000 evaluation and the first extension update. It exposed a loader-state omission, not a numerical, data, or training divergence.
