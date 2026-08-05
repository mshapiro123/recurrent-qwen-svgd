# Phase-2 Staged A1 Resume Amendment

Date: 2026-08-05

Status: `locked_before_resumed_training` when committed. This amendment restores
the inequality semantics that preceded the superseded symmetric share band. It
does not alter the A1 objective, static weights, optimizer, data, alpha, seeds,
or stage budget.

## Evidence authorizing the amendment

- Strategy resolution: Drive `1C-4h5v1OksmYVY5HR9IJnJVNv55R1JqA`.
- Matched-estimator audit commit: `c97aad94c0a913f1b0f05cf54eec98dcc63bf422`.
- Audit receipt: `outputs/stage5/stage5_paper2_phase2_a1_matched_estimator_audit_20260805/summary.json`.
- Audit receipt LF SHA-256: `5cd62b5b0c4cf951b20c17e65a826386269291b16c62bd07d71d69a18d706039`.
- Audit decision: `resume_saved_step_200`.
- Both seeds passed the amended contract on all 51 matched training batches at
  step 200. No model mutation, optimizer update, A2 launch, or confirmatory
  partition contact occurred during the audit.

The stopped run remains classified `protocol_bug_not_registered_attempt`.
Steps 0 through 200 were executed with the original fixed weights. Their
conformance with this amendment was established post hoc by the authorized
read-only matched audit.

## Binding A1 share contract

At optimizer steps 200, 400, 600, 800, and 1,000:

1. Population: the training partition.
2. Estimator: the exact seed-specific calibration measurement batches 50
   through 100, 51 batches of 128 rows, reconstructed from seed + 34001.
3. Hard inequalities: flow share at least 0.50 and functional-probe share at
   most 0.25.
4. A violation of either inequality stops the run with receipts. If the probe
   cap is the violated inequality, one recalibration at that point is the
   authorized response for a separately reviewed continuation; the current
   process does not recalibrate automatically.
5. Counterfactual-preservation share is descriptive. Its calibration weight
   remains fixed. Its behavioral guard is the existing quality-collapse
   tripwire.
6. Preservation loss is logged at every ordinary evaluation. A value above
   twice that seed's step-zero DEV value raises a log-only alarm. This telemetry
   threshold cannot stop or redirect training.
7. The original 60/20/20 shares remain the initialization target only. The
   superseded symmetric absolute-tolerance assertion is never used after this
   amendment.

The 51-batch fixed DEV share estimate remains descriptive population-shift
telemetry and cannot determine the verdict.

## Resume lineage

The exact source checkpoints are:

| Seed | Step | SHA-256 |
|---:|---:|---|
| 0 | 200 | `9815592e5358fbde535bec27d102717f4f9fe4a0beb9f649f0d0879f88db2c58` |
| 1 | 200 | `f3538465223c2f09f286bbb276631b3ce9e60a7c3ecd43bf677d4d4c4dfb6e4e` |

The launcher must preserve those files unchanged and write amended resume
checkpoints separately. The final receipt records both source hashes and the
amendment-lock commit. Frozen hashes, zero-loop identity, quality, non-finite,
trust-catastrophe, and clipping telemetry remain active.

## Boundary

A1 resumes only through step 1,000. It then stops for the registered A1 strategy
gate. The old automatic extension is disabled pending that review. A2 remains
closed, and this amendment contains no path that can launch it.
