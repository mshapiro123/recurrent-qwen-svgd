# Phase T1-lite Pre-Training Manifest Amendment

**Date:** 2026-07-24  
**Timing:** after preregistration lock, before launcher creation and before any T1-lite training step

The declared calibration procedure in Draft 4 is:

- `training.synthetic_depth_task.write_synthetic_depth_dataset`
- seed `2026072401`
- N16, depths 1-8, 64 rows per depth
- `test_chain_mcq.jsonl`
- prepend `t1_calibration_` to every generated row ID

The hashes in lock commit `44459f30` did not reproduce that declared procedure.
The generated rows were rebuilt twice and the reproducible hashes are:

- row-ID SHA-256: `3175178e33b56406d9b7147cd4af5a76f3e47027a414b67a62d804991c7715c7`
- canonical-row SHA-256: `9c4e7dacd30c720ed8b2ffba2770c39482e838544426219acce4760ee96e07ab`

This amendment corrects only those two integrity fields. It does not alter the
generator, seed, row count, depth distribution, calibration role, gates,
constants, curriculum, model lineage, or registered gated/extrapolation sets.
The launcher asserts these corrected hashes and fails before model loading on
any mismatch.
