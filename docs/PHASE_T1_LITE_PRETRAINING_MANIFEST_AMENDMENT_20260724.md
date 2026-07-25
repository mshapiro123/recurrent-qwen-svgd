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
An initial amendment check in `984a8fbb` also used an incorrectly escaped line
separator in an ad hoc shell command. Before any launcher was committed, all
three manifests were recomputed with the repository's established
`colab.run_stage5_depth_support_ladder.manifest_for_rows` implementation.

The reproducible calibration hashes are:

- row-ID SHA-256: `c58b779ce7b2fc00c8f66d82fdfe414433f6f9be0fe2e26278248eff6a0e016f`
- canonical-row SHA-256: `5df416b614c50eae8bbc44868a19eaa41d84db375563f4a166c8b968a96c1614`

The same audit corrected the existing-source depth 1-8 manifest to
`3a1cff...17746` / `529257...762e` and the depth 9-14 manifest to
`6e808f...8e26c` / `5c5f9b...4d636`.

This amendment corrects only those two integrity fields. It does not alter the
generator, seed, row count, depth distribution, calibration role, gates,
constants, curriculum, model lineage, or registered gated/extrapolation sets.
The launcher asserts these corrected hashes and fails before model loading on
any mismatch.
