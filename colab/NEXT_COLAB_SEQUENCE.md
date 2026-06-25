# Next Colab Sequence

This is the short execution queue for the current program state. It follows the
master sequence, but the re-entry repair stages have already passed their
immediate gates. The active blocker is deterministic recurrent competence, not
loop-closure liveness and not particle/SVGD geometry.

## Current Queue

1. `traced_sft_competence_preserving_pipeline`
   - Runtime: L4/T4 is preferred; A100 only if already attached and credits are
     acceptable.
   - Purpose: resume from the repaired Stage 4 recurrent checkpoint and run
     ARC-mixed deterministic recovery with stronger competence preservation.
   - Gate: checkpoint restore preflight must pass; ARC-mix child summary must
     be finite; full recurrent-vs-base assessment runs only if the ARC-mix proxy
     passes.
2. `review_stage5_competence_pipeline.py`
   - Runtime: CPU/local.
   - Purpose: summarize the competence wrapper, child statuses, selected
     checkpoint, and planner-selected next action after the GPU run lands.
3. `debiased_benchmark_suite` or recovery full assessment
   - Runtime: L4/T4.
   - Purpose: confirm recurrent-vs-base behavior under content and cyclic MCQ
     scoring after the competence-preserving proxy passes.
4. `dense_mcq_trace_sft_control`
   - Runtime: L4/T4.
   - Purpose: train/evaluate standard dense Qwen LoRA on the same curriculum so
     any later architecture claim is separated from data-recipe lift.
5. Phase 2 breadth diagnostics
   - Runtime: deferred.
   - Purpose: rerun `effective_pathways_diagnostic` and
     `candidate_conversion_diagnostic` only after deterministic recurrence is
     base-competitive and dense-control comparison is interpretable.
6. Particles/SVGD
   - Runtime: deferred.
   - Purpose: test particle breadth only after correct-bearing deterministic
     alternatives exist.

## Fresh Launcher

For the current front-of-queue action, prefer the tracked fresh launcher in
[`CURRENT_A100_ACTION.md`](CURRENT_A100_ACTION.md). It fetches
`colab/CURRENT_STAGE5_FRESH_LAUNCHER_CELL.py`, hard-resets the repo, mounts
Drive, and executes `traced_sft_competence_preserving_pipeline`.

Expected early output:

```text
launcher_version: fresh_launcher_v1
ee304c7 ... or newer
checkpoint_restore_preflight=ok ...
```

## Cheap Status

When a runtime restarts or the notebook state is unclear, use:

```python
TARGET = "master_sequence_status"
```

This target is still useful as a cheap pointer/reviewer readout, but it is not
the current GPU action. It should not be used as a substitute for the
competence pipeline review after the GPU run lands.

## Completed Context

The following targets are completed gates for the current checkpoint lineage and
should not be rerun unless deliberately doing archaeology:

```text
reentry_drift_diagnostic
reentry_norm_diagnostic
reentry_repair_smoke
reentry_recovery_training
```

Stage 3 made the bridge/re-entry adapter path live; Stage 4 kept re-entry health
sane after recovery SFT. The next question is whether deterministic recurrent
competence can be preserved and recovered enough to pass base comparisons.

## Parallel CPU/API Data Work

`claim_curriculum_scaleup_cpu` remains useful in a CPU or cheap non-GPU runtime,
but it is not a GPU gate. It prepares later direct/deep curriculum data and
should not hold an attached paid GPU runtime open.

## Stop Conditions

Stop and review if any of these occur:

- checkpoint restore preflight does not print `ok`;
- the competence pipeline fails before training;
- ARC-mix summary is missing or nonfinite;
- ARC-mix proxy does not pass;
- full assessment runs and still trails base under the balanced/debiased gate;
- any recommendation asks for dense control, particles, GPQA, or scale-up before
  deterministic recurrent competence has passed the base-comparison gate.
