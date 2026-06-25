# Stage 5 Re-entry Stage 3/4 Runbook

This runbook is the CPU-side contract for the next GPU work. It exists to keep
the re-entry repair sequence aimed at deterministic recurrent recovery before
we return to particles, SVGD, or broader architecture experiments.

This is Phase 0 in the master program sequence. See
[`PROGRAM_TRACK_MASTER_SEQUENCE.md`](PROGRAM_TRACK_MASTER_SEQUENCE.md) for the
full dependency chain: re-entry, then depth, then breadth/multistability, then
particles/SVGD and selector conversion.

## Current blocker

Stage 1 re-entry drift found that the current recovered recurrent checkpoint has
a dead bridge:

- `bridge_gate = 0.0`
- bridge delta RMS `= 0.0`
- bridge projection, bias, and gate gradients `= 0.0`
- loop norm drift is bounded, but entry/exit subspace overlap is low

This means the model can look numerically stable while still lacking a
trainable loop-closure path. Do not spend GPU on more particle-noise or SVGD
sweeps until this path is repaired and deterministic recurrence is
base-competitive again.

## Stage 2: eval-only re-entry normalization

Target:

```text
STAGE5_CURRENT_A100_TARGET=reentry_norm_diagnostic
```

Hypothesis: loop re-entry RMS normalization can reduce re-entry distribution
drift without materially reducing deterministic candidate conversion.

Success:

- assessment recommendation is `run_reentry_repair_smoke`;
- no major candidate-conversion regression from `none` to `entry_rms`;
- artifacts include drift, effective-pathway, candidate-conversion, and
  `reentry_assessment` outputs.

Failure:

- assessment recommendation is `review_before_trainable_repair`;
- candidate conversion regresses materially;
- artifacts are incomplete.

If the GPU run completed but failed before Git publish, use:

```text
STAGE5_CURRENT_A100_TARGET=reentry_norm_recover_only
```

The recover-only target can also salvage a late-interrupted run that has all
raw drift, effective-pathway, and candidate-conversion files but is missing the
final `summary.json`/`summary.md`; it rebuilds the summary and assessment
without rerunning GPU evaluation.

## Stage 3: trainable re-entry repair smoke

Target:

```text
STAGE5_CURRENT_A100_TARGET=reentry_repair_smoke
```

Hypothesis: an identity-preserving but gradient-live bridge plus the
identity-initialized re-entry adapter can move under a tiny continuation without
breaking loop-1 behavior.

Configured behavior:

- resumes from the current deterministic recovered checkpoint;
- resets the bridge to identity and overrides `bridge_gate=1.0`;
- trains only `bridge,reentry,halt`;
- uses `entry_rms` loop re-entry normalization;
- enables `use_reentry_adapter`;
- stops after a small smoke run and disconnects.

Success:

- final repair-smoke training metrics are present and finite;
- supervised depth metrics (`target_loop_abs_error`, `halting_target_nll`) are
  present when halt-depth supervision is enabled;
- bridge gradients are live;
- bridge delta changes measurably;
- re-entry adapter scale/bias gradients are live;
- re-entry adapter moves measurably;
- loop-1 preservation is present, comparable, informative on at least one
  source-correct example, and non-regressed;
- assessment recommendation is
  `run_bounded_recovery_training_with_reentry_repair`.

Failure responses:

- `fix_repair_smoke_training_log_before_recovery_training`: do not train; the
  repair smoke did not publish final step metrics.
- `fix_repair_smoke_training_before_recovery_training`: do not train; the
  final repair-smoke loss is missing or nonfinite.
- `fix_repair_smoke_depth_supervision_before_recovery_training`: do not train;
  depth-supervision metrics were not reported even though halt-depth loss was
  enabled.
- `fix_loop1_preservation_eval_before_recovery_training`: do not train; the
  preservation evidence is missing, mismatched, or source-zero and therefore
  uninformative.
- `review_or_reduce_repair_lr_before_recovery_training`: do not train; loop-1
  preservation regressed.
- `fix_reentry_adapter_before_recovery_training`: do not train; adapter
  gradients are not live.
- `extend_reentry_repair_smoke_or_increase_adapter_lr`: retry only a bounded
  Stage 3 variant; adapter is live but did not move.
- `extend_reentry_repair_smoke_or_increase_bridge_lr`: retry only a bounded
  Stage 3 variant; bridge is live but did not move.

## Stage 4: bounded recovery SFT

Target:

```text
STAGE5_CURRENT_A100_TARGET=reentry_recovery_training
```

Hypothesis: after the loop-closure path is live, deterministic recurrent SFT
can recover base behavior while using target-loop supervision to preserve the
depth gradient.

Configured behavior:

- refuses to run unless Stage 3 assessment recommends
  `run_bounded_recovery_training_with_reentry_repair`;
- refuses stale recommendation-only Stage 3 assessments that do not include
  finite train metrics, supervised depth metrics, loop-1 preservation evidence,
  source-correct loop-1 preservation signal, and live/moved bridge/re-entry
  repair evidence;
- resumes from the Stage 3 repaired checkpoint;
- trains `bridge,reentry,halt,lora`;
- enables learned loop control and target-loop NLL supervision;
- keeps `entry_rms` loop re-entry normalization active during training and
  validation;
- carries the re-entry adapter forward;
- uses strict target-loop row gates derived from the actual trace collection.
- backs up the run directory to Drive immediately after training, before
  validation, and refreshes that backup after the final summary is written.

Benchmark handoff:

- the maintained `debiased_benchmark_suite` target evaluates the repaired
  Stage 4 checkpoint with learned loop control enabled by default;
- use an explicit override only for older non-depth-router checkpoints.

Critical gate:

`STAGE5_CURRICULUM_MIN_TARGET_LOOP_ROWS` must preserve the real row counts, for
example:

```text
1=48,2=16,4=8
```

It must not collapse to presence-only gates such as:

```text
1=1,2=1,4=1
```

Success:

- finite loss and validation metrics;
- target-loop gradient is present;
- loop-depth behavior changes measurably;
- direct/loop-1 behavior does not collapse;
- recovered deterministic model is ready for broader base-vs-recurrent
  benchmark assessment.

Failure:

- nonfinite trainable params or gradients;
- target-loop gradient missing;
- target-loop row gates fail;
- direct behavior regresses sharply;
- checkpoint provenance cannot be restored from Drive/Git.

## Review command

After every re-entry stage, run:

```bash
python colab/review_stage5_reentry.py --no_write
```

The reviewer is intentionally CPU-only. It converts the latest committed
`reentry_assessment.json` artifacts into one next target or one explicit stop
reason.

When the reviewer prints a `Launch Env` section, use those key/value pairs for
the next Colab run. This is especially important for Stage 3 retry cases:
`extend_reentry_repair_smoke_or_increase_adapter_lr` and
`extend_reentry_repair_smoke_or_increase_bridge_lr` intentionally use a bounded
50-step retry with a modest LR increase instead of rerunning the same failed
smoke unchanged.

## Return to particles/SVGD

Resume Phase 2/SVGD only after Stage 4 produces a deterministic recurrent model
that is base-competitive again. The particle mechanism should then be evaluated
for candidate conversion, not superficial diversity:

- failed groups should gain correct candidates;
- correct-bearing pathway counts should increase;
- selector conversion should improve on hard slices.
