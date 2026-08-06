# Phase-2 A2 Step-237 Tripwire Amendment

Date: 2026-08-06

Status: `locked_before_a2_step237_continuation`

## Authority

This amendment implements the strategy resolution
`STRATEGY_TO_CODING_AGENT_A2_TRIPWIRE_RESOLUTION_20260806_r3.md` and the
binding `STRATEGY_GUARDRAIL_DOCTRINE_20260806.md`. It changes stopping and
resume mechanics only. It does not change the model, optimizer, losses, loss
weights, rows, evaluation sets, directional contract, endpoint gates, extension
rule, or paired comparison.

## Exact continuation

All four arms resume from their landed step-237 checkpoints. The batch generator
is reconstructed from common row seed `20260805` by replaying exactly 237 batch
draws, regardless of the generator state stored in the checkpoint. Before any
forward pass, the runner asserts that the next selected batch is attempt 238 with
row hash
`3fee92d729b86a2a354a259562343f5ba7cf084e2b01bae84012fdc5c4a3d716`.
It restores the reconstructed pre-selection state afterward, then training selects
and applies that same batch normally in every arm. The batch is neither skipped
nor clipped.

The four source checkpoint SHA-256 values are locked in the machine-readable
preregistration. Existing optimizer and model states are loaded unchanged. The
historical static-gradient abort is cleared only for the two full arms. Controls
must have no abort to clear.

## Gradient rule

The old seed-specific static raw-gradient thresholds become telemetry. Their
values and exceedances remain logged, but they cannot stop or shape training.

The replacement tripwire is raw gradient norm greater than ten times the median
of the prior 100 observed optimizer-step gradient norms on three consecutive
steps. The current norm is excluded from its own reference. The source
checkpoints do not contain historical per-step norms, and cross-sectional norms
evaluated at the step-237 checkpoint are not trajectory history. Therefore the
tripwire is telemetry-only for the first 100 newly observed continuation norms
and arms at the following step. Non-finite loss and gradient checks remain
immediate throughout this warmup. The rolling state is checkpointed so an
interrupted run does not restart the warmup.

## Guardrail inventory

The complete rule inventory is embedded in
`training/paper2_phase2_staged_repilot_preregistration.json`. Every entry records
its threshold, estimator, reference point, cadence, disposition, and named cliff.
The CPU grounding sweep validates that every stop-authority rule names one of the
four doctrine cliffs. Rules without a named cliff are launch-invalid and must be
demoted to telemetry before the lock can pass.

The armed stops are limited to non-finite loss or gradient, the trajectory-relative
explosion tripwire, frozen-lineage or source-identity mutation, frozen-partition
contact, control-path mutation, the registered Wilson quality floor, the
trajectory-grounded two-evaluation retention loss, and the locked two-tier
directional contract. Endpoint qualification and extension decisions execute only
at their registered endpoints. The obsolete static gradient ceiling, negative
retention slope, and intermediate endpoint metrics are telemetry or warnings.

## Receipt obligations

The completion receipt includes the validated inventory, exact source and next-row
hashes, pre- and post-reconstruction generator assertions, rolling gradient-norm
telemetry, static-threshold exceedance counts, relative-tripwire events, every
stop's resumable state, V1d acknowledgment and source receipt, and all pre-existing
paired endpoint outputs.
