# T1-lite EMA Audit Specification

**Date:** 2026-07-25  
**Mode:** post-hoc, read-only, descriptive  
**Registered T1-lite verdict:** immutable `registered_negative`

## Question

The registered final-step EMA checkpoint failed three gates, while the raw
final checkpoint learned exact depth control at all trained depths and missed
the chain-preservation threshold by eight rows. This audit localizes that
raw-versus-EMA divergence. It does not promote the raw result to the registered
primary result and does not alter any gate.

## Immutable inputs

- Raw final checkpoint SHA-256:
  `a83d056cb4fc366a0b3c3e95b10f00d59e2f624b554acbfac02d922119e5826c`
- EMA final checkpoint SHA-256:
  `1d674a14b7953d72031d72ac8dfd97744120809964b4234ece796efbce849a1e`
- Stage-boundary progress checkpoints at steps 500, 2,500, 6,500, and
  8,500 from the same Drive backup directory.
- Existing 256-row liveness pilot, with a fixed eight-row-per-depth screening
  subset selected using seed 20260725.

## Readouts

1. **Integrity:** verify checkpoint hashes, kinds, steps, matching key sets,
   finite tensors, and the scalar EMA update implementation against its exact
   recurrence.
2. **Stage lag:** evaluate both raw and EMA states from each stage-boundary
   progress checkpoint on rows restricted to the depths trained by that stage.
3. **Endpoint geometry:** report norm, difference, and cosine between raw and
   EMA tensors for the control rows, recurrent block, and bridge.
4. **Interpolation:** evaluate linear raw-to-EMA blends at alpha values
   0, 0.1, 0.25, 0.5, 0.75, 0.9, and 1 on the fixed screening rows.
5. **Group swaps:** exchange one of control rows, recurrent block, or bridge
   between raw and EMA endpoints in both directions. Confirm the strongest EMA
   rescue and strongest raw damage on all 256 pilot rows.

## Interpretations

- A hash, key-set, nonfinite, or scalar-recurrence failure is an implementation
  or artifact-integrity defect.
- Stage EMA that trails a green raw boundary is temporal lag under the staged
  curriculum.
- A sharp interpolation cliff is evidence of a nonlinear model-space averaging
  barrier; a smooth decline is ordinary lag.
- Recovery or damage from one group swap localizes the failure to that group.
  Failure of every single-group swap indicates distributed co-adaptation.

These are descriptive readings, not new thresholds. No training, optimizer
step, checkpoint mutation, seed-1 launch, or registered verdict change is
authorized.

## Artifact-availability amendment after first launch

The first audit launch on 2026-07-25 established that both endpoint
checkpoints are present with the registered hashes, but the four historical
stage-boundary progress checkpoints are absent from their recorded Drive
directory. Their absence is now a reported partial-evidence condition rather
than a fatal error. The audit preserves the archived raw boundary receipts,
marks raw-versus-EMA stage comparisons unavailable, and completes the endpoint
geometry, interpolation, and group-swap localization. It does not reconstruct,
infer, or retrain the missing EMA boundary states.
