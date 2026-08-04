# Paper Two Phase-2 Matched-Alpha Audit Handoff

Date: 2026-08-04  
Status: complete read-only audit  
Result commit: `cc78f823526a76b8aff3c5cb46f0adf4c6187056`

## 1. Executive verdict

The audit confirms the pilot's constrained-feasibility negative and identifies
the principal confounds. It does not select alpha and does not authorize E1.

The registered trust ceiling was a learning shaper, not a neutral safety
tripwire. At the actual beta-0.5 target, the demanded update exceeded the
registered permission on 92.1% of partial-whitening rows and 84.0% of
full-whitening rows. At beta 1.0, every row in every arm exceeded permission.
The hard trust stop therefore prevented the experiment from observing the
natural update distribution it was intended to measure.

The joint objective was also heavily distorted by gradient scale. On the fixed
16-anchor terminal atlas, the functional-probe loss contributed 94.0-97.8% of
post-clip gradient norm in the head/refiner groups for alpha 0 and 0.5. The
head and refiner groups clipped on every training step in those four arms. The
bridge did not clip and was almost entirely driven by final-token cross-entropy.

None of the six terminal checkpoints improved mean accepted length. Every
reported workload, position, and teacher-disagreement subgroup had a negative
mean acceptance delta. Gate magnitude had near-zero correlation with either
acceptance change or quality loss. The evidence therefore does not support an
arbitration-only amendment as the next move.

## 2. Audit design and integrity

The audit loaded all six exact terminal checkpoints and evaluated the same
8,031 document-isolated DEV anchors. It constructed no optimizer, performed no
parameter update, and touched no frozen E1 partition. Trainable parameter
hashes were unchanged before and after every evaluation. Student and 14B
teacher head hashes matched the locked sources.

The trust stop was reproduced from the stored per-step booleans: 51 active
steps for every alpha-0.5 and alpha-1.0 arm. Historical trust-rent magnitudes
were not stored and were not reconstructed. Scheduled-evaluation rent is
reported only as a proxy.

## 3. Exact terminal results

| Alpha | Seed | Step | Stop | Retained / 23,606 | Retention | Acceptance delta | Flow loss |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0.0 | 0 | 500 | quality qualification | 23,525 | 0.99657 | -0.00053 | 0.07053 |
| 0.0 | 1 | 600 | quality qualification | 23,529 | 0.99674 | -0.00309 | 0.06866 |
| 0.5 | 0 | 193 | trust saturation | 23,548 | 0.99754 | -0.00304 | 0.09353 |
| 0.5 | 1 | 184 | trust saturation | 23,569 | 0.99843 | -0.00192 | 0.09189 |
| 1.0 | 0 | 146 | trust saturation | 23,558 | 0.99797 | -0.00265 | 0.34564 |
| 1.0 | 1 | 146 | trust saturation | 23,543 | 0.99733 | -0.00244 | 0.33892 |

The partial- and full-whitening checkpoints still satisfied the registered
endpoint quality qualification at their exact stops. Their stop points were
not quality catastrophes. This does not establish that unconstrained training
would remain safe, but it confirms that the guessed trust shaper stopped
learning before an observed endpoint collapse.

Alpha 0 never contacted the trust threshold but failed endpoint qualification
after longer training. Removing trust alone is therefore not a solution.

## 4. Constraint geometry

| Alpha | Beta-0.5 demand above permission | Beta-1 demand above permission | Huber linear-regime occupancy |
|---:|---:|---:|---:|
| 0.0 | 30.9% | 100% | 0.049% |
| 0.5 | 92.1% | 100% | 0.438% |
| 1.0 | 84.0% | 100% | 18.2% |

This establishes two separate confounds. First, the trust constant was
incompatible with the target geometry for most whitened rows. Second, the
fixed coordinate-wise Huber delta changed the effective objective across
alpha: full whitening placed roughly two orders of magnitude more coordinates
in the linear regime than alpha 0.

## 5. Effective optimization pressure

Average post-clip norm shares on the fixed terminal atlas:

| Alpha | Group | Functional probe | Flow | Final CE |
|---:|---|---:|---:|---:|
| 0.0 | heads | 97.8% | 1.0% | 0.9% |
| 0.0 | refiner | 97.6% | 1.1% | 0.9% |
| 0.5 | heads | 95.7% | 2.9% | 1.0% |
| 0.5 | refiner | 94.0% | 4.5% | 1.1% |
| 1.0 | heads | 70.0% | 27.0% | 1.7% |
| 1.0 | refiner | 57.3% | 40.7% | 1.3% |
| all | bridge | 0% | 0% | about 99.8% |

The functional probe determined about 99% of the signed combined head/refiner
update for alpha 0 and 0.5. At alpha 1.0 the balance was less extreme, but one
seed showed the flow component opposing the final combined refiner direction.
These are terminal 16-anchor measurements, not a reconstruction of every
training step, but they agree with the recorded clip fractions: heads and
refiner clipped 100% of steps for alpha 0 and 0.5; alpha 1.0 clipped the
refiner on 96.6-99.3% and heads on about 31%.

## 6. Row-level localization

Across arms, 46.2-47.9% of rows improved accepted length while 52.0-53.7%
worsened. Mean acceptance remained negative in every code/general stratum,
every position bucket, and every teacher-disagreement-count bucket. The
three-disagreement bucket had a majority of individually improved rows, but
its mean delta was still negative in every arm.

Draft-gate correlations with acceptance delta ranged only from about -0.024
to 0.021. Correlations with quality loss ranged from about -0.035 to 0.017.
The existing gate therefore does not expose a usable benefit/harm selector.

From each last quality-passing scheduled evaluation to the exact stop,
acceptance worsened on average in all six arms. Partial and full whitening
retained endpoint qualification despite this decline; alpha 0 did not.

## 7. Guardrail interpretation

The r3 tripwire-versus-shaper principle is supported by the evidence.

- Hard tripwires remain appropriate for non-finite values, lineage mutation,
  and identity failure.
- The 0.997 point and 0.990 Wilson rules remain endpoint qualification gates,
  not catastrophe definitions.
- A quality-collapse tripwire needs a separately grounded, more generous
  threshold before a re-pilot.
- Trust rent and module clipping should begin the exploratory re-pilot in
  observe-and-log mode. Their natural distributions must be measured before
  they are allowed to shape optimization.

## 8. Adopted staged amendment

Strategy adopted the optimization-pressure amendment as the one identifying
change (Drive `1XjiWs4zQy7o_ygipy7oNA53BW-gMtP2u`, 8,574 bytes, SHA-256
`17d6d67f0e25613cca30ab8a94d37e9cda044144b7e5948f1b48b12602288c1d`).
It does not combine the repair with a trust projection, router redesign, or
alpha grid.

**Stage A1: state construction.** Arbitration/writeback/drafter gates are held
closed. Active losses are flow, functional probe, and preserve KL. A 100-step
calibration phase on the training partition sets static loss weights before
the A1 weights freeze. The locked targets are flow at least 50% and probe at
most 25% of refiner-path post-clip gradient norm. The Huber delta is the
training-partition p75 of per-coordinate target-increment magnitude, computed
for the alpha-0.5 arm before training. Coding recommends that calibration be
gradient-only so it cannot alter the registered initialization, but this is a
lock field rather than an inference from the strategy text.

**Stage A2: state use.** Flow parameters are frozen, not merely slowed.
Acceptance-facing losses and arbitration heads activate. The alpha-0.5 flow
learned in A1 cannot co-adapt, so an A2 failure localizes to use. A flow
learning-rate multiplier of 0.1 is held as a preregistered fallback only if A2
evidence shows that frozen flow is misaligned with use.

Trust has zero loss weight. Both ratios remain telemetry. The only trust
tripwire is sustained ratio above 5 over a 100-step window. Per-module clip
ceilings are set to approximately 10 times the calibration p99 norm; clip
activation is telemetry with an alarm above 1% of steps, not a continuous
optimizer shaper.

The re-pilot uses alpha 0.5 only and seeds 0 and 1. Alpha remains unselected;
the A35 matrix returns only after the repaired recipe qualifies. Exact A1/A2
budgets, the extension allocation, whether calibration updates parameters, the
static-weight solver, and the operational definition of sustained ratio remain
transcription fields that must be resolved in the amended preregistration
before its lock commit.

Pre-register a slope-based budget reading: if acceptance and qualification are
healthy but still improving at the planned endpoint, classify the result as
budget-limited rather than as a mechanism failure.

## 9. Decisions and boundaries

- Alpha remains unselected.
- No 0.25/0.75 refinement is justified.
- E1 remains blocked.
- A trust-constrained parameterization is not favored because the required
  beta-0.5 target itself lies outside the old permission for most whitened
  rows.
- A router-only amendment is not favored because no measured subgroup has a
  positive mean and the present gate has no useful outcome correlation.
- One staged optimization re-pilot is justified after its constants and
  catastrophe thresholds are empirically locked.
- The oracle-selector ceiling is a CPU-only post-processing receipt. It prices
  the benefit population but cannot authorize a router because it uses perfect
  hindsight.
- V1d is already complete, not missing: capped `c=0.15` produced 1,209/2,000
  teacher-token top-1 flips and retained 1,997/2,000 controls. Its Drive receipt
  is `1zH20VEuuc4myQl9pvFgv56iQ4tNXa4iQ`.

## 10. Canonical artifacts

- `outputs/stage5/stage5_paper2_phase2_matched_alpha_audit_20260804/summary.json`
- `outputs/stage5/stage5_paper2_phase2_matched_alpha_audit_20260804/receipt.md`
- six arm-level JSON files in the same directory
- private exact-row tensors in the matching Drive artifact directory
- result commit `cc78f823526a76b8aff3c5cb46f0adf4c6187056`
- strategy response Drive `1XjiWs4zQy7o_ygipy7oNA53BW-gMtP2u`
- `docs/PAPER2_PHASE2_V1D_CAPPED_RADIUS_HANDOFF_20260804.md`

## 11. Plain-language summary

The pilot did not fail because one whitening strength was slightly wrong. It
asked the model to make updates that the trust rule prohibited on most rows,
then let one auxiliary probe dominate nearly every trainable update. The
model's internal proxies moved, but accepted length became slightly worse in
every arm and in every broad subgroup. The safety stop was useful as a
diagnostic, but its guessed constant became part of the treatment.

The next experiment should separate learning the latent correction from
deciding when to use it. It should measure trust and gradient norms before
constraining them. That is a cleaner, single-mechanism test than relaxing the
trust ceiling, adding a router, or searching more alpha values.
