# Handoff: Phase-2 V1b Finite-Perturbation Causal Check

Date: 2026-08-01 (receipt published 2026-08-02 UTC)

## 0. Executive verdict

V1b completed successfully on the corrected batch-matched-neutral protocol. It
was DEV-only and read-only: no optimization occurred and no frozen evaluation
partition was touched.

The result separates two questions cleanly:

1. **Is the first-order analysis locally trustworthy? Yes, at the tested radii.**
   Predicted wrong-token-versus-teacher pair crossings and realized crossings
   agree within 0.1 percentage point at every tested radius.
2. **Is the permitted tube large enough to correct most oracle-help positions?
   No.** At the most permissive tested setting, `c = 0.05`, only 29.75% of
   oracle-help positions flip to the teacher token.

Collateral damage is very low. At `c = 0.05`, oracle-help perturbations caused
75 correctness losses among 930,625 other scored positions (0.0081%), while the
targeted preserve controls retained their target token in 2,000 of 2,000 cases.
Thus the tested local oracle intervention is usually safe but reaches only a
minority of needed corrections. The present constraint is local reach or
interface expressivity, not curvature and not broad collateral instability.

## 1. Why V1b was run

V1 established two derivative-based diagnostics at the Phase-2 insertion point:

- a sampled-maximum-gain compatibility diagnostic, which cannot establish
  reachability or impossibility; and
- the exact local gradient norm of the wrong-token-versus-teacher-token margin,
  which gives a first-order perturbation distance for the relevant margin.

V1b was authorized to test the derivative approximation directly. For sampled
oracle-help positions it applies the normalized steepest teacher-favoring
perturbation at three radii, then recomputes the model. It measures realized pair
crossings, actual teacher-token top-1 flips, and collateral changes elsewhere on
the same row. A matched preserve cohort measures baseline perturbation safety.

## 2. Experimental design

### Substrate and population

- Checkpoint: post-D0 EMA step 4,000.
- Checkpoint SHA-256:
  `8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf`.
- Data: DEV-C only; data SHA-256
  `05bca2ee3ba71421296b2e31a0439746eb9c1b0e15e2cea4471be202ab6ac29d`.
- Teacher-cache summary SHA-256:
  `bf6c0762221d07294fc6f45da0f5791a0241c4466d45b1efbb09d4b7fa3be5f4`.
- Oracle-help positions available: 2,388.
- Preserve-control positions available: 29,230.
- Sample: 2,000 positions per cohort, seed 20260731.
- Oracle-help sample: 713 code, 1,287 general.
- Preserve sample: 1,084 code, 916 general.

### Intervention

For each sampled position, V1b computes the exact gradient of the current
top-1-wrong minus teacher-token margin at the insertion point. It applies the
teacher-favoring perturbation

`delta = -r(c) * gradient / ||gradient||`

for `c` in `{0.01, 0.02, 0.05}`, with

`r(c) = gamma * c * RMS(h0) * sqrt(d) / (1 - rho)`,

`gamma = 0.05`, `rho = 0.8`, and hidden size inherited from the model.

The preserve cohort starts at positions correct under both the baseline and
trained-append path. It receives the same teacher-favoring construction against
the strongest non-teacher competitor.

### Corrected causal control

The preliminary V1b attempt compared a batch-size-8 perturbed forward against a
batch-size-1 reference. It stopped when an earlier argmax differed, but that
comparison could not separate causal leakage from batch-kernel numerical drift.
No scientific result was interpreted from that attempt.

V1b v2 compares every perturbation with an unmodified neutral forward using the
same batch size and batch index. Differences from the registered batch-1 path
are recorded separately. The strict causal assertion remains: any earlier-token
change relative to the matched neutral aborts the run.

## 3. Integrity checks

| Check | Result |
|---|---:|
| Status | `complete_no_training_dev_only` |
| Optimizer steps | 0 |
| Training started | false |
| Frozen evaluation partitions touched | none |
| Intervention records | 12,000 |
| Causal-prefix changes vs matched neutral | 0 |
| Neutral-vs-batch-1 target changes | 0 |
| GitHub receipt commit | `2c9eb343527515c875be5bfb0939ec71d5532785` |
| Public summary SHA-256 | `751df517abfb37fb736af3ea7f46c53c470dc6496dd46270b72c7041384cc186` |

The matched-neutral diagnostic also explains the preliminary failure. Small
batch-shape differences really existed: pooled neutral-versus-registered
prediction-change rates were 0.0015% in the oracle-help cohort and 0.0028% in
the preserve cohort, including two and four earlier-position changes per 2,000
interventions respectively. None occurred when perturbations were compared with
their matched neutral, and none changed the intervention target.

## 4. Primary results

### Oracle-help positions

| c | First-order pair-cross prediction | Realized pair crossing | Teacher-token top-1 flip | Collateral hurt rate | Rows with any collateral hurt |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 7.85% | 7.85% | 7.75% (155/2,000) | 0.0010% (9/930,625) | 8/2,000 |
| 0.02 | 13.65% | 13.75% | 13.70% (274/2,000) | 0.0015% (14/930,625) | 13/2,000 |
| 0.05 | 30.90% | 30.85% | 29.75% (595/2,000) | 0.0081% (75/930,625) | 52/2,000 |

At `c = 0.05`, there were also 105 collateral correctness gains, for a net of
30 gains over losses. All collateral prediction changes were in causally exposed
future positions; prior-position changes were zero.

### Preserve controls

| c | Target retained | Pair crossing | Collateral hurt rate | Rows with any collateral hurt |
|---:|---:|---:|---:|---:|
| 0.01 | 100% (2,000/2,000) | 0% | 0.0006% (6/940,023) | 5/2,000 |
| 0.02 | 100% (2,000/2,000) | 0% | 0.0015% (14/940,023) | 13/2,000 |
| 0.05 | 100% (2,000/2,000) | 0% | 0.0036% (34/940,023) | 31/2,000 |

At `c = 0.05`, preserve interventions produced 57 collateral gains and 34
losses, net +23.

### Domain split at c = 0.05

| Stratum | Oracle pair crossing | Oracle teacher flip | Oracle collateral hurt | Preserve target retained |
|---|---:|---:|---:|---:|
| Code | 32.12% | 30.86% | 0.0096% | 100% |
| General | 30.15% | 29.14% | 0.0072% | 100% |

The result is not being driven by one content stratum. Code is modestly more
compatible than general text, but both show the same qualitative reading.

## 5. Radius and linearity diagnostics

Typical radii were small and tightly concentrated:

| Cohort, c = 0.05 | Median radius | p95 radius | Maximum radius |
|---|---:|---:|---:|
| Oracle help | 0.1524 | 0.1846 | 21.8850 |
| Preserve | 0.1514 | 0.1799 | 21.7900 |

The rare maximum is caused by an extreme state-RMS tail: maximum RMS is about
58 versus medians near 0.4. It does not dominate the pooled counts, but the
private records should be stratified by RMS before any training specification
treats the state-scaled radius as uniformly benign.

Curvature is not the limiting factor at the registered radii. Realized pair
crossing differs from the first-order prediction by 0.00, +0.10, and -0.05
percentage points across increasing `c`. The teacher-token flip rate is slightly
below pair crossing at `c = 0.05` because crossing the selected pair does not
guarantee that no third token remains above the teacher.

## 6. Relation to V1 and V2

V1's 128-position exact-gradient sample estimated first-order compatibility at
4.69%, 14.06%, and 32.03%. V1b's larger 2,000-position oracle sample predicts
7.85%, 13.65%, and 30.90%, then realizes nearly those exact pair-cross rates.
The finite check therefore validates the local margin-gradient diagnostic as a
design instrument at these radii.

By contrast, V1's sampled-maximum-gain diagnostic called 97.66% of its sample
bound-compatible at `c = 0.05`. V1b confirms why that number must not be called
reachable: it is a non-falsification test over sampled directions, while the
position-specific target margin and direction are much more restrictive.

V2 found sampled recurrent-block directional gains with post-D0 medians of
0.691, 0.566, 0.553, and 0.559 over iterates 1 through 4, nearly unchanged from
pre-D0. Some first-iterate directions expanded (maximum 1.448), so this does not
certify contraction. Together with V1b, the data favor an interface-specific
reach problem over a newly introduced D0 instability.

## 7. Interpretation against the pre-stated readings

1. **First-order analysis:** validated locally. The predicted-versus-realized
   gap is negligible, so no curvature-triggered tube-arithmetic revision is
   required at `c <= 0.05`.
2. **Bounded-correction premise:** under pressure. Even an oracle direction
   reaches only 29.75% teacher-token flips at the largest tested radius. This is
   a strong negative signal for correcting most positions with the present
   tube, but it is not an impossibility proof.
3. **Collateral safety:** favorable. Targeted perturbations almost never damage
   other scored positions and preserve targets remain intact. Per-position
   arbitration may still be necessary for a learned controller, but broad local
   collateral is not the observed blocker.
4. **Main localization:** learning the correct direction and obtaining enough
   bounded reach are now the central design problems. Curvature and immediate
   collateral damage are secondary at the tested scale.

## 8. Limitations and do-not-claim boundaries

- This is DEV-only and cannot confirm frozen-slice performance.
- The intervention uses the exact teacher-token margin gradient. It is an oracle
  diagnostic, not a deployable controller and not evidence that a bridge can
  learn the direction.
- Matching the teacher token is agreement with the cached teacher, not an
  independent correctness proof.
- Pair crossing does not guarantee a teacher-token top-1 flip.
- The tested radii stop at `c = 0.05`; behavior outside that tube is unmeasured.
- Collateral positions within a row are correlated, so raw position-level rates
  should not be treated as independent Bernoulli trials.
- Rare high-RMS states produce radii two orders of magnitude above the typical
  range and need explicit stratification.
- Local finite perturbations do not establish global reachability or trainability.

## 9. Decisions requested from strategy

1. Does a 29.75% oracle flip ceiling at `c = 0.05` count as sufficiently low to
   activate the pre-named larger-`c` or E4 upper-layer-adaptation agenda before E1?
2. If the tube is enlarged, should a short DEV-only extension at `c = 0.075` and
   `0.10` be required before training, with the same collateral controls?
3. Should the state-scaled radius receive an absolute or percentile cap after a
   read-only audit of the RMS outliers?
4. Should E1 be designed to cover all oracle-help positions, or should a
   preregistered compatibility-stratified analysis be included without limiting
   the primary population after seeing the result?
5. Does the low collateral rate weaken the case for a heavy arbitration module,
   or should per-position gating remain mandatory because the learned direction
   will be less exact than this oracle direction?

## 10. Recommended next steps

1. Bank V1b as a completed pre-window diagnostic and update the Phase-2 ledger.
2. Run CPU-only/private-record post-processing by RMS quantile, original margin,
   gradient norm, content stratum, and sequence position. Confirm whether the
   high-RMS tail contributes disproportionately to flips or harms.
3. Make the radius-versus-adaptation decision before locking E1. Do not silently
   raise `c` or add upper-layer trainables.
4. If strategy authorizes a radius extension, perform one bounded DEV-only
   finite check with the existing exact protocol before training.
5. Lock E1 gates against the measured oracle ceiling. A learned interface cannot
   reasonably be expected to exceed its own oracle-direction diagnostic without
   changing the tube or trainable pathway.
6. Keep frozen-slice confirmation untouched until the design and training window
   are formally opened.

## 11. Plain-language summary

We asked whether a small, perfectly aimed change to one hidden state could fix a
wrong next-token decision without disrupting the rest of the sequence. The
answer is: often safely, but not often enough. At the largest allowed change,
the idealized intervention fixed about three in ten target positions and almost
never harmed another position. The mathematical local prediction was highly
accurate, so unexpected curvature is not hiding the answer. To fix the remaining
seven in ten, the next design likely needs more allowed movement, a more capable
interface, or limited adaptation above the insertion point.

## 12. Canonical artifacts

- Public V1b summary:
  `outputs/stage5/stage5_paper2_phase2_prewindow_20260731/v1b/summary.json`
- Receipt commit: `2c9eb343527515c875be5bfb0939ec71d5532785`
- Corrected-method amendment:
  `docs/PAPER2_PHASE2_V1B_BATCH_BASELINE_AMENDMENT_20260801.md`
- V1/V2 source summary:
  `outputs/stage5/stage5_paper2_phase2_prewindow_20260731/v1_v2/summary.json`
- Private corrected cache:
  `/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_phase2_prewindow_20260731/private/v1b_neutral_v2`
- Drive receipt copied by the launcher before GitHub publication:
  `/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_phase2_prewindow_20260731/receipts/v1b_summary.json`
