# Paper Two Phase 3 Linear-Decodability Forecast Receipt

Date: 2026-08-11. Status: complete, descriptive, no training.

## Purpose

This read-only job closes P3.3 pre-run assertion A5. It asks whether features
already present at the Phase 3 bridge input linearly predict the strict
14B-and-32B-concurrent oracle direction. It does not test trained bridge aim,
select a P3.3 threshold, or score a confirmation partition.

## Design and integrity

- Source: both migrated full-system E1-confirmation seed lineages.
- Population: 43,204 strict concurrent positions from 37,887 selected anchors.
- Inputs: 1,920-dimensional bridge features at loops 1 through 4.
- Target: 896-dimensional strict-concurrence oracle direction.
- Split: document-disjoint train, calibration, and holdout partitions.
- Model selection: ridge selected on calibration; holdout cosine reported with
  2,000 document-bootstrap replicates.
- The actual-LM-head optimized oracle matched the autograd reference exactly on
  the checked direction values. Batched versus single maximum direction
  difference was 0.0.
- Both seed lineages and all four loops completed. Optimizer steps were zero,
  no teachability threshold was selected, and CONFIRM remained unscored.

## Results

The initial ridge grid ended at 100 and selected that boundary in all eight
fits. Holdout cosine rose slightly with loop index:

| Seed | Loop 1 | Loop 2 | Loop 3 | Loop 4 |
|---:|---:|---:|---:|---:|
| 0 | 0.0743 | 0.0748 | 0.0754 | 0.0758 |
| 1 | 0.0702 | 0.0710 | 0.0714 | 0.0717 |

Because the calibration optimum sat at the grid boundary, the preregistered
post-hoc loop-4 extension evaluated ridge values through 1e8. Both seeds
selected 1e5:

| Seed | Selected ridge | Holdout cosine | Document-bootstrap 95% CI |
|---:|---:|---:|---:|
| 0 | 1e5 | 0.0952 | [0.0842, 0.1077] |
| 1 | 1e5 | 0.0874 | [0.0792, 0.0993] |

The extension corrects the capped-grid underestimate; it does not replace the
eight-loop table or turn the forecast into a gate.

## Interpretation

The bridge input contains weak but replicated linearly decodable information
about the strict concurrent oracle direction. The result is a planning prior
for P3.3, not evidence that the nonlinear bridge will capture a particular
fraction of oracle aim. It is neither an upper bound nor a lower bound on
trained nonlinear performance.

## Durable evidence

- Public summary:
  `outputs/stage5/stage5_paper2_phase3_oracle_forecast_20260810/summary.json`.
- Drive receipt root:
  `/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_phase3_oracle_forecast_20260810/receipts`.
- Drive consolidated summary SHA-256:
  `0cbe273a073d08be104dfc8d468b205653a1e2654ee5c7f9d7f168fb0788810d`.
- Drive oracle-cache summary SHA-256:
  `a4ee3bce8886f92a6a8f4a187a78936aa48b1f9cb1363abf6ad697210a0aa662`.
- Seed-0 ridge-extension SHA-256:
  `908956fdd4f04ca74bcbb837a626a9e574759999a279e3e964c2244f88d43ee4`.
- Seed-1 ridge-extension SHA-256:
  `9d6a1262c2305925f38d057612cfa59a184a660c7ea4f16aef1f9428b4f8beca`.

## Boundary

A5 is satisfied. P3.3 optimizer construction remains blocked by the four
source-to-lock discrepancies in
`docs/PAPER2_P33_LOCK_IMPLEMENTATION_AUDIT_20260811.md`; this receipt does not
resolve or waive them.
