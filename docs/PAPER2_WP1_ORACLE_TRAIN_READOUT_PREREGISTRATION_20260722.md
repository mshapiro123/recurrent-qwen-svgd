# Paper Two WP1: Oracle Train-Subset Readout Preregistration

**Locked:** 2026-07-22, before GPU evaluation.  
**Status:** post-hoc diagnostic only. No training or parameter mutation.

## Question

Did the existing additive and FiLM terminal oracle conditioners fit commanded
transitions on seen training variants, or did they fail to fit the command
mapping at all?

This readout cannot alter the registered held-out `BOTH_FAIL` verdict.

## Frozen inputs

- Keeper SHA: `0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f`.
- Additive EMA SHA: `4d5f2cb78f8bab14c6449b4cea8d971f59ad76a661a995ae7e62e883e125235c`.
- FiLM EMA SHA: `d551f1136cf1582e9a6e95be43be952a5f4da1ed5649c45ed29ecd4115e051de`.
- Source rows: the committed 1,899 training variants from the July 18 terminal probe.

All three hashes are asserted before and after evaluation.

## Cohorts

Primary matched cohort: seed `20260722`, 32 prompt groups, eight per depth.
All variants for the selected groups are retained. Group selection is
constrained to reproduce the held-out variant counts exactly:

| Depth | Groups | Variants | Transitions |
|---:|---:|---:|---:|
| 1 | 8 | 16 | 16 |
| 2 | 8 | 22 | 44 |
| 3 | 8 | 27 | 81 |
| 4 | 8 | 41 | 164 |
| **Total** | **32** | **106** | **305** |

Secondary cohort: all 1,899 training variants, 512 groups, 5,617 transitions.
The full cohort is reported to prevent a favorable or unfavorable matched
sample from determining the interpretation.

## Metrics and bands

The scorer and definitions are unchanged from the terminal probe:
non-default transition control, overall transition control, transition
legality, terminal validity, and per-loop localization.

- Non-default control at or above `0.85`: fit seen command mapping.
- Non-default control at or below `0.25`: did not fit command mapping.
- Between `0.25` and `0.85`: partial fit.

Both matched and full-cohort readings are reported. The result is descriptive,
does not reopen Phase G, and authorizes no automatic successor.
