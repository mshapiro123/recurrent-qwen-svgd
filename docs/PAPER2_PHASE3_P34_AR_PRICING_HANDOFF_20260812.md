# P3.4 A_r Pricing Audit Handoff

Date: 2026-08-12. Status: complete, read-only, no optimizer, no training. Governing charter: `docs/STRATEGY_P34_CHARTER_20260812.md`, SHA-256 `80cb1b13eb48ffff064ff7cc6c0d02de773dfec80924c1c50736115821c97ce4`.

## 1. Question and design

The audit prices the pre-registered P3.4 capacity-versus-slot fork before either arm spends GPU. On the 43,204 strict concurrent oracle rows, it measures the fraction of oracle-direction energy inside two rank-128 output-space subspaces:

1. The column span of the i1 bridge output projection, separately for each seed.
2. The leading rank-128 covariance subspace of the cached loop-4 base hidden states, matched in dimension.

The oracle cache, both loop-4 feature caches, and both i1 endpoint hashes were verified before computation. The endpoint output matrices had numerical rank 128. The audit used double-precision SVD/eigendecomposition on CPU. It constructed no model, optimizer, or training step.

## 2. Results

| Measure | Seed 0 | Seed 1 |
|---|---:|---:|
| Oracle energy in learned readout span | 25.546% | 26.311% |
| Oracle energy in matched state-covariance span | 14.164% | 14.164% |
| Oracle energy outside learned readout span | 74.454% | 73.689% |
| Readout energy / realized π_dir (14.901%) | 1.714 | 1.766 |
| Top-128 state variance explained | 84.299% | 84.299% |

The seed-mean learned-span fraction is 25.929%, with only 0.382 percentage points of half-range across seeds. A dimension-only isotropic reference is 128/896 = 14.286%; the matched PCA result is close to that reference, while the learned readout is about 1.8 times it. The learned bridge therefore acquired a reproducible aim-aligned output orientation rather than a generic high-variance subspace.

## 3. Interpretation

Three readings are supported.

First, the existing output projection is not geometrically inert. Its span contains substantially more oracle energy than the matched leading-variance state subspace, and this repeats across seeds.

Second, substantial geometric energy remains outside the rank-128 readout span. A wider or second-tower path could in principle address that component, but this measurement does not show that the current scratch state can predict the required coefficients.

Third, realized π_dir is below the energy fraction already inside the current span. Energy compatibility is not a flip-rate bound: margins, direction prediction, magnitude, and the gate all intervene. Still, the gap means a wider output space is not the only plausible bottleneck. The banked full-feature linear forecast is weak at loop 4 (holdout cosine 0.0952 and 0.0874), while the nonlinear bridge achieves 14.901% realized π_dir. Taken together, the evidence leans toward improving the information supplied to the scratch state before paying for a wider output path.

## 4. Recommended fork reading

Provisional recommendation: **slot supervision**, not automatic capacity expansion. The reasons are:

- the learned rank-128 span already carries reproducible aim alignment;
- realized capture has not exhausted the compatible energy already inside that span;
- the prior full-feature forecast says direction information is weakly linearly decodable; and
- LOTUS-style future-token supervision attacks that information bottleneck directly and produces an interpretable frozen-head slot decode.

This recommendation is not the registered verdict. The charter did not define a numerical high/low A_r threshold, and the matched PCA comparator cannot prove that information is absent because predictive signal may occupy low-variance state directions.

## 5. Strategy decisions requested

1. Ratify the slot-supervision arm, or select the capacity arm with the desired reading of the 74% outside-span energy.
2. Bind the high/low A_r rule in one sentence so the executed lock is reproducible.
3. Confirm whether the readout-span statistic plus the banked document-disjoint forecast is sufficient for the fork, or authorize one cheap supervised full-state ceiling before selecting.

No GPU arm should run until this one-line confirmation lands. The score-blind task-inference preflight is built and tested but remains unspent.

## 6. Artifacts and limitations

- Drive receipt: `P34_AR_PRICING_AUDIT_20260812.json`, Drive `1YkJYPo-jiVzkLEsWqXkhetEdgj9D3JwU`.
- Receipt SHA-256: `68c697812804b3e113fcb8cde8f1888821ac81199fcdbe0bad6e964ada7a7c8d` (5,139 bytes).
- Oracle cache: 43,204 rows, SHA-256 `611be787dea0438761d279aa035d5bfe2aa37e74710d880be1066d7ae80a45a2`.
- Seed 0 endpoint: `01c804bc69d35a01730fff236cf5a8d974899d2e4de7e15b92a227b2a9ce5d88`.
- Seed 1 endpoint: `2ed3296f510a6c3a66c451051ecbe2284de03b35dde4052827174a66a10c1d4a`.

Limitations: this is output-space geometry, not a causal capacity intervention; energy fraction is not a certified flip ceiling; the PCA comparator is unsupervised; the same cached base-hidden matrix is shared across seed-specific readout comparisons; and no task correctness was computed.

## 7. Plain-language summary

The bridge has learned a real aiming direction: its 128-dimensional output funnel contains about 26% of the directions that would correct the small model, almost twice what an ordinary same-size high-variance subspace contains. But the system only converts about 15% of oracle opportunities, so the funnel is not yet the sole demonstrated bottleneck. Most direction energy lies outside it, yet the information needed to use a larger funnel is weakly visible in the current state. The measured evidence therefore favors teaching the scratch slots more directly, while leaving the final arm choice to the pre-registered strategy confirmation.
