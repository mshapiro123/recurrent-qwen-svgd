# Strategy Addendum: V1 Terminology Correction Accepted; V1b Authorized

Date: 2026-07-31. This document amends section 2 and V1 of
`THEORETICAL_FOUNDATIONS_AUDIT_20260731.md`, and section 2.1 of
`STRATEGY_TO_CODING_AGENT_PHASE2_OPENING_20260731.md`. Until the audit text is
revised, this addendum governs.

## 1. Terminology correction

Sampled directional gains `||Jv|| / ||v||` are bounded above by the operator
norm. Their maximum is an empirical lower estimate of the local maximum gain,
not an upper bound. Centered finite differences add approximation error.

- A certified upper bound, which V1 does not provide, could prove that a margin
  above bound times radius is unreachable. It could not prove reachability.
- The sampled maximum gain establishes only bound compatibility. It cannot
  establish impossibility or certify a token flip.
- The position-specific norm of the gradient of the wrong-token-versus-teacher
  margin gives the locally optimal first-order rate. The ratio of margin to
  gradient norm is comparable to the permitted radius, but remains a local
  approximation subject to curvature and finite-radius direction constraints.

Receipts use `bound-compatible fraction using sampled maximum gain` and
`first-order compatibility using the margin-gradient norm`. The phrase
`reachable fraction` is retired.

## 2. Pre-stated V1 interpretation

Low first-order compatibility at `c = 0.05` is a strong negative signal. High
compatibility on both diagnostics is non-falsification, not validation. When
the sampled-gain and exact margin-gradient diagnostics diverge, the latter has
interpretive precedence.

## 3. V1b authorization

V1b is a separate DEV-only, no-update receipt sequenced after V1. On a seeded
sample of 2,000 oracle-help positions, it applies

`delta = -r(c) * grad(margin) / ||grad(margin)||`

for `c` in `{0.01, 0.02, 0.05}`, where

`r(c) = gamma * c * RMS(h0) * sqrt(d) / (1 - rho)`.

For each radius it reports the first-order predicted crossing of the original
wrong-versus-teacher pair, the realized pair crossing, the realized top-1 flip
to the teacher token, and correctness changes at every other scored position
on the same row. Pair crossing and teacher-token top-1 are separate because a
third competitor can remain above the teacher even after the original pair
crosses. The causally exposed suffix is also reported separately, and any
change before the perturbed position is a fatal causal-contract violation.

A matched-size preserve control uses positions where both the baseline and
trained append paths are teacher-correct. Its strongest non-teacher competitor
defines the same wrong-versus-teacher margin, and the same teacher-favoring
normalized perturbation is applied. This is a norm- and objective-matched
control; it does not deliberately attack a correct token.

Realized pair crossings near first-order predictions support the local linear
analysis. A large shortfall triggers tube-arithmetic revision before E1. High
collateral damage at flipping radii makes per-position gating load-bearing.

## 4. Boundaries

V1b touches DEV material only, changes no parameters, and does not contact a
frozen evaluation slice. It informs E1 design but does not independently gate
the Phase-2 window unless its curvature result triggers the pre-stated
tube-arithmetic revision. It does not convert a local perturbation into a
deployable controller claim.
