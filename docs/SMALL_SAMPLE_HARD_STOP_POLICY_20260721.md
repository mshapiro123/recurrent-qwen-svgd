# Small-Sample Hard-Stop Policy

**Effective:** prospectively after the E3b Arm S stop on 2026-07-21. Historical
decisions remain scored under their registered rules.

## Rule

A continuous margin must not be treated as continuously measurable when the
guardrail has a small, discrete sample. Future hard stops must record the metric's
resolution and use an uncertainty method appropriate to the observation model.

- For paired binary canaries, use row-aligned outcomes and an exact paired
  binomial/McNemar-style test or an explicitly registered binomial confidence
  bound. Do not use Student's t for binary hits.
- For approximately continuous measurements, use a small-sample interval based
  on Student's t when its assumptions are defensible.
- Convert a continuous accuracy boundary to counts permissively. A count that is
  only the first representable value beyond the nominal boundary is a review
  event unless the registered uncertainty criterion also supports the stop.
- Repeated checkpoint looks require a registered sequential correction, alpha
  spending rule, or an anytime-valid interval if inferential evidence is used.
- Always report the point estimate, row count, item resolution, confidence or
  exact-test result, and the rounded count boundary.

## E3b application

Arm S moved from 60/64 to 58/64. Each item is 1.5625 percentage points, and the
observed -3.125-point change exceeded the registered -3-point boundary by only
0.125 points. The historical hard stop remains valid because it was registered
and executed mechanically. Under this prospective policy, that outcome would be
reported as a near-boundary review event pending row-level paired evidence rather
than treated by its point estimate alone as established capability regression.

This policy changes future launch criteria. It does not authorize resuming Arm S,
changing E3b's endpoint, or reclassifying the recorded stop.
