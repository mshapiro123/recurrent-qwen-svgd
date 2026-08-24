# Bicameral W1 Phase B and Generative Contract Request

Date: 2026-08-24. Coding-agent pre-execution clarification request to strategy.

## 1. Purpose

Phase A can resume as soon as the separate injection-position ruling lands.
Three later operations in the W1 authorization are not mechanically complete,
however. Resolving them now avoids a second paid-compute stop after the
both-seed Phase A result.

No optimizer has been constructed. CONFIRM and EVAL-E remain sealed.

## 2. Phase B cluster assignments

The Stage-0 k=2 assignments cover the frozen 256-row arm-6 population. Phase B
is specified as L1/L2/L3 on the 2,048-row DEV-2 population, but the
authorization does not define assignments for the other 1,792 rows.

**Recommended ruling:** freeze each seed's Stage-0 feature transform and k=2
centroids, compute the identical features for all DEV-2 rows, and assign every
row to its nearest frozen centroid. Then compute L1/L2/L3 from the winning
target family over all 2,048 rows. Do not refit clusters on DEV-2. Report the
full-population cluster counts and per-battery composition before scoring.

Alternative: restrict L1/L2/L3 to the original 256 rows and provenance-tag
them as a separate population. This is cheaper but is not directly comparable
to the 2,048-row L0 result.

Required input either way: identify the exact feature transform, centroid,
and assignment artifacts by file, bytes, and SHA-256 for both seeds.

## 3. L6 residual directions

The authorization names `+/-u1` first and `+/-u2,u3` if budget permits, but
does not identify the authoritative tensors or whether the directions are
seed-specific, pooled, or sign-oriented against a reference correction.

**Recommended ruling:** use each seed's own cross-fitted R-S0-A residual
eigenvectors, ordered by descending residual eigenvalue. Score both signs for
each admitted direction as separate cells and retain the sign in every arm
name and receipt. A direction counts positive only under the already registered
both-seed rule; do not select the better sign post hoc. Provide the tensor
artifacts and hashes before staging.

## 4. Generative staging

The Phase A row targets are computed from teacher-forced answer-bearing
positions. The authorization requires a 461-row generative cell for any arm
whose margin CI clears zero, but it does not specify how a fixed oracle target
is used during autoregressive decoding.

**Recommended ruling:** for each registered 461 row, compute the arm's target
once using the same teacher-forced construction as the margin panel, then hold
that vector fixed and apply it through the frozen final-cell write convention
at every autoregressive decoding step. Use the same gamma=0.05, serving reader,
generation parameters, and execution schedule as the frozen depth-study
config-2 evaluator. Label every result `oracle-target-assisted`; it is a causal
capability read, not a deployable routing result. Score own shuffle and random
controls on the same rows whenever their parent arm is staged.

Alternative: omit generative staging because answer-derived directions leak
the reference answer into generation. If omitted, record that as a strategy
ruling rather than an execution-time trim.

Required input: identify the exact 461-row manifest and frozen generation
configuration by artifact hash.

## 5. Requested response

Bind one option in each section and provide the named artifacts. These choices
do not change Phase A's targets, constants, controls, winner rule, or cost cap.
They complete the already-authorized follow-on graph without allowing the
coding agent to invent an estimator after seeing the both-seed result.
