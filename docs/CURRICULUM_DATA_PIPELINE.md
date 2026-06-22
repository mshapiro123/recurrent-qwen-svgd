# Curriculum Data Pipeline for Wide and Deep Recurrent Training

This document is the data-generation contract for the depth/width curriculum.
It turns the strategy handoff into implementation constraints for future
dataset builders and training consumers.

## Purpose

Build one typed reasoning dataset where every problem has:

- a verified answer,
- measured width,
- measured depth,
- measured difficulty,
- mode labels for `direct`, `deep_narrow`, `wide`, or `both`,
- positive reasoning traces that are safe for SFT,
- negative and rationalization traces that are visible only to verifier or
  contrastive consumers.

The immediate Stage 5 blocker remains direct-mode repair on ARC rows. This
pipeline is the next data layer after direct calibration is restored.

## Hard Rule

Generation and labeling are separate. A model-generated trace is never trusted
because it sounds plausible.

A trace that reaches a wrong answer may train a verifier, selector, or
contrastive error detector, but it must never enter positive reasoning SFT.
Positive reasoning consumers must load only roles that begin with
`positive_`.

## Model Roles

Use at least two strong, diverse, non-Qwen models for generation and judging.
Do not use the student model or close teacher-lineage models as generators for
this dataset.

- `generator`: proposes problems and candidate methods.
- `solver`: independently solves each problem for ground truth.
- `method_solver`: attempts a solution under a named method constraint.
- `judge`: evaluates method naturalness, distinctness, and error location.
- `weak_reference`: estimates difficulty by pass rate over samples.

Ground truth is accepted only with cross-model agreement plus programmatic
verification when available.

## Method Taxonomy

Width is a structural count, not a surface-style count.

Math buckets should include concrete methods such as algebraic manipulation,
synthetic geometry, coordinate geometry, trigonometry, induction, modular
arithmetic, combinatorial or bijective argument, generating functions,
inequalities, extremal or pigeonhole arguments, complex numbers, calculus, and
bounded enumeration.

Code buckets should include iterative, recursive, dynamic programming, greedy,
divide and conquer, graph/search, mathematical closed-form, and hashing/set
based strategies.

Avoid vague labels such as `logic`; they make width unmeasurable.

## Pipeline Stages

1. **Seed sourcing**
   - Collect curated and generated candidate problems.
   - Decontaminate against evaluation sets before spending solver compute.
   - Treat generator-suggested methods as hints, not labels.

2. **Ground-truth verification**
   - Solve independently with two strong models.
   - Require agreement.
   - Confirm numeric/symbolic answers with programmatic checks where possible.
   - Set aside problems that cannot be verified.

3. **Width by method-constrained solving**
   - For each verified problem and each domain method, ask both models for a
     constrained solution.
   - Permit `METHOD DOES NOT APPLY`.
   - Accept a method only if a solution is correct and judged natural.
   - Use a distinctness judge to avoid counting relabeled variants.

4. **Depth measurement**
   - Decompose each natural correct solution into minimal necessary steps.
   - Store per-method depth.
   - Use minimum depth across applicable methods as the problem-level depth.

5. **Difficulty measurement**
   - Estimate difficulty by pass rate of a fixed weak reference model.
   - Do not accept a model's stated difficulty as a label.

6. **Mode assignment**
   - `direct`: low depth, width one, weak reference usually solves.
   - `deep_narrow`: high depth, width one.
   - `wide`: width at least two and low-to-moderate depth.
   - `both`: high depth and width at least two.

7. **Adversarial perturbation**
   - Present plausible false answers neutrally and under pressure.
   - Sort resulting traces against verified ground truth.
   - Correct resistance traces can become positives.
   - Rationalizations and slips become verifier or contrastive negatives only.

8. **Programmatic deep-narrow generation**
   - For controllable depth, construct problems by templates with known
     answer and step count.
   - Use strong models only to phrase constructed problems naturally.

## Record Schema

```json
{
  "id": "string",
  "domain": "math|code|arc|...",
  "statement": "string",
  "answer": {
    "value": "string",
    "verified_by": ["cross_model", "sympy|exec|numeric"],
    "confidence": "high"
  },
  "difficulty": {
    "pass_rate": 0.0,
    "reference_model": "string"
  },
  "width_signature": {
    "methods": ["algebra", "coordinate_geometry"],
    "width": 2
  },
  "depth": {
    "per_method": {"algebra": 8, "coordinate_geometry": 11},
    "min_steps": 8
  },
  "mode": "direct|deep_narrow|wide|both",
  "decontaminated": true,
  "traces": [
    {
      "role": "positive_wide",
      "method": "algebra",
      "correct": true,
      "natural": true,
      "steps": 8,
      "source_model": "string",
      "text": "string"
    },
    {
      "role": "negative_contrastive",
      "correct": false,
      "error_type": "genuine_slip",
      "source_model": "string",
      "text": "string"
    },
    {
      "role": "verifier_rationalization",
      "correct": false,
      "first_error_step": 3,
      "source_model": "string",
      "text": "string"
    }
  ]
}
```

## Training Consumers

- Direct calibration and deterministic Phase 1 repair consume `direct` rows
  with explicit `target_loop_count = 1`.
- Deep-narrow recovery consumes `deep_narrow` rows with larger explicit loop
  targets and particles off.
- Width/particle training consumes `wide` or `both` rows with method-labeled
  `positive_wide` traces.
- Verifier/selector training consumes `negative_contrastive`,
  `verifier_rationalization`, and `verifier_detection` roles.

No positive SFT loader should read `negative_` or `verifier_` roles.

## Sequencing

1. Finish direct-mode repair and reestablish base-competitive direct accuracy.
2. Build direct and deep-narrow data first, because these train the current
   deterministic failure mode.
3. Add wide and both data only after deterministic recurrent calibration is no
   longer the limiting factor.
4. Use method labels later to supervise particles toward distinct named
   solution families rather than arbitrary hidden-state separation.

## GPU Discipline

Strong-model API generation is not A100 work. It should run as a CPU/API
pipeline with decontamination and validation before any local GPU training.

GPU time is justified only after a dataset shard passes:

- answer verification,
- role validation,
- decontamination,
- positive/negative separation checks,
- a small dry-run loader test.

