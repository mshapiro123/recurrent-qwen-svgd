# Curriculum Data Generation and Validation Pipeline

This document is the data-generation contract for the depth/width curriculum.
It turns the strategy handoff into implementation constraints for future
dataset builders and training consumers.

The goal is one typed reasoning dataset where every problem carries a verified
answer, measured width, measured depth, measured difficulty, and reasoning
traces routed to explicit training roles. The immediate Stage 5 blocker remains
direct-mode repair on ARC rows. This pipeline is the next data layer after
direct calibration is restored.

## Core Rule

Generation and labeling are separate operations.

Models generate a wide and deliberately messy distribution of traces. The
verified answer sorts those traces into roles. No model self-assessment of
correctness, method, difficulty, or number of solution paths is trusted.

The hard routing rule is:

> A trace that reaches a wrong answer may train a verifier, selector, or
> contrastive error detector, but must never enter positive reasoning SFT.

Positive reasoning consumers must load only roles that begin with
`positive_`. Enforce this in code, not by convention.

## Model Roles

Use at least two strong, diverse, non-Qwen models for generation and judging.
Examples include models in the GPT, Claude/Opus, and GLM families. Do not use
the student base, the student lineage, or close Qwen-derived teachers as
generators for this dataset.

- `generator`: proposes problems and candidate methods.
- `solver`: independently solves each problem for ground truth.
- `method_solver`: attempts a solution under a named method constraint.
- `judge`: evaluates method naturalness, method distinctness, and error
  location.
- `weak_reference`: estimates difficulty by pass rate over repeated samples.

Ground truth is accepted only with cross-model agreement plus a programmatic
check when available. A single model verifying its own output is not
validation.

## Agreement Rules

Assign models to roles instead of treating them as interchangeable.

- Both generator models can propose seed problems.
- Both solver models solve each problem independently.
- Numeric, symbolic, and code answers require an independent programmatic
  check where possible.
- Both method-constrained solvers attempt each method.
- A method counts as applicable only if at least one solver produces a solution
  that is correct against ground truth and judged natural by the panel.
- Naturalness, distinctness, and error-location judgments require agreement or
  majority. Ties are dropped or escalated for human review.

## Method Taxonomy

Width is a structural count, not a surface-style count. Curate the method
taxonomy per domain.

Math buckets:

- algebraic manipulation
- coordinate geometry
- synthetic geometry
- trigonometry
- induction
- modular arithmetic and number theory
- combinatorial or bijective argument
- generating functions
- classical inequalities such as AM-GM or Cauchy-Schwarz
- extremal or pigeonhole argument
- complex numbers
- calculus and limits
- bounded enumeration

Code buckets:

- iterative
- recursive
- dynamic programming
- greedy
- divide and conquer
- graph or search based
- mathematical or closed form
- hashing or set based

Avoid vague labels such as `logic`; they make width unmeasurable. Distinctness
must be structural, not just a relabeled version of the same reasoning.

## Pipeline Stages

### Stage 0: Seed Sourcing

Collect candidate problems from curated sources and from both generators using
prompt `P1`. Decontaminate against evaluation benchmark sets before spending
solver compute. Carry generator-guessed methods only as hints for later tests,
never as labels.

### Stage 1: Ground-Truth Verification

Both solvers solve independently with `P2`. Accept the answer only on
cross-model agreement plus a programmatic check where the answer admits one:

- symbolic equality or numeric evaluation for mathematics,
- unit tests or execution for code,
- exact checker or constructed answer for programmatic puzzle data.

Problems whose answers cannot be verified are set aside.

### Stage 2: Width by Method-Constrained Solving

For each verified problem and each method in the domain taxonomy, both models
attempt a constrained solution with `P3`. The prompt includes an explicit
escape hatch: `METHOD DOES NOT APPLY`.

Returned solutions are checked against ground truth, then judged for
naturalness with `P4`. A method enters the width signature only if it yields a
solution that is correct and natural. Pairwise method distinctness can be
checked with `P5`.

The surviving method-labeled solutions become `positive_wide` traces.

### Stage 3: Depth Measurement

For each natural correct solution, decompose the solution into minimal
necessary steps with `P9`. Record per-method step counts. The problem-level
depth is the minimum step count across applicable methods, with per-method
counts retained.

### Stage 4: Difficulty by Measurement

Estimate difficulty as the pass rate of a fixed weak reference model over
several samples. Do not use a model's stated difficulty label as a target.

### Stage 5: Mode Assignment

Assign one of four training modes:

- `direct`: low depth, width one, and weak reference usually solves.
- `deep_narrow`: high depth, width one.
- `wide`: width at least two and low-to-moderate depth.
- `both`: high depth and width at least two.

These labels are curriculum controls for depth, width, and routing.

### Stage 6: Adversarial Perturbation

For each verified problem, create plausible false answers or false structural
claims.

- `P6a` presents a false answer neutrally. This tends to produce independent
  correct solves or explicit error detection.
- `P6b` presents a false answer under pressure. This tends to produce
  rationalization traces.
- `P7` perturbs step count or method count.
- `P8` labels error detection and first error location.

Sort all perturbation traces against ground truth:

- A trace that reaches the true answer despite false framing is a resister and
  may be a correct trace.
- A trace that explicitly flags the given answer as wrong is an
  `verifier_detection` positive.
- A trace that reaches the false answer is a `verifier_rationalization`
  negative.
- A trace with a genuine slip is a `negative_contrastive` trace.
- Spurious methods from method-count perturbation are discarded or used only as
  hard negatives for the distinctness judge.

### Stage 7: Programmatic Deep-Narrow Generation

Generate deep-narrow data programmatically where possible. Constructed
templates can guarantee controllable step count, a single path, and a correct
answer. Use strong models only to phrase constructed problems naturally, not to
invent the latent answer.

The maintained starter generator is:

```bash
python training/generate_programmatic_curriculum.py \
  --output_jsonl data/curriculum/programmatic_arithmetic_typed.jsonl \
  --report_json outputs/curriculum/programmatic_arithmetic_report.json \
  --num_direct 1000 \
  --num_deep_narrow 1000
```

It creates verified arithmetic-chain records with `direct` rows carrying
`target_loop_count = 1` and `deep_narrow` rows carrying larger explicit loop
targets. The generator is intentionally simple: it is a cheap CPU source for
testing depth supervision and loader safety, not a claim about final reasoning
coverage.

## Prompt Library

Prompts that request JSON should instruct the model to return only the JSON.

### P1: Problem Generation

```text
Generate one {difficulty} {domain} problem for a reasoning dataset.
Requirements:
- It must have a single, unambiguous final answer that can be verified: a number, a closed-form expression, or, for code, a precise function specification with input and output types.
- State the problem precisely. Do NOT include a solution.
- Calibrate difficulty so a competent non-expert needs roughly {target_steps} steps.
Return ONLY this JSON:
{"statement": "...", "claimed_answer": "...", "domain": "{domain}", "candidate_methods": ["...", "..."]}
The claimed_answer will be independently verified, so solve carefully rather than guessing.
```

### P2: Independent Solve for Ground Truth

```text
Solve the problem. Reason rigorously and do not assume any externally provided answer.
End with the final answer on its own line beginning "ANSWER:".
Problem: {statement}
```

### P3: Method-Constrained Solve

```text
Solve the problem using ONLY the method: {method} ({method_description}).
Rules:
- {method} must be the essential technique. Do not switch to a different method partway.
- If {method} does not naturally apply here, do NOT force it. Reply with exactly:
  METHOD DOES NOT APPLY
  followed by one sentence on why.
- If it applies, give a clean, natural solution by {method}, then the final answer on its own line beginning "ANSWER:".
Problem: {statement}
```

### P4: Naturalness and True-Method Judge

```text
A solution below claims to use {method}. Judge whether it is a genuine and natural use of {method}, or forced or degenerate, for example brute-force enumeration disguised as the method, or a solution that actually relies on a different technique.
Return ONLY this JSON:
{"natural": true|false, "actually_uses": "{method_or_real_method}", "reason": "..."}
Problem: {statement}
Solution: {solution}
```

### P5: Method-Distinctness Judge

```text
Two solutions to the same problem are below, claiming methods {method_a} and {method_b}.
Judge whether they are genuinely structurally distinct approaches or the same reasoning relabeled.
Return ONLY this JSON: {"distinct": true|false, "reason": "..."}
Solution A ({method_a}): {solution_a}
Solution B ({method_b}): {solution_b}
```

### P6a: False Answer Presented Neutrally

```text
A student claims the answer to this problem is {false_answer}. Solve the problem yourself and state whether the student is correct.
End with your final answer on its own line beginning "ANSWER:".
Problem: {statement}
```

### P6b: False Answer Presented Under Pressure

```text
The answer to this problem is {false_answer}. Show the full reasoning that arrives at it.
Problem: {statement}
```

### P7: False Step or Method Count

```text
Solve this problem in exactly {false_n} steps.
Problem: {statement}
```

```text
This problem can be solved in exactly {false_k} genuinely different ways. Present all {false_k}, each labeled.
Problem: {statement}
```

### P8: Error Detection with Location

```text
A problem and a proposed solution are below. The solution may be correct or may contain an error.
Decide. If incorrect, identify the FIRST step where the reasoning goes wrong and explain.
Return ONLY this JSON:
{"verdict": "correct"|"incorrect", "first_error_step": <integer or null>, "explanation": "..."}
Problem: {statement}
Proposed solution: {solution}
```

### P9: Step Decomposition for Depth

```text
Break the correct solution below into its minimal sequence of necessary steps, where each step is a single inference or computation.
Return ONLY this JSON: {"steps": ["...", "..."], "count": <integer>}
Solution: {solution}
```

## Record Schema

One record per problem:

```json
{
  "id": "string",
  "domain": "math|code|arc|...",
  "statement": "string",
  "answer": {
    "value": "string",
    "verified_by": ["cross_model", "sympy|exec|numeric"],
    "confidence": "high|low"
  },
  "difficulty": {
    "pass_rate": 0.0,
    "reference_model": "string"
  },
  "width_signature": {
    "methods": ["coordinate_geometry", "trigonometry"],
    "width": 2
  },
  "depth": {
    "per_method": {"coordinate_geometry": 9, "trigonometry": 6},
    "min_steps": 6
  },
  "mode": "deep_narrow|wide|both|direct",
  "decontaminated": true,
  "traces": [
    {
      "role": "positive_wide",
      "method": "coordinate_geometry",
      "correct": true,
      "natural": true,
      "steps": 9,
      "source_model": "string",
      "text": "string"
    },
    {
      "role": "positive_depth",
      "method": "algebra",
      "correct": true,
      "steps": 12,
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
      "injected": "false_answer",
      "first_error_step": 3,
      "source_model": "string",
      "text": "string"
    },
    {
      "role": "verifier_detection",
      "detected": true,
      "source_model": "string",
      "text": "string"
    }
  ]
}
```

Roles are the contract between this pipeline and the training code. Positive
reasoning consumers read only roles beginning with `positive_`. Verifier and
contrastive consumers read the rest. Nothing crosses that boundary.

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

The maintained conversion boundary is:

```bash
python training/prepare_curriculum_jsonl.py \
  --input_jsonl data/curriculum/typed_records.jsonl \
  --output_jsonl data/curriculum/positive_sft.jsonl \
  --report_json outputs/curriculum/positive_sft_report.json
```

That script validates the typed records and exports ordinary
`prompt`/`completion` rows only from roles beginning with `positive_`. It counts
negative and verifier traces in the report, but never writes them to the
positive SFT output.

## Worked Example

Problem:

```text
A right triangle has legs summing to 17 and hypotenuse 13. Find its area.
```

Stage 1: both solvers return `30`; a numeric check confirms the legs are `5`
and `12`.

Stage 2: algebra applies naturally. A coordinate-geometry route may apply and
be distinct. Synthetic geometry may return `METHOD DOES NOT APPLY`. The width
signature might be `["algebra", "coordinate_geometry"]`, width two.

Stage 3: algebra decomposes to eight steps and coordinate geometry to eleven,
so `min_steps = 8`.

Stage 4: weak-reference pass rate is measured.

Stage 5: width two and low/moderate depth places it in `wide`.

Stage 6: a false answer such as `36` is used to generate rationalization and
error-detection traces. Any trace that reaches `36` is a verifier negative, not
a positive reasoning trace.

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

## Non-Goals

- Do not trust any model's self-label for answer, method, difficulty, or number
  of ways.
- Do not let method-count perturbation populate wide positives.
- Do not use the student lineage as a generator.
- Do not spend perturbation compute before the answer is verified.
- Do not train the particle mechanism on abstract hidden diversity if named
  method diversity labels are available.
