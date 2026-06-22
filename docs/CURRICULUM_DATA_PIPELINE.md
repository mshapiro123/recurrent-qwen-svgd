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
Auxiliary traces are the inverse boundary: they may only carry `negative_` or
`verifier_` roles. Positive traces must be created only from verified,
natural, method/depth-measured solution candidates, never from perturbation or
auxiliary files.
Every positive trace must also carry an `answer_match` proof copied from the
method-solution collector. The required proof is `matched=true` plus the
normalized parsed answer and normalized verified answer. This makes the hard
rule auditable row by row: a role label alone is not enough evidence for
positive SFT.

## Model Roles

Use at least two strong, diverse, non-Qwen models for generation and judging.
Examples include models in the GPT, Claude/Opus, and GLM families. Do not use
the student base, the student lineage, or close Qwen-derived teachers such as
Qwen, QwQ, QvQ, or reuploaded Jackrong-style descendants as generators for
this dataset.
This restriction is enforced both when jobs are built and when API jobs resolve
logical names through `--model_map_json`, so a safe logical name such as
`opus-strong` cannot silently map to a Qwen-family model for generation.

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

## Running Provider Response Jobs

The maintained pipeline is artifact-driven and resumable. It is intentionally
CPU/network work; do not attach an A100 just to generate or collect provider
responses.

First run the artifact pipeline until it stops at the next missing response
file:

```bash
python training/run_curriculum_pipeline_from_artifacts.py \
  --work_dir data/curriculum/run_001 \
  --seed_models opus-strong,glm-strong \
  --solver_models opus-strong,glm-strong \
  --judge_models opus-strong,glm-strong \
  --require_programmatic_answer_check
```

Then submit the emitted jobs with either a custom command backend or an
OpenAI-compatible chat-completions endpoint. The command backend is the escape
hatch for Anthropic, hosted GLM, batch systems, or any provider whose API shape
does not match chat completions:

```bash
python training/run_curriculum_job_responses.py \
  --jobs_jsonl data/curriculum/run_001/jobs_seed.jsonl \
  --output_jsonl data/curriculum/run_001/responses_seed.jsonl \
  --backend command \
  --command "python scripts/my_provider_runner.py" \
  --resume \
  --fail_fast
```

For OpenAI-compatible endpoints, provide a concrete model map and an API key
through an environment variable or Colab secret:

```bash
python training/run_curriculum_job_responses.py \
  --jobs_jsonl data/curriculum/run_001/jobs_seed.jsonl \
  --output_jsonl data/curriculum/run_001/responses_seed.jsonl \
  --backend openai_compatible \
  --api_key_env OPENAI_API_KEY \
  --base_url https://api.openai.com/v1 \
  --model_map_json config/curriculum_model_map.example.json \
  --json_mode \
  --resume \
  --fail_fast
```

After each response file lands, rerun
`training/run_curriculum_pipeline_from_artifacts.py`. It will consume the
responses, write validated intermediate artifacts and per-stage reports, then
stop at the next required job file. The final handoff to GPU training is
`positive_sft.jsonl`; negative and verifier traces remain in `typed_records.jsonl`
and are not exported to positive SFT.

When the driver stops on a missing or partial response file, `summary.json`
includes a `pending_responses` list. Each entry gives the job file, target
response file, expected rows, existing rows, remaining rows, and a
`training/run_curriculum_job_responses.py` command template. Treat those
entries as CPU/API work. They are not A100 tasks.

Before using `positive_sft.jsonl` for any A100 fine-tune, run:

```bash
python training/check_curriculum_sft_gate.py \
  --work_dir data/curriculum/run_001 \
  --output_json data/curriculum/run_001/curriculum_sft_gate.json \
  --output_md data/curriculum/run_001/curriculum_sft_gate.md \
  --fail_on_no_go
```

The gate must report `go=true`; otherwise the shard is still CPU/API cleanup
work, not GPU training material.

Once the gate is green and the shard has enough positive rows to justify GPU
spend, the guarded deterministic recurrent SFT handoff is:

```bash
python colab/run_stage5_curriculum_sft.py
```

The runner reruns the gate into `outputs/stage5/<run_id>/`, refuses tiny or
unsafe shards, requires a visible Drive backup directory by default, writes a
deterministic train/validation split from `positive_sft.jsonl`, trains only the
Phase 1 recurrent model, validates the checkpoint on the held-out curriculum
split, and backs checkpoints/data up to Drive. It does not train Phase 2
particles or SVGD; those remain a separate post-recovery gate.

Useful Colab overrides:

```bash
STAGE5_CURRICULUM_WORK_DIR=data/curriculum/run_001
STAGE5_CURRICULUM_MIN_POSITIVE_ROWS=16
STAGE5_CURRICULUM_PHASE1_STEPS=150
STAGE5_CURRICULUM_MAX_LOOPS=4
STAGE5_CURRICULUM_RESUME_FROM=outputs/stage5/<recovered_run>/phase1/phase1_step_125.pt
```

The driver treats a response artifact as complete only when the number of
non-empty response rows is at least the number of emitted job rows for that
stage. This lets bounded provider batches resume safely: a partial
`responses_*.jsonl` remains pending rather than being interpreted as failed
model agreement or missing training signal.

The measured-difficulty stage is also artifact-driven. After ground-truth
verification the pipeline writes `jobs_reference_attempts.jsonl` and stops with
`pending_reference_attempt_responses`; run those jobs through the same response
runner to produce `responses_reference_attempts.jsonl`. The collector labels
each weak-reference attempt correct or incorrect against the verified answer,
then difficulty is computed as pass rate.

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

The current provider-neutral job path implements `P6a`, `P6b`, and both `P7`
structural perturbations. Structural perturbations remain non-positive even if
the final answer is correct: they are verifier/contrastive material unless the
response explicitly rejects the forced step or method count. This is intentional
because the failure being targeted is fluent rationalization of a false
structure, not only wrong final answers.

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
      "answer_match": {
        "matched": true,
        "source": "method_constrained_answer_line",
        "parsed_answer_normalized": "string",
        "verified_answer_normalized": "string"
      },
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

If `--allow_validation_issues` is used for auditing a partial artifact, records
with validation issues are still skipped by default. The debug-only
`--export_invalid_records` flag is required to export those rows, and outputs
created with that flag must not be used for GPU SFT.

Non-positive traces are validated even though this converter does not export
them to positive SFT. `negative_contrastive` and `verifier_rationalization`
must carry explicit `correct=false` supervision and text; `verifier_detection`
must carry a boolean `detected` label and text so later selector/verifier
training cannot consume unlabeled traces by accident.

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
