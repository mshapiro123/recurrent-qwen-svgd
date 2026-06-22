# Curriculum Data Generation And Validation Pipeline

This is the data contract for building the recurrent model's wide-and-deep
training curriculum with strong external generators. The goal is not to trust a
strong model's explanation. The goal is to generate a broad distribution of
candidate traces, verify the answer independently, then route each trace by
measured correctness and role.

## Core Rule

Generation and labeling are separate. A trace that reaches the wrong answer may
train a verifier or serve as a contrastive negative, but it must never enter a
positive reasoning set. The code boundary for this rule is
`training/prepare_curriculum_jsonl.py`: only roles beginning with `positive_`
are exported to causal SFT rows, and positive traces must be correct, natural,
and step-labeled.
The SFT gate adds a second boundary: every positive trace must include
`answer_match.matched=true`, copied from the method-solution collector after
its final `ANSWER:` line was matched to the verified answer. A positive role
without this proof is treated as unsafe, even if the rest of the record
validates.

## Model Roles

Use at least two strong non-Qwen models in separate roles where possible:

- Generators produce candidate problems and raw traces.
- Solvers independently derive answers for ground truth.
- Method-constrained solvers attempt named solution methods.
- Judges assess naturalness, true method use, distinctness, and error location.

Avoid using the student base, its adapters, or its teacher lineage as data
generators. We want curriculum signal, not leakage from the model family we are
evaluating.

## Pipeline

1. Seed candidate problems from curated sources and generators.
2. Decontaminate against evaluation sets before spending more compute.
3. Verify the answer through cross-model agreement plus a programmatic check
   when the domain permits it.
4. Measure width with method-constrained solving. A method counts only when a
   correct solution is also judged natural and structurally distinct.
5. Measure depth by decomposing correct solutions into minimal necessary steps.
6. Measure difficulty by pass rate of a fixed weak reference model, not by
   model self-report.
7. Assign mode:
   - `direct`: low depth, width one, easy.
   - `deep_narrow`: high depth, width one.
   - `wide`: width two or more, low to moderate depth.
   - `both`: high depth and width two or more.
8. Generate adversarial perturbations only after ground truth is verified.
   Correct resisters can become positives or verifier-detection examples.
   Rationalizations and slips become non-positive verifier/contrastive rows.
9. Generate deep-narrow rows programmatically when controllable depth is more
   important than natural problem variety.

## Method Taxonomy

Method labels should be structural, not cosmetic.

Math examples:

- algebraic manipulation
- coordinate geometry
- synthetic geometry
- trigonometry
- induction
- modular arithmetic or number theory
- combinatorial or bijective argument
- generating functions
- inequalities
- extremal or pigeonhole
- complex numbers
- calculus or limits
- bounded enumeration

Code examples:

- iterative
- recursive
- dynamic programming
- greedy
- divide and conquer
- graph or search based
- mathematical closed form
- hashing or set based

## Record Schema

One curriculum record represents one verified problem:

```json
{
  "id": "string",
  "domain": "math",
  "statement": "problem text",
  "answer": {
    "value": "verified answer",
    "verified_by": ["cross_model", "sympy"],
    "confidence": "high"
  },
  "difficulty": {"pass_rate": 0.42, "reference_model": "weak-ref"},
  "width_signature": {"methods": ["algebra", "unit_cancellation"], "width": 2},
  "depth": {"per_method": {"algebra": 6, "unit_cancellation": 4}, "min_steps": 4},
  "mode": "wide",
  "decontaminated": true,
  "traces": [
    {
      "role": "positive_wide",
      "method": "algebra",
      "correct": true,
      "natural": true,
      "steps": 6,
      "source_model": "strong-model-a",
      "answer_match": {
        "matched": true,
        "source": "method_constrained_answer_line",
        "parsed_answer_normalized": "42",
        "verified_answer_normalized": "42"
      },
      "text": "..."
    },
    {
      "role": "negative_contrastive",
      "correct": false,
      "error_type": "unit_slip",
      "source_model": "strong-model-b",
      "text": "..."
    },
    {
      "role": "verifier_rationalization",
      "correct": false,
      "first_error_step": 3,
      "source_model": "strong-model-b",
      "text": "..."
    }
  ]
}
```

Positive reasoning consumers read only:

- `positive_direct`
- `positive_depth`
- `positive_wide`

Verifier and contrastive consumers may read:

- `negative_contrastive`
- `verifier_rationalization`
- `verifier_detection`

## Prompt Templates

These are implementation prompts, not labels. Every output still goes through
verification and routing.

### Problem Generation

```text
Generate one {difficulty} {domain} problem for a reasoning dataset.
Requirements:
- It must have a single, unambiguous final answer that can be verified.
- State the problem precisely. Do NOT include a solution.
- Calibrate difficulty so a competent non-expert needs roughly {target_steps} steps.
Return ONLY JSON:
{"statement": "...", "claimed_answer": "...", "domain": "{domain}", "candidate_methods": ["...", "..."]}
```

### Independent Solve

```text
Solve the problem. Reason rigorously and do not assume any externally provided answer.
End with the final answer on its own line beginning "ANSWER:".
Problem: {statement}
```

### Method-Constrained Solve

```text
Solve the problem using ONLY the method: {method} ({method_description}).
If {method} does not naturally apply, reply exactly:
METHOD DOES NOT APPLY
followed by one sentence on why.
If it applies, give a clean, natural solution, then the final answer on its own line beginning "ANSWER:".
Problem: {statement}
```

### Naturalness Judge

```text
A solution below claims to use {method}. Judge whether it is a genuine and natural use of {method}.
Return ONLY JSON:
{"natural": true|false, "actually_uses": "{method_or_real_method}", "reason": "..."}
Problem: {statement}
Solution: {solution}
```

### Error Detection

```text
A problem and proposed solution are below. The solution may be correct or may contain an error.
If incorrect, identify the FIRST step where reasoning goes wrong.
Return ONLY JSON:
{"verdict": "correct"|"incorrect", "first_error_step": <integer or null>, "explanation": "..."}
Problem: {statement}
Proposed solution: {solution}
```

## Near-Term Implementation Order

1. Keep using `training/generate_programmatic_curriculum.py` for cheap
   direct/deep-narrow rows.
2. Use external-model generation only on CPU/API paths. Build provider-neutral
   prompt jobs with `training/build_curriculum_generation_jobs.py`; a separate
   runner can submit those jobs to GPT, Opus, GLM, or other non-student models.
3. Validate and export positive SFT rows with:

```bash
python training/prepare_curriculum_jsonl.py \
  --input_jsonl data/curriculum/typed_records.jsonl \
  --output_jsonl data/curriculum/positive_sft.jsonl \
  --report_json outputs/curriculum/positive_sft_report.json
```

4. Train only after the report confirms no validation issues and no non-positive
   roles were exported.

## Job Builder

The job builder writes JSONL records with `stage`, `role`, `model`, `prompt`,
`expects_json`, and metadata. It does not call any provider API and blocks Qwen
or other student-lineage ids by default.

Before running provider APIs, exercise the whole data path with the no-API
fixture:

```bash
python training/run_curriculum_pipeline_fixture.py \
  --output_dir outputs/curriculum_fixture \
  --overwrite
```

This writes fake raw responses, intermediate collection files, typed records,
and positive SFT rows. It should produce one `wide` typed record and two
`positive_wide` SFT rows.

For real API work, prefer the resumable artifact driver over manually chaining
each command in a notebook:

```bash
python training/run_curriculum_pipeline_from_artifacts.py \
  --work_dir data/curriculum/run_001 \
  --seed_models opus-strong,glm-strong \
  --solver_models opus-strong,glm-strong \
  --judge_models opus-strong,glm-strong \
  --require_programmatic_answer_check \
  --domains math \
  --difficulties medium,hard \
  --target_steps 4,8 \
  --count_per_combo 2 \
  --references_jsonl eval/smoke_exact_tasks_v2.jsonl \
  --references_jsonl eval/smoke_mcq_tasks.jsonl
```

The driver never calls a provider. It writes the next `jobs_*.jsonl`, checks for
the corresponding `responses_*.jsonl`, and stops with a `status` and
`next_action` in `summary.json`. After an API runner fills the missing response
file, run the same command again. The stop points are:

- `pending_seed_responses`
- `pending_ground_truth_responses`
- `pending_reference_attempt_responses`
- `pending_method_or_perturbation_responses`
- `pending_judgment_responses`
- `complete`

Pending summaries also include `pending_responses`: a structured list with each
needed job file, response file, expected row count, existing row count,
remaining rows, and a provider-runner command template. This is the handoff for
CPU/API generation work and should not be treated as an A100 action.

On completion it writes `typed_records.jsonl` and `positive_sft.jsonl`. Before
any GPU fine-tune, run the no-GPU SFT gate:

```bash
python training/check_curriculum_sft_gate.py \
  --work_dir data/curriculum/run_001 \
  --output_json data/curriculum/run_001/curriculum_sft_gate.json \
  --output_md data/curriculum/run_001/curriculum_sft_gate.md \
  --min_positive_rows 1 \
  --fail_on_no_go
```

This gate checks that the pipeline is complete, generated answers used strict
programmatic anchoring, records validate, non-positive roles did not leak into
`positive_sft.jsonl`, and enough positive rows exist. This is the preferred
Colab loop because it is restart-safe and burns no GPU while waiting on
API/provider artifacts.

After a shard passes the gate and has enough positive rows for a real run, use
the guarded GPU handoff:

```bash
python colab/run_stage5_curriculum_sft.py
```

It reruns the gate, requires Drive backup by default, writes a deterministic
train/validation split from `positive_sft.jsonl`, trains only Phase 1
deterministic recurrence, validates the checkpoint, and backs up run artifacts
to Drive. Use `colab/STAGE5_CURRICULUM_SFT_CELL.py` as the copy/paste Colab
cell for this path. Do not train Phase 2 particles from a generated shard until
the deterministic checkpoint is sane.

Seed problem-generation jobs:

```bash
python training/build_curriculum_generation_jobs.py \
  --stage seed \
  --models opus-strong,glm-strong \
  --domains math \
  --difficulties medium,hard \
  --target_steps 4,8,12 \
  --count_per_combo 5 \
  --output_jsonl data/curriculum/jobs_seed.jsonl \
  --report_json outputs/curriculum/jobs_seed_report.json
```

Run prompt jobs through a backend:

```bash
python training/run_curriculum_job_responses.py \
  --jobs_jsonl data/curriculum/jobs_seed.jsonl \
  --output_jsonl data/curriculum/responses_seed.jsonl \
  --report_json outputs/curriculum/responses_seed_report.json \
  --backend dry_run
```

For a real provider, wrap the provider call in a small command that reads one
job JSON object from stdin and writes the raw model response to stdout. Then use:

```bash
python training/run_curriculum_job_responses.py \
  --jobs_jsonl data/curriculum/jobs_seed.jsonl \
  --output_jsonl data/curriculum/responses_seed.jsonl \
  --report_json outputs/curriculum/responses_seed_report.json \
  --backend command \
  --command "python provider_runner.py" \
  --resume \
  --sleep_sec 0.5
```

The runner writes one response JSONL row per job with `job_id`, `response_text`,
`status`, `backend`, timing, and any command stderr. `--resume` skips job ids
already present in the output file, which matters for paid API batches.

Collect seed responses from an external API runner into candidate problems:

```bash
python training/collect_curriculum_job_outputs.py \
  --mode seed_candidates \
  --jobs_jsonl data/curriculum/jobs_seed.jsonl \
  --responses_jsonl data/curriculum/responses_seed.jsonl \
  --output_jsonl data/curriculum/candidates.jsonl \
  --report_json outputs/curriculum/candidates_report.json
```

The response JSONL should contain at least:

```json
{"job_id": "seed-000000-generator-opus-strong", "response_text": "{...raw model response...}"}
```

The collector also accepts `response`, `output_text`, `output`, `text`,
`content`, or `message.content` fields. It parses JSON from raw text or fenced
JSON blocks.

Decontaminate generated candidates before any more API or GPU compute is spent:

```bash
python training/decontaminate_curriculum_candidates.py \
  --candidates_jsonl data/curriculum/candidates.jsonl \
  --references_jsonl eval/smoke_exact_tasks_v2.jsonl \
  --references_jsonl eval/smoke_mcq_tasks.jsonl \
  --output_jsonl data/curriculum/candidates_decontaminated.jsonl \
  --rejected_jsonl data/curriculum/candidates_contaminated.jsonl \
  --annotated_jsonl data/curriculum/candidates_decontam_annotated.jsonl \
  --report_json outputs/curriculum/candidates_decontam_report.json \
  --ngram_size 5 \
  --min_ngram_size 3 \
  --threshold 0.5
```

This first pass is deterministic token 3-to-5-gram overlap with Jaccard and
containment checks. It is deliberately cheap and conservative: exact benchmark
copies and benchmark prompts embedded inside longer generated statements are
rejected. For real benchmark suites, pass every local eval JSONL prepared for
ARC, GPQA, GSM-style smoke tasks, and any held-out custom set. Only
`candidates_decontaminated.jsonl` should feed the next stage.

Ground-truth solve jobs from decontaminated candidates:

```bash
python training/build_curriculum_generation_jobs.py \
  --stage ground_truth \
  --models opus-strong,glm-strong \
  --input_jsonl data/curriculum/candidates_decontaminated.jsonl \
  --output_jsonl data/curriculum/jobs_ground_truth.jsonl
```

Collect ground-truth solver responses into verified candidates:

```bash
python training/collect_curriculum_job_outputs.py \
  --mode verified_candidates \
  --candidates_jsonl data/curriculum/candidates_decontaminated.jsonl \
  --jobs_jsonl data/curriculum/jobs_ground_truth.jsonl \
  --responses_jsonl data/curriculum/responses_ground_truth.jsonl \
  --output_jsonl data/curriculum/verified_candidates.jsonl \
  --report_json outputs/curriculum/verified_candidates_report.json
```

The verified-candidate collector requires at least two distinct solver models
to agree on the normalized `ANSWER:` line by default. Disagreements are written
to the report and are not promoted.

Build weak-reference attempt jobs from verified candidates:

```bash
python training/build_curriculum_generation_jobs.py \
  --stage reference_attempt \
  --models qwen-0.5b-base-greedy \
  --input_jsonl data/curriculum/verified_candidates.jsonl \
  --reference_samples 4 \
  --output_jsonl data/curriculum/jobs_reference_attempts.jsonl
```

Run the jobs through the response runner, then collect correctness labels
against the verified answer:

```bash
python training/collect_curriculum_job_outputs.py \
  --mode reference_attempts \
  --candidates_jsonl data/curriculum/verified_candidates.jsonl \
  --jobs_jsonl data/curriculum/jobs_reference_attempts.jsonl \
  --responses_jsonl data/curriculum/responses_reference_attempts.jsonl \
  --output_jsonl data/curriculum/reference_attempts.jsonl \
  --report_json outputs/curriculum/reference_attempts_report.json
```

Annotate difficulty from the collected weak-reference attempts:

```bash
python training/annotate_curriculum_difficulty.py \
  --candidates_jsonl data/curriculum/verified_candidates.jsonl \
  --attempts_jsonl data/curriculum/reference_attempts.jsonl \
  --output_jsonl data/curriculum/verified_candidates_difficulty.jsonl \
  --rejected_jsonl data/curriculum/verified_candidates_unmeasured.jsonl \
  --report_json outputs/curriculum/difficulty_report.json \
  --reference_model qwen-0.5b-base-greedy \
  --min_samples 4
```

The attempts file is runner-neutral. Each row should carry a problem id
(`record_id`, `id`, `problem_id`, or `curriculum_id`) and a boolean correctness
field (`correct`, `is_correct`, `matched`, or `success`). For example:

```json
{"record_id": "candidate-math-hard-8-seed-000001", "sample_id": 0, "correct": false}
```

Difficulty is the measured pass rate of the fixed weak reference model. Use
`--drop_unmeasured` when building a training set that must exclude candidates
without enough reference samples.

Annotate plausible false answers after ground-truth verification:

```bash
python training/annotate_curriculum_false_answers.py \
  --candidates_jsonl data/curriculum/verified_candidates_difficulty.jsonl \
  --output_jsonl data/curriculum/verified_candidates_false_answers.jsonl \
  --rejected_jsonl data/curriculum/verified_candidates_no_false_answer.jsonl \
  --report_json outputs/curriculum/false_answers_report.json
```

The annotator first reuses a seed `claimed_answer` or existing
`plausible_false_answers` entry when it disagrees with the verified answer. If
none exists, it creates a deterministic numeric near-miss while preserving
simple units or prefixes. Use `--drop_unannotated` when perturbation coverage is
required for every downstream row.

Method-constrained width jobs:

```bash
python training/build_curriculum_generation_jobs.py \
  --stage method_solve \
  --models opus-strong,glm-strong \
  --methods algebra,number_theory,bounded_enumeration \
  --input_jsonl data/curriculum/verified_candidates_false_answers.jsonl \
  --output_jsonl data/curriculum/jobs_methods.jsonl
```

Collect method-constrained responses into correct-answer solution candidates:

```bash
python training/collect_curriculum_job_outputs.py \
  --mode method_solutions \
  --candidates_jsonl data/curriculum/verified_candidates_false_answers.jsonl \
  --jobs_jsonl data/curriculum/jobs_methods.jsonl \
  --responses_jsonl data/curriculum/responses_methods.jsonl \
  --output_jsonl data/curriculum/method_solution_candidates.jsonl \
  --report_json outputs/curriculum/method_solution_candidates_report.json
```

Only responses whose final `ANSWER:` normalizes to the verified candidate answer
are emitted. `METHOD DOES NOT APPLY`, missing-answer, and wrong-answer responses
are retained in the report and are not forwarded to naturalness, distinctness, or
depth jobs.

Naturalness, pairwise method distinctness, depth, perturbation, and
error-detection jobs are available through `--stage naturalness`,
`--stage distinctness`, `--stage depth`, `--stage perturbation`, and
`--stage error_detection`. These are still job construction steps; responses must
be parsed and verified before typed curriculum records are created.

Build and collect naturalness, distinctness, and depth judge responses:

```bash
python training/build_curriculum_generation_jobs.py \
  --stage naturalness \
  --models opus-strong,glm-strong \
  --input_jsonl data/curriculum/method_solution_candidates.jsonl \
  --output_jsonl data/curriculum/jobs_naturalness.jsonl

python training/collect_curriculum_job_outputs.py \
  --mode naturalness_judgments \
  --candidates_jsonl data/curriculum/method_solution_candidates.jsonl \
  --jobs_jsonl data/curriculum/jobs_naturalness.jsonl \
  --responses_jsonl data/curriculum/responses_naturalness.jsonl \
  --output_jsonl data/curriculum/naturalness_judgments.jsonl \
  --report_json outputs/curriculum/naturalness_judgments_report.json

python training/build_curriculum_generation_jobs.py \
  --stage distinctness \
  --models opus-strong,glm-strong \
  --input_jsonl data/curriculum/method_solution_candidates.jsonl \
  --output_jsonl data/curriculum/jobs_distinctness.jsonl

python training/collect_curriculum_job_outputs.py \
  --mode distinctness_judgments \
  --candidates_jsonl data/curriculum/method_solution_candidates.jsonl \
  --jobs_jsonl data/curriculum/jobs_distinctness.jsonl \
  --responses_jsonl data/curriculum/responses_distinctness.jsonl \
  --output_jsonl data/curriculum/distinctness_judgments.jsonl \
  --report_json outputs/curriculum/distinctness_judgments_report.json

python training/build_curriculum_generation_jobs.py \
  --stage depth \
  --models opus-strong,glm-strong \
  --input_jsonl data/curriculum/method_solution_candidates.jsonl \
  --output_jsonl data/curriculum/jobs_depth.jsonl

python training/collect_curriculum_job_outputs.py \
  --mode depth_measurements \
  --candidates_jsonl data/curriculum/method_solution_candidates.jsonl \
  --jobs_jsonl data/curriculum/jobs_depth.jsonl \
  --responses_jsonl data/curriculum/responses_depth.jsonl \
  --output_jsonl data/curriculum/depth_measurements.jsonl \
  --report_json outputs/curriculum/depth_measurements_report.json
```

Build and collect adversarial perturbation traces only after answer verification:

```bash
python training/build_curriculum_generation_jobs.py \
  --stage perturbation \
  --models opus-strong,glm-strong \
  --input_jsonl data/curriculum/verified_candidates_false_answers.jsonl \
  --output_jsonl data/curriculum/jobs_perturbation.jsonl

python training/collect_curriculum_job_outputs.py \
  --mode perturbation_traces \
  --candidates_jsonl data/curriculum/verified_candidates_false_answers.jsonl \
  --jobs_jsonl data/curriculum/jobs_perturbation.jsonl \
  --responses_jsonl data/curriculum/responses_perturbation.jsonl \
  --output_jsonl data/curriculum/perturbation_traces.jsonl \
  --report_json outputs/curriculum/perturbation_traces_report.json
```

The perturbation job builder emits four probes per model when the required
metadata is available: neutral false answer (`P6a`), pressured false answer
(`P6b`), false step count (`P7`), and false method count (`P7`). The
perturbation collector routes them conservatively. A neutral false-answer
response that returns the verified answer becomes `verifier_detection`; a
pressure-prompt response or false-answer match becomes
`verifier_rationalization`; other wrong or missing answers become
`negative_contrastive`.

False step-count and false method-count traces are never promoted to positive
reasoning data just because they reach the right final answer. If a response
explicitly rejects the forced structure, it becomes `verifier_detection`.
Otherwise it becomes `verifier_rationalization` or `negative_contrastive`
depending on the answer. This keeps structurally bogus but fluent reasoning out
of causal SFT while preserving it for selector/verifier training.

Optionally build first-error-location judgments over perturbation traces:

```bash
python training/build_curriculum_generation_jobs.py \
  --stage error_detection \
  --models opus-strong,glm-strong \
  --input_jsonl data/curriculum/perturbation_traces.jsonl \
  --output_jsonl data/curriculum/jobs_error_detection.jsonl

python training/collect_curriculum_job_outputs.py \
  --mode error_detection_judgments \
  --candidates_jsonl data/curriculum/perturbation_traces.jsonl \
  --jobs_jsonl data/curriculum/jobs_error_detection.jsonl \
  --responses_jsonl data/curriculum/responses_error_detection.jsonl \
  --output_jsonl data/curriculum/error_detection_judgments.jsonl \
  --report_json outputs/curriculum/error_detection_judgments_report.json
```

Assemble typed curriculum records:

```bash
python training/assemble_curriculum_records.py \
  --verified_candidates_jsonl data/curriculum/verified_candidates_false_answers.jsonl \
  --solution_candidates_jsonl data/curriculum/method_solution_candidates.jsonl \
  --naturalness_jsonl data/curriculum/naturalness_judgments.jsonl \
  --depth_jsonl data/curriculum/depth_measurements.jsonl \
  --distinctness_jsonl data/curriculum/distinctness_judgments.jsonl \
  --auxiliary_traces_jsonl data/curriculum/perturbation_traces.jsonl \
  --auxiliary_traces_jsonl data/curriculum/error_detection_judgments.jsonl \
  --output_jsonl data/curriculum/typed_records.jsonl \
  --report_json outputs/curriculum/typed_records_report.json
```

Assembly rejects undecontaminated candidates by default. It also rejects method
solutions that lack both a naturalness agreement and a positive depth
measurement. When `--distinctness_jsonl` is provided, width counts only methods
whose best natural measured solutions are pairwise judged structurally distinct;
relabeled or degenerate duplicates collapse back to a single method before mode
assignment. Auxiliary perturbation and error-detection rows are attached to the
typed records as verifier/negative roles, but `prepare_curriculum_jsonl.py`
exports only roles beginning with `positive_`.
