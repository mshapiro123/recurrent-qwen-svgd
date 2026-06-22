# MCQ Debias Status

## Why This Exists

The latest ARC-Easy direct-regression diagnostic showed a large first/last
multiple-choice label drift under bare `A/B/C/D` scoring. That pattern is a
known MCQ selection-bias failure mode, not clean evidence that the recurrent
model lost content competence. From this point forward, bare-label MCQ scores
are diagnostic only; they are not sufficient evidence for training decisions.

## Current Evidence

Completed diagnostic:

```text
outputs/stage5/stage5_mcq_debias_direct_20260622_194346/summary.json
```

Result:

```text
status = selection_bias_likely
passed = true
ARC config = ARC-Easy
ARC limit = 128
```

Key table:

| arm | bare label | content-only | cyclic label aggregated |
|---|---:|---:|---:|
| base | `87/128` | `74/128` | `96/128` |
| start loop 1 | `88/128` | `74/128` | `97/128` |
| start loop 4 | `82/128` | `62/128` | `97/128` |
| best loop 4 | `81/128` | `62/128` | `96/128` |

The bare-label recurrent loop-4 result appeared to trail base by `-5`, but the
cyclic-permutation aggregate was `+1` for the start checkpoint and `0` for the
best conservative checkpoint. The direct interpretation is:

- a large part of the apparent regression is option-label/position bias;
- the recurrent model may be amplifying the bias through repeated computation;
- content-only scoring remains weaker and should be treated as a separate
  diagnostic, not as the current proof metric;
- direct-preservation training is premature until ARC-Challenge is re-measured
  under cyclic scoring.

## Active Gate

Run the same bounded diagnostic on ARC-Challenge:

```text
STAGE5_CURRENT_A100_TARGET=arc_challenge_mcq_debias_confirm
```

The launcher is:

```text
colab/STAGE5_ARC_CHALLENGE_MCQ_DEBIAS_CELL.py
```

It pins `STAGE5_MCQ_DEBIAS_ARC_CONFIG=ARC-Challenge`, enables quiet/resumable
output, pushes summary artifacts, runs
`colab/assess_stage5_mcq_debias_pair.py` to combine ARC-Easy and ARC-Challenge
into one planner-readable gate, and disconnects the Colab runtime.

## Decision Rule

After ARC-Challenge cyclic scoring:

| Outcome | Interpretation | Next action |
|---|---|---|
| Cyclic score closes the base/recurrent gap | Prior ARC MCQ regressions were mostly scoring artifacts. | Standardize MCQ evaluation on cyclic/permutation scoring, update benchmark claims, and resume depth/particle planning. |
| Cyclic score leaves a material recurrent gap | True content or calibration degradation remains. | Run bounded depth-1 preservation or bypass training before particles/SVGD. |
| Cyclic and content-only disagree strongly | The harness is measuring two separable effects. | Inspect rows and add a richer MCQ scorer before training. |

The preferred artifact after the ARC-Challenge run is now
`kind=stage5_mcq_debias_pair_assessment`:

- `status=mcq_selection_bias_confirmed`: do not spend A100 time on
  direct-preservation training; standardize MCQ claims on debiased scoring.
- `status=mcq_content_gap_persists`: one bounded depth-1 preservation probe is
  justified from the blocking split.
- `status=mcq_debias_mixed_or_inconclusive`: inspect rows before choosing a
  GPU action.

If the pair assessment confirms selection bias, run
`colab/apply_stage5_mcq_scoring_policy.py`. It writes
`kind=stage5_mcq_scoring_policy`, makes cyclic/content scoring the MCQ policy
for future claims, and flags stale label-only artifacts in `outputs/stage5`.

## Policy

Do not use bare `A/B/C/D` accuracy alone to justify:

- direct-preservation training;
- Phase 2/SVGD training;
- scale-up to 1.5B/3B;
- claims that recurrent Qwen beats or trails base Qwen on MCQ benchmarks.

For MCQ tasks, report at least:

- bare-label score;
- cyclic-permutation aggregated score;
- content-question-only score when the benchmark format supports it;
- prediction-count drift by label;
- edge-minus-middle drift;
- paired wins/losses against base.

Content-only scoring remains useful but is not yet the primary proof metric
because the current ARC-Easy run showed it can underperform both base and
recurrent while cyclic scoring removes the apparent recurrent gap.

The benchmark suite can now emit the policy metrics directly by setting:

```text
STAGE5_BENCHMARK_SCORE_TARGETS=label,content_question_only,cyclic_label_aggregated
```

`content_question_only` scores option content without showing the option list in
the prompt. `cyclic_label_aggregated` runs label scoring across cyclic option
permutations, then averages scores back into the original content-label space.
Use these as measurement tools. Do not treat them as a training objective by
themselves.
