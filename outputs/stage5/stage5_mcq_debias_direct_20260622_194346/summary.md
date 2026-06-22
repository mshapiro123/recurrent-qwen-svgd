# MCQ Debias Diagnostic - stage5_mcq_debias_direct_20260622_194346

- Status: `selection_bias_likely`
- Source summary: `outputs/stage5/stage5_arc_agi_next_action_20260622_181850_plan_conservative_direct_preservation/answer_prior_diagnosis.json`
- ARC config: `ARC-Easy`
- ARC limit: `128`
- Decision: Do not train direct preservation yet. Regenerate MCQ benchmark claims with content/permutation scoring and then decide whether any residual degradation remains.

| arm | method | correct | total | edge-minus-middle | delta vs base |
|---|---|---:|---:|---:|---:|
| `base` | `label` | 87 | 128 | -22 |  |
| `base` | `content_question_only` | 74 | 128 | 2 |  |
| `base` | `cyclic_label_aggregated` | 96 | 128 | 10 |  |
| `start_loop1` | `label` | 88 | 128 | -18 | 1 |
| `start_loop1` | `content_question_only` | 74 | 128 | 4 | 0 |
| `start_loop1` | `cyclic_label_aggregated` | 97 | 128 | 10 | 1 |
| `start_loop4` | `label` | 82 | 128 | 54 | -5 |
| `start_loop4` | `content_question_only` | 62 | 128 | 12 | -12 |
| `start_loop4` | `cyclic_label_aggregated` | 97 | 128 | 6 | 1 |
| `best_loop4` | `label` | 81 | 128 | 40 | -6 |
| `best_loop4` | `content_question_only` | 62 | 128 | 14 | -12 |
| `best_loop4` | `cyclic_label_aggregated` | 96 | 128 | 8 | 0 |

## Decision Payload

```json
{
  "status": "selection_bias_likely",
  "passed": true,
  "label_delta": -5,
  "content_delta": -12,
  "cyclic_delta": 1,
  "best_debiased_delta": 1,
  "closure_vs_label": 6,
  "thresholds": {
    "min_label_gap": 3,
    "max_debiased_gap": 2,
    "min_closure": 3
  },
  "next_step": "Do not train direct preservation yet. Regenerate MCQ benchmark claims with content/permutation scoring and then decide whether any residual degradation remains."
}
```
