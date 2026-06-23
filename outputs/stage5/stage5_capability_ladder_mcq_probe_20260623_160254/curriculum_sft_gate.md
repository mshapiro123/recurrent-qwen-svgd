# Curriculum SFT Gate - curriculum_sft_gate_20260623_160524

- Summary: `/content/recurrent-qwen-svgd/data/curriculum/stage5_capability_ladder_mcq_probe_20260623_160254/summary.json`
- Go: `True`
- Status: `go_train_recurrent_sft`

## Report Checks

```json
{
  "reports": {
    "exported_examples": 66,
    "typed_records": 66,
    "verified": 66,
    "decontaminated": 66,
    "method_solutions": 66,
    "naturalness_judgments": 66,
    "min_natural_agree": 2,
    "min_distinct_agree": 2,
    "distinctness_required": false,
    "depth_measurements": 66,
    "difficulty_measured": 66,
    "programmatic_answer_check_required": false,
    "answer_line_verification_allowed": false
  },
  "typed_records": {
    "rows": 66,
    "invalid_rows": 0,
    "positive_missing_answer_match": 0,
    "mode_counts": {
      "deep_narrow": 32,
      "direct": 34
    },
    "role_counts": {
      "positive_depth": 32,
      "positive_direct": 34
    }
  },
  "positive_sft": {
    "rows": 66,
    "bad_rows": 0,
    "role_counts": {
      "positive_depth": 32,
      "positive_direct": 34
    },
    "mode_counts": {
      "deep_narrow": 32,
      "direct": 34
    },
    "source_model_counts": {
      "qwen_0_5b": 34,
      "qwen_1_5b": 20,
      "qwen_3b": 12
    },
    "mode_requirements": {
      "deep_narrow": {
        "required": 1,
        "observed": 32,
        "passed": true
      },
      "direct": {
        "required": 1,
        "observed": 34,
        "passed": true
      }
    }
  }
}
```

## Issues

- None.
