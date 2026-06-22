# Curriculum SFT Gate - curriculum_sft_gate_20260622_134618

- Summary: `/content/recurrent-qwen-svgd/data/curriculum/programmatic_direct_deep_001/summary.json`
- Go: `True`
- Status: `go_train_recurrent_sft`

## Report Checks

```json
{
  "reports": {
    "exported_examples": 2000,
    "typed_records": 2000,
    "verified": 2000,
    "decontaminated": 2000,
    "method_solutions": 2000,
    "naturalness_judgments": 2000,
    "min_natural_agree": 2,
    "min_distinct_agree": 2,
    "distinctness_required": false,
    "depth_measurements": 2000,
    "difficulty_measured": 2000
  },
  "typed_records": {
    "rows": 2000,
    "invalid_rows": 0,
    "positive_missing_answer_match": 0,
    "mode_counts": {
      "deep_narrow": 1000,
      "direct": 1000
    },
    "role_counts": {
      "positive_depth": 1000,
      "positive_direct": 1000
    }
  },
  "positive_sft": {
    "rows": 2000,
    "bad_rows": 0,
    "role_counts": {
      "positive_depth": 1000,
      "positive_direct": 1000
    },
    "mode_counts": {
      "deep_narrow": 1000,
      "direct": 1000
    },
    "source_model_counts": {
      "programmatic_generator": 2000
    },
    "mode_requirements": {
      "deep_narrow": {
        "required": 1000,
        "observed": 1000,
        "passed": true
      },
      "direct": {
        "required": 1000,
        "observed": 1000,
        "passed": true
      }
    }
  }
}
```

## Issues

- None.
