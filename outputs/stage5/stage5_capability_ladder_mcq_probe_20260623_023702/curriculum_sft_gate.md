# Curriculum SFT Gate - curriculum_sft_gate_20260623_023901

- Summary: `/content/recurrent-qwen-svgd/data/curriculum/stage5_capability_ladder_mcq_probe_20260623_023702/summary.json`
- Go: `True`
- Status: `go_train_recurrent_sft`

## Report Checks

```json
{
  "reports": {
    "exported_examples": 73,
    "typed_records": 73,
    "verified": 73,
    "decontaminated": 73,
    "method_solutions": 73,
    "naturalness_judgments": 73,
    "min_natural_agree": 2,
    "min_distinct_agree": 2,
    "distinctness_required": false,
    "depth_measurements": 73,
    "difficulty_measured": 73
  },
  "typed_records": {
    "rows": 73,
    "invalid_rows": 0,
    "positive_missing_answer_match": 0,
    "mode_counts": {
      "deep_narrow": 40,
      "direct": 33
    },
    "role_counts": {
      "positive_depth": 40,
      "positive_direct": 33
    }
  },
  "positive_sft": {
    "rows": 73,
    "bad_rows": 0,
    "role_counts": {
      "positive_depth": 40,
      "positive_direct": 33
    },
    "mode_counts": {
      "deep_narrow": 40,
      "direct": 33
    },
    "source_model_counts": {
      "qwen_0_5b": 33,
      "qwen_1_5b": 22,
      "qwen_3b": 10,
      "qwen_7b": 8
    },
    "mode_requirements": {
      "deep_narrow": {
        "required": 1,
        "observed": 40,
        "passed": true
      },
      "direct": {
        "required": 1,
        "observed": 33,
        "passed": true
      }
    }
  }
}
```

## Issues

- None.
