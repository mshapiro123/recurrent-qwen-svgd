# Curriculum SFT Gate - curriculum_sft_gate_20260623_040338

- Summary: `/content/recurrent-qwen-svgd/data/curriculum/stage5_capability_ladder_trace_collection_20260623_040333/summary.json`
- Go: `False`
- Status: `no_go`

## Report Checks

```json
{
  "reports": {
    "exported_examples": 32,
    "typed_records": 32,
    "verified": 32,
    "decontaminated": 32,
    "method_solutions": 32,
    "naturalness_judgments": 32,
    "min_natural_agree": 2,
    "min_distinct_agree": 2,
    "distinctness_required": false,
    "depth_measurements": 32,
    "difficulty_measured": 32
  },
  "typed_records": {
    "rows": 32,
    "invalid_rows": 0,
    "positive_missing_answer_match": 0,
    "mode_counts": {
      "deep_narrow": 24,
      "direct": 8
    },
    "role_counts": {
      "positive_depth": 24,
      "positive_direct": 8
    }
  },
  "positive_sft": {
    "rows": 32,
    "bad_rows": 0,
    "role_counts": {
      "positive_depth": 24,
      "positive_direct": 8
    },
    "mode_counts": {
      "deep_narrow": 24,
      "direct": 8
    },
    "source_model_counts": {
      "Qwen/Qwen2.5-7B-Instruct": 32
    },
    "mode_requirements": {}
  }
}
```

## Issues

- `programmatic_check_not_required`: verified_candidates_report did not require programmatic answer checks.
