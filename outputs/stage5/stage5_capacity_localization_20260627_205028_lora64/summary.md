# Stage 5 Re-entry Recovery Training - stage5_capacity_localization_20260627_205028_lora64

- Cell version: `reentry_recovery_training_v5_fixed_tail_damper`
- Child curriculum summary: `outputs/stage5/stage5_capacity_localization_20260627_205028_lora64_curriculum_sft/summary.json`
- Trace summary: `outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json`
- Stage 3 repair assessment: `outputs/stage5/stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/summary.json`
- Status: `validation_sane`
- Passed: `True`
- Checkpoint: `outputs/stage5/stage5_capacity_localization_20260627_205028_lora64_curriculum_sft/phase1/phase1_step_100.pt`
- Fixed tail damper: `outputs/stage5/stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/tail_damper.pt`
- Fixed tail damper strength: `1.0`

## Validation Checks
```json
{
  "status": "validation_sane",
  "issues": [],
  "nonfinite_metrics": [],
  "min_mean_expected_loops": 1.05,
  "mean_expected_loops": 1.773289,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.367033,
    "deep_narrow_mean_expected_loops": 1.976418,
    "required_margin": 0.25,
    "observed": true
  },
  "require_target_loop_gradient": false,
  "target_loop_gradient": {
    "available": true,
    "points": [
      {
        "target_loop_count": 1.0,
        "mean_expected_loops": 1.367033,
        "examples": 2.0
      },
      {
        "target_loop_count": 2.0,
        "mean_expected_loops": 1.987881,
        "examples": 3.0
      },
      {
        "target_loop_count": 3.0,
        "mean_expected_loops": 1.94203,
        "examples": 1.0
      }
    ],
    "required_margin": 0.1,
    "adjacent_margins": [
      0.6208480000000001,
      -0.045851000000000086
    ],
    "observed": false
  }
}
```

## Post-Recovery Re-entry Health
- Drift summary: `outputs/stage5/stage5_capacity_localization_20260627_205028_lora64/post_reentry_drift.json`
- Health status: `reentry_health_sane`
```json
{
  "status": "reentry_health_sane",
  "issues": [],
  "thresholds": {
    "min_bridge_gate_abs": 0.05,
    "max_loop8_output_over_entry_rms": 3.0
  },
  "metrics": {
    "bridge_gate": 1.0008043050765991,
    "bridge_delta_rms": 0.1564834862947464,
    "bridge_weight_grad_rms": 0.3893272578716278,
    "bridge_bias_grad_rms": 0.0075753177516162395,
    "reentry_adapter_mode": "spectral",
    "reentry_adapter_delta_rms": 0.005735846236348152,
    "reentry_adapter_scale_grad_rms": 0.0,
    "reentry_adapter_bias_grad_rms": 0.0,
    "reentry_adapter_spectral_u_grad_rms": 0.11836444586515427,
    "reentry_adapter_spectral_v_grad_rms": 0.026174556463956833,
    "reentry_adapter_spectral_theta_grad_abs": 0.11842305213212967,
    "reentry_adapter_gradient_live": true,
    "loop8_output_over_entry_rms": 1.0044348239898682,
    "loop8_output_over_input_rms": 1.0051334698994954,
    "mean_exit_over_entry_rms": 1.0017465750376384,
    "subspace_overlap": 0.3539472818374634
  }
}
```

## Fixed-Damper Depth Readout
- Summary: `outputs/stage5/stage5_capacity_localization_20260627_205028_lora64/fixed_tail_damper_depth_readout/summary.json`
```json
[
  {
    "strength": 0.0,
    "tail_trace": {
      "entry": {
        "tail_trace": 26.049549387810607,
        "ratio_vs_entry": 1.0
      },
      "loop1": {
        "tail_trace": 66.81692244087517,
        "ratio_vs_entry": 2.5649934072234233
      },
      "loop2": {
        "tail_trace": 130.90490053959024,
        "ratio_vs_entry": 5.025227062117424
      },
      "loop3": {
        "tail_trace": 214.0849948797108,
        "ratio_vs_entry": 8.218376129757077
      },
      "loop4": {
        "tail_trace": 314.9092892133705,
        "ratio_vs_entry": 12.088857451051586
      },
      "loop8": {
        "tail_trace": 852.6406260101554,
        "ratio_vs_entry": 32.731492330885864
      }
    },
    "score_summary": {
      "strength": 0.0,
      "examples": 512,
      "loop_results": {
        "1": {
          "correct": 182,
          "total": 512,
          "accuracy": 0.35546875
        },
        "2": {
          "correct": 157,
          "total": 512,
          "accuracy": 0.306640625
        },
        "3": {
          "correct": 158,
          "total": 512,
          "accuracy": 0.30859375
        }
      },
      "oracle_correct": 229,
      "oracle_accuracy": 0.447265625,
      "oracle_gap_vs_loop1": 47,
      "rescued_vs_loop1": 47,
      "harmed_vs_loop1": 68,
      "stable_correct": 114,
      "stable_wrong": 283,
      "pattern_counts": {
        "000": 283,
        "001": 18,
        "010": 5,
        "011": 24,
        "100": 52,
        "101": 2,
        "110": 14,
        "111": 114
      }
    }
  },
  {
    "strength": 1.0,
    "tail_trace": {
      "entry": {
        "tail_trace": 26.049549387810607,
        "ratio_vs_entry": 1.0
      },
      "loop1": {
        "tail_trace": 66.81692244087517,
        "ratio_vs_entry": 2.5649934072234233
      },
      "loop2": {
        "tail_trace": 82.32711467327496,
        "ratio_vs_entry": 3.160404560080351
      },
      "loop3": {
        "tail_trace": 89.22027410886383,
        "ratio_vs_entry": 3.42502178370167
      },
      "loop4": {
        "tail_trace": 90.13194238582776,
        "ratio_vs_entry": 3.460019251926227
      },
      "loop8": {
        "tail_trace": 71.17609831508982,
        "ratio_vs_entry": 2.732335107047776
      }
    },
    "score_summary": {
      "strength": 1.0,
      "examples": 512,
      "loop_results": {
        "1": {
          "correct": 182,
          "total": 512,
          "accuracy": 0.35546875
        },
        "2": {
          "correct": 157,
          "total": 512,
          "accuracy": 0.306640625
        },
        "3": {
          "correct": 158,
          "total": 512,
          "accuracy": 0.30859375
        }
      },
      "oracle_correct": 235,
      "oracle_accuracy": 0.458984375,
      "oracle_gap_vs_loop1": 53,
      "rescued_vs_loop1": 53,
      "harmed_vs_loop1": 74,
      "stable_correct": 108,
      "stable_wrong": 277,
      "pattern_counts": {
        "000": 277,
        "001": 22,
        "010": 4,
        "011": 27,
        "100": 55,
        "101": 1,
        "110": 18,
        "111": 108
      }
    }
  }
]
```

## Next Step
Review fixed-damper depth readout for separate rescued/harmed movement. Depth selection remains open; this run tests whether training on a damped recurrent manifold improves recovery. After review, use debiased_benchmark_suite for the broader competence check.
