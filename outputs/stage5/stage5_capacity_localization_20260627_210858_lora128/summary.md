# Stage 5 Re-entry Recovery Training - stage5_capacity_localization_20260627_210858_lora128

- Cell version: `reentry_recovery_training_v5_fixed_tail_damper`
- Child curriculum summary: `outputs/stage5/stage5_capacity_localization_20260627_210858_lora128_curriculum_sft/summary.json`
- Trace summary: `outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json`
- Stage 3 repair assessment: `outputs/stage5/stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/summary.json`
- Status: `validation_sane`
- Passed: `True`
- Checkpoint: `outputs/stage5/stage5_capacity_localization_20260627_210858_lora128_curriculum_sft/phase1/phase1_step_100.pt`
- Fixed tail damper: `outputs/stage5/stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/tail_damper.pt`
- Fixed tail damper strength: `1.0`

## Validation Checks
```json
{
  "status": "validation_sane",
  "issues": [],
  "nonfinite_metrics": [],
  "min_mean_expected_loops": 1.05,
  "mean_expected_loops": 1.77308,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.367357,
    "deep_narrow_mean_expected_loops": 1.975941,
    "required_margin": 0.25,
    "observed": true
  },
  "require_target_loop_gradient": false,
  "target_loop_gradient": {
    "available": true,
    "points": [
      {
        "target_loop_count": 1.0,
        "mean_expected_loops": 1.367357,
        "examples": 2.0
      },
      {
        "target_loop_count": 2.0,
        "mean_expected_loops": 1.987316,
        "examples": 3.0
      },
      {
        "target_loop_count": 3.0,
        "mean_expected_loops": 1.941816,
        "examples": 1.0
      }
    ],
    "required_margin": 0.1,
    "adjacent_margins": [
      0.6199590000000001,
      -0.045500000000000096
    ],
    "observed": false
  }
}
```

## Post-Recovery Re-entry Health
- Drift summary: `outputs/stage5/stage5_capacity_localization_20260627_210858_lora128/post_reentry_drift.json`
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
    "bridge_gate": 1.000808596611023,
    "bridge_delta_rms": 0.15300339460372925,
    "bridge_weight_grad_rms": 0.3893328905105591,
    "bridge_bias_grad_rms": 0.0075752693228423595,
    "reentry_adapter_mode": "spectral",
    "reentry_adapter_delta_rms": 0.006337281316518784,
    "reentry_adapter_scale_grad_rms": 0.0,
    "reentry_adapter_bias_grad_rms": 0.0,
    "reentry_adapter_spectral_u_grad_rms": 0.15144726634025574,
    "reentry_adapter_spectral_v_grad_rms": 0.039701223373413086,
    "reentry_adapter_spectral_theta_grad_abs": 0.11867845058441162,
    "reentry_adapter_gradient_live": true,
    "loop8_output_over_entry_rms": 1.0061067342758179,
    "loop8_output_over_input_rms": 1.005152940750122,
    "mean_exit_over_entry_rms": 1.0017465750376384,
    "subspace_overlap": 0.3539472818374634
  }
}
```

## Fixed-Damper Depth Readout
- Summary: `outputs/stage5/stage5_capacity_localization_20260627_210858_lora128/fixed_tail_damper_depth_readout/summary.json`
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
        "tail_trace": 130.6778800282757,
        "ratio_vs_entry": 5.016512112467632
      },
      "loop3": {
        "tail_trace": 213.4400654996682,
        "ratio_vs_entry": 8.193618335660863
      },
      "loop4": {
        "tail_trace": 312.89091796729696,
        "ratio_vs_entry": 12.01137544873265
      },
      "loop8": {
        "tail_trace": 839.3088703266695,
        "ratio_vs_entry": 32.21970782801365
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
          "correct": 156,
          "total": 512,
          "accuracy": 0.3046875
        },
        "3": {
          "correct": 159,
          "total": 512,
          "accuracy": 0.310546875
        }
      },
      "oracle_correct": 229,
      "oracle_accuracy": 0.447265625,
      "oracle_gap_vs_loop1": 47,
      "rescued_vs_loop1": 47,
      "harmed_vs_loop1": 67,
      "stable_correct": 115,
      "stable_wrong": 283,
      "pattern_counts": {
        "000": 283,
        "001": 20,
        "010": 5,
        "011": 22,
        "100": 51,
        "101": 2,
        "110": 14,
        "111": 115
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
        "tail_trace": 82.21212712960735,
        "ratio_vs_entry": 3.155990374561986
      },
      "loop3": {
        "tail_trace": 88.91271887866743,
        "ratio_vs_entry": 3.413215236662499
      },
      "loop4": {
        "tail_trace": 89.16231380655951,
        "ratio_vs_entry": 3.4227967815935165
      },
      "loop8": {
        "tail_trace": 70.37545394191885,
        "ratio_vs_entry": 2.7015996666280033
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
          "correct": 158,
          "total": 512,
          "accuracy": 0.30859375
        },
        "3": {
          "correct": 159,
          "total": 512,
          "accuracy": 0.310546875
        }
      },
      "oracle_correct": 235,
      "oracle_accuracy": 0.458984375,
      "oracle_gap_vs_loop1": 53,
      "rescued_vs_loop1": 53,
      "harmed_vs_loop1": 73,
      "stable_correct": 109,
      "stable_wrong": 277,
      "pattern_counts": {
        "000": 277,
        "001": 22,
        "010": 4,
        "011": 27,
        "100": 54,
        "101": 1,
        "110": 18,
        "111": 109
      }
    }
  }
]
```

## Next Step
Review fixed-damper depth readout for separate rescued/harmed movement. Depth selection remains open; this run tests whether training on a damped recurrent manifold improves recovery. After review, use debiased_benchmark_suite for the broader competence check.
