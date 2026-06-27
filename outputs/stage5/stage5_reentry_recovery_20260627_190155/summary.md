# Stage 5 Re-entry Recovery Training - stage5_reentry_recovery_20260627_190155

- Cell version: `reentry_recovery_training_v5_fixed_tail_damper`
- Child curriculum summary: `outputs/stage5/stage5_reentry_recovery_20260627_190155_curriculum_sft/summary.json`
- Trace summary: `outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json`
- Stage 3 repair assessment: `outputs/stage5/stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/summary.json`
- Status: `validation_sane`
- Passed: `True`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260627_190155_curriculum_sft/phase1/phase1_step_100.pt`
- Fixed tail damper: `outputs/stage5/stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/tail_damper.pt`
- Fixed tail damper strength: `1.0`

## Validation Checks
```json
{
  "status": "validation_sane",
  "issues": [],
  "nonfinite_metrics": [],
  "min_mean_expected_loops": 1.05,
  "mean_expected_loops": 1.774222,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.367029,
    "deep_narrow_mean_expected_loops": 1.977819,
    "required_margin": 0.25,
    "observed": true
  },
  "require_target_loop_gradient": false,
  "target_loop_gradient": {
    "available": true,
    "points": [
      {
        "target_loop_count": 1.0,
        "mean_expected_loops": 1.367029,
        "examples": 2.0
      },
      {
        "target_loop_count": 2.0,
        "mean_expected_loops": 1.989368,
        "examples": 3.0
      },
      {
        "target_loop_count": 3.0,
        "mean_expected_loops": 1.943173,
        "examples": 1.0
      }
    ],
    "required_margin": 0.1,
    "adjacent_margins": [
      0.622339,
      -0.046194999999999986
    ],
    "observed": false
  }
}
```

## Post-Recovery Re-entry Health
- Drift summary: `outputs/stage5/stage5_reentry_recovery_20260627_190155/post_reentry_drift.json`
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
    "bridge_gate": 1.0008302927017212,
    "bridge_delta_rms": 0.16522078216075897,
    "bridge_weight_grad_rms": 0.38929417729377747,
    "bridge_bias_grad_rms": 0.007574875373393297,
    "reentry_adapter_mode": "spectral",
    "reentry_adapter_delta_rms": 0.00679776119068265,
    "reentry_adapter_scale_grad_rms": 0.0,
    "reentry_adapter_bias_grad_rms": 0.0,
    "reentry_adapter_spectral_u_grad_rms": 0.17443014681339264,
    "reentry_adapter_spectral_v_grad_rms": 0.04415825381875038,
    "reentry_adapter_spectral_theta_grad_abs": 0.11808851361274719,
    "reentry_adapter_gradient_live": true,
    "loop8_output_over_entry_rms": 1.0046992301940918,
    "loop8_output_over_input_rms": 1.005111853281657,
    "mean_exit_over_entry_rms": 1.0017465750376384,
    "subspace_overlap": 0.3539472818374634
  }
}
```

## Fixed-Damper Depth Readout
- Summary: `outputs/stage5/stage5_reentry_recovery_20260627_190155/fixed_tail_damper_depth_readout/summary.json`
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
        "tail_trace": 130.9274745682461,
        "ratio_vs_entry": 5.026093642506965
      },
      "loop3": {
        "tail_trace": 214.97214943625278,
        "ratio_vs_entry": 8.252432555967548
      },
      "loop4": {
        "tail_trace": 316.983683376467,
        "ratio_vs_entry": 12.168490082396339
      },
      "loop8": {
        "tail_trace": 870.0444717877531,
        "ratio_vs_entry": 33.399597775571266
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
          "correct": 159,
          "total": 512,
          "accuracy": 0.310546875
        }
      },
      "oracle_correct": 228,
      "oracle_accuracy": 0.4453125,
      "oracle_gap_vs_loop1": 46,
      "rescued_vs_loop1": 46,
      "harmed_vs_loop1": 67,
      "stable_correct": 115,
      "stable_wrong": 284,
      "pattern_counts": {
        "000": 284,
        "001": 18,
        "010": 4,
        "011": 24,
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
        "tail_trace": 82.4970101926505,
        "ratio_vs_entry": 3.1669265738336887
      },
      "loop3": {
        "tail_trace": 90.04025926884023,
        "ratio_vs_entry": 3.4564996855942876
      },
      "loop4": {
        "tail_trace": 91.52169565849897,
        "ratio_vs_entry": 3.5133696286249316
      },
      "loop8": {
        "tail_trace": 73.72130176281186,
        "ratio_vs_entry": 2.8300413441051058
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
          "correct": 163,
          "total": 512,
          "accuracy": 0.318359375
        },
        "3": {
          "correct": 159,
          "total": 512,
          "accuracy": 0.310546875
        }
      },
      "oracle_correct": 234,
      "oracle_accuracy": 0.45703125,
      "oracle_gap_vs_loop1": 52,
      "rescued_vs_loop1": 52,
      "harmed_vs_loop1": 71,
      "stable_correct": 111,
      "stable_wrong": 278,
      "pattern_counts": {
        "000": 278,
        "001": 18,
        "010": 5,
        "011": 29,
        "100": 52,
        "101": 1,
        "110": 18,
        "111": 111
      }
    }
  }
]
```

## Next Step
Review fixed-damper depth readout for separate rescued/harmed movement. Depth selection remains open; this run tests whether training on a damped recurrent manifold improves recovery. After review, use debiased_benchmark_suite for the broader competence check.
