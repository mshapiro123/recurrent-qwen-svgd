# Stage 5 Re-entry Recovery Training - stage5_reentry_recovery_20260627_131940_readout

- Cell version: `reentry_recovery_training_v5_fixed_tail_damper`
- Child curriculum summary: `outputs/stage5/stage5_reentry_recovery_20260627_131940_curriculum_sft/summary.json`
- Trace summary: `outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json`
- Stage 3 repair assessment: `outputs/stage5/stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/summary.json`
- Status: `validation_sane`
- Passed: `True`
- Checkpoint: `outputs/stage5/stage5_reentry_recovery_20260627_131940_curriculum_sft/phase1/phase1_step_100.pt`
- Fixed tail damper: `outputs/stage5/stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/tail_damper.pt`
- Fixed tail damper strength: `1.0`

## Validation Checks
```json
{
  "status": "validation_sane",
  "issues": [],
  "nonfinite_metrics": [],
  "min_mean_expected_loops": 1.05,
  "mean_expected_loops": 1.771071,
  "require_depth_gradient": true,
  "depth_gradient": {
    "available": true,
    "direct_mean_expected_loops": 1.339174,
    "deep_narrow_mean_expected_loops": 1.98702,
    "required_margin": 0.25,
    "observed": true
  },
  "require_target_loop_gradient": false,
  "target_loop_gradient": {
    "available": true,
    "points": [
      {
        "target_loop_count": 1.0,
        "mean_expected_loops": 1.339174,
        "examples": 2.0
      },
      {
        "target_loop_count": 2.0,
        "mean_expected_loops": 1.997607,
        "examples": 3.0
      },
      {
        "target_loop_count": 3.0,
        "mean_expected_loops": 1.955259,
        "examples": 1.0
      }
    ],
    "required_margin": 0.1,
    "adjacent_margins": [
      0.6584329999999998,
      -0.04234799999999983
    ],
    "observed": false
  }
}
```

## Post-Recovery Re-entry Health
- Drift summary: `outputs/stage5/stage5_reentry_recovery_20260627_131940_readout/post_reentry_drift.json`
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
    "bridge_gate": 1.0008049011230469,
    "bridge_delta_rms": 0.1509188562631607,
    "bridge_weight_grad_rms": 0.38931503891944885,
    "bridge_bias_grad_rms": 0.007570483721792698,
    "reentry_adapter_mode": "spectral",
    "reentry_adapter_delta_rms": 0.006415508687496185,
    "reentry_adapter_scale_grad_rms": 0.0,
    "reentry_adapter_bias_grad_rms": 0.0,
    "reentry_adapter_spectral_u_grad_rms": 0.15641836822032928,
    "reentry_adapter_spectral_v_grad_rms": 0.043190859258174896,
    "reentry_adapter_spectral_theta_grad_abs": 0.11866993457078934,
    "reentry_adapter_gradient_live": true,
    "loop8_output_over_entry_rms": 1.0032675663630168,
    "loop8_output_over_input_rms": 1.0051012833913167,
    "mean_exit_over_entry_rms": 1.0017683903376262,
    "subspace_overlap": 0.34538424015045166
  }
}
```

## Fixed-Damper Depth Readout
- Summary: `outputs/stage5/stage5_reentry_recovery_20260627_131940_readout/fixed_tail_damper_depth_readout/summary.json`
```json
[
  {
    "strength": 0.0,
    "tail_trace": {
      "entry": {
        "tail_trace": 26.08695414921531,
        "ratio_vs_entry": 1.0
      },
      "loop1": {
        "tail_trace": 68.12258700684418,
        "ratio_vs_entry": 2.6113660727576087
      },
      "loop2": {
        "tail_trace": 133.85550122154973,
        "ratio_vs_entry": 5.131128013485472
      },
      "loop3": {
        "tail_trace": 220.12722947690258,
        "ratio_vs_entry": 8.438211230709124
      },
      "loop4": {
        "tail_trace": 323.87111065556996,
        "ratio_vs_entry": 12.415060370906197
      },
      "loop8": {
        "tail_trace": 864.7926109432204,
        "ratio_vs_entry": 33.15038643440991
      }
    },
    "score_summary": {
      "strength": 0.0,
      "examples": 512,
      "loop_results": {
        "1": {
          "correct": 179,
          "total": 512,
          "accuracy": 0.349609375
        },
        "2": {
          "correct": 173,
          "total": 512,
          "accuracy": 0.337890625
        },
        "3": {
          "correct": 159,
          "total": 512,
          "accuracy": 0.310546875
        }
      },
      "oracle_correct": 228,
      "oracle_accuracy": 0.4453125,
      "oracle_gap_vs_loop1": 49,
      "rescued_vs_loop1": 49,
      "harmed_vs_loop1": 68,
      "stable_correct": 111,
      "stable_wrong": 284,
      "pattern_counts": {
        "000": 284,
        "001": 11,
        "010": 5,
        "011": 33,
        "100": 40,
        "101": 4,
        "110": 24,
        "111": 111
      }
    }
  },
  {
    "strength": 1.0,
    "tail_trace": {
      "entry": {
        "tail_trace": 26.08695414921531,
        "ratio_vs_entry": 1.0
      },
      "loop1": {
        "tail_trace": 68.12258700684418,
        "ratio_vs_entry": 2.6113660727576087
      },
      "loop2": {
        "tail_trace": 84.6943084666186,
        "ratio_vs_entry": 3.2466154531561586
      },
      "loop3": {
        "tail_trace": 93.33210603298959,
        "ratio_vs_entry": 3.5777310566476004
      },
      "loop4": {
        "tail_trace": 94.59517317771882,
        "ratio_vs_entry": 3.6261486349323087
      },
      "loop8": {
        "tail_trace": 78.1278410206142,
        "ratio_vs_entry": 2.9949008448333654
      }
    },
    "score_summary": {
      "strength": 1.0,
      "examples": 512,
      "loop_results": {
        "1": {
          "correct": 179,
          "total": 512,
          "accuracy": 0.349609375
        },
        "2": {
          "correct": 168,
          "total": 512,
          "accuracy": 0.328125
        },
        "3": {
          "correct": 159,
          "total": 512,
          "accuracy": 0.310546875
        }
      },
      "oracle_correct": 233,
      "oracle_accuracy": 0.455078125,
      "oracle_gap_vs_loop1": 54,
      "rescued_vs_loop1": 54,
      "harmed_vs_loop1": 70,
      "stable_correct": 109,
      "stable_wrong": 279,
      "pattern_counts": {
        "000": 279,
        "001": 16,
        "010": 9,
        "011": 29,
        "100": 44,
        "101": 5,
        "110": 21,
        "111": 109
      }
    }
  }
]
```

## Next Step
Review fixed-damper depth readout for separate rescued/harmed movement. Depth selection remains open; this run tests whether training on a damped recurrent manifold improves recovery. After review, use debiased_benchmark_suite for the broader competence check.
