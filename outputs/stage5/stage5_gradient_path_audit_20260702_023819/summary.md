# Gradient-Path Audit

status: `graph_connected`
issues: `[]`

## Selected Batch

- row: `train_d02_00000`
- depth: `2`

## Per-Loop Gradient Matrix

| loop | active_tokens | loss | bridge_prelude_rms | bridge_state_rms | recurrent_rms | coda_rms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.000004 | 0.000e+00 | 0.000e+00 | 2.068e-07 | 1.453e-07 |
| 2 | 1 | 0.203779 | 4.999e-03 | 9.082e-03 | 2.518e-03 | 3.908e-03 |

## Finite Difference

| loop | base_loss | perturbed_loss | abs_delta | delta_per_epsilon |
|---:|---:|---:|---:|---:|
| 1 | 0.000004 | 0.000004 | 0.000e+00 | 0.000e+00 |
| 2 | 0.203779 | 0.228022 | 2.424e-02 | 2.424e+00 |
