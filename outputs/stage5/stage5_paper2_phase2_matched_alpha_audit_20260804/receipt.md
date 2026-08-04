# Phase-2 Matched-Alpha Read-Only Audit Receipt

Status: `complete_read_only`

This receipt evaluates the six terminal pilot checkpoints without constructing an optimizer or updating model parameters.

| Arm | Abort step | Exact retention | Exact acceptance delta | Demand above permission | Huber linear regime |
|---|---:|---:|---:|---:|---:|
| alpha_0p0_seed_0 | 500 | 0.996569 | -0.000530 | 30.11% | 0.05% |
| alpha_0p0_seed_1 | 600 | 0.996738 | -0.003086 | 31.66% | 0.05% |
| alpha_0p5_seed_0 | 193 | 0.997543 | -0.003038 | 91.67% | 0.44% |
| alpha_0p5_seed_1 | 184 | 0.998433 | -0.001924 | 92.63% | 0.43% |
| alpha_1p0_seed_0 | 146 | 0.997967 | -0.002648 | 84.20% | 18.52% |
| alpha_1p0_seed_1 | 146 | 0.997331 | -0.002438 | 83.83% | 17.93% |

## Boundaries

- DEV-only; no frozen E1 evaluation partition was touched.
- This audit does not select alpha or authorize E1.
- Per-training-step trust magnitudes were not stored. Scheduled evaluation rent is reported only as a proxy.
- The old 0.997 retention rule is reported as endpoint qualification, not relabeled as a catastrophe tripwire.
