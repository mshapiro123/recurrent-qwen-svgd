# Paper Two DC1 Stage A Resource Note

**Date:** 2026-07-30  
**Status:** proposed preregistration inputs only; no training authority  
**Required runtime:** NVIDIA A100-SXM4-80GB or equivalent 80GB accelerator

## Proposed locked values

| Field | Proposed value |
|---|---:|
| Step ceiling | 2,000 optimizer steps |
| Microbatch | 1 row |
| Gradient accumulation | 1 |
| Effective batch | 1 row |
| Maximum sequence length | 512 tokens |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `0.0` |
| Gradient clipping | `0.5` global norm |
| Precision | full fp32 for model, feedback boundary, gradients, and optimizer |
| Passive checkpoints | steps 500, 1,000, 1,500, and 2,000 |
| Expected wall time | approximately 2 to 4 hours on one A100-SXM4-80GB |

The learning rate is the bridge-only rate already exercised by the composite
optimizer-coverage smoke in `tests/test_coconut_composite.py`. It is ten times
the `1e-5` full-block rate used in the installed-mechanism stages, which is
appropriate here because only the zero-initialized 802,816-parameter horizontal
delta matrix moves. The proposal keeps batch and accumulation at one so that a
"step" remains one observed row, matching the program's existing T1 training
accounting and avoiding an unrecorded eightfold dose increase.

## Memory and timing basis

The trainable optimizer state is small; the cost is retaining a full fp32
backward graph through the frozen 0.5B model for the baseline and one appended
slot under recompute. The A100 requirement follows RG-11, not merely capacity:
both tested bfloat16 policies failed the per-example gradient-direction
criterion, while full fp32 passed. The 2 to 4 hour estimate includes four
checkpoint receipts and startup assertions. The launcher should print measured
peak allocated memory and a 20-step throughput projection before proceeding
beyond step 20. A measured projection outside this range is reported, not used
to change the locked step count.

## Boundary

This note does not authorize training and is not a preregistration. The Stage A
launcher may be created only after the separate governing preregistration is
stored in Drive with its SHA-256 and `locked_before_training` receipt.
