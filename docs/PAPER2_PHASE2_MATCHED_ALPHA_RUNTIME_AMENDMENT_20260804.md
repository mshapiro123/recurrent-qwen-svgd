# Phase-2 Matched Alpha Runtime Amendment

Date: 2026-08-04

Status: `locked_before_training`. This amendment was written after the launch
preflight rejected an NVIDIA A100-SXM4-40GB and before any optimizer step ran.
It changes only the minimum accelerator-memory guard. No experimental arm,
seed, data row, model parameter, precision, loss, optimizer setting, budget,
gate, stopping rule, or decision rule changes.

## Reason

The original resource note inherited an A100-80GB requirement from the Stage
0A teacher-collection path. The matched-alpha implementation does not load a
teacher or student transformer. It consumes the already frozen Stage 0A hidden
states and sparse candidate lattice, keeps the large immutable cache in system
RAM, and stages only each training batch to CUDA. Its persistent CUDA objects
are the frozen 14B LM head (about 3.1 GB in fp32), frozen 0.5B tied embedding
(about 0.55 GB in fp32), the 1.185M-parameter student module and AdamW state,
and batch-local tensors. The full recurrent-gradient path remains fp32.

## Amended Contract

- Minimum CUDA memory: 35 GiB, admitting an A100-SXM4-40GB with approximately
  39.5 GiB visible memory.
- Required architecture class: NVIDIA A100 or equivalent CUDA accelerator.
- Local scratch and high-system-RAM staging remain mandatory.
- The allowance applies only to the cached-state sparse-logit matched-alpha
  launcher. Any path loading a transformer backbone retains its own larger
  runtime requirement.
- The launcher records GPU name and visible memory in the status receipt.

The rejected preflight performed no training and consumed no registered run.
