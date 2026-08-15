---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: transformers
tags:
  - recurrent-depth
  - latent-reasoning
  - qwen2.5
  - research
---

# recurrent-qwen2.5-0.5b-full-block

The full-block recurrent-depth retrofit of Qwen2.5-0.5B-Instruct: the deep-characterization system of the paper *Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets* (arXiv:2608.11233). This is the consolidated N24 support-12 keeper at step 6,000.

## Architecture

The 24-layer base model is split into a Prelude (layers 0-5), a weight-tied Recurrent Block (layers 6-17), and a Coda (layers 18-23). The block executes T times per pass, and a trained split re-entry bridge combines the carried state with the re-injected Prelude output under an identity-biased gate on loops 2 through T. At T = 1 the recurrent additions are bypassed and the model reproduces the base computation exactly. Trained parameters: the 12-layer block plus the bridge, 180,556,929 forward-active. The recurrent wrapper is custom architecture code: load with `trust_remote_code=True`, using the modeling code shipped in this repository.

## Usage contract

Evaluation is forced-depth: the loop count is set externally to the task depth. The model has no learned halting, and running more loops than a problem's depth overshoots the answer, because the installed operation keeps applying the rule. Off the trained task, extra loops are not free: deep forced loops degrade general-capability behavior, so general use should run at T = 1, where the model is preregistered non-inferior to its base on ARC-Easy and ARC-Challenge.

## Measured results (receipts in the companion repository)

Threshold-crossing validity frontier 17.93 at supervised support 12, a frontier-to-support ratio of 1.49, holding 70.3% accuracy at depth 18. On the frozen 1,792-row Phase A family the full-block arm scored 84.04% overall and 53.13% beyond depth 10, against 72.10% and 2.54% for a same-size serialized-scratchpad control, answering 7.6 times faster at depth 14. The stepwise mechanism persists after outcome-only annealing (625/640 diagonal, 93.0% of above-diagonal states continuing).

## Limitations — claims this model does not support

- No broad natural-language reasoning gains. Competence is demonstrated on a synthetic symbolic family and controlled verbal renderings of it, nothing wider.
- No learned depth selection. Depth is forced; the tested controller was closed as a registered bounded negative.
- No stochastic-width claims in either direction.
- Preservation is non-inferiority on the evaluated ARC battery at loop 1, not universal capability preservation.
- The dense-baseline comparison is a system comparison, with training lineage and inference compute unmatched, not an architecture-only causal estimate.
- Continued training toward new operations breaches an acquisition-retention boundary: the paper's registered continuations could not acquire an inverse operation while preserving the installed mechanism and general capability.

## Provenance

Checkpoint SHA-256: `898a259db2ab344ece4545e2910b051840e8408dbe4927f799e7cdb3cdd8c7dc`. Training and evaluation receipts, the claim ledger, and preregistration documents: https://github.com/mshapiro123/recurrent-qwen-svgd. Base model: Qwen2.5-0.5B-Instruct (Apache 2.0), see the Qwen2.5 Technical Report (arXiv:2412.15115).

## Citation

Shapiro, M. (2026). Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets. arXiv:2608.11233.
