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

# recurrent-qwen2.5-0.5b-natural-keeper

The frozen natural keeper of the full-block recurrent retrofit: the verbally trained step-2,000 checkpoint from *Retrofitting Recurrent Depth into a Pretrained Language Model* (arXiv:2608.11233), and the substrate designated for the registered companion study on guided stochastic width.

## What this checkpoint is

The same full-block architecture as the companion full-block release (Prelude 0-5, weight-tied Recurrent Block 6-17 executed T times, Coda 18-23, split re-entry bridge; 180,556,929 forward-active trained parameters; T = 1 reproduces the base computation exactly; loads with `trust_remote_code=True`, using the modeling code shipped in this repository). After verbal fine-tuning on generated relay surfaces, checkpoints past step 2,000 kept improving aggregate accuracy while the worst-case deep tail contracted monotonically. This checkpoint was frozen at the deep-tail peak: the worst-case deep-tail minimum across the two verbal surfaces peaked here at 54.69%, against 19.53% by step 6,000. It was selected to preserve worst-case deep behavior rather than to win on aggregate accuracy.

## Usage contract

Forced-depth evaluation only: loops = task depth, no learned halting. General use should run at T = 1. The keeper lineage is preregistered non-inferior to the base model on ARC-Easy and ARC-Challenge at loop 1.

## Measured results (receipts in the companion repository)

At step 6,000 the same lineage reached 86.0% (relay) and 79.0% (pointer) on the controlled verbal surfaces; this step-2,000 keeper trades aggregate accuracy for the strongest deep tail. It passed the branching-relations validity screen, 389/512 (75.98%) pooled with a minimum depth accuracy of 62.5%, which qualifies it as the substrate for the companion stochastic-width study. Verbal competence comes from verbal training: without it, zero-shot transfer to these surfaces was 16-20% at both budgets.

## Limitations — claims this model does not support

- The verbal surfaces are templated and distractor-free renderings of the synthetic task, not broad natural reasoning benchmarks. No broad natural-language reasoning gains are claimed.
- No learned depth selection and no stochastic-width claims. The width screen establishes substrate competence only; no latent prior or posterior head was trained.
- Preservation is battery-scoped non-inferiority at loop 1, not universal.
- Continued training toward new operations breaches the acquisition-retention boundary documented in the paper.

## Provenance

Checkpoint SHA-256: `0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f`. Receipts, claim ledger, preregistrations: https://github.com/mshapiro123/recurrent-qwen-svgd. Base model: Qwen2.5-0.5B-Instruct (Apache 2.0), Qwen2.5 Technical Report (arXiv:2412.15115).

## Citation

Shapiro, M. (2026). Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets. arXiv:2608.11233.
