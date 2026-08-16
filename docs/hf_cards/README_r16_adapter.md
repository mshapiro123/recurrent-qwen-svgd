---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: transformers
tags:
  - recurrent-depth
  - latent-reasoning
  - lora
  - qwen2.5
  - research
---

# recurrent-qwen2.5-0.5b-r16-adapter

The parameter-efficient recurrent-depth retrofit from *Retrofitting Recurrent Depth into a Pretrained Language Model* ([arXiv:2608.11233](https://arxiv.org/abs/2608.11233); [PDF](https://arxiv.org/pdf/2608.11233)): rank-16 LoRA over all Recurrent Block projections plus the split re-entry bridge, 6,007,425 forward-active trained parameters (4,399,104 LoRA over 84 projections plus a 1,608,321-parameter bridge), with every pretrained base weight frozen. This repository contains the adapter and bridge weights only — a small download that attaches to Qwen2.5-0.5B-Instruct through the recurrent wrapper.

## Architecture

![Architecture comparison from the paper](./figure1_architecture_comparison.svg)

The identical surgery as the full-block release: Prelude (layers 0-5), weight-tied Recurrent Block (layers 6-17) executed T times, Coda (layers 18-23). The LoRA deltas are weight-tied across loop iterations, so the low-rank correction applies at every depth and the adapter is part of the recurrent operator. The base weights never change, so the adapter is detachable and recovery of the base model is guaranteed by construction. Load with `trust_remote_code=True`, using the recurrent wrapper shipped in this repository.

## Usage contract

Forced-depth evaluation only: loops = task depth, no learned halting. General capability is preserved by construction at the base weights; the one-loop identity difference at run end was exactly 0.0 with the base-weight hash unchanged.

## Measured results (receipts in the companion repository)

At 3.3 percent of the full-block trainable budget, this adapter installs the same looping mechanism: 83.76% overall on the frozen 1,792-row Phase A family against the full-block arm's 84.04% (paired p = 0.813), ahead on trained support and near extrapolation through depth 11, behind at depths 12-14. Its threshold-crossing frontier at support 8 was 11.56 against the full-block arm's 11.61. Persistence after outcome-only annealing slightly exceeds the full-block reference (636/640 diagonal, 99.0% of above-diagonal states continuing). Verbal training begun from this installed adapter outpaced matched training from a fresh adapter by 18.6 points at the last comparable checkpoint, including on a held-out surface (917/1,536 versus 617/1,536).

## Limitations — claims this model does not support

- No broad natural-language reasoning gains; zero-shot transfer to verbal surfaces is minimal (16-17%) without verbal training.
- No learned depth selection and no stochastic-width claims.
- The far-tail deficit at depths 12-14 is not attributed among rank, frozen-base geometry, bridge capacity, and optimization; no rank sweep was run.
- The verbal-transfer advantage is measured to step 3,000 only; the comparison's planned endpoint was not reached, and no asymptotic superiority is claimed.
- Training a second operation into this single adapter breached the acquisition-retention boundary in the registered probe; the adapter guarantees recovery, not coexistence.
- Single training seed for the adapter arm and the retention probe.

## Provenance

Adapter SHA-256: `bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839`. Receipts, claim ledger, preregistrations: https://github.com/mshapiro123/recurrent-qwen-svgd. Base model: Qwen2.5-0.5B-Instruct (Apache 2.0), Qwen2.5 Technical Report (arXiv:2412.15115).

## Citation

Shapiro, M. (2026). Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets. arXiv:2608.11233.
