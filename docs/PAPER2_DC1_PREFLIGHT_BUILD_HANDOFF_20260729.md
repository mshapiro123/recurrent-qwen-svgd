# Paper Two DC1-P Build Handoff

Date: 2026-07-29. Status: implementation complete; no GPU run has occurred.

## Authority

The implementation is governed by the byte-verified Drive artifacts now landed in the repository:

- `docs/COMPOSITE_TRAINING_DESIGN_20260729.md`, SHA-256 `0ae848f560dda18abc89deb7716b53b24f40b49f5a7d44a6d5f2e514c9d5ed7b`.
- `docs/STRATEGY_ADDENDUM_DC1_ROADMAP_20260729.md`, SHA-256 `67a38f52529fadf79a9b229e8a88d045a645a1f36cdfc2be89a1effec953a78b`.
- `docs/figures/composite_architecture_20260729.svg`, SHA-256 `444aa15ae4210096a7082d23ec9ec88380f25b1c96624808fc3107ee7907cf9f`.

The earlier DC1 authorization remains controlling for Stage A. This build advances only its preconditions.

## What was built

1. A hard horizontal cap, `k <= 3`, on both composite execution paths.
2. A machine-readable stage policy fixing global vertical depth to `L=1` in Stages A-C and `L=2` in Stage D. Per-position vertical routing is prohibited.
3. Stage-C-ready continue/stop readouts at the real decision point and after each horizontal slot. The logits are observed only. They cannot change execution in this build and remain excluded from visible generation by the existing control-token contract.
4. DC1-P scale interpolation, position-id ablation, and layer-resolved slot-attention instrumentation.
5. A fresh, reusable 500,000-token DEV-C builder with a 50/50 general/code mix, document-hash disjointness from D0 and EVAL-B, and one cached 7B teacher pass.
6. RG-4 with the adjacent-epsilon rule and RG-11 over the full newly authorized range `k = 1, 2, 3` under all three precision policies.
7. One resumable Colab target, `paper2_dc1_preflight`, which publishes public aggregates to GitHub and copies receipts to Drive.

## Important receipt limitation

The requested fragility probe named the model's baseline top-two logit margin on DC0 hurt positions. DC0 saved row-aligned predictions but not those logits. EVAL-B is spent and remains closed. The evaluator therefore records the banked teacher-logprob quartile pattern as a clearly labeled proxy and records the exact model-margin request as not recoverable from saved outputs. It does not reread EVAL-B or manufacture a substitute measure.

## What the target does not do

- It does not create or read EVAL-C.
- It does not train a bridge, adapter, controller, or backbone parameter.
- It does not create an optimizer or write a checkpoint.
- It does not authorize RG-12, DC1 training, Stage B, Stage C, or Stage D.
- It does not resolve any of the four markup-open design choices.

## Exit interpretation

- Exit `0`: DC1-P completed and RG-4/RG-11 are green. Strategy may draft and lock the DC1 preregistration.
- Exit `2`: DC1-P completed, but the numerical precondition needs review. Receipts still land; training remains blocked.
- Any other exit: implementation or environment failure. Do not substitute a dataset, checkpoint, precision policy, or evaluation partition.

## Verification

The implementation followed red/green tests. The focused architecture and launcher suite passed, followed by the complete notebook and recurrent-wrapper regression set. GPU behavior remains to be established by the authorized Colab run.
