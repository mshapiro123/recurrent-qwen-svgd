# Phase-A Surpass Preregistration - stage5_phase_a_surpass_prereg_20260707_122918

- Status: `preregistered`
- Frozen eval set: `stage5_synthetic_depth_frozen_eval_v2_depth14`
- Primary gate: A beats B at >=3 consecutive depths with one-sided Fisher p<0.05 per depth

## Arms
- `A_looped`: support-8 dose-arm checkpoint, same-reader final-symbol metric
- `B_dense_direct`: dense Qwen2.5-0.5B LoRA direct final-symbol SFT, 4000 steps
- `C_dense_scratchpad`: dense Qwen2.5-0.5B LoRA serialized-orbit scratchpad SFT, 4000 steps
- `D_dense_1p5b_direct_optional`: dense Qwen2.5-1.5B direct-answer exchange-rate arm

## Compute Ledger
- Looped arm: zero text context growth across latent loops
- Scratchpad arm: linear serialized-orbit context growth
- Policy: do not claim raw FLOP advantage; report one parallel pass per latent step and sequence/context tradeoff
