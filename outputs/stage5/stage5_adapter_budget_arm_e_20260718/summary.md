# Adapter-Budget Arm E - stage5_adapter_budget_arm_e_20260718

- Status: `finished`
- Verdict: `deficit`
- Initialization: `fresh_base_qwen_surgery`
- Trainable set: R16 LoRA plus active repaired split bridge (`6,007,425` parameters)
- Optimizer: `AdamW`

## Stages

| Stage | Steps | Checkpoint SHA | Canary | Dose receipts |
|---|---:|---|---|---:|
| primitive_depth1 | 500 | `24cd7b8f1308` | manual/final | 0 |
| chain_depth_le2 | 2000 | `7433f73d06ef` | green_continue | 4 |
| chain_depth_le4 | 4000 | `e68dd86005fd` | green_continue | 8 |
| chain_depth_le8 | 2000 | `cd4b6a02793c` | green_continue | 4 |
| chain_depth_le8_dose | 2000 | `bffa8c4277ce` | green_continue | 4 |

## Paired Phase A Read

- Arm A: `1506/1792`.
- Arm E: `1501/1792`.
- Verdict: `deficit`.
- Deficit shape: `tail_concentrated`.
