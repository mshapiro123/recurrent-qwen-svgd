# Phase A Dense Full-Model Controls

This amendment replaces the unimplemented LoRA placeholders in the earlier
Phase A preregistration with matched full-model AdamW arms. B and C use pinned
Qwen2.5-0.5B-Instruct; D uses pinned Qwen2.5-1.5B-Instruct. All arms use FP32
parameters and AdamW moments, BF16 forward/backward compute, effective batch 8,
2e-6 learning rate, and 4,000 optimizer steps.

The locked training source has SHA256
`260d5c11c0b6e97d1f09c9356b1eaedbde86cceac4053cc6bf561e53d0176bde`.
The locked depth-1..14 evaluation source has SHA256
`aaa71c3d4cc500f68fac7ee6f5f0e31d9e11570bdff90adb805c769c12c66cd3`.

- B trains direct final symbols.
- C trains serialized orbit steps followed by the same final symbol.
- D trains direct final symbols at 1.5B and is the scale exchange-rate arm.

Evaluation greedily generates and applies the same reader to all arms: prefer
an explicit `answer:` marker, otherwise take the first valid full symbol. B/C
may run sequentially on one L4. D requires at least 35 GiB and
runs independently on an A100 or larger GPU. These jobs are independent of the
inverse-table rebase gate and may execute concurrently. Checkpoints live on
Drive; only lightweight receipts are published to GitHub.
