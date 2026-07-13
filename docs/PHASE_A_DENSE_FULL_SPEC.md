# Phase A Dense Full-Model Controls

This amendment replaces the unimplemented LoRA placeholders in the earlier
Phase A preregistration with matched full-model AdamW arms. B and C use pinned
Qwen2.5-0.5B-Instruct; D uses pinned Qwen2.5-1.5B-Instruct. All arms use FP32
parameters and AdamW moments, BF16 forward/backward compute, effective batch 8,
2e-6 learning rate, and 4,000 optimizer steps.

The locked training source has SHA256
`cf61c14c2629f2caa7e1b6bd100adb122a468d5285b74970aaa4aebfbb56fd12`.
The locked depth-1..14 evaluation source has SHA256
`3de844669aba303063e6932f5852914ee0993e531c8e65c2a4c4b18e219b3fc8`.

These are SHA256 hashes of the JSONL bytes after normalizing CRLF/CR to LF, so
the same Git rows have identical receipts on Windows and Linux/Colab.

- B trains direct final symbols.
- C trains serialized orbit steps followed by the same final symbol.
- D trains direct final symbols at 1.5B and is the scale exchange-rate arm.

Evaluation greedily generates and applies the same reader to all arms: prefer
an explicit `answer:` marker, otherwise take the first valid full symbol. B/C
may run sequentially on one L4. D requires at least 35 GiB and
runs independently on an A100 or larger GPU. These jobs are independent of the
inverse-table rebase gate and may execute concurrently. Checkpoints live on
Drive; only lightweight receipts are published to GitHub.
