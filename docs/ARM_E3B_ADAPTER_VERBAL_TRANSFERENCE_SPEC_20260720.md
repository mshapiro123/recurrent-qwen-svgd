# Arm E3b: Adapter Verbal Transference

## Question

Does an installed symbolic transition mechanism improve subsequent verbal-task learning at a frozen-base R16 adapter budget, and what synthetic competence is lost during that learning?

## Locked Arms

- **Arm T:** exact Arm E final checkpoint, SHA `bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839`.
- **Arm S:** fresh Qwen2.5-0.5B surgery with a fresh deterministic rank-16, alpha-32 adapter and repaired bridge. It must pass the `1e-3` one-loop identity gate.
- Pretrained Qwen parameters remain frozen and must retain SHA `960f8bf265ba2850c9cdd60a388a00f8f366464babe0507521f010cb7f34971f`.
- The only experimental variable is whether the adapter initialization has symbolic training history.

## P1: Archived Zero-Shot Receipt

The full-block pre-verbal synthetic keeper is read from `stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812`; Arm E is read from `stage5_adapter_parity_e3a_20260719`. Both use the same-reader full-symbol diagonal with forced loops equal to row depth. No GPU rerun is required.

## P2: Matched Verbal Training

The original full-block natural program is copied exactly where it defines the task: 2,048 relay rows plus 2,048 synthetic-rehearsal rows, depths 1-8. Pointer is held out and measures cross-surface transfer; it is not included in training. Both arms use AdamW, learning rate `1e-5`, batch size 1, seed 0, per-loop labels, fixed eight-loop compute, 6,000 optimizer steps, and checkpoints every 1,000 steps.

At every registered checkpoint, both arms are evaluated on the identical frozen 1,536-row relay and 1,536-row pointer sets across depths 1-12. The primary endpoint is pooled Arm-T-minus-Arm-S accuracy with the exact paired sign/McNemar test on row-aligned binary outcomes. `POSITIVE` requires Arm T to exceed Arm S and two-sided `p < 0.05`; otherwise the registered reading is `NULL`. First pooled crossing of 0.71 is reported for each arm. Full-block relay `1321/1536` and pointer `1213/1536` endpoints are descriptive references only.

## P3: Regression

Arm T is evaluated on the frozen 2,048-row synthetic forward battery and the 64-row Tier-1 arithmetic canary at step 0 and every 1,000 steps. Synthetic performance is measurement-only and cannot stop training. It is `retained` only if every stratum remains at or above 0.93 throughout, `collapsed` if the final minimum stratum falls below 0.10 (the E4 near-chance signature was 0.09375), and `partial` otherwise. Tier-1 is a hard stop at three points below Arm T's own step-zero baseline. Arm S receives the same Tier-1 rule against its own step-zero baseline. Base-hash drift is always fatal.

## Operational Rules

The two 6,000-step training runs execute sequentially on one GPU and publish intermediate receipts. Every 1,000-step checkpoint is copied to Drive. The run is disposable, never promotes a keeper, uses one seed, performs no rank sweep, and returns exit code 2 with written tables after a registered hard stop.

