# Recurrent Baseline Ladder

The central model question is **base Qwen 0.5B versus recurrent Qwen 0.5B**.
Because the recurrent wrapper changes the computation path, the recurrent model
can be weaker before it is trained well. We should therefore separate internal
recurrent progress from the final base-model comparison.

## Model Roles

1. **Base Qwen 0.5B**
   - Unmodified `Qwen/Qwen2.5-0.5B-Instruct`.
   - This is the outer target.
   - A release candidate should eventually match or beat this baseline on at
     least one non-toy benchmark slice without losing exact-task competence.

2. **Identity wrapper**
   - Recurrent wrapper with `max_loops=1`.
   - This is a correctness gate, not a research baseline.
   - It should remain logit-identical to base Qwen under the strict identity
     settings.

3. **Phase1 deterministic recurrent**
   - Recurrent depth with sequence-level PonderNet halting.
   - This is the first meaningful recurrent baseline.
   - It may trail base Qwen at first because it has changed architecture and
     only small adapter/controller training.

4. **Phase2 stochastic/SVGD recurrent**
   - Recurrent model with multiple particle trajectories and SVGD-style
     trajectory separation.
   - This is the recurrent candidate.
   - Its first job is to beat Phase1; its second job is to close and then cross
     the gap to base Qwen.

## Evaluation Questions

Report these separately:

- **Internal recurrent lift:** Phase2 minus Phase1.
- **Base gap:** best recurrent variant minus base Qwen.
- **Trajectory value:** best-of-K or selected-K minus K=1 deterministic recurrent.
- **Diversity health:** candidate diversity and unique-correct counts, not just
  raw unique generations.

This prevents the project from discarding a useful recurrent improvement simply
because the recurrent architecture is not yet trained enough to beat the base
model.

## Current Read

As of the 2026-06-20 smoke runs:

- Within-group SVGD improves exact-task heldout diagnostics versus random
  projected repulsion.
- Phase2 sometimes recovers Phase1 loss on MCQ/ARC, especially with max
  aggregation.
- Base Qwen still beats the current recurrent checkpoints on ARC-128.

That means the SVGD mechanism is not disproven, but the current checkpoints are
diagnostic artifacts, not release candidates.

## Next Gate

The Stage 4 Opus fine-tune should produce a stronger deterministic recurrent
baseline first. The gate after Stage 4 is:

1. Phase1 trained on modified Opus traces has finite validation loss and
   non-collapsed loop depth.
2. Phase2/SVGD improves over that Phase1 baseline on exact tasks and at least
   one MCQ/ARC slice.
3. The best recurrent variant is closer to base Qwen than the previous
   recurrent checkpoints.

Only after those pass should we spend serious A100 time on GPQA Diamond or HF
release packaging.
