# Benchmark Status - 2026-06-20

## Main Result
Current recurrent/SVGD checkpoints are diagnostically useful, but not ready to claim benchmark improvement over base Qwen 0.5B.

## Completed Runs

### Exact-task SVGD heldout diagnostics
- Random32 baseline, seeds 0-9: best_hits 77/140, candidate_hits 251/560.
- Within-group PCA dim8 repulsion=2, seeds 0-9: best_hits 89/140, candidate_hits 264/560.
- Interpretation: within-group SVGD produces useful diversity on the exact synthetic task suite.

### Tiny MCQ smoke, 8 questions
- base label: {'mean': '4/8 (50.0%)'}
- phase1 label: {'mean': '4/8 (50.0%)'}
- phase2 label: {'max': '4/8 (50.0%)', 'mean': '5/8 (62.5%)', 'vote': '5/8 (62.5%)'}
- base option_text: {'mean': '3/8 (37.5%)'}
- phase1 option_text: {'mean': '1/8 (12.5%)'}
- phase2 option_text: {'max': '2/8 (25.0%)', 'mean': '2/8 (25.0%)', 'vote': '3/8 (37.5%)'}
- base label_and_text: {'mean': '4/8 (50.0%)'}
- phase1 label_and_text: {'mean': '1/8 (12.5%)'}
- phase2 label_and_text: {'max': '3/8 (37.5%)', 'mean': '3/8 (37.5%)', 'vote': '3/8 (37.5%)'}
- Interpretation: label scoring is the only stable scoring mode in this harness; option_text and label_and_text are brittle for these adapters.

### ARC-Challenge validation smoke
- ARC-32 base: {'mean': '20/32 (62.5%)'}
- ARC-32 phase1: {'mean': '16/32 (50.0%)'}
- ARC-32 phase2: {'max': '19/32 (59.4%)', 'mean': '18/32 (56.2%)', 'vote': '18/32 (56.2%)'}
- ARC-128 base: {'mean': '72/128 (56.2%)'}
- ARC-128 phase1: {'mean': '66/128 (51.6%)'}
- ARC-128 phase2: {'max': '68/128 (53.1%)', 'mean': '63/128 (49.2%)', 'vote': '63/128 (49.2%)'}
- Interpretation: the current recurrent checkpoints regress versus base on ARC. Phase2 max aggregation recovers part of the Phase1 loss but still trails base.

## Blockers
- GPQA-Diamond is gated on Hugging Face for the current token/account. The runner exists, but access must be approved before Colab can download it.
- The current Phase1/Phase2 checkpoints were trained as small diagnostic smoke artifacts, not benchmark-targeted adapters.

## Next Actions
1. Treat current checkpoints as diagnostic baselines only.
2. Build the modified Opus reasoning-trace fine-tuning pass with stronger heldout validation.
3. Add benchmark-aware validation during fine-tuning: exact tasks, ARC subset, and GPQA once access is approved.
4. Only package/push a Hugging Face adapter after it matches or beats base on at least one non-toy benchmark slice without collapsing exact-task diversity.

