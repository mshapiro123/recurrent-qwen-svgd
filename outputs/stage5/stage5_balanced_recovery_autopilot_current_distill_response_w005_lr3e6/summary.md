# Stage 5 Phase1 Recovery Ladder - stage5_balanced_recovery_autopilot_current_distill_response_w005_lr3e6

## Question
Can continued deterministic recurrent training close or reverse the remaining gap to base Qwen before further particle/SVGD work?

## ARC-Challenge Proxy
- Base Qwen: `{'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}}`
- Phase1 start (outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_150.pt): `{'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}}`
- Best checkpoint: `outputs/stage5/stage5_balanced_recovery_autopilot_current_distill_response_w005_lr3e6/phase1/phase1_step_100.pt`
- Best checkpoint ARC: `{'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}}`
- Start lift: `0.0`
- Gap to base: `-0.0078125`

## Checkpoint Ladder
- outputs/stage5/stage5_balanced_recovery_autopilot_current_distill_response_w005_lr3e6/phase1/phase1_step_50.pt: arc={'mean': {'correct': 70, 'total': 128, 'accuracy': 0.546875}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.318866, 'halting_kl': 0.167366, 'loss': 2.338949, 'mean_expected_loops': 2.403297, 'mean_halt_entropy': 1.373937} vs_start={'helped': 1, 'hurt': 2, 'tied': 125, 'prediction_changes': 10} vs_base={'helped': 8, 'hurt': 10, 'tied': 110, 'prediction_changes': 27}
- outputs/stage5/stage5_balanced_recovery_autopilot_current_distill_response_w005_lr3e6/phase1/phase1_step_100.pt: arc={'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.304135, 'halting_kl': 0.154814, 'loss': 2.322712, 'mean_expected_loops': 2.371335, 'mean_halt_entropy': 1.371974} vs_start={'helped': 2, 'hurt': 2, 'tied': 124, 'prediction_changes': 7} vs_base={'helped': 9, 'hurt': 10, 'tied': 109, 'prediction_changes': 28}

## Decision Rule
If the best checkpoint improves over Phase1 start, keep extending deterministic recovery. If it regresses while validation improves, add base-logit distillation before more Opus training.

Note: this is not ARC-AGI. It is the competence recovery proxy before building the ARC-AGI-specific solver/eval loop.
