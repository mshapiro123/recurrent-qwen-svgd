# Stage 5 Phase1 Recovery Ladder - stage5_phase1_distill_continue_250_arc128_20260621_164533

## Question
Can continued deterministic recurrent training close or reverse the remaining gap to base Qwen before further particle/SVGD work?

## ARC-Challenge Proxy
- Base Qwen: `{'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}}`
- Phase1 start (outputs/stage5/stage5_phase1_distill_recovery_250_arc128_20260621_163546/phase1/phase1_step_250.pt): `{'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}}`
- Best checkpoint: `outputs/stage5/stage5_phase1_distill_continue_250_arc128_20260621_164533/phase1/phase1_step_125.pt`
- Best checkpoint ARC: `{'mean': {'correct': 70, 'total': 128, 'accuracy': 0.546875}}`
- Start lift: `-0.0078125`
- Gap to base: `-0.015625`

## Checkpoint Ladder
- outputs/stage5/stage5_phase1_distill_continue_250_arc128_20260621_164533/phase1/phase1_step_125.pt: arc={'mean': {'correct': 70, 'total': 128, 'accuracy': 0.546875}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.257159, 'halting_kl': 0.113579, 'loss': 2.266245, 'mean_expected_loops': 2.234148, 'mean_halt_entropy': 1.351465} vs_start={'helped': 5, 'hurt': 6, 'tied': 117, 'prediction_changes': 13} vs_base={'helped': 5, 'hurt': 7, 'tied': 116, 'prediction_changes': 21}
- outputs/stage5/stage5_phase1_distill_continue_250_arc128_20260621_164533/phase1/phase1_step_250.pt: arc={'mean': {'correct': 69, 'total': 128, 'accuracy': 0.5390625}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.211254, 'halting_kl': 0.092928, 'loss': 2.218688, 'mean_expected_loops': 2.088575, 'mean_halt_entropy': 1.308049} vs_start={'helped': 8, 'hurt': 10, 'tied': 110, 'prediction_changes': 23} vs_base={'helped': 3, 'hurt': 6, 'tied': 119, 'prediction_changes': 16}

## Decision Rule
If the best checkpoint improves over Phase1 start, keep extending deterministic recovery. If it regresses while validation improves, add base-logit distillation before more Opus training.

Note: this is not ARC-AGI. It is the competence recovery proxy before building the ARC-AGI-specific solver/eval loop.
