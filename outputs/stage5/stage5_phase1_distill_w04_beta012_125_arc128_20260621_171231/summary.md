# Stage 5 Phase1 Recovery Ladder - stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231

## Question
Can continued deterministic recurrent training close or reverse the remaining gap to base Qwen before further particle/SVGD work?

## ARC-Challenge Proxy
- Base Qwen: `{'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}}`
- Phase1 start (outputs/stage4/stage4_opus_a100_20260620/phase1/phase1_step_500.pt): `{'mean': {'correct': 70, 'total': 128, 'accuracy': 0.546875}}`
- Best checkpoint: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt`
- Best checkpoint ARC: `{'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}}`
- Start lift: `0.015625`
- Gap to base: `0.0`

## Checkpoint Ladder
- outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt: arc={'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.389109, 'halting_kl': 0.244998, 'loss': 2.418509, 'mean_expected_loops': 2.561408, 'mean_halt_entropy': 1.368269} vs_start={'helped': 8, 'hurt': 6, 'tied': 114, 'prediction_changes': 15} vs_base={'helped': 13, 'hurt': 13, 'tied': 102, 'prediction_changes': 36}

## Decision Rule
If the best checkpoint improves over Phase1 start, keep extending deterministic recovery. If it regresses while validation improves, add base-logit distillation before more Opus training.

Note: this is not ARC-AGI. It is the competence recovery proxy before building the ARC-AGI-specific solver/eval loop.
