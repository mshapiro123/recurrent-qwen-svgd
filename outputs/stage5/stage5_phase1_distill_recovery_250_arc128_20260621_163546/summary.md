# Stage 5 Phase1 Recovery Ladder - stage5_phase1_distill_recovery_250_arc128_20260621_163546

## Question
Can continued deterministic recurrent training close or reverse the remaining gap to base Qwen before further particle/SVGD work?

## ARC-Challenge Proxy
- Base Qwen: `{'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}}`
- Phase1 start (outputs/stage4/stage4_opus_a100_20260620/phase1/phase1_step_500.pt): `{'mean': {'correct': 70, 'total': 128, 'accuracy': 0.546875}}`
- Best checkpoint: `outputs/stage5/stage5_phase1_distill_recovery_250_arc128_20260621_163546/phase1/phase1_step_250.pt`
- Best checkpoint ARC: `{'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}}`
- Start lift: `0.0078125`
- Gap to base: `-0.0078125`

## Checkpoint Ladder
- outputs/stage5/stage5_phase1_distill_recovery_250_arc128_20260621_163546/phase1/phase1_step_250.pt: arc={'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.306299, 'halting_kl': 0.158244, 'loss': 2.318958, 'mean_expected_loops': 2.381554, 'mean_halt_entropy': 1.372872} vs_start={'helped': 12, 'hurt': 11, 'tied': 105, 'prediction_changes': 31} vs_base={'helped': 10, 'hurt': 11, 'tied': 107, 'prediction_changes': 30}

## Decision Rule
If the best checkpoint improves over Phase1 start, keep extending deterministic recovery. If it regresses while validation improves, add base-logit distillation before more Opus training.

Note: this is not ARC-AGI. It is the competence recovery proxy before building the ARC-AGI-specific solver/eval loop.
