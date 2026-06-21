# Stage 5 Phase1 Recovery Ladder - stage5_phase1_recovery_500_arc128_20260621_161620

## Question
Can continued deterministic recurrent training close or reverse the remaining gap to base Qwen before further particle/SVGD work?

## ARC-Challenge Proxy
- Base Qwen: `{'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}}`
- Phase1 start (outputs/stage4/stage4_opus_a100_20260620/phase1/phase1_step_500.pt): `{'mean': {'correct': 70, 'total': 128, 'accuracy': 0.546875}}`
- Best checkpoint: `outputs/stage5/stage5_phase1_recovery_500_arc128_20260621_161620/phase1/phase1_step_250.pt`
- Best checkpoint ARC: `{'mean': {'correct': 69, 'total': 128, 'accuracy': 0.5390625}}`
- Start lift: `-0.0078125`
- Gap to base: `-0.0234375`

## Checkpoint Ladder
- outputs/stage5/stage5_phase1_recovery_500_arc128_20260621_161620/phase1/phase1_step_250.pt: arc={'mean': {'correct': 69, 'total': 128, 'accuracy': 0.5390625}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.298474, 'halting_kl': 0.15079, 'loss': 2.310537, 'mean_expected_loops': 2.364228, 'mean_halt_entropy': 1.371882} vs_start={'helped': 12, 'hurt': 13, 'tied': 103, 'prediction_changes': 33} vs_base={'helped': 9, 'hurt': 12, 'tied': 107, 'prediction_changes': 32}
- outputs/stage5/stage5_phase1_recovery_500_arc128_20260621_161620/phase1/phase1_step_500.pt: arc={'mean': {'correct': 69, 'total': 128, 'accuracy': 0.5390625}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.17377, 'halting_kl': 0.093583, 'loss': 2.181257, 'mean_expected_loops': 1.971476, 'mean_halt_entropy': 1.25644} vs_start={'helped': 18, 'hurt': 19, 'tied': 91, 'prediction_changes': 50} vs_base={'helped': 3, 'hurt': 6, 'tied': 119, 'prediction_changes': 16}

## Decision Rule
If the best checkpoint improves over Phase1 start, keep extending deterministic recovery. If it regresses while validation improves, add base-logit distillation before more Opus training.

Note: this is not ARC-AGI. It is the competence recovery proxy before building the ARC-AGI-specific solver/eval loop.
