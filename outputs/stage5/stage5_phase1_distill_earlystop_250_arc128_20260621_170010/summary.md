# Stage 5 Phase1 Recovery Ladder - stage5_phase1_distill_earlystop_250_arc128_20260621_170010

## Question
Can continued deterministic recurrent training close or reverse the remaining gap to base Qwen before further particle/SVGD work?

## ARC-Challenge Proxy
- Base Qwen: `{'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}}`
- Phase1 start (outputs/stage4/stage4_opus_a100_20260620/phase1/phase1_step_500.pt): `{'mean': {'correct': 70, 'total': 128, 'accuracy': 0.546875}}`
- Best checkpoint: `outputs/stage5/stage5_phase1_distill_earlystop_250_arc128_20260621_170010/phase1/phase1_step_125.pt`
- Best checkpoint ARC: `{'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}}`
- Start lift: `0.0078125`
- Gap to base: `-0.0078125`

## Checkpoint Ladder
- outputs/stage5/stage5_phase1_distill_earlystop_250_arc128_20260621_170010/phase1/phase1_step_125.pt: arc={'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.376142, 'halting_kl': 0.240009, 'loss': 2.395343, 'mean_expected_loops': 2.553754, 'mean_halt_entropy': 1.369442} vs_start={'helped': 8, 'hurt': 7, 'tied': 113, 'prediction_changes': 17} vs_base={'helped': 13, 'hurt': 14, 'tied': 101, 'prediction_changes': 37}
- outputs/stage5/stage5_phase1_distill_earlystop_250_arc128_20260621_170010/phase1/phase1_step_250.pt: arc={'mean': {'correct': 70, 'total': 128, 'accuracy': 0.546875}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.304367, 'halting_kl': 0.15717, 'loss': 2.316941, 'mean_expected_loops': 2.37862, 'mean_halt_entropy': 1.372817} vs_start={'helped': 11, 'hurt': 11, 'tied': 106, 'prediction_changes': 27} vs_base={'helped': 9, 'hurt': 11, 'tied': 108, 'prediction_changes': 31}

## Decision Rule
If the best checkpoint improves over Phase1 start, keep extending deterministic recovery. If it regresses while validation improves, add base-logit distillation before more Opus training.

Note: this is not ARC-AGI. It is the competence recovery proxy before building the ARC-AGI-specific solver/eval loop.
