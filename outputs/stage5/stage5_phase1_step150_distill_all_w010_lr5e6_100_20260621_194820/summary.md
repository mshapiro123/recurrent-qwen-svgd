# Stage 5 Phase1 Recovery Ladder - stage5_phase1_step150_distill_all_w010_lr5e6_100_20260621_194820

## Question
Can continued deterministic recurrent training close or reverse the remaining gap to base Qwen before further particle/SVGD work?

## ARC-Challenge Proxy
- Base Qwen: `{'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}}`
- Phase1 start (outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_150.pt): `{'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}}`
- Best checkpoint: `outputs/stage5/stage5_phase1_step150_distill_all_w010_lr5e6_100_20260621_194820/phase1/phase1_step_50.pt`
- Best checkpoint ARC: `{'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}}`
- Start lift: `0.0`
- Gap to base: `-0.0078125`

## Checkpoint Ladder
- outputs/stage5/stage5_phase1_step150_distill_all_w010_lr5e6_100_20260621_194820/phase1/phase1_step_50.pt: arc={'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.311049, 'halting_kl': 0.158239, 'loss': 2.330038, 'mean_expected_loops': 2.379905, 'mean_halt_entropy': 1.372618} vs_start={'helped': 2, 'hurt': 2, 'tied': 124, 'prediction_changes': 8} vs_base={'helped': 8, 'hurt': 9, 'tied': 111, 'prediction_changes': 27}
- outputs/stage5/stage5_phase1_step150_distill_all_w010_lr5e6_100_20260621_194820/phase1/phase1_step_100.pt: arc={'mean': {'correct': 70, 'total': 128, 'accuracy': 0.546875}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.290425, 'halting_kl': 0.140015, 'loss': 2.307227, 'mean_expected_loops': 2.32825, 'mean_halt_entropy': 1.367601} vs_start={'helped': 1, 'hurt': 2, 'tied': 125, 'prediction_changes': 7} vs_base={'helped': 8, 'hurt': 10, 'tied': 110, 'prediction_changes': 28}

## Decision Rule
If the best checkpoint improves over Phase1 start, keep extending deterministic recovery. If it regresses while validation improves, add base-logit distillation before more Opus training.

Note: this is not ARC-AGI. It is the competence recovery proxy before building the ARC-AGI-specific solver/eval loop.
