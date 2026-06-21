# Stage 5 Phase1 Recovery Ladder - stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143

## Question
Can continued deterministic recurrent training close or reverse the remaining gap to base Qwen before further particle/SVGD work?

## ARC-Challenge Proxy
- Base Qwen: `{'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}}`
- Phase1 start (outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt): `{'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}}`
- Best checkpoint: `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_200.pt`
- Best checkpoint ARC: `{'mean': {'correct': 73, 'total': 128, 'accuracy': 0.5703125}}`
- Start lift: `0.0078125`
- Gap to base: `0.0078125`

## Checkpoint Ladder
- outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_50.pt: arc={'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.37199, 'halting_kl': 0.22353, 'loss': 2.398813, 'mean_expected_loops': 2.522481, 'mean_halt_entropy': 1.37204} vs_start={'helped': 2, 'hurt': 2, 'tied': 124, 'prediction_changes': 6} vs_base={'helped': 11, 'hurt': 11, 'tied': 106, 'prediction_changes': 32}
- outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_100.pt: arc={'mean': {'correct': 69, 'total': 128, 'accuracy': 0.5390625}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.355478, 'halting_kl': 0.200935, 'loss': 2.37959, 'mean_expected_loops': 2.478272, 'mean_halt_entropy': 1.374481} vs_start={'helped': 1, 'hurt': 4, 'tied': 123, 'prediction_changes': 8} vs_base={'helped': 9, 'hurt': 12, 'tied': 107, 'prediction_changes': 30}
- outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_150.pt: arc={'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.339362, 'halting_kl': 0.180615, 'loss': 2.361036, 'mean_expected_loops': 2.434324, 'mean_halt_entropy': 1.374814} vs_start={'helped': 2, 'hurt': 3, 'tied': 123, 'prediction_changes': 11} vs_base={'helped': 10, 'hurt': 11, 'tied': 107, 'prediction_changes': 32}
- outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_200.pt: arc={'mean': {'correct': 73, 'total': 128, 'accuracy': 0.5703125}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.319192, 'halting_kl': 0.161216, 'loss': 2.338538, 'mean_expected_loops': 2.387442, 'mean_halt_entropy': 1.373019} vs_start={'helped': 5, 'hurt': 4, 'tied': 119, 'prediction_changes': 15} vs_base={'helped': 10, 'hurt': 9, 'tied': 109, 'prediction_changes': 29}
- outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_250.pt: arc={'mean': {'correct': 71, 'total': 128, 'accuracy': 0.5546875}} val={'lora_recurrent_modules': 84.0, 'examples': 208.0, 'expected_ce': 2.303063, 'halting_kl': 0.144602, 'loss': 2.320416, 'mean_expected_loops': 2.341969, 'mean_halt_entropy': 1.369176} vs_start={'helped': 5, 'hurt': 6, 'tied': 117, 'prediction_changes': 19} vs_base={'helped': 7, 'hurt': 8, 'tied': 113, 'prediction_changes': 25}

## Full ARC-Challenge Final
{
  "base": {
    "mean": {
      "correct": 167,
      "total": 299,
      "accuracy": 0.5585284280936454
    }
  },
  "phase1_start": {
    "mean": {
      "correct": 168,
      "total": 299,
      "accuracy": 0.5618729096989966
    }
  },
  "phase1_best": {
    "mean": {
      "correct": 170,
      "total": 299,
      "accuracy": 0.568561872909699
    }
  },
  "best_checkpoint": "outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_200.pt",
  "best_vs_start": {
    "helped": 11,
    "hurt": 9,
    "tied": 279,
    "prediction_changes": 32
  },
  "best_vs_base": {
    "helped": 22,
    "hurt": 19,
    "tied": 258,
    "prediction_changes": 63
  }
}

## Decision Rule
If the best checkpoint improves over Phase1 start, keep extending deterministic recovery. If it regresses while validation improves, add base-logit distillation before more Opus training.

Note: this is not ARC-AGI. It is the competence recovery proxy before building the ARC-AGI-specific solver/eval loop.
