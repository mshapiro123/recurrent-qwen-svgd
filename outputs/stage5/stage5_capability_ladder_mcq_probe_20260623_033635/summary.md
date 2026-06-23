# Capability-Ladder MCQ Probe: stage5_capability_ladder_mcq_probe_20260623_033635

- Status: `capability_ladder_probe_gate_ready`
- Score mode: `content_question_only`
- ARC: `ARC-Challenge` `train` limit `96`
- Typed records: `73`
- Positive SFT rows: `73`
- Mode counts: `{'deep_narrow': 40, 'direct': 33}`
- Target loop counts: `{'1': 33, '2': 22, '3': 10, '4': 8}`

## Model Scores
- `qwen_0_5b`: `33/96` accuracy `0.3438`
- `qwen_1_5b`: `46/96` accuracy `0.4792`
- `qwen_3b`: `55/96` accuracy `0.5729`
- `qwen_7b`: `60/96` accuracy `0.6250`

## Caveat

This probe uses answer-only MCQ predictions as minimal traces. Use it to select/size a depth curriculum, then replace or enrich rows with verified reasoning traces before claiming reasoning SFT quality.
