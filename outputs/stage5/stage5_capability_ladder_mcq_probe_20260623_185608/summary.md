# Capability-Ladder MCQ Probe: stage5_capability_ladder_mcq_probe_20260623_185608

- Status: `capability_ladder_probe_gate_ready`
- Score mode: `content_question_only`
- ARC: `ARC-Challenge` `train` limit `96`
- Typed records: `66`
- Positive SFT rows: `66`
- Mode counts: `{'deep_narrow': 32, 'direct': 34}`
- Target loop counts: `{'1': 34, '2': 20, '3': 12}`

## Model Scores
- `qwen_0_5b`: `34/96` accuracy `0.3542`
- `qwen_1_5b`: `44/96` accuracy `0.4583`
- `qwen_3b`: `57/96` accuracy `0.5938`

## Caveat

This probe uses answer-only MCQ predictions as minimal traces. Use it to select/size a depth curriculum, then replace or enrich rows with verified reasoning traces before claiming reasoning SFT quality.
