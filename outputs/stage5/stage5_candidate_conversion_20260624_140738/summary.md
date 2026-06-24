# Stage 5 Candidate Conversion - stage5_candidate_conversion_20260624_140738

- Checkpoint: `outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt`
- JSONL: `outputs/stage5/stage5_candidate_conversion_20260624_140738/candidate_conversion/candidate_conversion.jsonl`

## Setting Summary

| noise | loops | best | candidates | mean unique | all q2 | correct q2 | wrong q2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 21/42 | 84/168 | 1.000 | 1.0 | 1.0 | 1.0 |
| 0 | 8 | 21/42 | 84/168 | 1.000 | 1.0 | 1.0 | 1.0 |
| 0.005 | 4 | 21/42 | 84/168 | 1.048 | 1.0285714297067552 | 1.0 | 1.0571428594135104 |
| 0.005 | 8 | 21/42 | 84/168 | 1.048 | 1.0285714297067552 | 1.0 | 1.0571428594135104 |
| 0.01 | 4 | 21/42 | 84/168 | 1.024 | 1.0142857148533775 | 1.0 | 1.0285714297067552 |
| 0.01 | 8 | 21/42 | 84/168 | 1.024 | 1.0142857148533775 | 1.0 | 1.0285714297067552 |
| 0.02 | 4 | 21/42 | 84/168 | 1.071 | 1.0428571445601327 | 1.0 | 1.0857142891202654 |
| 0.02 | 8 | 21/42 | 84/168 | 1.071 | 1.0428571445601327 | 1.0 | 1.0857142891202654 |
| 0.05 | 4 | 21/42 | 83/168 | 1.095 | 1.0571428594135104 | 1.0 | 1.0818181850693442 |
| 0.05 | 8 | 21/42 | 83/168 | 1.095 | 1.0571428594135104 | 1.0 | 1.0818181850693442 |

## Interpretation Prompt

Use this to decide whether particle breadth is correct-bearing. If candidate hits rise with noise and correct q2 rises, selector work is justified. If uniqueness/q2 rise while candidate hits fall, noise is fragmentation and the next move is training-time pathway shaping, not stronger inference noise.
