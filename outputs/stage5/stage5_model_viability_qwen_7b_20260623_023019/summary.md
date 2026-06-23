# Stage 5 Model Viability Probe - stage5_model_viability_qwen_7b_20260623_023019

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Model label: `qwen_7b`
- Layer split: `auto`
- Identity passed: `True`
- Identity max abs diff: `0.0`
- Loops: `[1, 2]`
- Score targets: `['label', 'content_question_only']`

## Loop Sweep

### arc_easy
- score target `label`
  - loop `1` aggregate `mean`: recurrent `23/24`, base `23/24`, delta `0`, W/L/T `0/0/24`, p `None`, mean loops `1.0`
  - loop `2` aggregate `mean`: recurrent `23/24`, base `23/24`, delta `0`, W/L/T `0/0/24`, p `None`, mean loops `1.750552773475647`
- score target `content_question_only`
  - loop `1` aggregate `mean`: recurrent `21/24`, base `21/24`, delta `0`, W/L/T `0/0/24`, p `None`, mean loops `1.0`
  - loop `2` aggregate `mean`: recurrent `20/24`, base `21/24`, delta `-1`, W/L/T `0/1/23`, p `1.0`, mean loops `1.750552773475647`

### arc_challenge
- score target `label`
  - loop `1` aggregate `mean`: recurrent `24/24`, base `24/24`, delta `0`, W/L/T `0/0/24`, p `None`, mean loops `1.0`
  - loop `2` aggregate `mean`: recurrent `22/24`, base `24/24`, delta `-2`, W/L/T `0/2/22`, p `0.5`, mean loops `1.750552773475647`
- score target `content_question_only`
  - loop `1` aggregate `mean`: recurrent `13/24`, base `13/24`, delta `0`, W/L/T `0/0/24`, p `None`, mean loops `1.0`
  - loop `2` aggregate `mean`: recurrent `14/24`, base `13/24`, delta `1`, W/L/T `1/0/23`, p `1.0`, mean loops `1.750552773475647`
