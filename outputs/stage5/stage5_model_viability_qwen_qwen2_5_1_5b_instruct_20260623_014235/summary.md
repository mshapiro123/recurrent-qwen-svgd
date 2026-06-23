# Stage 5 Model Viability Probe - stage5_model_viability_qwen_qwen2_5_1_5b_instruct_20260623_014235

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Model label: `qwen_1_5b`
- Layer split: `auto`
- Identity passed: `True`
- Identity max abs diff: `0.0`
- Loops: `[1, 2]`
- Score targets: `['label', 'content_question_only']`

## Loop Sweep

### arc_easy
- score target `label`
  - loop `1` aggregate `mean`: recurrent `27/32`, base `28/32`, delta `-1`, W/L/T `0/1/31`, p `1.0`, mean loops `1.0`
  - loop `2` aggregate `mean`: recurrent `28/32`, base `28/32`, delta `0`, W/L/T `0/0/32`, p `None`, mean loops `1.750552773475647`
- score target `content_question_only`
  - loop `1` aggregate `mean`: recurrent `25/32`, base `25/32`, delta `0`, W/L/T `0/0/32`, p `None`, mean loops `1.0`
  - loop `2` aggregate `mean`: recurrent `25/32`, base `25/32`, delta `0`, W/L/T `1/1/30`, p `1.0`, mean loops `1.750552773475647`

### arc_challenge
- score target `label`
  - loop `1` aggregate `mean`: recurrent `27/32`, base `26/32`, delta `1`, W/L/T `1/0/31`, p `1.0`, mean loops `1.0`
  - loop `2` aggregate `mean`: recurrent `25/32`, base `26/32`, delta `-1`, W/L/T `1/2/29`, p `1.0`, mean loops `1.750552773475647`
- score target `content_question_only`
  - loop `1` aggregate `mean`: recurrent `17/32`, base `17/32`, delta `0`, W/L/T `0/0/32`, p `None`, mean loops `1.0`
  - loop `2` aggregate `mean`: recurrent `17/32`, base `17/32`, delta `0`, W/L/T `1/1/30`, p `1.0`, mean loops `1.750552773475647`
