# Stage 5 Model Viability Probe - stage5_model_viability_qwen_3b_20260623_020135

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Model label: `qwen_3b`
- Layer split: `auto`
- Identity passed: `True`
- Identity max abs diff: `0.0`
- Loops: `[1, 2]`
- Score targets: `['label', 'content_question_only']`

## Loop Sweep

### arc_easy
- score target `label`
  - loop `1` aggregate `mean`: recurrent `27/32`, base `28/32`, delta `-1`, W/L/T `0/1/31`, p `1.0`, mean loops `1.0`
  - loop `2` aggregate `mean`: recurrent `27/32`, base `28/32`, delta `-1`, W/L/T `0/1/31`, p `1.0`, mean loops `1.750552773475647`
- score target `content_question_only`
  - loop `1` aggregate `mean`: recurrent `26/32`, base `26/32`, delta `0`, W/L/T `0/0/32`, p `None`, mean loops `1.0`
  - loop `2` aggregate `mean`: recurrent `25/32`, base `26/32`, delta `-1`, W/L/T `0/1/31`, p `1.0`, mean loops `1.750552773475647`

### arc_challenge
- score target `label`
  - loop `1` aggregate `mean`: recurrent `27/32`, base `27/32`, delta `0`, W/L/T `0/0/32`, p `None`, mean loops `1.0`
  - loop `2` aggregate `mean`: recurrent `24/32`, base `27/32`, delta `-3`, W/L/T `3/6/23`, p `0.5078125`, mean loops `1.750552773475647`
- score target `content_question_only`
  - loop `1` aggregate `mean`: recurrent `16/32`, base `16/32`, delta `0`, W/L/T `0/0/32`, p `None`, mean loops `1.0`
  - loop `2` aggregate `mean`: recurrent `9/32`, base `16/32`, delta `-7`, W/L/T `1/8/23`, p `0.0390625`, mean loops `1.750552773475647`
