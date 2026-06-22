# Stage 5 ARC-Mix Result Review - stage5_arc_mix_review_20260622_031727

- Source summary: `outputs/stage5/stage5_arc_mix_recovery_once_20260622_030628/summary.json`
- Source status: `no_proxy_lift`
- Source passed: `False`
- Decision: `stop_and_revise_objective`
- Decision basis: `explicit_decision`
- Blocked reason: Best ARC-mix checkpoint did not improve over the recurrent start or close the base gap.
- Next A100 spend: **NO: stop A100 work and repair locally.**

## Best Arm

- Arm: `arc_mix_response_w01_lr2e6`
- Checkpoint: `outputs/stage5/stage5_arc_mix_recovery_once_20260622_030628/arc_mix_response_w01_lr2e6/phase1/phase1_step_50.pt`
- Proxy score: `66/128`
- Start score: `68/128`
- Base score: `68/128`
- Lift vs start: `-2`
- Gap vs base: `-2`
- Mean margin delta vs base: `-0.308232`
- Max prediction-count shift: `10`
- Calibration OK: `False`

## Planner Action

- Name: `Inspect ARC-mix proxy gate no_proxy_lift`
- Priority: `10`
- Command: `cat outputs/stage5/stage5_arc_mix_recovery_once_20260622_030628/summary.md`
