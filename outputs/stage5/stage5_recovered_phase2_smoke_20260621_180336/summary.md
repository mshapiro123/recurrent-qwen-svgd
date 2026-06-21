# Recovered Phase2 Smoke - stage5_recovered_phase2_smoke_20260621_180336

- Parent checkpoint: `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt`
- Phase2 checkpoint: `outputs/stage5/stage5_recovered_phase2_smoke_20260621_180336/phase2/phase2_step_50.pt`
- ARC limit: `128`
- K: `4`
- Repulsion: `2`
- Distillation: `True` weight `0.4` target `trajectories`

## ARC-Challenge
- `max`: recurrent `66/128` vs base `72/128`, delta `-6`, W/L/T `5/11/112`, p `0.210113525390625`
- `mean`: recurrent `69/128` vs base `72/128`, delta `-3`, W/L/T `6/9/113`, p `0.60723876953125`
- `vote`: recurrent `68/128` vs base `72/128`, delta `-4`, W/L/T `6/10/112`, p `0.454498291015625`
