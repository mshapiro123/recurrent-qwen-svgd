# Stage 5 Balanced MCQ Checkpoint Assessment - stage5_balanced_mcq_20260621_160637

- Status: `needs_competence_recovery`
- Passed: `False`
- ARC-Easy sweep: `outputs\stage5\stage5_arceasy_sweep_full_20260621_185841\summary.json`
- ARC-Challenge summaries: `['outputs\\stage5\\stage5_phase1_step150_arcchallenge_full_20260621_194028\\summary.json', 'outputs\\stage5\\stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143\\summary.json']`
- Required benchmarks: `['arc_easy', 'arc_challenge']`
- Next step: Use the selected checkpoint as the current balanced baseline, then train with a competence-preserving mixed objective before returning to particles/SVGD.

## Ranked Checkpoints

| rank | label | full coverage | macro delta | micro delta | recurrent/base | W/L/T | checkpoint |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | `step_150` | `True` | -0.004550 | -7 | 581/588 | 40/47/782 | `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_150.pt` |
| 2 | `step_200` | `True` | -0.005510 | -9 | 579/588 | 37/46/786 | `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_200.pt` |
| 3 | `parent` | `True` | -0.014994 | -18 | 570/588 | 16/35/519 | `outputs/stage5/stage5_phase1_distill_w04_beta012_125_arc128_20260621_171231/phase1/phase1_step_125.pt` |
| 4 | `step_250` | `False` | -0.022807 | -13 | 408/421 | 15/28/527 | `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_250.pt` |
| 5 | `step_100` | `False` | -0.024561 | -14 | 407/421 | 15/29/526 | `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_100.pt` |
| 6 | `step_050` | `False` | -0.029825 | -17 | 404/421 | 16/33/521 | `outputs/stage5/stage5_phase1_recovered_extend_w04_beta012_lr5e6_250_20260621_181143/phase1/phase1_step_50.pt` |

## Benchmark Details

### `step_150`

- `arc_easy`: recurrent `412/570`, base `421/570`, delta `-9`, accuracy delta `-0.015789`, W/L/T `16/25/529`, p `0.21102359760880063`
- `arc_challenge`: recurrent `169/299`, base `167/299`, delta `2`, accuracy delta `0.006689`, W/L/T `24/22/253`, p `0.8829959121223965`

### `step_200`

- `arc_easy`: recurrent `409/570`, base `421/570`, delta `-12`, accuracy delta `-0.021053`, W/L/T `15/27/528`, p `0.08842954698775429`
- `arc_challenge`: recurrent `170/299`, base `167/299`, delta `3`, accuracy delta `0.010033`, W/L/T `22/19/258`, p `0.755228657504631`

### `parent`

- `arc_easy`: recurrent `402/570`, base `421/570`, delta `-19`, accuracy delta `-0.033333`, W/L/T `16/35/519`, p `0.010973562899720513`
- `arc_challenge`: recurrent `168/299`, base `167/299`, delta `1`, accuracy delta `0.003344`, W/L/T `None/None/None`, p `None`

### `step_250`

- `arc_easy`: recurrent `408/570`, base `421/570`, delta `-13`, accuracy delta `-0.022807`, W/L/T `15/28/527`, p `0.0659940344557981`
- `arc_challenge`: missing

### `step_100`

- `arc_easy`: recurrent `407/570`, base `421/570`, delta `-14`, accuracy delta `-0.024561`, W/L/T `15/29/526`, p `0.048766765904474596`
- `arc_challenge`: missing

### `step_050`

- `arc_easy`: recurrent `404/570`, base `421/570`, delta `-17`, accuracy delta `-0.029825`, W/L/T `16/33/521`, p `0.021294114141387155`
- `arc_challenge`: missing

