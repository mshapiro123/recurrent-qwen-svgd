# Phase A Dense First-Response Reader Audit

- Status: `corrected_reader_required`
- Finding: The registered reader can overwrite a completed response with later untrained continuation. Dense accuracy and figures must use the corrected first-response readout.

| Arm | Registered | Corrected | Delta | D1 | D2 | D4 |
|---|---:|---:|---:|---:|---:|---:|
| B_step4000 | 470 | 496 | +26 | 128/128 | 95/128 | 29/128 |
| C_step4000 | 952 | 1292 | +340 | 128/128 | 128/128 | 128/128 |
| D_step4000 | 322 | 656 | +334 | 125/128 | 107/128 | 75/128 |

## Paper-One Marker Closure

- B tail, depths 11-14: `60/512`.
- D tail, depths 11-14: `58/512`.
- A-versus-B primary gate: `True` over consecutive depths `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`.
- Depth 1 is a corrected tie (`128/128`); do not retain the old all-14-depth wording.

The correction is evaluation-only. It does not alter checkpoints, frozen rows, or model outputs.
