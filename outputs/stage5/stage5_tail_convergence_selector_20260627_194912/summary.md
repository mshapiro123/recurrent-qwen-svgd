# Stage 5 Tail-Convergence Selector Diagnostic

- Run: `stage5_tail_convergence_selector_20260627_194912`
- Discovery sweep: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260625_194738/summary.json`
- Held-out sweep: `outputs/stage5/stage5_heldout_router_validation_20260625_230408/summary.json`
- Checkpoint: `outputs/stage5/stage5_depth_signal_recovery_20260625_180014_curriculum_sft/phase1/phase1_step_100.pt`
- Tail basis: drop top `1`, keep `7`

## Discovery

- Tail probe clears permutation null p95: `False`
- Observed/null p95 alignment: `0.804239229711722` / `0.8613766904382041`

| feature | oriented AUC | pos mean | neg mean | best zero-harm delta | best max-net delta |
|---|---:|---:|---:|---:|---:|
| `tail_rel_disp_12` | 0.5324248120300752 | 0.5361761258521879 | 0.5383089489057948 | None | 3 |
| `tail_rel_disp_23` | 0.5513784461152882 | 0.25651793900758885 | 0.2539145745570351 | None | 3 |
| `tail_cos_12` | 0.5587406015037594 | 0.9981500121402929 | 0.9978451329614558 | None | 2 |
| `tail_cos_23` | 0.5869360902255639 | 0.9993863782723104 | 0.9992096373972007 | 0 | 3 |
| `tail_deceleration_12_minus_23` | 0.5134711779448622 | 0.279658186844599 | 0.2843943743487597 | None | 3 |
| `tail_disp_ratio_23_over_12` | 0.5222431077694235 | 0.48226178478616843 | 0.47701109441346284 | None | 2 |

## Held-Out Transfer

### arc_easy

| feature | zero-harm delta | max-net delta | max-net W/L |
|---|---:|---:|---:|
| `tail_rel_disp_12` | None | -1 | 9/10 |
| `tail_rel_disp_23` | 0 | 0 | 10/10 |
| `tail_cos_12` | None | 0 | 10/10 |
| `tail_cos_23` | 0 | 0 | 10/10 |
| `tail_deceleration_12_minus_23` | None | -1 | 9/10 |
| `tail_disp_ratio_23_over_12` | 3 | 4 | 10/6 |

### arc_challenge

| feature | zero-harm delta | max-net delta | max-net W/L |
|---|---:|---:|---:|
| `tail_rel_disp_12` | 0 | 0 | 0/0 |
| `tail_rel_disp_23` | 0 | 0 | 2/2 |
| `tail_cos_12` | None | -1 | 0/1 |
| `tail_cos_23` | 0 | 0 | 0/0 |
| `tail_deceleration_12_minus_23` | 0 | 0 | 0/0 |
| `tail_disp_ratio_23_over_12` | 2 | 2 | 3/1 |

### open_hard_arc_challenge

| feature | zero-harm delta | max-net delta | max-net W/L |
|---|---:|---:|---:|
| `tail_rel_disp_12` | 1 | 1 | 2/1 |
| `tail_rel_disp_23` | 2 | 4 | 7/3 |
| `tail_cos_12` | None | 1 | 3/2 |
| `tail_cos_23` | 0 | 2 | 4/2 |
| `tail_deceleration_12_minus_23` | 1 | 1 | 2/1 |
| `tail_disp_ratio_23_over_12` | None | 3 | 9/6 |
