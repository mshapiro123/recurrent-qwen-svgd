# Experiment Log

## 2026-06-18: Phase 2 SVGD Smoke Controls

Model:

- Base: `Qwen/Qwen2.5-0.5B-Instruct`
- Phase 1 checkpoint: `outputs/qwen_0_5b_phase1_a100_beta008_continue_150/phase1_step_150.pt`
- Phase 2 checkpoint: `outputs/qwen_0_5b_phase2_svgd_smoke25/phase2_step_25.pt`
- Eval task set: `eval/smoke_exact_tasks.jsonl`
- Trajectories: `K=4`
- Max generated tokens: `140`

Known Phase 1 baseline:

- Phase 1 K=1: about `1/5` exact-task hits on this smoke suite.

Temperature-only control:

- `particle_update_mode=none`
- `temperature=0.7`
- Mean oracle best-of-K: `2.6/5`
- Mean candidate hits: `4.0/20`
- Interpretation: plain sampling can raise oracle score, but most sampled candidates are low-quality.

SVGD drift/noise control:

- `particle_update_mode=svgd`
- `particle_init_noise=0.05`
- `particle_noise_steps=16`
- `svgd_eps=1.0`
- `svgd_repulsion_max_norm=1.0`
- `svgd_repulsion_scale=0.0`
- Mean oracle best-of-K: `1.8/5`
- Mean candidate hits: `6.8/20`
- Interpretation: learned recurrent drift plus repeated particle noise improves candidate density relative to temperature-only, but loses oracle breadth.

SVGD repulsion result:

- Same as drift/noise control, except `svgd_repulsion_scale=1.0`
- Mean oracle best-of-K: `2.4/5`
- Mean candidate hits: `9.6/20`
- Interpretation: kernel repulsion improves both oracle and candidate-density metrics relative to the no-repulsion control. This is the first clean positive signal that the SVGD-style update is doing useful work beyond stochastic decoding/noise.

SVGD repulsion scale sweep:

| Repulsion scale | Mean oracle best-of-K | Mean candidate hits |
| --- | ---: | ---: |
| `0.25` | `1.8/5` | `6.8/20` |
| `0.5` | `1.8/5` | `7.2/20` |
| `1.0` | `2.4/5` | `9.6/20` |
| `1.5` | `2.2/5` | `8.6/20` |
| `2.0` | `2.2/5` | `8.4/20` |

Interpretation: `svgd_repulsion_scale=1.0` is the current smoke-suite winner. Larger values do not improve oracle score and reduce candidate density.

Current best practical setting:

```text
phase2_particle_update_mode=svgd
particle_init_noise=0.05
particle_noise_every_step=true
particle_noise_steps=16
svgd_eps=1.0
svgd_repulsion_scale=1.0
svgd_repulsion_max_norm=1.0
temperature=0.0
phase2_num_trajectories=4
```

Open cautions:

- This is still a tiny five-task exact-pattern smoke suite, not a general benchmark.
- Pharmacy and anagram tasks remain systematic failures and should not drive hyperparameter tuning alone.
- Temperature-only has higher oracle than drift/noise, but much lower candidate density.
- The current SVGD implementation uses learned recurrent drift plus kernel repulsion, not a verifier/log-probability posterior gradient.

## 2026-06-18: Broader Exact Smoke Suite v2

Task set:

- `eval/smoke_exact_tasks_v2.jsonl`
- 14 exact-pattern tasks
- Seeds: `0,1,2`
- Trajectories: `K=4`
- Max generated tokens: `140`

Results:

| Method | Mean oracle best-of-K | Mean candidate hits |
| --- | ---: | ---: |
| Temperature-only, `temp=0.7` | `7.333/14` | `18.667/56` |
| SVGD drift/noise, `repulsion=0` | `8.333/14` | `26.667/56` |
| SVGD drift/noise/repulsion, `repulsion=1` | `8.667/14` | `28.667/56` |

Interpretation:

- The SVGD-style recurrent particle path generalizes beyond the original five-task smoke suite.
- Candidate density is the clearest gain: `repulsion=1` improves from `18.667/56` to `28.667/56` versus temperature-only.
- Repulsion still helps beyond drift/noise alone, but the v2 delta is modest: `26.667/56` to `28.667/56`.
- Temperature-only remains broad but sparse; SVGD is less diverse textually, but yields a substantially higher fraction of usable candidates.

Persistent failures:

- `pharmacy tubs`, `rates word`, `n queens small`, and often `letter count` remain weak or unsolved.
- These look like model/checkpoint capability or prompt pathology issues, not just trajectory-diversity issues.

Structured diagnostics:

| Setting | Best hits | Candidate hits | Drift RMS | Repulsion RMS | Clip fraction | Pairwise distance | Trajectory diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `repulsion=0` | `25/42` | `80/168` | `0.570344` | `0.0056865` | `0.0045773` | `1.77413` | `0.00756361` |
| `repulsion=1` | `26/42` | `86/168` | `0.569031` | `0.00326463` | `0.0` | `2.02359` | `0.00797702` |

Interpretation:

- The repulsion path is not being limited by the `max_norm=1.0` clip; clip fraction is effectively zero.
- The repulsion vector is tiny relative to recurrent drift, roughly `0.6%` of drift RMS for `repulsion=1`.
- Despite the small RMS, `repulsion=1` increases pairwise distance, trajectory diversity, best hits, and candidate hits.
- This suggests the hidden-space kernel has a real but weak steering effect. More scale alone is unlikely to be the answer because the previous repulsion-scale sweep already degraded beyond `1.0`.
- The next highest-value diagnostic is a low-dimensional projected kernel, not further tuning of raw hidden-space repulsion scale.
