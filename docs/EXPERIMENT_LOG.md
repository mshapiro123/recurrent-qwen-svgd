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

## 2026-06-19: Projected Hidden-Kernel Diagnostics

Task set:

- `eval/smoke_exact_tasks_v2.jsonl`
- Seeds: `0,1,2`
- Projection seed: `123`
- Shared settings: `K=4`, `particle_init_noise=0.05`, `particle_noise_steps=16`, `temperature=0.0`

Projection dimension sweep at `repulsion=1.0`:

| Kernel projection | Best hits | Candidate hits | Repulsion/drift | Pairwise distance | Trajectory diversity | Clip fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw hidden, `0` | `26/42` | `86/168` | `0.005737` | `2.02359` | `0.00797702` | `0.0` |
| projected, `32` | `26/42` | `87/168` | `0.015364` | `3.22081` | `0.00820761` | `0.065191` |
| projected, `64` | `23/42` | `78/168` | `0.013304` | `2.99058` | `0.00856586` | `0.0570577` |

Interpretation:

- A 32D random projection strengthens repulsion and slightly improves density.
- 64D hurts quality despite stronger repulsion than raw hidden space.
- Projection helps with force magnitude, but random projection alone is not a semantic breakthrough.

32D projection, repulsion scale sweep:

| Repulsion scale | Best hits | Candidate hits | Notes |
| --- | ---: | ---: | --- |
| `0.5` | `25/42` | `93/168` | Best density so far; lower oracle than raw baseline. |
| `1.0` | `26/42` | `87/168` | Slight density gain over raw. |
| `2.0` | `25/42` | `91/168` | Higher density, solves train speed strongly, loses some arithmetic. |
| `4.0` | `27/42` | `90/168` | Best oracle so far; high density; solves some letter-count cases but hurts arithmetic add/multiply. |

Interpretation:

- Projected 32D kernels now clearly beat raw hidden-space density after scale tuning.
- `repulsion=0.5` is the best density setting.
- `repulsion=4.0` is the best oracle setting and may be the best selector-facing setting if majority/verifier can handle the changed task profile.
- The effect is task-redistributive, not uniformly better: higher repulsion helps train speed and letter count but can damage arithmetic add/multiply.
- Next step is a robustness sweep over seeds `0..4` for `proj_dim=32`, `repulsion=0.5` and `4.0`, with raw hidden `repulsion=1.0` retained as the baseline.

## 2026-06-23: ARC-Mix Content-Surface Recovery Confirmation

Checkpoint:

```text
outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/
  arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt
```

Confirmation run:

```text
outputs/stage5/stage5_content_arcmix_qonly_optiontext_arc256_check_20260623_123424/summary.json
```

Results on the 256-example confirmation slice:

| Benchmark | Scoring | Base | Recurrent | Delta |
| --- | --- | ---: | ---: | ---: |
| ARC-Easy | cyclic option permutation | `202/256` | `204/256` | `+2` |
| ARC-Easy | content question-only | `146/256` | `155/256` | `+9` |
| ARC-Challenge | cyclic option permutation | `154/256` | `154/256` | `0` |
| ARC-Challenge | content question-only | `87/256` | `97/256` | `+10` |

Interpretation:

- This is the first bounded non-toy recurrent-vs-base win after the model
  surgery.
- The result is still bounded; paired sign-test p-values are positive but not
  decisive.
- The next GPU-worthy action is an offset-256 confirmation with the same
  checkpoint and scoring surfaces before further training or Phase 2/SVGD
  scaling.

## 2026-06-24: Stage 5 Control Ledger

This section is the human-readable actions, decisions, and issues ledger for
the current Stage 5 work. The repo also keeps machine-readable summaries under
`outputs/stage5/**/summary.json`, current-pointer files under `config/`, and
Colab/Drive backups for selected runs.

### Current GPU Action

- Target: `traced_sft_score_alignment_repair`.
- Runtime class: L4 is expected to be sufficient; A100/G4 is not required for
  this bounded 0.5B score-level repair unless the run OOMs.
- Bootstrap source: `colab/CURRENT_A100_BOOTSTRAP_CELL.py` from GitHub `main`.
- Launcher: `colab/STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL.py`.
- Runner: `colab/run_stage5_surface_alignment_repair.py`.
- Source summary:
  `outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm/summary.json`.
- Objective: repair ARC-Easy content-route MCQ score behavior using direct
  option-score cross entropy, without repeating the failed SFT surface repair.
- Key training settings:
  - `STAGE5_SURFACE_ALIGN_TRAINER=score_ce`
  - `STAGE5_SURFACE_ALIGN_MAX_STEPS=75`
  - `STAGE5_SURFACE_ALIGN_LR=5e-7`
  - `STAGE5_SURFACE_ALIGN_DISTILL_WEIGHT=0.0`
  - `STAGE5_SURFACE_ALIGN_SCORE_DISTILL_WEIGHT=0.05`
  - `STAGE5_SURFACE_ALIGN_SCORE_MARGIN=0.05`
  - `STAGE5_SURFACE_ALIGN_SCORE_MARGIN_WEIGHT=0.1`
- Expected artifact if successful:
  `outputs/stage5/stage5_score_alignment_repair_content_route_20260624/summary.json`.

### Decisions

- Do not repeat the previous failed SFT-style content/cyclic surface repair. The
  planner now stops if a score-level repair also produces no easy content lift.
- Treat base preservation and route repair as necessary gates before scaling
  SVGD, depth routing, or larger model variants.
- Use L4/T4 for bounded 0.5B repair and diagnostic runs. Reserve A100/G4/H100
  for 3B/7B capability-ladder probes, longer SFT, or memory-heavy particle
  experiments.
- Continue developing the capability-ladder curriculum in parallel with GPU
  runs so the next depth-training pass is not blocked on bookkeeping.

### Current Issues

- The recurrent 0.5B checkpoint can beat base on some bounded ARC content
  surfaces, but it remains fragile across scoring routes.
- Surface mismatch is still the immediate blocker for honest broader benchmark
  claims.
- Capability-ladder data generation supports arbitrary model scales, but the
  quality gate needs explicit per-depth row requirements before we should launch
  paid depth-router SFT.
- Existing run summaries are authoritative but scattered; this log should be
  updated when decisions change, when a run lands, or when a target is retired.

### Next Checks

- Inspect the score-repair run summary when it lands.
- If score repair improves content accuracy without cyclic collapse, update the
  current source summary pointer and proceed to a held-out confirmation.
- If score repair fails, use the score-margin diagnostics to determine whether
  the repair changed option scores in the right direction but failed to flip
  enough predictions, or whether it drifted into another label/content prior.
- Before depth-router SFT, require enough positive SFT rows at each intended
  `target_loop_count`, especially for `1`, `2`, `3`, and `4` when using a
  0.5B/1.5B/3B/7B ladder.

### Local Verification Notes

- 2026-06-24: `traced_sft_score_alignment_repair` completed and pushed
  `outputs/stage5/stage5_score_alignment_repair_content_route_20260624/summary.json`.
  Status: `surface_alignment_not_passed`; assessment status:
  `needs_recurrent_recovery`.
- Score-level repair outcome:
  - ARC-Easy content: source recurrent `140/256`, repaired recurrent `139/256`,
    base `146/256`; repair delta versus source `-1`.
  - ARC-Easy cyclic: source recurrent `203/256`, repaired recurrent `204/256`,
    base `202/256`; repair delta versus source `+1`.
  - ARC-Challenge content: source recurrent `86/256`, repaired recurrent
    `91/256`, base `87/256`; repair delta versus source `+5`, repaired
    recurrent delta versus base `+4`.
  - ARC-Challenge cyclic: source recurrent `151/256`, repaired recurrent
    `153/256`, base `154/256`; repair delta versus source `+2`.
- Interpretation: score-level repair is not a dead end; it moved the harder
  ARC-Challenge content slice in the desired direction and did not collapse
  cyclic scoring. It did not fix the ARC-Easy content regression, so it fails
  the current all-slice preservation gate.
- Decision: do not repeat this same 75-step score repair unchanged. Next action
  should inspect score-margin/prediction-change diagnostics and either revise
  the repair objective/data mix or move to a held-out capability-ladder/depth
  curriculum experiment if the surface route is considered good enough for the
  harder-slice hypothesis.
- Follow-on decision: use the stronger ARC-mix recovered checkpoint for the
  next depth experiment rather than the weaker score-repair checkpoint. The
  next GPU target is `STAGE5_CURRENT_A100_TARGET=arc_mix_offset_then_depth_chain`;
  it confirms the ARC-mix checkpoint on offset-256 examples, then launches
  learned-depth ARC-mix SFT only if the offset gate passes.
- 2026-06-24: locally verified the maintained score-repair path compiles:
  `colab/STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL.py`,
  `colab/run_stage5_surface_alignment_repair.py`,
  `training/prepare_mcq_score_alignment_jsonl.py`,
  `training/train_phase1_mcq_score_align.py`, and
  `colab/assess_stage5_surface_repair.py`.
- 2026-06-24: locally reran focused score-repair tests:
  `tests/test_prepare_mcq_score_alignment_jsonl.py`,
  `tests/test_train_phase1_mcq_score_align.py`,
  `tests/test_stage5_surface_alignment_repair.py`,
  `tests/test_stage5_surface_repair_assessment.py`, and
  `tests/test_stage5_notebooks.py`.
- Score-repair runner behavior confirmed from source: on success it writes the
  run summary, backs up artifacts to Drive when mounted, commits/pushes with
  `[skip ci]`, and disconnects; on error the outer Colab cell leaves the runtime
  connected for inspection.
- 2026-06-24: Chrome inspection found the visible Colab notebook disconnected,
  showing stale `run_stage5_surface_alignment_repair.py` output rather than an
  active `arc_mix_offset_then_depth_chain` run. No GPU appeared to be burning at
  inspection time. The next Colab action remains the maintained
  `arc_mix_offset_then_depth_chain` bootstrap cell against
  `stage5_content_arcmix_qonly_optiontext_arc256_check_20260623_123424`.
- 2026-06-24: added a CPU-only offset/depth-chain reviewer:
  `colab/review_stage5_offset_depth_chain.py`. It classifies a completed chain
  summary into one of the next operational actions: stop on failed offset,
  inspect failed depth, run missing post-depth debiased gate, inspect post-depth
  warning, or run dense MCQ control when both mixed ARC rows and an upstream
  `positive_sft` source are visible.
- The reviewer deliberately separates two requirements for dense control:
  `data.mixed_train_jsonl` from the ARC-mix depth run and a compatible upstream
  `positive_sft` source summary. If mixed rows exist but no `positive_sft`
  source is discoverable, it blocks dense control rather than launching an
  expensive run that will fail inside `run_stage5_mcq_dense_sft_control.py`.
- Local verification for the reviewer:
  `.venv\Scripts\python.exe -m pytest -q tests\test_review_stage5_offset_depth_chain.py tests\test_stage5_offset_then_depth.py tests\test_stage5_balanced_arc_mix_gate.py tests\test_stage5_mcq_dense_sft_control.py tests\test_stage5_notebooks.py`
  -> `94 passed`.
- GitHub Actions budget note: reviewer and latest log changes are local-only
  unless/until pushed. The repo is intentionally ahead of `origin/main`; push in
  a batch with `[skip ci]` only when Colab needs the new reviewer or when a run
  artifact needs to be synchronized.
