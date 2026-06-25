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

## 2026-06-24: Effective-Pathway Dynamics Gate

- Strategy update: before spending more GPU time on SVGD/kernel geometry, measure
  whether the deterministic recurrent map itself preserves multiple latent
  pathways for a fixed prompt. The concern is that particle collapse may be a
  single-attractor dynamical regime, not a bad repulsion scale.
- Added `eval/pathway_diversity.py`, a Leinster-Cobbold
  similarity-sensitive diversity implementation over q in `{0,1,2,inf}` with a
  nearest-neighbor local bandwidth. This is the standing "effective number of
  distinct pathways" diagnostic for breadth.
- Added `eval/eval_effective_pathways.py`, which initializes many particles by
  embedded-input noise, disables latent sampling and SVGD, runs the deterministic
  recurrent wrapper, and reports effective pathway counts, final/initial
  particle spread, a Lyapunov-style spread proxy, and next-token trajectory
  uniqueness.
- Updated the sequenced experiment plan and Stage 5 training recipe so particle
  tuning re-enters only after this gate distinguishes non-collapsed dynamics
  from single-attractor contraction.
- Local verification:
  `.venv\Scripts\python.exe -m pytest -q tests\test_pathway_diversity.py tests\test_eval_effective_pathways.py`
  -> `10 passed`.
- First Colab L4 diagnostic run:
  `stage5_effective_pathways_20260624_024236`, checkpoint
  `stage5_content_arcmix_qonly_optiontext_20260623_121707/arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt`.
  The GPU/eval work completed and artifacts were backed up to Drive, but the
  launcher failed after backup because `outputs/` is gitignored and it attempted
  a plain `git add`. The launcher has been patched to use `git add -f` for
  selected Stage 5 evidence artifacts.
- Loops=4 aggregate over 8 smoke prompts, K=16, noise=0.05:
  initial spread `0.4747`, final spread `45.8323`, spread ratio `97.31`,
  Lyapunov proxy per loop `1.1417`, mean unique next-token argmax `8.0`,
  effective pathways q0/q1/q2/qinf = `2.403/2.011/1.818/1.473`.
- Loops=8 aggregate: final spread `83.4485`, spread ratio `179.75`,
  Lyapunov proxy per loop `0.6462`, mean unique next-token argmax `7.875`,
  effective pathways q0/q1/q2/qinf = `2.428/1.942/1.728/1.421`.
- Interpretation: the recurrent map is not immediately single-attractor
  contractive; it preserves/amplifies perturbation-dependent pathway variation.
  However, the huge spread expansion plus many next-token argmaxes suggests an
  expansive or chaotic regime rather than clean multistable attractor basins.
  The next diagnostic should sweep lower particle-init noise and include a
  zero-noise control before further SVGD tuning.
- Follow-up noise sweep landed as
  `outputs/stage5/stage5_effective_pathways_noise_sweep_20260624/`.
  Zero-noise control is clean: final spread `0`, unique next-token argmax `1`,
  and effective pathway count `1` for loops 4 and 8.
- Nonzero perturbations are strongly amplified even at very low scale:
  - noise `0.005`: spread ratio `91.7` at 4 loops and `153.1` at 8 loops;
    q2 effective pathways `1.318` and `1.281`; unique argmax `1.38`.
  - noise `0.01`: spread ratio `95.1` and `151.4`; q2 `1.373` and `1.282`;
    unique argmax `2.88`.
  - noise `0.02`: spread ratio `116.9` and `191.6`; q2 `1.348` and `1.251`;
    unique argmax `4.88`.
  - noise `0.05`: spread ratio `97.3` and `179.8`; q2 `1.818` and `1.728`;
    unique argmax `8.0` and `7.875`.
- Updated interpretation: the current recurrent map is not collapsed, but the
  breadth mostly appears as local expansive sensitivity. The q2 effective count
  stays low and generally decreases with more loops, while output argmaxes
  fragment as noise rises. This is not yet evidence of useful multistable
  reasoning basins. The next paid diagnostic should ask whether any of this
  expansion converts into correct candidate coverage; if not, move to
  regime/pathway supervision rather than more SVGD kernel geometry.
- Strategy-agent interpretation accepted: the diagnostic closed the
  single-attractor-collapse branch but did not open the clean-multistability
  branch. Large spread expansion with low/noise-flat effective pathway count is
  best read as mono-unstable expansion: one or a few unstable directions amplify
  small perturbations and fragment the answer surface without creating stable
  basins. The next gate is therefore a correctness-split candidate-conversion
  sweep. Compute effective pathway counts separately for correct and wrong
  candidates. If diversity is correct-bearing, proceed to selector conversion;
  if diversity lives in wrong candidates, stop inference-time noise/SVGD tuning
  and move to spectral/Jacobian regime shaping plus method-anchored pathway
  supervision.
- Implemented the next GPU gate as `candidate_conversion_diagnostic`. The
  evaluator now supports particle-init-noise sweeps with SVGD disabled, max-loop
  sweeps in the same one-load run, and per-task correctness-split pathway
  diagnostics (`all`, `correct`, `wrong`). The Colab launcher runs the bounded
  L4/T4-friendly sweep over `noise={0,0.005,0.01,0.02,0.05}` and
  `loops={4,8}` with `K=4`, then writes a summary table that compares candidate
  conversion against pathway expansion.

## 2026-06-25: Re-entry Architecture Repair Gate

- Strategy update: stop spending GPU on particle noise/SVGD geometry until the
  deterministic recurrent loop-closure path is verified. The candidate
  conversion runs showed that perturbations can create output diversity, but not
  enough correct-bearing alternatives. That made the next blocker architectural:
  is the recurrent block being re-entered through a trainable, live translation
  path, or are we repeatedly feeding recurrent-block exits back as if they were
  entries?
- Added a staged re-entry reset:
  1. `reentry_drift_diagnostic`: read-only entry/exit and loop-drift
     measurement.
  2. `reentry_norm_diagnostic`: eval-only loop re-entry RMS rescale comparison.
  3. `reentry_repair_smoke`: tiny trainable bridge/re-entry repair smoke.
  4. `reentry_recovery_training`: gated recovery SFT after the repair smoke
     passes.
- Stage 1 landed as
  `outputs/stage5/stage5_reentry_drift_20260625_011444/`.
  Key findings on the current recovered recurrent checkpoint:
  - mean entry RMS `~11.867`, exit RMS `~11.892`, exit/entry RMS `~1.0024`;
  - pooled entry/exit cosine `~0.9757`;
  - entry/exit subspace overlap `~0.3703`, with only one principal dimension
    above cosine `0.8`;
  - loop-8 output/entry RMS `~1.0373`, so gross norm drift is bounded but
    subspace mismatch is real;
  - `bridge_gate=0.0`, bridge delta RMS `0.0`, and bridge projection/bias/gate
    gradients `0.0`.
- Interpretation: the bridge in this checkpoint is dead. Norm drift alone is
  not catastrophic, but the bridge cannot learn a distribution translation
  because the gate-zero identity path kills projection gradients. This means the
  recurrent loop can appear numerically stable while still lacking a trainable
  re-entry repair mechanism.
- Implementation repair: `IdentityGatedBridge` now defaults to
  `gate_init=1.0`, preserving identity output while making the identity
  projection gradient-live. Existing dead checkpoints are not mutated by default;
  repair is opt-in through `bridge_reset_identity: true` and
  `bridge_gate_override: 1.0` in `training/reentry_repair.py`.
- Stage 2 status: the current Colab `reentry_norm_diagnostic` run is expected
  to compare `none` against `entry_rms` on drift, effective pathways, and
  candidate conversion. The first observed output completed the `none` drift and
  pathway sections and then entered the slower candidate-conversion sweep. The
  repo has not yet received a Stage 2 artifact.
- Follow-up hardening: added a default-on bound for future Stage 2 candidate
  conversion. The old `9b81ead` launcher runs all 14 smoke tasks across the
  loop/noise/seed sweep; current `main` limits candidate conversion to the first
  8 tasks by default via `STAGE5_REENTRY_NORM_CANDIDATE_TASK_LIMIT`, matching
  the drift and effective-pathway readouts. Current `main` also defaults this
  candidate-conversion gate to seed `0` and `80` generated tokens. This makes a
  restart cheaper if the old Colab run stalls or disconnects; expand seeds or
  token budget only if the quick gate is borderline.
- Stage 3 hardening: added a tiny `ReentryAffineAdapter`, initialized to exact
  identity but gradient-live. Stage 3 now trains `bridge,reentry,halt` with
  `entry_rms` re-entry normalization and records adapter scale/bias movement
  alongside bridge liveness and loop-1 preservation. Stage 4 recovery defaults
  carry the same re-entry adapter forward only after Stage 3 passes.
- Stage 3 assessment hardening: recovery training is now blocked unless an
  enabled re-entry adapter has live scale/bias gradients and measurable
  movement, in addition to bridge liveness/movement and loop-1 preservation.
  This prevents a bridge-only pass from being mistaken for a full loop-entry
  repair when the adapter is part of the configured smoke.
- Operational guard: added `reentry_norm_recover_only`. If the Stage 2 run
  completed and backed up to Drive but failed before Git publish, this target
  copies the completed `stage5_reentry_norm_*` artifact from Drive, regenerates
  `reentry_assessment` if needed, and pushes it without rerunning GPU eval.
- Stage 3 guard: `reentry_repair_smoke` now refuses to run unless Stage 2
  recommends `run_reentry_repair_smoke`, except under an explicit override. It
  is resumable and incrementally backs up to Drive after pre-drift, training,
  post-drift, and loop-1 preservation checks.
- Stage 4 guard: `reentry_recovery_training` now exists but is locked behind a
  passed Stage 3 assessment. When cleared, it resumes from the repaired Stage 3
  checkpoint and uses the existing capability-ladder curriculum SFT path with
  learned loop-control and target-loop validation enabled.
- Stage 4 depth-count hardening: `reentry_recovery_training` now derives
  `STAGE5_CURRICULUM_MIN_TARGET_LOOP_ROWS` from a tested helper that preserves
  the actual row count per target loop, for example `1=48,2=16,4=8`. It no
  longer collapses observed depths to presence-only gates such as
  `1=1,2=1,4=1`, which would erase the intended depth-curriculum signal.
- Stage 3 preservation hardening: `assess_stage5_reentry.py` now refuses to
  clear recovery training unless loop-1 preservation evidence is present on
  both source and trained checkpoints and covers matching task groups. Missing
  or mismatched preservation evidence now routes to
  `fix_loop1_preservation_eval_before_recovery_training`.
- Reviewer routing hardening: `review_stage5_reentry.py` now distinguishes
  missing loop-1 evidence, loop-1 regression, adapter-not-live, adapter-live but
  unmoved, bridge-live but unmoved, and full Stage 3 pass. The bootstrap stale
  marker for `reentry_recovery_training` was updated to
  `reentry_recovery_training_v2_depth_count_gate`.
- Added the operational contract
  `docs/STAGE5_REENTRY_STAGE3_STAGE4_RUNBOOK.md`, which defines the hypotheses,
  success criteria, failure responses, and exact next targets for Stage 2,
  Stage 3, and Stage 4.
- Current reviewer output remains Stage 1 only until Stage 2 lands:
  `bridge_dead -> run_reentry_norm_then_repair_smoke -> next target
  reentry_norm_diagnostic`.
- Tests after this reset:
  - `137 passed` after refreshing re-entry recovery routing markers.
  - `133 passed` after requiring loop-1 evidence before recovery training.
  - `91 passed` after hardening the Stage 4 depth-count recovery gate.
  - `1294 passed` after adding the Stage 2 recover-only target.
  - `1293 passed` after adding gated Stage 4 recovery training.
  - `1292 passed` after hardening Stage 3 repair smoke.
- Next allowed GPU actions:
  1. Let the current Stage 2 run finish and publish.
  2. If Stage 2 did finish but did not publish, run
     `STAGE5_CURRENT_A100_TARGET=reentry_norm_recover_only`.
  3. If Stage 2 assessment recommends repair, run
     `STAGE5_CURRENT_A100_TARGET=reentry_repair_smoke`.
  4. Only after Stage 3 passes, run
     `STAGE5_CURRENT_A100_TARGET=reentry_recovery_training`.
- Do not return to Phase 2/SVGD particle training until deterministic recurrence
  has a gradient-live re-entry path and recovery SFT produces base-competitive
  deterministic depth behavior.
