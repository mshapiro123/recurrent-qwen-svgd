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

- Target: `reentry_repair_smoke`.
- Runtime class: L4/T4 is sufficient for the current 0.5B repair smoke. A100,
  G4, or H100 is not required unless Colab availability makes it effectively
  cheaper.
- Bootstrap source: `colab/CURRENT_A100_BOOTSTRAP_CELL.py` from GitHub `main`.
- Maintained notebook: `colab/00_single_a100_runbook.ipynb`.
- Launcher: `colab/STAGE5_REENTRY_REPAIR_SMOKE_CELL.py`.
- Source summary:
  `outputs/stage5/stage5_reentry_norm_20260625_013527/summary.json`.
- Objective: make the loop-closure path gradient-live and verify that the
  bridge/re-entry adapter moves without damaging loop-1 preservation. This is
  Phase 0, not another particle or surface-alignment experiment.
- Added Stage 3 guard: `bridge_gate_active=true` is required. Bridge projection
  movement alone is not enough if the scalar `bridge_gate` collapsed near zero.
- Expected artifact if successful:
  `outputs/stage5/stage5_reentry_repair_smoke_<timestamp>/summary.json` plus
  `reentry_assessment.json`.
- Immediate next target, only if the reviewer recommends it:
  `reentry_recovery_training`.
- Follow-on maintained queue:
  `reentry_repair_smoke -> reentry_recovery_training ->
  debiased_benchmark_suite -> dense_mcq_trace_sft_control`.

### Decisions

- Do not launch more particle/SVGD, inference-time noise, or kernel-geometry
  sweeps until the deterministic loop closure is repaired and Phase 1 depth
  recovery is interpretable.
- Treat Stage 3 repair smoke and Stage 4 deterministic recovery as Phase 0/1
  prerequisites. Breadth and SVGD sit downstream of evidence that deterministic
  depth can recover or improve hard rows without easy regression.
- Compare repaired recurrent training against a standard dense Qwen LoRA trained
  on the same curriculum before making any architecture claim.
- When Phase 2 opens, run breadth diagnostics only on the checkpoint resolved
  from the passed Phase 1 source-summary chain. Manual checkpoint overrides are
  for archaeology, not the normal program track.
- Stage 3 repair smoke must inherit its checkpoint from the passed Stage 2 norm
  assessment. If that lineage is missing, rerun or recover Stage 2; use
  `STAGE5_REENTRY_REPAIR_CHECKPOINT` only for an intentional artifact test.
- Use L4/T4 for bounded 0.5B repair, recovery, and benchmark slices when
  practical. Reserve A100/G4/H100 for 1.5B/3B/7B probes, longer SFT, or
  memory-heavy particle experiments.
- Continue developing the capability-ladder curriculum in parallel with GPU
  runs so the next depth-training pass is not blocked on bookkeeping.

### Current Issues

- Stage 1/2 re-entry diagnostics showed the bridge path was effectively dead
  before repair (`bridge_gate=0`, zero bridge delta, and no useful bridge
  gradient signal). Stage 3 must prove that this path is live.
- The model has shown depth-shaped signal on hard ARC content, but easy-route
  preservation and debiased scoring remain the binding benchmark constraints.
- Inference-time particles/noise produced superficial diversity without reliable
  correct-candidate conversion, so particles are currently a downstream
  training objective, not the next mechanism to tune.
- Existing historical run summaries remain useful, but the current source
  pointer and master sequence are authoritative for the next GPU action.

### Next Checks

- Inspect the Stage 3 repair summary and `reentry_assessment.json` when it
  lands.
- Continue to Stage 4 only if `review_stage5_reentry.py --no_write` recommends
  `run_bounded_recovery_training_with_reentry_repair`.
- If Stage 3 recommends adapter or bridge extension, rerun only the bounded
  repair smoke with adjusted settings; do not skip to recovery training.
- After Stage 4 recovery, run `master_sequence_status`; continue to
  `debiased_benchmark_suite` only if validation is sane and
  `post_reentry_health_checks.status` is `reentry_health_sane`, then run
  `dense_mcq_trace_sft_control` if the Phase 1 reviewer asks for it. This is
  the minimum evidence chain for asking whether recurrence adds value beyond
  the same training data.
- Before any larger depth-router SFT, require enough positive SFT rows at each
  intended `target_loop_count`, especially for `1`, `2`, `3`, and `4` when
  using a 0.5B/1.5B/3B/7B ladder.

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
- Stage 3 publish hardening: `reentry_repair_smoke` now uses
  `colab.stage5_publish_utils.publishable_artifact_paths` instead of
  force-adding the whole output directory. GitHub receives lightweight evidence
  artifacts (`.json`, `.jsonl`, `.md`, `.yaml`, `.log`, etc.); checkpoints stay
  in Drive and are restored by Stage 4 from the `trained_checkpoint` path in the
  Stage 3 summary.
- Stage 4/dense-control publish hardening: `run_stage5_curriculum_sft.py`
  now defaults `STAGE5_CURRICULUM_SFT_COMMIT_CHECKPOINTS=0`, and the dense
  MCQ control reuses the same lightweight artifact allowlist. Checkpoints can
  still be explicitly committed for a special run, but the default Colab path
  backs model weights up to Drive and publishes only evidence to GitHub.
- Stage 3 assessment hardening: recovery training is now blocked unless an
  enabled re-entry adapter has live scale/bias gradients and measurable
  movement, in addition to bridge liveness/movement and loop-1 preservation.
  This prevents a bridge-only pass from being mistaken for a full loop-entry
  repair when the adapter is part of the configured smoke.
- Operational guard: added `reentry_norm_recover_only`. If the Stage 2 run
  completed and backed up to Drive but failed before Git publish, this target
  copies the completed `stage5_reentry_norm_*` artifact from Drive, regenerates
  `reentry_assessment` if needed, and pushes it without rerunning GPU eval.
- Stage 2 recovery hardening: recover-only can now salvage late-interrupted
  runs that have all raw drift, effective-pathway, and candidate-conversion
  files but are missing `summary.json`/`summary.md`. It rebuilds the summary and
  assessment from raw outputs, so an overnight Colab run that dies after the
  expensive GPU work should not need to be rerun.
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
- Stage 4 consistency fix: `reentry_recovery_training` now passes
  `STAGE5_CURRICULUM_REENTRY_RESCALE_MODE=entry_rms` into the curriculum SFT
  runner, and `eval_jsonl.py` now evaluates with the same re-entry rescale mode
  and adapter flags used during training. This keeps recovery SFT aligned with
  the loop-closure regime validated in Stage 2 and trained in Stage 3.
- Added `docs/PROGRAM_TRACK_MASTER_SEQUENCE.md` as the umbrella dependency
  sequence: Phase 0 re-entry, Phase 1 depth, Phase 2 breadth/multistability,
  and Phase 3 particles/SVGD plus selector. Current work remains Phase 0.
- Added `colab/NEXT_COLAB_SEQUENCE.md` as the short Colab target queue and
  tightened the post-Stage-4 depth-control path. The generic
  `STAGE5_CURRENT_A100_SOURCE_SUMMARY` override now fans out to
  `STAGE5_DEBIASED_BENCHMARK_SOURCE_SUMMARY` and
  `STAGE5_DENSE_MCQ_SOURCE_SUMMARY`, so a repaired Stage 4 summary can be used
  consistently for base-vs-recurrent benchmarking and the dense
  same-curriculum control. The dense-control path also forwards the override to
  `STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY`; after the repaired recurrent
  benchmark lands, point the override at that benchmark summary for the dense
  run so the control compares against the intended recurrent artifact rather
  than a stale default.
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
- Phase 0 publishing hardening: Stage 1 re-entry drift, Stage 2 eval-only
  re-entry norm, and Stage 2 recover-only now publish through
  `publishable_artifact_paths` instead of force-adding the entire output
  directory. This keeps the current Colab execution chain aligned with the
  low-GitHub-churn policy and prevents accidental checkpoint publication while
  preserving summaries, JSONL diagnostics, logs, and markdown readouts.
- Phase 0 progress-ledger hardening: `summarize_stage5_progress.py` now reads
  re-entry drift, eval-only norm, and repair-smoke summaries plus their
  `reentry_assessment.json` files. The generated ledger includes a
  "Re-entry Phase 0" section with recommendation, bridge/adapter liveness, and
  loop-1 or entry-RMS deltas, so the current master-sequence head is visible in
  the same status report as older Stage 5 artifacts.
- Phase 0 planner hardening: `plan_stage5_next_run.py` now recognizes
  re-entry drift, eval-only norm, and repair-smoke summaries. It maps
  assessment recommendations to the next maintained bootstrap target
  (`reentry_norm_diagnostic`, `reentry_repair_smoke`, or
  `reentry_recovery_training`) and emits a read-only runbook/target command
  instead of falling through to older ARC/SVGD planner branches. This keeps
  generic safe-continue status checks aligned with the master sequence without
  allowing the generic planner to launch opaque bootstrap targets.
- Checkpoint-publication hardening: the Stage 4 re-entry recovery launcher and
  older depth/capability-ladder SFT launchers now leave
  `STAGE5_CURRICULUM_SFT_COMMIT_CHECKPOINTS=0` by default, with explicit
  per-launcher opt-in env vars for exceptional cases. Checkpoints should remain
  Drive-backed/HF-packaged; GitHub commits should carry lightweight evidence
  artifacts, current-summary pointers, configs, logs, and summaries.
- Stage 3 bridge-movement hardening: the re-entry repair smoke summary now
  records direct bridge projection movement (`proj_identity_max_abs_diff` and
  `proj_bias_max_abs`) in addition to sampled bridge output delta. The
  assessment gate accepts either direct projection movement or output movement
  when gradients are live, so a tiny but real bridge update is not rejected
  because a sampled delta happens to be small.
- Post-Stage-4 control wiring hardening: the dense standard-Qwen same-curriculum
  control now separates the comparison source summary from the curriculum/SFT
  source summary. Benchmark and assessment wrappers are still accepted as the
  front-of-queue source, but the control follows their source chain to inherit
  positive-SFT rows, depth-hint style, max steps, and learning rate from the
  underlying training/curriculum summary. This prevents a repaired recurrent
  benchmark wrapper from silently falling back to stale dense-control defaults
  during the standard-Qwen control run.
- Colab restart/readout hardening: added `master_sequence_status` as a cheap
  CPU bootstrap target. It fetches the latest repo, prints the current
  source-summary pointer, runs the planner and re-entry reviewer, prints the
  next Colab queue excerpt, and disconnects without Drive mount, model
  downloads, training, or evaluation. This gives restarted notebooks a
  low-cost way to recover the next Phase 0/Phase 1 target before spending GPU.
- Phase 0 pointer alignment: updated `config/stage5_current_source_summary.txt`
  to the committed Stage 1 re-entry drift summary and force-published its
  `reentry_assessment.json/md`. The cheap status target, generic planner, and
  re-entry reviewer now agree that the next maintained target is
  `reentry_norm_diagnostic`, rather than falling back to stale score-alignment
  work.
- Stage 2 restart hardening: `reentry_norm_diagnostic` now restores any
  incremental Drive backup for the selected run ID before beginning its drift,
  effective-pathway, and candidate-conversion steps. On an interrupted L4/T4
  run, relaunching with the same `STAGE5_REENTRY_NORM_RUN_ID` can reuse valid
  partial outputs through the existing `resume_skip=*` checks instead of
  rerunning the whole diagnostic.
- Stage 3 restart hardening: `reentry_repair_smoke` now restores any
  incremental Drive backup for the selected run ID before pre/post drift,
  smoke training, and loop-1 preservation checks. Relaunching with the same
  `STAGE5_REENTRY_REPAIR_RUN_ID` can reuse completed pre-drift, train, and
  preservation artifacts rather than repeating the whole smoke.
- Stage 4 checkpoint-first backup hardening: `run_stage5_curriculum_sft.py`
  now backs the Stage 4 run directory up to Drive immediately after Phase 1
  training succeeds, before validation begins. It refreshes the Drive backup
  again after `summary.json`/`summary.md` are written. This preserves the
  repaired recovery checkpoint even if validation, Colab disconnect, or Git
  push fails after training.
- Stage 2 readout and pointer advance: `stage5_reentry_norm_20260625_013527`
  reduced loop-8 output/entry RMS under `entry_rms` from about `1.037` to
  `1.008` without changing candidate hits or best hits. The assessment is
  `entry_rms_safe_for_smoke -> run_reentry_repair_smoke`, so the current source
  pointer and `NEXT_COLAB_SEQUENCE` paste-anywhere default now advance to the
  Stage 3 trainable re-entry repair smoke.
- Re-entry publisher pointer hardening: Stage 1 drift, Stage 2 norm,
  Stage 2 recover-only, and Stage 3 repair-smoke publishers now write and stage
  `config/stage5_current_source_summary.txt` in the same commit as their
  summary artifacts. This prevents successful Colab runs from leaving a
  restarted notebook pointed at a stale earlier phase.
- Stage 4 wrapper-summary handoff: `reentry_recovery_training` now runs the
  generic curriculum SFT child under a child run ID, then publishes a
  `stage5_reentry_recovery_training` wrapper summary as the current source.
  The wrapper preserves the repaired checkpoint path, Stage 3 provenance,
  validation checks, dataset counts, and child summary path. The planner routes
  this explicit Phase 0/1 recovery wrapper to `debiased_benchmark_suite`
  before dense control, breadth diagnostics, particles, or SVGD. This fixes the
  previous mismatch where a completed Stage 4 run would look like a generic
  `stage5_curriculum_sft` artifact and could fall back to the older routing
  diagnostic branch instead of the master-sequence benchmark gate.
- Colab handoff alignment: refreshed `STAGED_NOTEBOOKS.md`,
  `00_stage_launcher.ipynb`, `00_single_a100_runbook.ipynb`, and the
  current-order section of `SEQUENCED_EXPERIMENT_PLAN.md` so human-facing
  launchers match the master sequence. The maintained notebook path is now an
  explicit GitHub-fetched bootstrap target queue:
  `reentry_repair_smoke -> reentry_recovery_training ->
  debiased_benchmark_suite -> dense_mcq_trace_sft_control`, with Phase 2/3
  breadth/SVGD gated behind deterministic depth evidence. This removes stale
  ARC-mix/direct-preservation "current action" affordances from the main
  notebooks while preserving those historical notebooks for provenance.
- Stage 3 train-metric hardening: `reentry_repair_smoke` now parses the final
  `train_phase1_ponder.py` step metrics into `summary.json` and `summary.md`.
  The Stage 3 assessment blocks Stage 4 if final training metrics are missing,
  if final loss is nonfinite, or if supervised depth metrics
  (`target_loop_abs_error`, `halting_target_nll`) are absent while halt-depth
  supervision is enabled. This keeps the Stage 3 -> Stage 4 decision tied to a
  visible repair-training signal, not only checkpoint existence.
- Stage 4 stale-assessment guard: `reentry_recovery_training` now revalidates
  the Stage 3 repair assessment before launching recovery SFT. It refuses
  older recommendation-only repair artifacts and requires the metric-hardened
  evidence fields: finite final train loss, depth-supervision metrics,
  comparable loop-1 preservation, bridge live/moved, and re-entry adapter
  live/moved when enabled. The wrapper summary also preserves those Stage 3
  metrics for downstream benchmark and dense-control provenance.
- Post-recovery benchmark alignment: the maintained
  `debiased_benchmark_suite` target now enables learned loop control by
  default. Stage 4 explicitly trains the depth router, so the first
  base-vs-recurrent benchmark should evaluate the repaired recurrent model
  with that learned routing path active unless an older non-router checkpoint
  is intentionally being tested.
- Master-sequence planner alignment: a passed post-recovery debiased benchmark
  now routes next to `dense_mcq_trace_sft_control`, not directly to capability
  ladder, scale probes, claim packaging, particles, or SVGD. This keeps the
  Phase 1 architecture question clean: first compare repaired recurrent Qwen
  against standard Qwen trained on the same curriculum, then decide whether
  depth-label/capability-ladder work is needed.
- Dense-control guard alignment: the safe-continue/go-no-go classifier now
  allows `dense_mcq_trace_sft_control` after any passed broader benchmark
  assessment, matching the planner route above. The dense-control runner is
  covered by a planner-style source-chain test that resolves the benchmark
  assessment to both the recurrent benchmark suite and the underlying Stage 4
  curriculum defaults.
- Stage 3 resume safety: `reentry_repair_smoke` now skips Phase 1 repair
  training only when both the trained checkpoint and parseable train-log
  metrics exist. If Colab restores a checkpoint without a usable
  `train_phase1_ponder.log`, the cell reruns the tiny repair training instead
  of publishing a metric-empty artifact that Stage 4 would later reject.
- Re-entry Drive compatibility: Stage 3 now searches both the current artifact
  Drive root and the legacy `recurrent-qwen-svgd` Drive root for Stage 2 norm
  assessments, and Stage 4 does the same for Stage 3 repair assessments. This
  reduces manual recovery when Colab output exists on Drive but not under the
  newest artifact prefix.
- Stage 4 current-pointer preference: `reentry_recovery_training` now prefers
  the repair assessment implied by `config/stage5_current_source_summary.txt`
  when that pointer names a `stage5_reentry_repair_smoke` summary. Broad Drive
  glob fallback remains available, but it no longer shadows the planner-selected
  repair artifact just because an unrelated failed or partial run has a newer
  modification time.
- Stage 3 current-pointer preference: `reentry_repair_smoke` now applies the
  same source-selection rule one phase earlier. If
  `config/stage5_current_source_summary.txt` names a
  `stage5_reentry_norm_eval_only` summary or assessment, Stage 3 uses that norm
  assessment before broad Drive glob fallback. This prevents a newer unrelated
  Stage 2 norm diagnostic from silently becoming the trainable repair source.
- Re-entry reviewer current-pointer preference: `review_stage5_reentry.py` now
  follows the current source pointer when it names a re-entry summary or
  assessment. If the pointer names a re-entry summary but the sibling
  `reentry_assessment.json` is missing or unreadable, the reviewer stops and
  requests recovery/rerun instead of falling back to an older broad-scan
  assessment.
- Single-runtime master-sequence notebook: `00_single_a100_runbook.ipynb` now
  defines one GitHub-SHA-resolved bootstrap helper plus explicit cells for
  `master_sequence_status`, `reentry_repair_smoke`,
  `reentry_recovery_training`, `debiased_benchmark_suite`, and
  `dense_mcq_trace_sft_control`. The helper exposes `KEEP_RUNTIME_OPEN`; the
  debiased benchmark target now honors `STAGE5_DEBIASED_BENCHMARK_DISCONNECT`
  so bounded L4/T4 cells can be batched intentionally without changing target
  code.
- Result-publishing hardening: Stage 4 recovery SFT, post-recovery debiased
  benchmark, and dense same-curriculum MCQ control now all treat rejected GitHub
  pushes as recoverable once, not silent. Each runner now attempts a direct
  push, then one `git pull --rebase --autostash` plus retry, and then fails
  loudly if publication still cannot land. This preserves the Drive-backed
  checkpoint policy while reducing the chance that overnight GPU work finishes
  but leaves no current-source pointer or result artifact on GitHub.
- Queue-drift guard: `tests/test_stage5_notebooks.py` now checks that the
  maintained user-facing handoffs keep the paid-GPU target order synchronized:
  `reentry_repair_smoke -> reentry_recovery_training ->
  debiased_benchmark_suite -> dense_mcq_trace_sft_control`. The cheap
  `master_sequence_status` target can appear separately, but the operational
  GPU queue should not silently diverge across README, current action,
  next-sequence, staged-notebook, or single-runtime runbook surfaces.
- Stage 3 publish guard coverage: the re-entry repair smoke bootstrap test now
  asserts that Stage 3 itself uses the same rejected-push recovery pattern as
  the later Stage 4/benchmark/control runners: direct push, one
  `git pull --rebase --autostash`, and a final checked push. This keeps the
  current front-of-queue GPU action covered against the publication failure
  mode that previously cost manual recovery time.
- Dense-control source-chain hardening: `dense_mcq_trace_sft_control` now
  follows `child_summary` and `trace_summary` links in addition to the older
  source/nested/benchmark links when resolving the curriculum owner. This
  matches the real Stage 4 wrapper shape, where the wrapper carries the
  repaired checkpoint while the child SFT or trace-collection summary owns the
  positive-SFT rows and training defaults. New tests cover both
  wrapper-to-trace and wrapper-to-child resolution, and explicitly prefer the
  child SFT summary when both links are present so the dense control inherits
  the actual Stage 4 split/defaults.
- Stage 3 optimizer-module guard: the tiny recurrent wrapper tests now assert
  that `optimizer_modules=bridge,reentry,halt` selects exactly the bridge,
  loop re-entry adapter, and halting predictor parameters. This protects the
  repair-smoke objective from silently dropping one of the trainable loop
  closure components.
- Re-entry diagnostic adapter coverage: `tests/test_eval_reentry_drift.py` now
  verifies that the eval diagnostic emits identity-at-init movement stats and
  live scale/bias gradients for the re-entry adapter. These are the fields the
  Stage 3 assessment uses to decide whether the loop re-entry adapter is
  gradient-live and actually moved.
- Master-sequence handoff refresh: added
  `docs/DEEP_RESEARCH_HANDOFF_2026_06_25_MASTER_SEQUENCE.md` and linked it
  from the README. This is the compact strategy-agent packet for the current
  dependency chain: re-entry repair first, deterministic depth recovery second,
  dense same-curriculum control third, breadth diagnostics fourth, and
  particles/SVGD only after correct-bearing breadth exists.
- Stage 4 curriculum readiness readout: added
  `colab/review_stage5_recovery_curriculum.py` and wired it into the cheap
  `master_sequence_status` target. The current Stage 4 trace curriculum is
  usable for bounded recovery (`63` positive rows, target loops `1=26,2=28,3=9`)
  but explicitly warns that it is not claim-sized and has a sparse highest-loop
  bucket. This keeps Stage 4 data adequacy visible before spending GPU.
- Stage 4 curriculum source-resolution hardening: the readiness readout now
  follows the same source-selection shape as Stage 4 recovery training:
  explicit trace-summary override first, then latest gate-ready local/Drive
  trace collection, then the known default. Tests cover latest-ready selection,
  explicit override, and rejection of non-gate-ready summaries.
- Stage 4 claim-size readiness: the recovery-curriculum readout now separates
  bounded-smoke readiness from performance-claim readiness. The current trace
  collection remains green for Stage 4 smoke, but against the default
  claim-sized direct/deep threshold (`2000` positives, `direct=1000`,
  `deep_narrow=1000`) it reports a `1937` positive-row deficit, with `974`
  missing direct rows and `963` missing deep-narrow rows.
- Stage 4 curriculum scale-up planner: added
  `colab/plan_stage5_curriculum_scaleup.py` and wired it into the cheap
  `master_sequence_status` target. The readout now prints the current bounded
  Stage 4 readiness, claim-sized deficits, and concrete CPU/API commands for
  building the next direct/deep curriculum shard while the GPU queue remains
  on `reentry_repair_smoke`. This preserves the master-sequence dependency
  order: data can scale in parallel, but Stage 4 remains locked behind the
  Stage 3 re-entry repair gate.
- Claim-sized curriculum CPU target: updated the resumable curriculum artifact
  pipeline to default to the claim-sized direct/deep shard
  (`data/curriculum/claim_direct_deep_001`, `2000` positives,
  `direct=1000,deep_narrow=1000`, `math,science`, target steps `1,2,5,9`,
  `count_per_combo=122`) while keeping provider calls disabled unless
  `STAGE5_CURRICULUM_RUN_PROVIDER_RESPONSES=1` is explicitly set. The bootstrap
  now exposes `claim_curriculum_scaleup_cpu`, so the same single Colab launcher
  can run the CPU/API data path without copy/pasting raw commands.
- Claim curriculum provider-map ergonomics: the claim-sized CPU/API target now
  accepts a provider model map from `STAGE5_CURRICULUM_MODEL_MAP_JSON` or from
  individual `STAGE5_CURRICULUM_OPUS_MODEL`,
  `STAGE5_CURRICULUM_GLM_MODEL`, and
  `STAGE5_CURRICULUM_WEAK_REFERENCE_MODEL` values. This removes the need to
  edit the Colab cell body before enabling provider responses, while preserving
  the placeholder check that refuses paid calls when concrete model ids are not
  configured.
- Stage 3 GPU preflight: `reentry_repair_smoke` now checks for an attached GPU
  runtime before repo sync, Drive checkpoint restoration, requirements install,
  or training setup. A CPU-only or disconnected Colab runtime now fails with a
  clear instruction to reconnect L4/T4/A100/H100, reducing late setup failures
  on the current front-of-queue GPU action.
- Single-runtime CPU/GPU split: `00_single_a100_runbook.ipynb` now includes an
  explicit `claim_curriculum_scaleup_cpu` cell next to the GPU target queue.
  This makes the master-sequence parallelism concrete: claim-sized direct/deep
  curriculum generation can run on CPU/API time while the paid GPU sequence
  remains locked on `reentry_repair_smoke -> reentry_recovery_training ->
  debiased_benchmark_suite -> dense_mcq_trace_sft_control`.
- Stage 3 loop-1 preservation signal hardening: `reentry_repair_smoke`
  assessment now requires the source loop-1 preservation comparison to contain
  at least one correct source example. A zero-hit source comparison proves only
  non-regression on an uninformative slice, so Stage 4 now refuses that artifact
  and asks for a better loop-1 preservation eval before recovery SFT.
- Claim curriculum readiness artifact: the CPU/API claim-sized curriculum
  target now writes `curriculum_readiness.json` into its work directory after
  each pass. The file records provider enabled/disabled state, API-key presence,
  whether concrete model ids replaced placeholders, pending provider
  job/response pairs, row requirements, and the next safe action. A local
  no-provider preflight with the claim-scale defaults created `3904` seed jobs
  and stopped cleanly at `pending_seed_responses`, proving the CPU path starts
  without GPU or provider spend.
- Model-size modularity for re-entry/depth path: Stage 3 repair smoke and the
  Stage 4 curriculum SFT runner now pass `MODEL_NAME` and
  `STAGE5_RECURRENT_LAYER_SPLIT`/`STAGE5_CURRICULUM_LAYER_SPLIT` through their
  train and eval commands. The default split is `auto`, which preserves the
  prior Qwen2.5-0.5B `6,18` partition while avoiding stale hardcoded splits for
  1.5B/3B/7B scale probes.
- Phase 1 gate reviewer: added `colab/review_stage5_phase1_gate.py` and wired
  it into the cheap `master_sequence_status` target. The reviewer is
  phase-aware: while the current pointer is still Stage 2/3 it ignores stale
  benchmark artifacts and waits for Stage 4 recovery; after Stage 4, it routes
  passed recurrent-vs-base benchmarks to the dense same-curriculum control and
  only treats `hard_tail_lift_vs_dense` as an architecture signal. This keeps
  the master sequence honest: recovery is not architecture proof, and particles
  stay blocked until the deterministic Phase 1 control clears.
- Stage 3 bridge-gate hardening: the re-entry repair assessment now requires
  `bridge_gate_active=true`. Bridge projection movement is not enough if the
  scalar `bridge_gate` collapsed back near zero, because that repaired path
  would still be effectively disconnected from loop re-entry. Stage 4 recovery
  refuses Stage 3 assessments that lack this active-gate evidence.
- Stage 3 gate-collapse retry mode: the re-entry reviewer now names
  `bridge_gate_collapsed` separately and emits a bounded retry with
  `STAGE5_REENTRY_REPAIR_OPTIMIZER_MODULES=bridge_proj,reentry,halt`. This
  keeps the scalar gate active and checkpointed while testing the bridge
  projection plus re-entry adapter.
- Stage 4 post-recovery re-entry health gate: `reentry_recovery_training` now
  runs a cheap `eval/eval_reentry_drift.py` probe after recovery SFT and
  publishes `post_reentry_health_checks`. The Stage 4 reviewer and next-run
  planner now block the debiased benchmark unless that health status is
  `reentry_health_sane`, preventing a recovery run from silently closing the
  bridge/re-entry path after Stage 3 repaired it.
- Phase 2 checkpoint lineage hardening: `effective_pathways_diagnostic` and
  `candidate_conversion_diagnostic` no longer default to the old ARC-mix
  checkpoint. Once the Phase 1 architecture gate passes, the master-sequence
  gate walks the current source-summary chain back through the recurrent
  benchmark suite to the Stage 4 recurrent checkpoint and hands that checkpoint
  to the breadth launchers. Explicit `STAGE5_EFFECTIVE_PATHWAYS_CHECKPOINT` or
  `STAGE5_CANDIDATE_CONVERSION_CHECKPOINT` overrides remain available only for
  intentional archaeology runs.
- Stage 4 trace-curriculum source unification: `reentry_recovery_training` now
  resolves its capability-ladder trace summary through the same
  `review_stage5_recovery_curriculum` helper used by the CPU status/review
  path. This removes the duplicated in-cell trace-summary scan and keeps the
  paid recovery SFT launcher aligned with the tested reviewer: explicit
  `STAGE5_REENTRY_RECOVERY_TRACE_SOURCE_SUMMARY` still wins, newer gate-ready
  local/Drive summaries are scanned next, and the historical collection remains
  only the reviewed helper's last fallback.
- Phase 1 benchmark source guard: `debiased_benchmark_suite` now refuses the
  normal current-pointer path unless the resolved checkpoint-bearing source is
  a `stage5_reentry_recovery_training` summary with
  `post_reentry_health_checks.status == "reentry_health_sane"`. This keeps the
  master-sequence seam hard: recurrent-vs-base benchmarking starts only after
  Stage 4 recovery has passed its re-entry health gate. Explicit
  `STAGE5_DEBIASED_BENCHMARK_SOURCE_SUMMARY` still bypasses this check for
  intentional older-artifact comparisons.
- Stage 3 repair-smoke resume safety: `reentry_repair_smoke` now compares the
  existing training config with the freshly derived config before reusing
  cached drift, loop-1 preservation, train logs, or checkpoints. A rerun with
  the same `STAGE5_REENTRY_REPAIR_RUN_ID` but different checkpoint, LR,
  optimizer modules, layer split, model, or loop settings now regenerates the
  diagnostics/training instead of silently mixing stale cached artifacts into
  the Stage 4 handoff.
- Dense-control source gate: `dense_mcq_trace_sft_control` now requires the
  normal source pointer to be a passed `stage5_broader_benchmark_suite`
  recurrent-vs-base assessment before spending GPU on the same-curriculum dense
  LoRA. This keeps the Phase 1 claim sequence in order: Stage 4 recovery, then
  recurrent-vs-base benchmark, then dense control. Intentional archaeology can
  still set `STAGE5_DENSE_MCQ_ALLOW_UNPASSED_BENCHMARK=1`, which prints an
  override marker.
- Stage 3 full-topology gate: the Stage 3 repair assessment and Stage 4
  recovery gate now require the current loop-closure repair topology, not just
  a live bridge. A recovery-unlocking repair smoke must report
  `reentry_rescale_mode="entry_rms"`, `use_reentry_adapter=true`, adapter
  gradients live, adapter movement, bridge liveness/movement, active
  `bridge_gate`, finite train/depth metrics, and loop-1 preservation. A stale
  or legacy bridge-only repair smoke is now routed to a bounded Stage 3 rerun
  instead of Stage 4 recovery.
- Stage 3 checkpoint provenance: the re-entry repair smoke summary now records
  whether the source checkpoint came from the passed Stage 2 norm assessment,
  an explicit user override, or an explicit fallback. This keeps later Stage 4
  and benchmark readouts from conflating the normal front-of-queue path with an
  archaeology run.
- Stage 3/4 re-entry repair result: the L4 chained run produced a passing
  `stage5_reentry_repair_smoke_20260625_114554` artifact. The bridge gate moved
  from `0.0` to about `1.0002`, bridge and re-entry adapter gradients were live,
  adapter/bridge deltas moved, and loop-1 preservation did not regress on the
  small smoke slice. Stage 4 then produced
  `stage5_reentry_recovery_20260625_114836`, whose post-recovery health check
  was `reentry_health_sane` with loop-8 output/input RMS about `1.0061`. This
  resolves the immediate loop-closure/re-entry liveness concern and restores
  permission to evaluate deterministic recurrent competence.
- Stage 4 depth-routing readout: the same recovery run remained
  `validation_needs_review` because target-loop routing was only partially
  learned. Direct rows averaged about `1.33` expected loops and deep-narrow rows
  about `1.97`, so a broad depth-gradient signal is present. The explicit
  target loop ladder was not monotone from target 2 to target 3 (`~1.99` then
  `~1.93`), so deeper-loop supervision still needs a larger or better separated
  curriculum before it can support depth claims.
- Debiased benchmark readout: `stage5_debiased_benchmark_suite_20260625_115004`
  compared the repaired recurrent checkpoint against base Qwen2.5-0.5B on
  ARC-Easy/ARC-Challenge. Recurrent was positive on ARC-Easy label/cyclic
  variants and ARC-Challenge content-only, but slightly negative on
  ARC-Challenge cyclic-label (`67/128` vs base `68/128`). GPQA-lite did not run
  because `Idavidrein/gpqa` is gated for the active HF account. The assessment
  therefore stayed `needs_review`, and the Phase 1 gate correctly blocked dense
  control and Phase 2/SVGD.
- Current next branch: because deterministic recurrent competence is close but
  not yet base-competitive under the strict gate, the planner routes to
  `traced_sft_competence_preserving_pipeline`: ARC-Easy-weighted mixed recovery
  with response distillation before any dense control, particles, or SVGD. The
  target now defaults to the fresh re-entry benchmark assessment
  (`stage5_debiased_benchmark_assessment_20260625_121302`) instead of an older
  June 23 direct-preservation artifact.
- Runtime hardening: Phase 1, Phase 2, dense LoRA, and MCQ score-align training
  now defensively cast numeric config values. This prevents generated JSON/YAML
  scientific notation such as `1e-05` from being read as a string and crashing
  `AdamW` or gradient clipping during long Colab chains.
- Competence pipeline launch/restore hardening: the first
  `stage5_competence_recovery_from_reentry_benchmark` attempt failed before
  training because the selected Stage 4 checkpoint was not local and Drive was
  not mounted in the top-level Colab process. The child subprocess could not
  mount Drive (`NoneType.kernel`), so checkpoint restore failed. The launcher
  now mounts Drive before child subprocesses, the wrapper records child-log
  tails and diagnoses `checkpoint_restore_or_drive_mount_failed`, and the
  planner maps that diagnosis to a same-run-id resume instead of a vague manual
  inspect action. The current bootstrap also prefers the synced local HEAD over
  a possibly stale GitHub ref response, and `CURRENT_STAGE5_FRESH_LAUNCHER_CELL`
  provides a tracked blank-notebook launcher so future Colab sessions clone,
  hard-reset, mount Drive, verify the freshness/Drive fixes, and then execute
  the current Stage 5 target without hand-assembled setup cells.
- Competence pipeline stale-failure routing: older failed competence summaries
  did not contain the newer `failure_diagnosis` field, so the planner could
  still route the current stale checkpoint/Drive failure to manual inspection.
  The planner now reads durable `arc_mix.log` evidence and recognizes the same
  missing-checkpoint/Drive-restore signature. It resumes the competence
  wrapper with the same run ids after the top-level Drive mount instead of
  forcing another hand triage loop. A CPU-only
  `review_stage5_competence_pipeline.py` helper now prints the pipeline status,
  child statuses/checkpoints, and planner-selected next action for immediate
  post-Colab triage.
- Capacity localization arm: after tail-damper, forced-depth, selector, and
  tail-convergence diagnostics failed to produce a cheap selection fix, the next
  decisive question is whether the low-rank recurrent operator is the ceiling.
  The new `reentry_capacity_localization_rank64` target keeps the fixed
  strength-1.0 tail damper, data, optimizer modules, trace source, and recovery
  procedure fixed, then changes only recurrent LoRA capacity from the existing
  rank-32 baseline to rank 64. The run writes a separate
  `stage5_current_capacity_localization_summary.txt` pointer rather than
  moving the generic checkpoint-bearing source pointer to an aggregate summary.
  The summary reports trainable parameter ledger, loop-1/2/3 scores, oracle,
  rescued, harmed, tail trace ratio, and deltas versus the June 27 rank-32
  fixed-damper recovery baseline. Rank 128 remains an explicit follow-up only
  if rank 64 shows rescued/oracle/depth movement; otherwise the meaningful next
  escalation is the unfreeze+Muon recurrence-curriculum bundle, not more
  inference-time geometry.
- 2026-07-12 Phase G track reunification: the verified GRAM audit removes the
  old split between the deterministic queue and an eventual stochastic-width
  return. The abductive-injective gate is now both the next deterministic rung
  and G-alpha's substrate gate. The final frozen evaluation is N=24 arbitrary
  non-bijective mappings with disjoint calibration/test splits and 128 rows in
  each exact-preimage stratum (1, 2-4, >=5). Exact coverage is recomputed from
  the forward orbit. The answer-head comparator is entropy-matched per row;
  fixed-temperature output is provisional. G-alpha freezes the entire
  deterministic block and trains only prior/posterior heads and injection
  scale, with exact zero-gradient assertions. Numeric gate margins remain the
  sole preregistration blank pending the calibration-split power calculation.
  The current N=20 constructive-fan run is retained as a screening gate; a
  deterministic N=24 arbitrary-table calibration pass is additionally required
  before the substrate is frozen, so stochastic guidance cannot be credited
  with repairing a task-distribution competence gap.
- 2026-07-12 invalid Phase G Experiment 1 attempt: run
  `stage5_phase_g_experiment1_20260712` is a no-op receipt, not a negative model
  result. Its prompt ended in a trailing space while completions lacked a
  leading space. Separate tokenization made every outcome and loop label
  inactive: training logged loss `0`, active loop labels `0`, and gradients
  `0` for all 1,000 steps. The 9/128 injective smoke score is therefore an
  untrained baseline. The corrected run uses ID
  `stage5_phase_g_experiment1_fixed_boundary_20260712`, canonical `Answer:` +
  `" Name"` boundaries, and hard failures on zero supervision or zero gradient.
- 2026-07-12 valid Phase G deterministic injective screen: the corrected run
  restored active supervision and nonzero gradients, trained 1,000 updates, and
  produced a sharp one-step/composition split. Depth 1 was 16/16, while depths
  2-8 were 6/112 (5.36%), essentially the N=20 chance rate of 5%; pooled smoke
  was 22/128 (17.19%) against the locked 50% smoke floor. The runner therefore
  recorded `blocked_injective_smoke` and intentionally returned exit code 2.
  This blocks abductive and G-alpha work but is not yet a final substrate
  negative because the run used less than one pass-equivalent of updates and
  disabled the recurrence curriculum.
- 2026-07-12 Phase G injective curriculum recovery: commit `d705ba8` adds one
  bounded continuation from the exact corrected step-1000 checkpoint, verified
  by SHA `0d6cf119...a1a6`. It changes only dose and recurrent curriculum: 2,000
  additional updates, a linear 2-to-8 loop/compute ramp, frozen row hashes, and
  deterministic K=1 competence gates. Latent heads, learned halting, LPRM, and
  SVGD remain disabled. If the same depth-1-only pattern survives, the next
  decision is a short train-versus-held-out/micro-overfit autopsy versus one
  clean curriculum-from-keeper restart, not stochastic-head construction.
- 2026-07-12 Phase G injective curriculum recovery result: training completed
  all 2,000 additional updates and backed up checkpoint SHA `fc98feb5...53d1`.
  Smoke improved nominally from 22/128 to 26/128, but paired rows were 10
  helped, 6 hurt, and 112 tied (exact two-sided sign p=0.4545). Depth 1 stayed
  16/16; depths 2-8 totaled 10/112 (8.93%) with a non-monotonic pattern and
  zero greedy hits at depths 4, 5, and 7. The runner correctly recorded
  `blocked_injective_smoke` and returned 2 after training. This does not open
  abductive or G-alpha work and does not justify another automatic dose run.
- 2026-07-12 invalidator B10 and curriculum-autopsy amendment: the original
  Phase G attempt with zero active labels remains invalid under B10, with
  active-supervision preflight and nonzero-gradient abort now standing
  countermeasures. Static verification of the later valid runs found no second
  construction invalidator: fixed-eight compute was used initially, but
  labels after row depth were masked and evaluation read loop `d`, so no hold
  objective or end-reader mismatch existed. Saved diagonal predictions also
  falsify an invert-once-then-hold policy (one-step-preimage counts 4/128 and
  6/128); most errors were other legal names. Target
  `phase_g_curriculum_autopsy` now runs the missing train/held-out loop matrix,
  deep-row loop-one, above-diagonal, state-query, and corrected uniform-
  coverage diagnostics on both exact checkpoint hashes. It also prepares but
  does not train the inverse-table control. The K=20 N=20 uniform baseline is
  `1-(19/20)^20 = 0.6415`, above the prior observed `0.5703`; future coverage
  summaries report this baseline and raw per-sample validity explicitly.
- 2026-07-13 Phase G curriculum autopsy result: the read-only two-checkpoint
  matrix materially revises the final-diagonal interpretation. On rows from
  the exact training-order prefix seen by both checkpoints, recovery moved
  loop 2 from 10/112 to 51/112; on held-out rows it moved from 5/112 to
  42/112, with 40 paired improvements and 3 regressions (two-sided sign
  p=3.02e-9). Held-out loop 3 moved from 1/96 to 10/96 (10 helped, 1 hurt,
  p=0.0117). Loop 1 remained 125/128 and loops 4-8 remained unsupported. The
  final diagonal hid these gains because later rows require every later loop.
  Exact recovery exposure explains the staircase: raw active labels by loop
  were 2000, 1749, 1377, 950, 594, 310, 114, and 19; after per-row active-loss
  averaging, loop-8 weight was only about 2.4 full-row equivalents. The next
  causal test is therefore inverse-table direction control plus a loop-balanced
  stagewise depth-1-to-4 restart, not another nominal-step extension under the
  same linear 2-to-8 ramp. Full handoff:
  `docs/PHASE_G_CURRICULUM_AUTOPSY_HANDOFF_20260713.md`.
- 2026-07-13 inverse-composition staircase preregistration: the next GPU job is
  one matched two-arm AdamW staircase from the locked natural-surface step-2000
  keeper. Arm F retains the forward table and reverse-search transition; Arm C
  rerenders the identical rows as inverse tables so each recurrent transition
  is a forward lookup. The trainer now supports fixed-batch weighted per-loop
  loss and true gradient accumulation to effective batch 8. Loop weights
  inverse exposure, double the newest loop, and are logged as raw and weighted
  active labels; adjusted realized mass must remain within 0.8x-1.25x every
  200 optimizer steps. Stages 2-4 gate at 46/64 and approximately 1,500 newest-
  loop weighted labels; stages 5-8 open only after both Phase-1 arms and the
  0.93 guardrail pass. The active matrix, conditional transition success,
  target decodability, and stratified Phase-1 CKA are the diagnostic receipts.
  Muon is excluded from this matched causal test and remains a later optimizer
  ablation only. Full spec: `docs/INVERSE_COMPOSITION_STAIRCASE_SPEC.md`.
- 2026-07-13 inverse-composition staircase result: the run stopped validly at
  cap 2. The inverse-table/forward-lookup control reached 62/64 at 1,598.4
  newest-loop weighted labels and preserved the synthetic guardrail at a
  0.9375 minimum. The forward-table/reverse-search arm reached only 3/64 at
  1,603.2 weighted labels and reduced the guardrail minimum to 0.21875. Its
  first inverse transition still reached 55/64 on depth-2 rows, but the second
  transition reached only 3/64, localizing the failure to repeated inverse
  composition. The receipt label was corrected post-run from the overstrong
  `non_native_position_cost` to `experiment_stalled_at_matched_dose`: the run
  did not observe the preregistered five-fold dose ratio. Phase G-alpha remains
  closed.
- 2026-07-13 inverse-table rebase and parallel Phase A implementation: target
  `inverse_table_rebase_caps3_4` starts from exact C cap-2 SHA
  `bc1de1cd...5b01`, runs only caps 3 and 4, retains the 46/64 and 0.93 gates,
  and adds a final natural-surface canary before pausing. Independent target
  `phase_a_dense_full` implements dense arms B/C on an L4 and D on an A100.
  All dense arms now use full-model AdamW with FP32 parameters/moments, BF16
  compute, effective batch 8, 2e-6 LR, 4,000 steps, pinned Qwen revisions, and
  locked train/eval hashes. B is direct, C is serialized scratchpad, and D is
  the 1.5B direct exchange-rate arm. Checkpoints are backed up to Drive and
  lightweight receipts publish per arm so the lanes can run concurrently.
- 2026-07-13 Phase A dense preflight correction: both dense lanes stopped
  before model load because their dataset locks used byte hashes generated
  from a Windows CRLF checkout, while Colab checked out identical JSONL rows
  with LF endings. The lock now hashes newline-normalized JSONL bytes and is
  regression-tested against both conventions. The same logs exposed that the
  outer launcher SHA did not pin the nested target fetch; `STAGE5_BOOTSTRAP_REF`
  now accepts an immutable 40-character commit SHA so concurrent result pushes
  cannot change the code executed by another lane. No dense training occurred
  before either correction.
- 2026-07-13 inverse-table rebase and dense Phase A results: the rebase reached
  63/64 at cap 3, including 63/64 conditional third-transition success, but
  reduced the locked synthetic guardrail minimum to 0.8125 against the 0.93
  floor; cap 4 did not run and G-alpha remains closed. Dense direct arm B
  reached 470/1792 (26.23%), dense 1.5B direct arm D reached 320/1792
  (17.86%), and dense 0.5B serialized-scratchpad arm C reached 952/1792
  (53.12%). C held 70.0% through depths 1-10 despite training through depth 8,
  then fell to 10.94% on depths 11-14. Its training loss saturated almost
  immediately. Eval-only target `phase_a_checkpoint_comparison` now compares
  all B/C/D step-2000 and step-4000 backups on the same frozen rows, retains
  full compressed continuations and paired predictions, diagnoses C depth-2
  errors, and requires exact reproduction of the landed step-4000 summaries.
- 2026-07-14 Phase A comparison resume correction: all six checkpoint
  evaluations completed, but finalization stopped because the independently
  reloaded D step-4000 greedy BF16 GPU run scored 322/1792 versus the prior
  320/1792, with a maximum absolute depth-stratum delta of 2. This is a
  repeatability discrepancy, not a scientific result or checkpoint mismatch;
  B and C reproduced exactly. The runner now records exact-versus-within-
  envelope repeatability (4 total correct, 3 per depth, 1 parse failure), keeps
  structural checks exact, and reuses completed raw rows after interruption
  instead of rerunning D step-4000.
- 2026-07-13 F9 multi-channel bridge precursor: the architecture remains
  banked while an eval-only battery tests its necessary premise. The locked
  channel basis is the 14 query-head write subspaces from final-recurrent-layer
  `o_proj` column blocks, not residual-coordinate or 2-KV-head slices. M1
  measures loop-1-envelope drift concentration, M2 measures answer-query table
  attention with layer-head identity stability, and M3 removes only the
  selected subspace of the prelude contribution behind a bit-exact-off flag.
  All three use at least 20 dimension-matched random controls. M1/M2 require
  replication on N24 step 6000 and backward recovery; M3 uses N24 active-label
  damage. Activation still requires at least two positive measurements **and**
  staircase reading one. The landed staircase is
  `experiment_stalled_at_matched_dose`, so this battery cannot presently move
  F9 out of BANKED status. Full spec:
  `docs/MULTICHANNEL_BRIDGE_PRECURSOR_SPEC.md`.
- 2026-07-14 Phase A checkpoint comparison finalized: on the same 1,792 frozen
  rows, direct 0.5B scored 464 at step 2,000 and 470 at step 4,000; serialized-
  scratchpad 0.5B scored 930 and 952; direct 1.5B scored 350 and 322. From
  step 2,000 to 4,000, paired net changes were +6 for B (`p=0.771`), +22 for C
  (`p=0.00319`), and -28 for D (`p=0.161`). At step 4,000, C beat B by 482
  paired rows and D by 630; D trailed B by 148. C held 896/1,280 (70.0%) over
  depths 1-10 but fell to 56/512 (10.94%) at depths 11-14. B and C reproduced
  exactly; D's independently reloaded run differed by two total rows and stayed
  inside the locked GPU repeatability envelope. Full synthesis and decision
  questions: `docs/STRATEGY_HANDOFF_INVERSE_STAIRCASE_PHASE_A_20260714.md`.
- 2026-07-14 two-lane surpass and width re-base amendment: arm A's same-reader
  recurrent result is 1,506/1,792 on the same frozen depth-1-through-14 rows,
  versus 470 for dense direct B and 952 for dense serialized-scratchpad C at
  their preregistered step-4,000 checkpoints. A retains 272/512 at depths
  11-14 versus C's 56/512. The original primary gate is A over B, which A
  clears at all 14 depths under the registered one-sided Fisher form; A over C
  is a labeled extension and also clears at all depths. Final receipts must add
  paired row-level tests and must not call Fisher paired. C step 2,000 remains
  an efficiency secondary, not a post hoc replacement for step 4,000. The
  claim is limited to the evaluated synthetic family and recipes; training
  lineage and FLOPs are not matched. G-alpha's deterministic prerequisite is
  re-based to inverse-rendered non-injective validity, while canonical
  forward-table abduction moves to a separate curriculum-science lane. The
  historical 42/112 recovery versus arm-F 3/64 contrast verifies a performance
  gap but not a unique cause; primitive-first and rehearsal are now explicit
  causal tests. F9 remains diagnostic-only and BANKED. Amendment of record:
  `docs/TWO_LANE_SURPASS_REBASE_AMENDMENT_20260714.md`.
- 2026-07-14 multi-channel precursor pilot: the corrected bounded M1/M2
  evaluation completed on the exact N24 step-6000 checkpoint, using 14 frozen
  rows, loops 1-14, and 20 matched random rotations. M1 was beyond the random
  p95 at all nine eligible late loops but reached only 1.370x-1.419x versus
  its locked 2.0x concentration bar, so it is recorded as `smeared`. M2 found
  37 stable layer-head retrieval positions and beat the aggregate random null
  (0.7404 versus 0.5229 p95), a local positive pending its required
  backward-recovery replication. The architecture remains BANKED: M3 did not
  run, the two-positive-measurement rule is unmet, and the staircase reading
  remains `experiment_stalled_at_matched_dose`. Full handoff:
  `docs/stage5_multichannel_bridge_precursor_pilot_20260714_handoff.md`.
- 2026-07-14 active execution queue: F9 is frozen as diagnostic-only. The
  stable-checkpoint path is W3, zero-shot inverse-rendered non-injective
  validity of exact C cap-3 SHA `83767ebf...9ac5`, followed by C1, one
  predeclared 25% forward-rehearsal cap-3 replay. Both launchers force their
  locked run IDs and environment settings to prevent stale Colab variables
  from changing checkpoint lineage or experiment scope. W3's result controls
  whether one bounded deterministic tune may be designed; C1's task,
  retention, and natural-canary verdict controls cap-4 authorization.
- 2026-07-14 F9 amendment: authorize exactly one descriptive
  `backward_recovery` M1/M2 replication with one frozen row per depth, 20
  matched random rotations, the same locked query-head basis and thresholds,
  and an immutable N24 pilot receipt imported only after summary and checkpoint
  SHA validation. This replication cannot activate a bridge intervention:
  F9 remains BANKED unless a replicated battery and an independently priced
  corrected-design reading-one both exist. Bridge-SVD and intervention-derived
  bases are predeclared but not run; F9 closes if a corrected design removes
  the installation-cost problem.
- 2026-07-14 F9 replication verdict: the bounded `backward_recovery`
  replication completed with zero locked measurement votes. M1 remained
  `smeared`: its loop-6-through-8 concentration ratios were `1.37447`,
  `1.37352`, and `1.36925`, all below the predeclared `2.0x` bar. M2 did not
  replicate: zero stable retrieval heads qualified, and aggregate
  concentration `0.272197` was below the matched-random p95 `0.603922`.
  M3 was not run. The aggregate battery therefore records
  `battery_specialization=false`, while the independent staircase remains
  `experiment_stalled_at_matched_dose`. F9 is closed as `remain_banked`; do
  not run M3, search alternative bases, or build a multi-channel bridge on
  these results. The active queue returns to W3 inverse-rendered validity and
  C1 cap-3 rehearsal/retention repair.
- 2026-07-15 deterministic Part 1 closeout: the canonical forward-table
  inverse branch, explicit inverse-table branch, and inverse-rendered W3/W4
  branch are closed as one retention-boundary finding. The isolated operation
  is learnable (`63/64`), but no swept checkpoint entered the joint permitted
  region, and W4 regressed to `208/384` calibration, `0.125` synthetic
  retention minimum, and `171/256` natural canary. F9 is closed, not banked:
  M1 smeared on both checkpoints and M2 failed replication, so its two-of-three
  gate is unsatisfiable. Two mechanical policies now apply: launch-time checks
  must fail before any continuation whose resolved source is below a guardrail
  floor, and clean keepers are frozen assets. Full-block work is legal only in
  non-promotable, source-forbidden disposable measurement branches. Record:
  `docs/PART1_DETERMINISTIC_PROGRAM_CLOSEOUT_20260715.md`.
- 2026-07-15 post-closeout execution contract: one shared L4 target runs the
  non-gating loop-position transfer micro-test and two zero-shot
  branching-relations screens. The micro-test trains inverse positions 1-2,
  requires `>=0.71` on 64 held-out rows each, and measures positions 3-4 on
  128 rows each with locked `0.55`/`0.15` interpretation bands. The branching
  gate uses exact reachable sets, 128 rows per depth 1-4, and requires pooled
  validity `>=0.70` plus every depth `>=0.55` on either the natural N20 verbal
  or N24 symbolic keeper. A double miss does not auto-launch an adapter because
  no numeric near-miss band was preregistered. Phase G-alpha opens only after a
  green screen and powered-margin lock.
- 2026-07-15 paper-one consolidation: replaced the stale June project-status
  narrative and historical master queue with the closed deterministic record
  and post-closeout dependency chain. The manuscript now includes the Phase A
  paired receipts, N24 support-depth frontier, same-reader natural transfer,
  inverse-task acquisition-retention boundary, and F9 closure. A machine-
  readable claim ledger links every supported or open claim to durable local
  artifacts and tests the Phase A arithmetic. Primary-literature review bounds
  novelty to the pretrained-Qwen retrofit, forensic loop-closure repair, and
  controlled mechanism study; recurrent depth, adaptive halting, equilibrium
  depth, transformer conversion, and stochastic recursive reasoning all have
  prior art. Canonical manuscript:
  `docs/PAPER_ONE_DETERMINISTIC_RECURRENT_QWEN_20260715.md`.
- 2026-07-15 Part 1 pivot session completed. The disposable loop-position arm
  passed its trained-position prerequisite at step 500 (`0.71875`, `1.0`) but
  hit the registered synthetic hard stop at step 1,000 (`0.8125 < 0.93`), so
  transfer positions 3-4 were not measured and all disposable checkpoints were
  deleted. The natural N20 verbal branching screen passed at `389/512 =
  75.98%`, with depth accuracies `127/128`, `95/128`, `87/128`, and `80/128`.
  The N24 symbolic screen narrowly missed at `355/512 = 69.34%`, with depth 3
  at `67/128 = 52.34%`. The frozen natural keeper satisfies the deterministic
  substrate gate; no adapter is needed. Phase G-alpha is ready for the powered
  margin lock and then launch. Receipt:
  `outputs/stage5/stage5_part1_closeout_pivot_20260715/summary.json`.
- 2026-07-17 parameter-efficient closure: corrected-loop R16 LoRA plus the
  repaired bridge trained 7,613,953 parameters and first cleared the registered
  depth-1-through-4 gate at 4,000 cumulative steps. At step 6,000 it scored
  `64/64`, `64/64`, `60/64`, and `53/64`; the pretrained-base hash was
  unchanged and the arithmetic canary improved from `60/64` to `61/64`.
  This is a bounded synthetic installation result, not full-block parity:
  the full-block reference remained 3 rows better at depth 3 and 6 rows better
  at depth 4. The attached controller failed, selecting loop 2 for every row
  and scoring `191/256` versus `241/256` under forced depth. Receipt:
  `outputs/stage5/stage5_peft_ponder_closure_20260717_182113/summary.json`.
- 2026-07-17 bounded selector closure: the frozen N24 step-6,000 mechanism
  retained `759/768 = 98.83%` oracle forced-depth answer accuracy, but the
  controller failed both registered arms. S1 supervised stated-depth reading
  selected the correct depth on `70/768 = 9.11%` and the correct final answer
  on `73/768 = 9.51%`. S2 outcome-only Ponder training saturated at depth 12
  for all 768 rows, with zero depth correlation and `173/768 = 22.53%`
  selected-answer accuracy. Startup gradients were live; late zero gradients
  were boundary saturation, not a graph cut. The line is closed as a bounded
  negative. Receipt:
  `outputs/stage5/stage5_depth_selector_bounded_20260717_204109/summary.json`.
- 2026-07-18 Phase G oracle-interface decision: the additive A0
  `NO-CHANNEL` verdict is ratified. One terminal, frozen-keeper capacity probe
  is authorized before any further variational work: parameter-matched
  additive and FiLM routes receive the true next selected-chain symbol at
  every transition and train only `oracle_reentry_conditioner.*` with
  per-loop chain CE. The locked held-out set has 106 variants, 32 prompt
  groups, and 305 transitions. An arm must clear non-default control `0.85`,
  overall control `0.90`, legality `0.95`, terminal validity `0.71`, exact
  zeroed identity, and exact frozen lineage. No KL, stochastic sampling,
  coverage, selector, halting, particles, or SVGD is part of this probe, and
  no outcome automatically authorizes a successor. Spec:
  `docs/STAGE5_PHASE_G_ORACLE_INTERFACE_PROBE_SPEC_20260718.md`.
- 2026-07-18 Phase G oracle-interface result: both parameter-matched routes
  failed the locked held-out gate, producing the terminal reading `BOTH_FAIL`
  and interpretation `reentry_conditioning_closed_on_frozen_substrate`.
  Additive controlled `31/216 = 14.35%` of non-default transitions and
  `94/305 = 30.82%` overall; FiLM controlled `34/216 = 15.74%` non-default
  and `87/305 = 28.52%` overall. Transition legality was `54.10%` and
  `56.39%`, respectively, versus the locked `95%` floor. FiLM alone retained
  terminal validity above the `71%` floor (`79/106 = 74.53%`), but neither
  route approached the command-control thresholds. Both preserved exact
  zero-conditioning identity and exact frozen-keeper lineage, with 1,500
  gradient-liveness and frozen-gradient assertions per arm. No variational
  successor, coverage, selection, halting, particles, or SVGD run is
  authorized automatically. Receipt:
  `outputs/stage5/stage5_phase_g_oracle_interface_probe_20260718/summary.json`.
- 2026-07-19 Arm E adapter-parity battery closed. The R16-plus-bridge arm had
  already matched Arm A in pooled frozen depth accuracy (`1501/1792` versus
  `1506/1792`, paired `p=0.813`) while failing registered profile parity only
  in the far tail, with frontier `11.56` or `1.44x` trained support. E3a found
  minimal zero-shot verbal transfer: relay `249/1536 = 16.21%`, pointer
  `264/1536 = 17.19%`. E2 was a strong persistence positive after 1,000
  outcome-only steps: active diagonal `636/640 = 99.38%`, continuation
  `380/384 = 98.96%`, and zero holds. E4 hard-stopped at step 100: Tier-1
  remained green at `59/64`, but the own-baseline natural canary fell from
  `60/256` to `49/256`, the synthetic minimum was `0.09375`, and inverse
  acquisition was only `2/64`; the registered verdict is `wall_holds`. E3b
  remains unrun and unauthorized as a separate natural-training question, not
  an Arm E closure requirement. Consolidated handoff:
  `docs/ARM_E_ADAPTER_PARITY_CLOSURE_HANDOFF_20260719.md`.
- 2026-07-19 Paper One finishing receipts closed without new experiments.
  The loop-1 ARC guardrail battery is recorded as bounded noninferiority across
  four promoted keeper checkpoints with a `-0.03` margin and Bonferroni
  correction over eight primary comparisons. Early-era configuration,
  aggregate halting archaeology, the zero-gradient bridge failure, and the
  retired PC-tail damper are traced to their source artifacts. Primary-source
  checks verified HRM and TRM metadata, retained hedged procedural-knowledge
  wording, rejected mathematical equivalence between the retired damper and
  Parcae, and narrowed the LoRA novelty language in light of McLeish et al.'s
  cited prior adapter retrofit. Canonical Arm E artifacts supersede provisional
  handoff values: persistence `636/640` with `380/384` continuation; zero-shot
  relay/pointer `249/1536` and `264/1536`; E4 `wall_holds`, not a joint pass.
  Figure 4 now has five curves, trained-support shading, and the Arm A/E
  depth-11-to-12 crossover. Receipt:
  `docs/PAPER_ONE_FINISHING_RECEIPTS_20260719.md`.
- 2026-07-21 E3b adapter-budget verbal transference completed with a
  preregistered guardrail truncation. This entry supersedes the 2026-07-19
  note that E3b was unrun. At the last matched checkpoint (step 3,000), the
  installed R16-plus-bridge arm scored `1852/3072 = 60.29%` versus
  `1282/3072 = 41.73%` for fresh R16 surgery, a `+18.55` point paired
  advantage (763 installed-only wins, 193 control-only wins,
  `p = 9.74e-81`). Gains were concentrated at depths 3-11 and were present on
  pointer, which was held out from verbal training. The installed arm
  completed 6,000 steps, first crossing 0.71 pooled at step 4,000 before a
  nonmonotonic endpoint of 64.68%; all rehearsed synthetic strata remained at
  or above 94.53%, and Tier-1 remained at 59-60/64. The fresh arm stopped at
  step 3,000 after a near-boundary `60/64 -> 58/64` canary change; its paired
  exact `p = 0.50`, so the stop is honored but not interpreted as demonstrated
  regression. The registered 6,000-step T-versus-S endpoint is unavailable.
  Receipt:
  `outputs/stage5/stage5_adapter_verbal_transference_e3b_20260720/summary.json`.
  Handoff:
  `docs/ARM_E3B_ADAPTER_VERBAL_TRANSFERENCE_HANDOFF_20260721.md`.
- 2026-07-22 Phase A dense-reader audit corrected a semantic boundary error in
  the registered dense evaluator. Dense models were not trained to emit EOS
  after their short completion, and the old reader could overwrite the first
  completed response with the last `Answer:` marker in later untrained
  continuation. Re-reading the archived, hash-locked continuations at the
  first completed response changed B from `470/1792` to `496/1792`, C from
  `952/1792` to `1292/1792`, and D from `322/1792` to `656/1792`. The two
  flagged cells were artifacts: D depth 1 is `125/128`, not `13/128`, and C
  depth 2 is `128/128`, not `57/128`. The corrected result strengthens the
  crossover reading: C is perfect through depth 9 and `127/128` at depth 10,
  then falls to `13/128` at depth 11 and zero at depths 12-14, while A retains
  `272/512` over depths 11-14. A remains higher overall (`1506/1792` versus
  `1292/1792`; paired A-only `262`, C-only `48`, two-sided exact
  `p=7.81e-37`). No checkpoint or model output changed. Receipt:
  `outputs/stage5/stage5_phase_a_dense_reader_audit_20260722/summary.json`.
  The corrected preregistered A-over-B gate passes over depths 2-14; depth 1
  is a `128/128` tie. Corrected 2,000-to-4,000 net changes are B `+1`, C `+1`,
  and D `+13`, with no paired two-sided result below `0.50`. Paper-One marker
  closure: `docs/PAPER_ONE_DENSE_READER_AUDIT_CLOSURE_20260722.md`.
- 2026-07-22 Paper Two record reconciliation: the stale live Phase G queue is
  closed on the tested frozen high-level re-entry interface. The initial
  guided-width result is exploratory and non-identifying because 2,048 rows
  contained zero repeated-prompt groups. The corrected posterior-control arms
  reached only `22.6%` to `23.6%` target fidelity and `4/32` switching groups
  against `24/32` required. Forced residual amplification retained the
  registered `NO-CHANNEL` verdict. Parameter-matched terminal additive and
  FiLM conditioners supplied with the true next symbol retained the registered
  held-out `BOTH_FAIL` verdict. The active queue is now a post-hoc train-row
  readout, executable T0 contracts, and a locked T1 preregistration. The
  train-row readout cannot alter `BOTH_FAIL`, and the coded intra-block probe
  remains unrun. Canonical ledger: `docs/paper2_claim_evidence_ledger.json`.
- 2026-07-23 Paper Two WP1 oracle train-row readout: the frozen additive and
  FiLM EMA conditioners were evaluated without mutation on both a seeded,
  depth-stratified 106-variant cohort (`305` transitions) and all `1,899`
  training variants (`5,617` transitions). Matched non-default transition
  control was additive `48/225 = 21.33%` and FiLM `59/225 = 26.22%`. The full
  readout was additive `820/4,064 = 20.18%` and FiLM `947/4,064 = 23.30%`.
  Under the locked descriptive bands, both full-cohort results are
  `did_not_fit_command_mapping`. The result localizes the failure to terminal
  interface fit rather than held-out generalization; it is post-hoc and does
  not change the registered `BOTH_FAIL` verdict. Keeper and conditioner hashes
  were identical before and after. Receipt:
  `outputs/stage5/stage5_phase_g_oracle_train_readout_20260722/summary.json`.
- 2026-07-23 Paper Two Phase T0 preflight: all five no-training contracts
  passed on an NVIDIA L4. Qwen's tokenizer length (`151,665`) was aligned to
  its padded model vocabulary (`151,936`) without adding model parameters;
  the three controls then occupied genuinely new IDs `151936`-`151938`.
  The tied input/LM-head policy and every old row were preserved, exactly
  `2,688` parameters were added, control logits were excluded from visible
  generation, inactive one-loop logit difference was `0.0 < 1e-3`, and
  requested/executed/selected loops all equaled four. No training occurred and
  no checkpoint was written. Receipt:
  `outputs/stage5/stage5_paper2_internal_token_t0_preflight_20260722/summary.json`.
- 2026-07-23 Paper Two T1 design review: external primary-source review and
  gate analysis retained the four-part causal design but identified four
  preregistration details that must be resolved before lock: exact selected
  depth rather than micro transition accuracy for gate 3, class-balanced
  continue/stop supervision, intervention at the control logits for gate 4,
  and integer/statistical readings for the three-point and 90-percent gates.
  The review also pre-writes failure-localized successor paths: frozen
  post-hoc control, randomized-depth backbone training, convergence exits,
  shortcut consistency, exposure-bias repair, and content-determined synthetic
  halting. Design memo:
  `docs/PAPER2_T1_DESIGN_RATIONALE_AND_FORWARD_STRATEGY_20260723.md`.
- 2026-07-23 Paper Two Draft 3 pivot: the positive-seeking program now targets
  useful natural-data training in the Qwen series. Registered T1 is descoped
  to one full-block T1-lite actuator qualification. The already authorized
  adapter P0 grid remains uncitable and may select candidate loss constants,
  but it is not matched-lineage evidence for T1-lite. D0 speculative-decoding
  depth recoverability is next after the T1-lite verdict and its own locked
  preregistration. Paper Two packaging is deferred until the D0 pilot. Arm G
  remains closed under `NO-CHANNEL` and `BOTH_FAIL`; width and natural-trace
  training remain unauthorized. Record:
  `docs/PAPER2_EXPERIMENTAL_PLAN_DRAFT3_20260723.md`.
- 2026-07-24 Paper Two T1 P0 calibration: all ten fixed adapter cells completed
  on the dedicated 256-row pilot set. Seven of nine controlled cells cleared
  both `0.60` recall floors. The fixed selection rule chose
  `lambda0p5_ratio1`: stop recall `177/256 = 0.6914`, continue recall
  `885/896 = 0.9877`, exact selected depth `166/256 = 0.6484`, and answer
  accuracy `151/256 = 0.5898`, versus `136/256 = 0.5312` for lambda zero.
  The selected normalized class weights are continue `1.0`, stop `1.0`.
  Pretrained embedding rows remained hash-identical, and the A-P loop-target
  alignment contract passed. P0 is uncitable, uses a different trainable
  lineage from registered full-block T1-lite, and calibrates constants only.
  Full-block T1-lite remains locked pending strategy ratification and a committed
  `locked_before_training` preregistration. Receipt:
  `outputs/stage5/stage5_paper2_internal_token_t1_p0_letter_v2_20260724/summary.json`.
  Handoff:
  `docs/PAPER2_T1_P0_CALIBRATION_STRATEGY_HANDOFF_20260724.md`.
- 2026-07-24 Phase T1-lite lock: strategy ratified the P0-selected control-loss
  lambda `0.5`, equal class weights, the 10,500-step full-block curriculum,
  the seed/replication policy, the `1005/1024` reference and checkpoint hash,
  and all four unchanged gates. The standalone 1,500-step confirmation cell
  was withdrawn as stage-mismatched and replaced by descriptive liveness
  readouts at steps 500, 2,500, 6,500, and 8,500. The only boundary abort
  requires both flat stage control loss and exactly zero stop recall on trained
  pilot depths. A manifest audit corrected the stale gated-row placeholder to
  the canonical Phase A hash `7aa673d0...1fdcbe`; a disjoint 512-row
  calibration manifest was locked at seed `2026072401`. Human contract:
  `docs/PHASE_T1_LITE_PREREGISTRATION_DRAFT4_20260724.md`. Machine contract:
  `outputs/stage5/stage5_paper2_t1_lite_preregistration_20260724/preregistration.json`.
  No registered training had occurred at lock.
- 2026-07-24 Paper Two S1 tokenizer audit: the official Qwen2.5-0.5B and
  Qwen3-0.6B tokenizer artifacts share all 151,643 model-vocabulary entries
  with identical IDs. Qwen3 alone adds `<think>` at 151667, `</think>` at
  151668, and a tool-response pair. Qwen3 teacher traces are therefore
  transferable to a Qwen2.5 drafter as text after retokenization, but the raw
  think-token IDs and embedding rows are not compatible. Receipt:
  `outputs/stage5/stage5_paper2_s1_tokenizer_audit_20260724/summary.json`.
- 2026-07-25 registered T1-lite result: the final-step continuous-EMA primary
  returned a registered negative with `202/1024` forced answers, `104/1024`
  self-halted answers, and `128/1024` exact selections. The raw final-step
  secondary achieved `967/1024` forced and self-halted answers and exact
  selection on `1024/1024`; it missed the `975/1024` preservation floor by
  eight rows. The full causal override sweep was exact on `5632/5632`.
  Receipt: `outputs/stage5/stage5_paper2_t1_lite_20260724/summary.json`.
- 2026-07-25 T1-lite EMA localization: reciprocal group transplants localized
  the endpoint divergence to the recurrent block. EMA plus the raw block
  restored `256/256` exact selection, while raw plus the EMA block reduced it
  to `32/256`. Linear interpolation remained exact through alpha `0.25` and
  collapsed by `0.50`. The EMA scalar recurrence passed to `1.43e-8`; the
  registered negative is unchanged. Receipt:
  `outputs/stage5/stage5_paper2_t1_lite_ema_audit_20260725/summary.json`.
- 2026-07-25 T1-lite-R lock: seed 1 is authorized under the standing
  replication rule. The only amended policy factor is raw final-step primary;
  continuous EMA and stage-reset EMA at `0.999` are passive descriptive
  shadows. All original data, curriculum, loss, optimizer, and four gates are
  structurally unchanged. Atomic hashed raw and shadow states are required at
  steps `500`, `2500`, `6500`, `8500`, and `10500`. Lock commit `ae2793ac`.
- 2026-07-25 COCONUT composite no-training preflight: eight of eleven
  contracts passed, including exact H=0 identity, horizontal and prompt
  gradient reachability, adapter transparency, grid accounting, anomaly
  detection, and checkpointing. RG-4 narrowly missed its one-epsilon
  derivative tolerance, RG-5 rejected sliced-cache strict equivalence despite
  gradient cosine `1.0`, and RG-11 measured bf16/fp32 gradient cosine
  `0.983584 < 0.99`. Sliced cache is retired; the bounded numerical follow-up
  is authorized; RG-12 remains unauthorized. Receipt:
  `outputs/stage5/stage5_coconut_composite_rg1_rg11_20260725/summary.json`.
- 2026-07-25 D0 build-only implementation: the updated draft, exact-match and
  signal contract, depth-calibration branch, added-token probability masking,
  and depth-recoverable-fraction scorer are implemented and unit-tested. The
  CPU receipt records zero models loaded, teacher forwards, optimizer steps,
  and checkpoints. D0 remains unlocked; GPU labeling and training are not
  authorized. Receipt:
  `outputs/stage5/stage5_paper2_d0_build_only_20260725/summary.json`.
- 2026-07-25 T1-lite-R launch preflight correction: implementation commit
  `368bd2a0` stopped before model load or training because the locked original
  preregistration hash had been computed over a Windows CRLF checkout while
  Colab checked out Git's LF bytes. The registered `4e55...` hash remains
  historical metadata; launch integrity now verifies the equivalent canonical
  LF hash `69cc...` after newline normalization. No experimental factor or
  registered attempt changed. Receipt:
  `docs/PAPER2_T1_LITE_R_LAUNCH_HASH_CORRECTION_20260725.md`.
