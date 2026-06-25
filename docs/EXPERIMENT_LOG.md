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
- After Stage 4 recovery, run `debiased_benchmark_suite`, then
  `dense_mcq_trace_sft_control`. This is the minimum evidence chain for asking
  whether recurrence adds value beyond the same training data.
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
