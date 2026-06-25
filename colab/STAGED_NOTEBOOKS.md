# Single-Runtime Colab Runbook

The preferred workflow is **one Colab notebook attached to one runtime** plus
`STAGE5_CURRENT_A100_TARGET`. Do not hop between old stage notebooks while a
paid runtime is active. The older split notebooks remain in `colab/` for
provenance, but the maintained execution path is the bootstrap target queue in
[`NEXT_COLAB_SEQUENCE.md`](NEXT_COLAB_SEQUENCE.md).

For the shortest current instruction, use
[`CURRENT_A100_ACTION.md`](CURRENT_A100_ACTION.md). For the full phase order,
use [`../docs/PROGRAM_TRACK_MASTER_SEQUENCE.md`](../docs/PROGRAM_TRACK_MASTER_SEQUENCE.md).

## Current Target

The current target is:

```text
traced_sft_competence_preserving_pipeline
```

The program has moved past the immediate re-entry repair gate. Stage 3 made the
bridge/re-entry adapter live; Stage 4 kept re-entry health sane after recovery
SFT. The active question is whether deterministic recurrent competence can be
preserved and recovered enough to pass the balanced recurrent-vs-base gate.

Use the fresh launcher in `CURRENT_A100_ACTION.md`. It fetches the tracked
launcher from GitHub, hard-resets the repo, mounts Drive, and executes the
current target.

## Current Sequence

1. `traced_sft_competence_preserving_pipeline`
   - Runtime: L4/T4 preferred; A100 only if already attached.
   - Gate: checkpoint restore preflight, finite ARC-mix summary, and proxy pass
     before any full assessment spend.
2. `review_stage5_competence_pipeline.py`
   - Runtime: CPU/local.
   - Gate: route according to the planner-selected next action.
3. `debiased_benchmark_suite` or recovery full assessment
   - Runtime: L4/T4.
   - Gate: recurrent must be base-competitive under debiased MCQ scoring.
4. `dense_mcq_trace_sft_control`
   - Runtime: L4/T4.
   - Gate: architecture lift requires recurrent-vs-dense evidence under the
     same curriculum.
5. `effective_pathways_diagnostic` / `candidate_conversion_diagnostic`
   - Runtime: deferred until the deterministic Phase 1 gate passes.
6. Phase 3 particles/SVGD
   - Runtime: deferred until correct-bearing breadth exists.

## CPU/API Parallel Work

`claim_curriculum_scaleup_cpu` remains a valid CPU/API data-prep target. It can
build or resume the claim-sized direct/deep curriculum shard while GPU work
focuses on deterministic recurrent competence. It is not a GPU gate and should
not hold an attached paid runtime open.

Provider calls should stay disabled until provider API secrets and concrete
model ids are configured. Use a tiny provider smoke before filling all pending
rows.

## Optional Information-Value Probes

`model_viability_probe` and `model_viability_queue` are standing information
probes for larger Qwen-family checkpoints. They do not unlock Stage 4,
particles, or release claims by themselves.

## Historical Notebooks

The following notebooks are retained for provenance and old-run reproduction,
not as the current front-of-queue path:

1. `00_single_a100_runbook.ipynb`
2. `00_stage_launcher.ipynb`
3. `01_stage1_svgd_seed_replication.ipynb`
4. `02_stage2_benchmark_harness.ipynb`
5. `03_stage3_hf_packaging.ipynb`
6. `04_stage4_modified_opus_finetune.ipynb`
7. `05_stage5_benchmarks.ipynb`
8. `06_stage6_writeup_and_release.ipynb`
9. `07_stage5_full_arc_assessment.ipynb`
10. `08_stage5_safe_continue.ipynb`
11. `09_stage5_arc_mix_recovery_once.ipynb`
12. `10_stage5_direct_preservation_precheck.ipynb`
13. `11_stage5_direct_preservation_g4_auto.ipynb`

If a historical notebook conflicts with `CURRENT_A100_ACTION.md`, trust the
current action card and the tracked bootstrap target.
