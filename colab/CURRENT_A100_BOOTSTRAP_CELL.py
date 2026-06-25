import base64, json, os, time, urllib.request
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
REF = "main"
BOOTSTRAP_VERSION = "sha_resolved_nested_fetch_v3"

# Safe default: verify Drive/checkpoint visibility on a CPU/cheap runtime.
# Other options:
#   "programmatic_curriculum_cpu" - generate/publish the direct/deep curriculum gate on CPU.
#   "safe_continue_dry_run" - fetch safe-continue but do not spend GPU.
#   "safe_continue_execute" - fetch safe-continue and opt in to the guarded paid action.
#   "arc_challenge_mcq_debias_confirm" - bounded no-training cyclic MCQ confirmation on ARC-Challenge.
#   "debiased_benchmark_suite" - bounded ARC-Challenge/GPQA-lite benchmark with debiased MCQ scoring.
#   "depth_balanced_benchmark" - balanced ARC content/cyclic benchmark for learned-depth checkpoints.
#   "arc_mix_offset_confirm" - bounded ARC-Easy/Challenge offset-256 confirmation for the latest ARC-mix checkpoint.
#   "arc_mix_offset_then_depth_chain" - offset confirmation, then learned-depth ARC-mix SFT only if confirmed.
#   "arc_mix_depth_routing_probe" - bounded learned-depth ARC-mix SFT probe from the latest recovered checkpoint.
#   "effective_pathways_diagnostic" - bounded deterministic recurrent pathway-collapse diagnostic.
#   "candidate_conversion_diagnostic" - bounded particle-noise candidate conversion with correctness-split pathways.
#   "reentry_drift_diagnostic" - bounded read-only recurrent loop-closure drift diagnostic.
#   "reentry_norm_diagnostic" - bounded eval-only loop re-entry RMS normalization comparison.
#   "reentry_repair_smoke" - bounded trainable bridge/re-entry repair smoke.
#   "capability_ladder_mcq_probe" - bounded Qwen 0.5B/1.5B/3B ARC MCQ scoring for depth-label data.
#   "capability_ladder_7b_mcq_probe" - high-memory Qwen 0.5B/1.5B/3B/7B ARC MCQ scoring for depth-4 data.
#   "capability_ladder_7b_trace_chain" - high-memory 7B ladder probe followed by trace-job build.
#   "capability_ladder_trace_jobs_cpu" - CPU-only trace-job build from latest capability ladder probe.
#   "capability_ladder_trace_responses_cpu" - CPU/network provider responses for trace jobs.
#   "capability_ladder_trace_collect_cpu" - CPU-only trace-response collection into gated SFT data.
#   "capability_ladder_trace_response_collect_cpu" - CPU/network provider responses then immediate collection.
#   "capability_ladder_local_hf_trace_collect" - GPU local-HF responses then immediate collection.
#   "capability_ladder_local_hf_trace_sft" - GPU local-HF traces, collection, then bounded recurrent SFT.
#   "traced_sft_scale64_benchmark" - benchmark the completed scale64 traced SFT checkpoint.
#   "traced_sft_direct_preservation_precheck" - quick loop1/base preservation check, no training.
#   "traced_sft_direct_preservation_probe" - content-route direct preservation from the scale64 checkpoint.
#   "traced_sft_direct_preservation_recover_only" - publish surviving direct-preservation output without rerunning training.
#   "traced_sft_direct_preservation_confirm" - larger loop-1 ARC confirmation after direct preservation passes.
#   "traced_sft_surface_alignment_repair" - repair ARC-Easy content/cyclic surface mismatch.
#   "traced_sft_score_alignment_repair" - repair ARC-Easy content route with direct MCQ score CE.
#   "dense_mcq_trace_sft_control" - train/evaluate standard Qwen LoRA on the same traced MCQ curriculum.
#   "traced_sft_competence_preserving_pipeline" - mixed recovery after confirmation still trails base.
#   "traced_sft_depth_router_after_direct_preserve" - learned-depth continuation from a passed direct-preservation checkpoint.
#   "traced_capability_ladder_sft" - GPU Phase 1 SFT from the latest gate-ready traced capability ladder.
#   "direct_preservation_probe" - bounded max_loops=1 base-preservation training probe.
#   "depth_sweep_heldout" - L4/T4 held-out ARC tail loop-depth sweep for routing validation.
#   "model_viability_probe" - no-training Qwen model scale probe; defaults to 1.5B and is env-configurable for 3B+.
#   "model_viability_queue" - queued no-training Qwen 3B/7B probes with VRAM-aware skipping.
TARGET = os.environ.get("STAGE5_CURRENT_A100_TARGET", "preflight")
SOURCE_SUMMARY_OVERRIDE = os.environ.get("STAGE5_CURRENT_A100_SOURCE_SUMMARY", "").strip()

if TARGET == "traced_sft_scale64_benchmark" and os.environ.get(
    "STAGE5_ALLOW_STALE_SCALE64_BENCHMARK", "0"
).strip().lower() not in {"1", "true", "yes", "y"}:
    print(
        "traced_sft_scale64_benchmark is complete; rerouting to "
        "traced_sft_direct_preservation_probe. Set "
        "STAGE5_ALLOW_STALE_SCALE64_BENCHMARK=1 to intentionally rerun "
        "the benchmark-only target.",
        flush=True,
    )
    TARGET = "traced_sft_direct_preservation_probe"

TARGETS = {
    "preflight": {
        "path": "colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.py",
        "markers": [
            "stage5_drive_checkpoint_preflight",
            "checkpoint_preflight",
            "STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY",
            "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY",
            "drive.mount",
            "runtime.unassign",
            "colab/check_stage5_a100_go_no_go.py",
            "colab/run_stage5_next_action.py",
            "next_action_guard",
            "stage5_current_source_summary.txt",
            "Using current source summary pointer",
        ],
        "env": {},
    },
    "safe_continue_dry_run": {
        "path": "colab/STAGE5_SAFE_CONTINUE_CELL.py",
        "markers": [
            "STAGE5_SAFE_CONTINUE_RUN_A100_ACTION",
            "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY",
            "RUN_A100_ACTION",
            "colab/check_stage5_a100_go_no_go.py",
            "colab/run_stage5_next_action.py",
            "Skipping requirements install because no paid action will execute.",
        ],
        "env": {"STAGE5_SAFE_CONTINUE_RUN_A100_ACTION": "0"},
    },
    "programmatic_curriculum_cpu": {
        "path": "colab/STAGE5_PROGRAMMATIC_CURRICULUM_CELL.py",
        "markers": [
            "REFUSE_GPU_RUNTIME",
            "ALLOW_GPU_RUNTIME_FOR_CPU_WORK",
            "training/run_programmatic_curriculum_pipeline.py",
            "training/check_curriculum_sft_gate.py",
            "colab/publish_stage5_curriculum_gate.py",
            "REQUIRE_DRIVE_BACKUP_FOR_PUBLISH",
            "PUBLISH_GATE_TO_GITHUB",
            "stage5_current_source_summary",
            "PROGRAMMATIC_CURRICULUM_CELL_VERSION",
            "shutil.which(\"nvidia-smi\")",
            "FileNotFoundError",
            "OSError",
            "Refusing to run CPU-only programmatic curriculum generation",
        ],
        "env": {},
    },
    "safe_continue_execute": {
        "path": "colab/STAGE5_SAFE_CONTINUE_CELL.py",
        "markers": [
            "STAGE5_SAFE_CONTINUE_RUN_A100_ACTION",
            "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY",
            "STAGE5_SAFE_CONTINUE_PREFER_TRAINING_SOURCE",
            "RUN_A100_ACTION",
            "mount_drive_for_paid_action",
            "tests/test_stage5_routing_repair.py",
            "tests/test_filter_mcq_sft_by_eval.py",
            "tests/test_mcq_debias.py",
            "colab/run_stage5_next_action.py",
        ],
        "env": {
            "STAGE5_SAFE_CONTINUE_RUN_A100_ACTION": "1",
            "STAGE5_SAFE_CONTINUE_PREFER_TRAINING_SOURCE": "1",
        },
    },
    "arc_challenge_mcq_debias_confirm": {
        "path": "colab/STAGE5_ARC_CHALLENGE_MCQ_DEBIAS_CELL.py",
        "markers": [
            "STAGE5_ARC_CHALLENGE_MCQ_DEBIAS_CELL_VERSION",
            "ARC-Challenge",
            "STAGE5_MCQ_DEBIAS_QUIET_EVAL",
            "STAGE5_MCQ_DEBIAS_RESUME_EXISTING",
            "STAGE5_MCQ_DEBIAS_PUSH",
            "colab/run_stage5_mcq_debias_diagnostic.py",
            "colab/assess_stage5_mcq_debias_pair.py",
            "colab/apply_stage5_mcq_scoring_policy.py",
            "tests/test_mcq_debias.py",
            "tests/test_stage5_next_plan.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "debiased_benchmark_suite": {
        "path": "colab/STAGE5_DEBIASED_BENCHMARK_SUITE_CELL.py",
        "markers": [
            "STAGE5_DEBIASED_BENCHMARK_SUITE_CELL_VERSION",
            "STAGE5_DEBIASED_MOUNT_DRIVE_FIRST",
            "STAGE5_DEBIASED_USE_LEARNED_LOOP_CONTROL",
            "STAGE5_BENCHMARK_SCORE_TARGETS",
            "label,content_question_only,cyclic_label_aggregated",
            "cyclic_label_aggregated",
            "permutation_mean",
            "STAGE5_DEBIASED_BENCHMARKS",
            "arc_challenge,gpqa_lite",
            "colab/run_stage5_benchmark_suite.py",
            "colab/assess_stage5_benchmark_suite.py",
            "tests/test_stage5_benchmark_suite.py",
            "tests/test_stage5_benchmark_assessment.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "depth_balanced_benchmark": {
        "path": "colab/STAGE5_DEBIASED_BENCHMARK_SUITE_CELL.py",
        "markers": [
            "STAGE5_DEBIASED_BENCHMARK_SUITE_CELL_VERSION",
            "STAGE5_DEBIASED_MOUNT_DRIVE_FIRST",
            "STAGE5_DEBIASED_USE_LEARNED_LOOP_CONTROL",
            "Skipping upfront Drive mount",
            "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL",
            "content_question_only",
            "cyclic_label_aggregated",
            "permutation_mean",
            "colab/run_stage5_benchmark_suite.py",
            "colab/assess_stage5_benchmark_suite.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_DEBIASED_MOUNT_DRIVE_FIRST": "0",
            "STAGE5_DEBIASED_BENCHMARKS": "arc_easy,arc_challenge",
            "STAGE5_DEBIASED_ARC_EASY_LIMIT": "512",
            "STAGE5_DEBIASED_ARC_CHALLENGE_LIMIT": "512",
            "STAGE5_DEBIASED_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated",
            "STAGE5_DEBIASED_USE_LEARNED_LOOP_CONTROL": "1",
            "STAGE5_BENCHMARK_ASSESS_ALLOWED_NEGATIVE_DELTA": "0",
        },
    },
    "arc_mix_offset_confirm": {
        "path": "colab/STAGE5_DEBIASED_BENCHMARK_SUITE_CELL.py",
        "markers": [
            "STAGE5_DEBIASED_BENCHMARK_SUITE_CELL_VERSION",
            "STAGE5_DEBIASED_ARC_CHALLENGE_OFFSET",
            "STAGE5_DEBIASED_ARC_EASY_OFFSET",
            "STAGE5_DEBIASED_SCORE_TARGETS",
            "content_question_only",
            "cyclic_label_aggregated",
            "permutation_mean",
            "colab/run_stage5_benchmark_suite.py",
            "tests/test_stage5_benchmark_suite.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_DEBIASED_BENCHMARKS": "arc_easy,arc_challenge",
            "STAGE5_DEBIASED_ARC_EASY_LIMIT": "256",
            "STAGE5_DEBIASED_ARC_CHALLENGE_LIMIT": "256",
            "STAGE5_DEBIASED_ARC_EASY_OFFSET": "256",
            "STAGE5_DEBIASED_ARC_CHALLENGE_OFFSET": "256",
            "STAGE5_DEBIASED_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated",
            "STAGE5_BENCHMARK_SUITE_RUN_ID": "stage5_content_arcmix_qonly_optiontext_arc256_offset256_confirm",
        },
    },
    "arc_mix_depth_routing_probe": {
        "path": "colab/STAGE5_ARC_MIX_DEPTH_ROUTING_CELL.py",
        "markers": [
            "STAGE5_ARC_MIX_DEPTH_ROUTING_CELL_VERSION",
            "target_loop_count ARC-Easy=1 ARC-Challenge=3",
            "STAGE5_ARC_MIX_USE_LEARNED_LOOP_CONTROL",
            "STAGE5_ARC_MIX_LOOP_CONTROL_CE_WEIGHT",
            "STAGE5_ARC_MIX_HALT_TARGET_NLL_WEIGHT",
            "STAGE5_ARC_MIX_PROMPT_STYLE",
            "question_only",
            "option_text",
            "colab/run_stage5_balanced_arc_mix_gate.py",
            "tests/test_stage5_balanced_arc_mix_gate.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "arc_mix_offset_then_depth_chain": {
        "path": "colab/STAGE5_ARC_MIX_OFFSET_THEN_DEPTH_CELL.py",
        "markers": [
            "STAGE5_ARC_MIX_OFFSET_THEN_DEPTH_CELL_VERSION",
            "STAGE5_ARC_MIX_CHAIN_EXECUTE_DEPTH",
            "STAGE5_ARC_MIX_CHAIN_ALLOWED_NEGATIVE_DELTA",
            "STAGE5_ARC_MIX_CHAIN_MIN_EXAMPLES",
            "STAGE5_ARC_MIX_CHAIN_MOUNT_DRIVE_FIRST",
            "STAGE5_ARC_MIX_CHAIN_RUN_POST_DEPTH_DEBIASED_GATE",
            "Offset gate: ARC-Easy and ARC-Challenge, offset=256",
            "target_loop_count ARC-Easy=1 ARC-Challenge=3",
            "Post-depth gate: debiased cyclic scoring is primary",
            "colab/run_stage5_arc_mix_offset_then_depth.py",
            "tests/test_stage5_offset_then_depth.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_ARC_MIX_CHAIN_EXECUTE_DEPTH": "1",
            "STAGE5_ARC_MIX_CHAIN_ALLOWED_NEGATIVE_DELTA": "0",
            "STAGE5_ARC_MIX_CHAIN_MIN_EXAMPLES": "256",
            "STAGE5_ARC_MIX_CHAIN_MOUNT_DRIVE_FIRST": "0",
            "STAGE5_ARC_MIX_CHAIN_RUN_POST_DEPTH_DEBIASED_GATE": "1",
            "STAGE5_ARC_MIX_CHAIN_POST_DEPTH_MIN_EXAMPLES": "128",
        },
    },
    "effective_pathways_diagnostic": {
        "path": "colab/STAGE5_EFFECTIVE_PATHWAYS_CELL.py",
        "markers": [
            "STAGE5_EFFECTIVE_PATHWAYS_CELL_VERSION",
            "stage5_effective_pathways_v1",
            "STAGE5_EFFECTIVE_PATHWAYS_CHECKPOINT",
            "eval/eval_effective_pathways.py",
            "tests/test_pathway_diversity.py",
            "outputs/stage5",
            "particle_init_noise",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_EFFECTIVE_PATHWAYS_LOOP_SWEEP": "4,8",
            "STAGE5_EFFECTIVE_PATHWAYS_NUM_PARTICLES": "16",
            "STAGE5_EFFECTIVE_PATHWAYS_NOISE": "0.05",
            "STAGE5_EFFECTIVE_PATHWAYS_LIMIT": "8",
            "STAGE5_EFFECTIVE_PATHWAYS_DISCONNECT": "1",
        },
    },
    "reentry_drift_diagnostic": {
        "path": "colab/STAGE5_REENTRY_DRIFT_CELL.py",
        "markers": [
            "STAGE5_REENTRY_DRIFT_CELL_VERSION",
            "stage5_reentry_drift_v1_readonly",
            "eval/eval_reentry_drift.py",
            "colab/assess_stage5_reentry.py",
            "tests/test_eval_reentry_drift.py",
            "bridge_gradient_liveness",
            "entry_exit_subspace",
            "loop_summary",
            "Readout Pause",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_REENTRY_DRIFT_MAX_LOOPS": "8",
            "STAGE5_REENTRY_DRIFT_LIMIT": "8",
            "STAGE5_REENTRY_DRIFT_MAX_LENGTH": "256",
            "STAGE5_REENTRY_DRIFT_SUBSPACE_RANK": "8",
            "STAGE5_REENTRY_DRIFT_DISCONNECT": "1",
        },
    },
    "reentry_norm_diagnostic": {
        "path": "colab/STAGE5_REENTRY_NORM_CELL.py",
        "markers": [
            "STAGE5_REENTRY_NORM_CELL_VERSION",
            "stage5_reentry_norm_v1_eval_only",
            "eval/eval_reentry_drift.py",
            "eval/eval_effective_pathways.py",
            "eval/eval_best_of_k_jsonl.py",
            "colab/assess_stage5_reentry.py",
            "reentry_rescale_mode",
            "entry_rms",
            "Candidate Conversion",
            "incremental_backup",
            "Readout Pause",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_REENTRY_NORM_LOOP_SWEEP": "4,8",
            "STAGE5_REENTRY_NORM_NOISE_SWEEP": "0,0.05",
            "STAGE5_REENTRY_NORM_SEEDS": "0,1,2",
            "STAGE5_REENTRY_NORM_K": "4",
            "STAGE5_REENTRY_NORM_LIMIT": "8",
            "STAGE5_REENTRY_NORM_DISCONNECT": "1",
        },
    },
    "reentry_repair_smoke": {
        "path": "colab/STAGE5_REENTRY_REPAIR_SMOKE_CELL.py",
        "markers": [
            "STAGE5_REENTRY_REPAIR_SMOKE_CELL_VERSION",
            "stage5_reentry_repair_smoke_v1_trainable",
            "bridge_gate_override",
            "bridge_reset_identity",
            "reentry_rescale_mode",
            "training/train_phase1_ponder.py",
            "eval/eval_reentry_drift.py",
            "Loop-1 Preservation",
            "loop1_preservation",
            "STAGE5_REENTRY_REPAIR_REQUIRE_NORM_PASS",
            "stage2_norm_assessment",
            "colab/assess_stage5_reentry.py",
            "tests/test_bridge.py",
            "Readout Pause",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_REENTRY_REPAIR_MAX_STEPS": "25",
            "STAGE5_REENTRY_REPAIR_MAX_LOOPS": "4",
            "STAGE5_REENTRY_REPAIR_DRIFT_MAX_LOOPS": "8",
            "STAGE5_REENTRY_REPAIR_LIMIT": "8",
            "STAGE5_REENTRY_REPAIR_OPTIMIZER_MODULES": "bridge,halt",
            "STAGE5_REENTRY_REPAIR_DISCONNECT": "1",
        },
    },
    "candidate_conversion_diagnostic": {
        "path": "colab/STAGE5_CANDIDATE_CONVERSION_CELL.py",
        "markers": [
            "STAGE5_CANDIDATE_CONVERSION_CELL_VERSION",
            "stage5_candidate_conversion_v3_chunk_merge",
            "candidate_conversion",
            "STAGE5_CANDIDATE_CONVERSION_NOISE_SWEEP",
            "max_loops_sweep",
            "pathway_split_diagnostics",
            "eval/eval_best_of_k_jsonl.py",
            "tests/test_eval_best_of_k_generation.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_CANDIDATE_CONVERSION_SEEDS": "0,1,2",
            "STAGE5_CANDIDATE_CONVERSION_NOISE_SWEEP": "0,0.005,0.01,0.02,0.05",
            "STAGE5_CANDIDATE_CONVERSION_MAX_LOOPS_SWEEP": "4,8",
            "STAGE5_CANDIDATE_CONVERSION_K": "4",
            "STAGE5_CANDIDATE_CONVERSION_DISCONNECT": "1",
        },
    },
    "capability_ladder_mcq_probe": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL_VERSION",
            "capability_ladder_mcq_probe",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "STAGE5_CAPABILITY_LADDER_MODEL_LADDER",
            "STAGE5_CAPABILITY_LADDER_MIN_TARGET_LOOP_ROWS",
            "STAGE5_CAPABILITY_LADDER_ARC_LIMIT",
            "content_question_only",
            "colab/run_stage5_capability_ladder_mcq_probe.py",
            "tests/test_stage5_capability_ladder_mcq_probe.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_CAPABILITY_LADDER_MODEL_LADDER": "qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3",
            "STAGE5_CAPABILITY_LADDER_MIN_TARGET_LOOP_ROWS": "1=1,2=1,3=1",
            "STAGE5_CAPABILITY_LADDER_ARC_LIMIT": "96",
            "STAGE5_CAPABILITY_LADDER_SCORE_MODE": "content_question_only",
            "STAGE5_CAPABILITY_LADDER_BACKUP_DRIVE": "0",
            "STAGE5_CAPABILITY_LADDER_DISCONNECT": "1",
        },
    },
    "capability_ladder_7b_mcq_probe": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL_VERSION",
            "capability_ladder_mcq_probe",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "STAGE5_CAPABILITY_LADDER_MODEL_LADDER",
            "STAGE5_CAPABILITY_LADDER_MIN_TARGET_LOOP_ROWS",
            "STAGE5_CAPABILITY_LADDER_ARC_LIMIT",
            "content_question_only",
            "colab/run_stage5_capability_ladder_mcq_probe.py",
            "tests/test_stage5_capability_ladder_mcq_probe.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_CAPABILITY_LADDER_MODELS": (
                "qwen_0_5b=Qwen/Qwen2.5-0.5B-Instruct,"
                "qwen_1_5b=Qwen/Qwen2.5-1.5B-Instruct,"
                "qwen_3b=Qwen/Qwen2.5-3B-Instruct,"
                "qwen_7b=Qwen/Qwen2.5-7B-Instruct"
            ),
            "STAGE5_CAPABILITY_LADDER_MODEL_LADDER": (
                "qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3,qwen_7b:4"
            ),
            "STAGE5_CAPABILITY_LADDER_MIN_TARGET_LOOP_ROWS": "1=1,2=1,3=1,4=1",
            "STAGE5_CAPABILITY_LADDER_ARC_LIMIT": "96",
            "STAGE5_CAPABILITY_LADDER_SCORE_MODE": "content_question_only",
            "STAGE5_CAPABILITY_LADDER_BACKUP_DRIVE": "0",
            "STAGE5_CAPABILITY_LADDER_DISCONNECT": "1",
        },
    },
    "capability_ladder_7b_trace_chain": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_CELL_VERSION",
            "capability_ladder_7b_trace_chain",
            "Qwen/Qwen2.5-7B-Instruct",
            "qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3,qwen_7b:4",
            "STAGE5_CAPABILITY_LADDER_MIN_TARGET_LOOP_ROWS",
            "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_ARC_LIMIT",
            "colab/run_stage5_capability_ladder_mcq_probe.py",
            "colab/run_stage5_capability_ladder_trace_jobs.py",
            "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_RUN_PROVIDER",
            "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_RUN_SFT",
            "colab/run_stage5_capability_ladder_trace_responses.py",
            "colab/run_stage5_capability_ladder_trace_collect.py",
            "colab/run_stage5_curriculum_sft.py",
            "tests/test_capability_ladder_trace_jobs.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_CAPABILITY_LADDER_MIN_TARGET_LOOP_ROWS": "1=1,2=1,3=1,4=1",
        },
    },
    "capability_ladder_trace_jobs_cpu": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL_VERSION",
            "capability_ladder_trace_jobs_cpu",
            "STAGE5_CAPABILITY_LADDER_TRACE_ALLOW_GPU",
            "colab/run_stage5_capability_ladder_trace_jobs.py",
            "training/build_capability_ladder_trace_jobs.py",
            "tests/test_capability_ladder_trace_jobs.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "capability_ladder_trace_responses_cpu": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL_VERSION",
            "capability_ladder_trace_responses_cpu",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_GPU",
            "colab/run_stage5_capability_ladder_trace_responses.py",
            "training/run_curriculum_job_responses.py",
            "tests/test_stage5_capability_ladder_trace_responses_runner.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "capability_ladder_trace_collect_cpu": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL_VERSION",
            "capability_ladder_trace_collect_cpu",
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_ALLOW_GPU",
            "colab/run_stage5_capability_ladder_trace_collect.py",
            "training/collect_capability_ladder_trace_outputs.py",
            "tests/test_stage5_capability_ladder_trace_collect_runner.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "capability_ladder_trace_response_collect_cpu": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_CELL_VERSION",
            "capability_ladder_trace_response_collect_cpu",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_ALLOW_GPU",
            "colab/run_stage5_capability_ladder_trace_responses.py",
            "colab/run_stage5_capability_ladder_trace_collect.py",
            "tests/test_stage5_capability_ladder_trace_responses_runner.py",
            "tests/test_stage5_capability_ladder_trace_collect_runner.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "capability_ladder_local_hf_trace_collect": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_CELL_VERSION",
            "capability_ladder_trace_response_collect_cpu",
            "hf_local",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_LOCAL_HF",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_MODEL_NAME",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_STUDENT_LINEAGE",
            "colab/run_stage5_capability_ladder_trace_responses.py",
            "colab/run_stage5_capability_ladder_trace_collect.py",
            "tests/test_stage5_capability_ladder_trace_responses_runner.py",
            "tests/test_stage5_capability_ladder_trace_collect_runner.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BACKEND": "hf_local",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_LOCAL_HF": "1",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_MODEL_NAME": "Qwen/Qwen2.5-7B-Instruct",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_DTYPE": "bfloat16",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_DEVICE": "cuda",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_TOP_P": "0.95",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_STUDENT_LINEAGE": "1",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT": "32",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MAX_TOKENS": "1536",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_TEMPERATURE": "0.2",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_TIMEOUT_SEC": "300",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_FAIL_FAST": "0",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_GPU": "1",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_ALLOW_GPU": "1",
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_ALLOW_GPU": "1",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BACKUP_DRIVE": "0",
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_BACKUP_DRIVE": "0",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_DISCONNECT": "1",
        },
    },
    "capability_ladder_local_hf_trace_sft": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_CELL_VERSION",
            "capability_ladder_local_hf_trace_sft",
            "hf_local",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_LOCAL_HF",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_MODEL_NAME",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_STUDENT_LINEAGE",
            "colab/run_stage5_capability_ladder_trace_responses.py",
            "colab/run_stage5_capability_ladder_trace_collect.py",
            "stage5_capability_ladder_trace_collection",
            "colab/run_stage5_curriculum_sft.py",
            "colab/run_stage5_benchmark_suite.py",
            "colab/assess_stage5_traced_sft.py",
            "STAGE5_CURRICULUM_USE_LEARNED_LOOP_CONTROL",
            "STAGE5_CURRICULUM_LOOP_CONTROL_CE_WEIGHT",
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_RUN_BENCHMARK",
            "tests/test_stage5_curriculum_sft.py",
            "tests/test_curriculum_sft_gate.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT": "32",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_MODEL_NAME": "Qwen/Qwen2.5-7B-Instruct",
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_RUN_SFT": "1",
            "STAGE5_TRACED_CAPABILITY_SFT_PHASE1_STEPS": "150",
            "STAGE5_TRACED_CAPABILITY_SFT_ALLOW_NO_DRIVE_BACKUP": "1",
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_DISCONNECT": "1",
        },
    },
    "capability_ladder_local_hf_trace_sft_scale64": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_CELL_VERSION",
            "capability_ladder_local_hf_trace_sft",
            "local_hf_trace_vram_preflight",
            "hf_local",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_LOCAL_HF",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_MODEL_NAME",
            "colab/run_stage5_capability_ladder_trace_responses.py",
            "colab/run_stage5_capability_ladder_trace_collect.py",
            "colab/run_stage5_curriculum_sft.py",
            "colab/run_stage5_benchmark_suite.py",
            "colab/assess_stage5_traced_sft.py",
            "STAGE5_CURRICULUM_USE_LEARNED_LOOP_CONTROL",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_capability_ladder_trace_jobs_20260623_150116/summary.json"
            ),
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_ID": (
                "stage5_capability_ladder_trace_responses_20260623_191545"
            ),
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT": "64",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RESUME": "1",
            "STAGE5_TRACED_CAPABILITY_SFT_MIN_TRACE_ROWS": "48",
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_MIN_POSITIVE_ROWS": "48",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_MODEL_NAME": "Qwen/Qwen2.5-7B-Instruct",
            "STAGE5_TRACED_CAPABILITY_SFT_RESUME_FROM": (
                "outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_191843/phase1/phase1_step_150.pt"
            ),
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_RUN_SFT": "1",
            "STAGE5_TRACED_CAPABILITY_SFT_PHASE1_STEPS": "200",
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_RUN_BENCHMARK": "1",
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_BENCHMARKS": "arc_easy,arc_challenge",
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_ARC_EASY_LIMIT": "128",
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_ARC_CHALLENGE_LIMIT": "128",
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_SCORE_TARGETS": (
                "content_question_only,cyclic_label_aggregated"
            ),
            "STAGE5_TRACED_CAPABILITY_SFT_ALLOW_NO_DRIVE_BACKUP": "1",
            "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_DISCONNECT": "1",
        },
    },
    "traced_sft_scale64_benchmark": {
        "path": "colab/STAGE5_TRACED_SFT_BENCHMARK_CELL.py",
        "markers": [
            "STAGE5_TRACED_SFT_BENCHMARK_CELL_VERSION",
            "traced_sft_benchmark_v1",
            "stage5_local_hf_traced_capability_sft_20260623_194543",
            "STAGE5_BENCHMARK_SOURCE_SUMMARY",
            "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL",
            "content_question_only,cyclic_label_aggregated",
            "colab/run_stage5_benchmark_suite.py",
            "colab/assess_stage5_traced_sft.py",
            "tests/test_stage5_benchmark_suite.py",
            "tests/test_stage5_traced_sft_assessment.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_TRACED_SFT_BENCHMARK_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_194543/summary.json"
            ),
            "STAGE5_TRACED_SFT_BENCHMARKS": "arc_easy,arc_challenge",
            "STAGE5_TRACED_SFT_BENCHMARK_ARC_EASY_LIMIT": "128",
            "STAGE5_TRACED_SFT_BENCHMARK_ARC_CHALLENGE_LIMIT": "128",
            "STAGE5_TRACED_SFT_BENCHMARK_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated",
            "STAGE5_TRACED_SFT_BENCHMARK_DISCONNECT": "1",
        },
    },
    "traced_capability_ladder_sft": {
        "path": "colab/STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL.py",
        "markers": [
            "STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL_VERSION",
            "traced_capability_ladder_sft",
            "stage5_capability_ladder_trace_collection",
            "STAGE5_TRACED_CAPABILITY_SFT_SOURCE_SUMMARY",
            "STAGE5_CURRICULUM_WORK_DIR",
            "STAGE5_CURRICULUM_MIN_POSITIVE_ROWS",
            "STAGE5_CURRICULUM_MIN_MODE_ROWS",
            "colab/run_stage5_curriculum_sft.py",
            "tests/test_stage5_curriculum_sft.py",
            "tests/test_curriculum_sft_gate.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "direct_preservation_probe": {
        "path": "colab/STAGE5_DIRECT_PRESERVATION_PROBE_CELL.py",
        "markers": [
            "STAGE5_DIRECT_PRESERVATION_PROBE_CELL_VERSION",
            "direct_preservation_probe",
            "STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY",
            "STAGE5_DIRECT_PRESERVE_MAX_STEPS",
            "colab/run_stage5_direct_preservation_probe.py",
            "stage5_arc_agi_next_action_20260622_181850_plan_conservative_direct_preservation",
            "runtime.unassign",
        ],
        "env": {},
    },
    "traced_sft_direct_preservation_probe": {
        "path": "colab/STAGE5_DIRECT_PRESERVATION_PROBE_CELL.py",
        "markers": [
            "STAGE5_DIRECT_PRESERVATION_PROBE_CELL_VERSION",
            "direct_preservation_probe",
            "STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY",
            "STAGE5_DIRECT_PRESERVE_MAX_STEPS",
            "STAGE5_DIRECT_PRESERVE_SWEEP",
            "STAGE5_DIRECT_PRESERVE_CHAIN_CONFIRM",
            "STAGE5_DIRECT_PRESERVE_CHAIN_DEPTH_ROUTER",
            "colab/assess_stage5_benchmark_suite.py",
            "colab/run_stage5_direct_preservation_probe.py",
            "colab/STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_CELL.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_local_hf_traced_sft_scale64_assessment_20260623_202446/summary.json"
            ),
            "STAGE5_DIRECT_PRESERVE_RUN_ID": "stage5_traced_sft_direct_preservation_20260623_scale64",
            "STAGE5_DIRECT_PRESERVE_ARC_TRAIN_LIMIT": "512",
            "STAGE5_DIRECT_PRESERVE_ARC_EVAL_LIMIT": "128",
            "STAGE5_DIRECT_PRESERVE_MAX_STEPS": "75",
            "STAGE5_DIRECT_PRESERVE_MIN_BASE_MARGIN": "1.0",
            "STAGE5_DIRECT_PRESERVE_PROMPT_STYLE": "question_only",
            "STAGE5_DIRECT_PRESERVE_SCORE_TARGET": "option_text",
            "STAGE5_DIRECT_PRESERVE_LR": "5e-7",
            "STAGE5_DIRECT_PRESERVE_DISTILL_WEIGHT": "1.0",
            "STAGE5_DIRECT_PRESERVE_DISTILL_TEMPERATURE": "2.0",
            "STAGE5_DIRECT_PRESERVE_SWEEP": (
                "baseline:lr=5e-7,steps=75,distill=1.0;"
                "lr1e6:lr=1e-6,steps=100,distill=1.0;"
                "lr2e6_distill2:lr=2e-6,steps=100,distill=2.0"
            ),
            "STAGE5_DIRECT_PRESERVE_CHAIN_CONFIRM": "1",
            "STAGE5_DIRECT_CONFIRM_ARC_EASY_LIMIT": "256",
            "STAGE5_DIRECT_CONFIRM_ARC_CHALLENGE_LIMIT": "256",
            "STAGE5_DIRECT_CONFIRM_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated",
            "STAGE5_DIRECT_CONFIRM_ASSESS_SCORE_TARGET": "content_question_only",
            "STAGE5_DIRECT_PRESERVE_CHAIN_DEPTH_ROUTER": "1",
            "STAGE5_DIRECT_PRESERVE_DRIVE_BACKUP": "1",
            "STAGE5_DIRECT_PRESERVE_DISCONNECT": "1",
        },
    },
    "traced_sft_direct_preservation_precheck": {
        "path": "colab/STAGE5_DIRECT_PRESERVATION_PROBE_CELL.py",
        "markers": [
            "STAGE5_DIRECT_PRESERVATION_PROBE_CELL_VERSION",
            "direct_preservation_probe",
            "STAGE5_DIRECT_PRESERVE_PRECHECK_ONLY",
            "direct_route_precheck_needs_training",
            "STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY",
            "STAGE5_DIRECT_PRESERVE_PROMPT_STYLE",
            "STAGE5_DIRECT_PRESERVE_SCORE_TARGET",
            "colab/run_stage5_direct_preservation_probe.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_local_hf_traced_sft_scale64_assessment_20260623_202446/summary.json"
            ),
            "STAGE5_DIRECT_PRESERVE_RUN_ID": "stage5_traced_sft_direct_preservation_precheck_20260623_scale64",
            "STAGE5_DIRECT_PRESERVE_ARC_TRAIN_LIMIT": "512",
            "STAGE5_DIRECT_PRESERVE_ARC_EVAL_LIMIT": "128",
            "STAGE5_DIRECT_PRESERVE_MIN_BASE_MARGIN": "1.0",
            "STAGE5_DIRECT_PRESERVE_PROMPT_STYLE": "question_only",
            "STAGE5_DIRECT_PRESERVE_SCORE_TARGET": "option_text",
            "STAGE5_DIRECT_PRESERVE_PRECHECK_ONLY": "1",
            "STAGE5_DIRECT_PRESERVE_RESUME_EXISTING": "1",
            "STAGE5_DIRECT_PRESERVE_CHAIN_CONFIRM": "0",
            "STAGE5_DIRECT_PRESERVE_CHAIN_DEPTH_ROUTER": "0",
            "STAGE5_DIRECT_PRESERVE_DRIVE_BACKUP": "1",
            "STAGE5_DIRECT_PRESERVE_DISCONNECT": "1",
        },
    },
    "traced_sft_direct_preservation_recover_only": {
        "path": "colab/STAGE5_DIRECT_PRESERVATION_PROBE_CELL.py",
        "markers": [
            "STAGE5_DIRECT_PRESERVATION_PROBE_CELL_VERSION",
            "direct_preservation_probe",
            "STAGE5_DIRECT_PRESERVE_RESUME_EXISTING",
            "STAGE5_DIRECT_PRESERVE_RESUME_ONLY",
            "direct_preservation_resume_existing",
            "direct_preservation_resume_missing",
            "STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY",
            "STAGE5_DIRECT_PRESERVE_SWEEP",
            "colab/run_stage5_direct_preservation_probe.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_local_hf_traced_sft_scale64_assessment_20260623_202446/summary.json"
            ),
            "STAGE5_DIRECT_PRESERVE_RUN_ID": "stage5_traced_sft_direct_preservation_20260623_scale64",
            "STAGE5_DIRECT_PRESERVE_SWEEP": (
                "baseline:lr=5e-7,steps=75,distill=1.0;"
                "lr1e6:lr=1e-6,steps=100,distill=1.0;"
                "lr2e6_distill2:lr=2e-6,steps=100,distill=2.0"
            ),
            "STAGE5_DIRECT_PRESERVE_RESUME_EXISTING": "1",
            "STAGE5_DIRECT_PRESERVE_RESUME_ONLY": "1",
            "STAGE5_DIRECT_PRESERVE_CHAIN_CONFIRM": "0",
            "STAGE5_DIRECT_PRESERVE_CHAIN_DEPTH_ROUTER": "0",
            "STAGE5_DIRECT_PRESERVE_DISCONNECT": "0",
        },
    },
    "traced_sft_direct_preservation_confirm": {
        "path": "colab/STAGE5_DIRECT_PRESERVATION_CONFIRM_CELL.py",
        "markers": [
            "STAGE5_DIRECT_PRESERVATION_CONFIRM_CELL_VERSION",
            "direct_preservation_confirm_v2",
            "stage5_latest_direct_preservation_summary.txt",
            "STAGE5_DIRECT_CONFIRM_SOURCE_SUMMARY",
            "STAGE5_BENCHMARK_PUSH",
            "colab/assess_stage5_benchmark_suite.py",
            "content_question_only,cyclic_label_aggregated",
            "colab/run_stage5_benchmark_suite.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_DIRECT_CONFIRM_RUN_ID": "stage5_traced_sft_direct_preservation_confirm_20260623_scale64",
            "STAGE5_DIRECT_CONFIRM_ARC_EASY_LIMIT": "256",
            "STAGE5_DIRECT_CONFIRM_ARC_CHALLENGE_LIMIT": "256",
            "STAGE5_DIRECT_CONFIRM_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated",
            "STAGE5_DIRECT_CONFIRM_ASSESS_SCORE_TARGET": "content_question_only",
            "STAGE5_DIRECT_CONFIRM_DISCONNECT": "1",
        },
    },
    "traced_sft_surface_alignment_repair": {
        "path": "colab/STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL.py",
        "markers": [
            "STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL_VERSION",
            "surface_alignment_repair_v1",
            "traced_sft_surface_alignment_repair",
            "STAGE5_SURFACE_ALIGN_SOURCE_SUMMARY",
            "stage5_traced_sft_direct_preservation_20260623_scale64_confirm",
            "eval/analyze_mcq_order_sensitivity.py",
            "eval/analyze_mcq_surface_mismatch.py",
            "training/prepare_mcq_conditional_invariance_jsonl.py",
            "training/prepare_mcq_surface_alignment_jsonl.py",
            "surface_alignment_train_jsonl",
            "colab/assess_stage5_surface_repair.py",
            "colab/run_stage5_surface_alignment_repair.py",
            "tests/test_prepare_mcq_surface_alignment_jsonl.py",
            "tests/test_prepare_mcq_conditional_invariance_jsonl.py",
            "tests/test_analyze_mcq_order_sensitivity.py",
            "tests/test_stage5_surface_alignment_repair.py",
            "tests/test_stage5_surface_repair_assessment.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_SURFACE_ALIGN_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm/summary.json"
            ),
            "STAGE5_SURFACE_ALIGN_MAX_STEPS": "50",
            "STAGE5_SURFACE_ALIGN_LR": "5e-7",
            "STAGE5_SURFACE_ALIGN_DISTILL_WEIGHT": "0.05",
            "STAGE5_SURFACE_ALIGN_PUSH": "1",
            "STAGE5_SURFACE_ALIGN_DISCONNECT": "1",
        },
    },
    "traced_sft_score_alignment_repair": {
        "path": "colab/STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL.py",
        "markers": [
            "STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL_VERSION",
            "surface_alignment_repair_v1",
            "training/prepare_mcq_score_alignment_jsonl.py",
            "training/train_phase1_mcq_score_align.py",
            "STAGE5_SURFACE_ALIGN_TRAINER",
            "score_ce",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_SURFACE_ALIGN_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm/summary.json"
            ),
            "STAGE5_SURFACE_ALIGN_RUN_ID": "stage5_score_alignment_repair_content_route_20260624",
            "STAGE5_SURFACE_ALIGN_TRAINER": "score_ce",
            "STAGE5_SURFACE_ALIGN_MAX_STEPS": "75",
            "STAGE5_SURFACE_ALIGN_LR": "5e-7",
            "STAGE5_SURFACE_ALIGN_DISTILL_WEIGHT": "0.0",
            "STAGE5_SURFACE_ALIGN_SCORE_DISTILL_WEIGHT": "0.05",
            "STAGE5_SURFACE_ALIGN_SCORE_MARGIN": "0.05",
            "STAGE5_SURFACE_ALIGN_SCORE_MARGIN_WEIGHT": "0.1",
            "STAGE5_SURFACE_ALIGN_PUSH": "1",
            "STAGE5_SURFACE_ALIGN_DISCONNECT": "1",
        },
    },
    "dense_mcq_trace_sft_control": {
        "path": "colab/STAGE5_DENSE_MCQ_TRACE_SFT_CONTROL_CELL.py",
        "markers": [
            "STAGE5_DENSE_MCQ_TRACE_SFT_CONTROL_CELL_VERSION",
            "dense_mcq_trace_sft_control_v1",
            "dense_mcq_trace_sft_control",
            "STAGE5_DENSE_MCQ_SOURCE_SUMMARY",
            "stage5_local_hf_traced_capability_sft_20260623_194543",
            "training/train_dense_lora.py",
            "eval/eval_mcq.py --mode base --checkpoint",
            "STAGE5_DENSE_MCQ_EXTRA_TRAIN_JSONL",
            "colab/run_stage5_mcq_dense_sft_control.py",
            "colab/assess_stage5_mcq_recipe_control.py",
            "tests/test_eval_mcq_dense_lora.py",
            "tests/test_stage5_mcq_dense_sft_control.py",
            "tests/test_stage5_mcq_recipe_control_assessment.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_DENSE_MCQ_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_local_hf_traced_capability_sft_20260623_194543/summary.json"
            ),
            "STAGE5_DENSE_MCQ_RUN_ID": "stage5_dense_mcq_trace_sft_control_20260623",
            "STAGE5_DENSE_MCQ_BENCHMARKS": "arc_easy,arc_challenge",
            "STAGE5_DENSE_MCQ_ARC_EASY_LIMIT": "256",
            "STAGE5_DENSE_MCQ_ARC_CHALLENGE_LIMIT": "256",
            "STAGE5_DENSE_MCQ_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated",
            "STAGE5_DENSE_MCQ_AGGREGATES": "mean",
            "STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY": (
                "outputs/stage5/stage5_local_hf_traced_sft_scale64_benchmark_20260623_201923/summary.json"
            ),
            "STAGE5_DENSE_MCQ_COMMIT_CHECKPOINT": "0",
            "STAGE5_DENSE_MCQ_PUSH": "1",
            "STAGE5_DENSE_MCQ_DISCONNECT": "1",
        },
    },
    "traced_sft_competence_preserving_pipeline": {
        "path": "colab/STAGE5_COMPETENCE_PRESERVING_PIPELINE_CELL.py",
        "markers": [
            "STAGE5_COMPETENCE_PRESERVING_PIPELINE_CELL_VERSION",
            "competence_preserving_pipeline_v1",
            "traced_sft_competence_preserving_pipeline",
            "STAGE5_COMPETENCE_SOURCE_SUMMARY",
            "stage5_traced_sft_direct_preservation_20260623_scale64_confirm_assessment",
            "colab/run_stage5_competence_preserving_pipeline.py",
            "tests/test_stage5_competence_preserving_pipeline.py",
            "tests/test_stage5_balanced_arc_mix_gate.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_COMPETENCE_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_confirm_assessment/summary.json"
            ),
            "STAGE5_COMPETENCE_PIPELINE_RUN_ID": (
                "stage5_competence_preserving_from_direct_confirm_20260623"
            ),
            "STAGE5_COMPETENCE_PIPELINE_PUSH": "1",
            "STAGE5_COMPETENCE_PIPELINE_DISCONNECT": "1",
        },
    },
    "traced_sft_depth_router_after_direct_preserve": {
        "path": "colab/STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_CELL.py",
        "markers": [
            "STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_CELL_VERSION",
            "traced_sft_depth_router_after_direct_preserve",
            "STAGE5_DEPTH_ROUTER_TRACE_SOURCE_SUMMARY",
            "STAGE5_DEPTH_ROUTER_DIRECT_SOURCE_SUMMARY",
            "stage5_latest_direct_preservation_summary.txt",
            "STAGE5_CURRICULUM_USE_LEARNED_LOOP_CONTROL",
            "STAGE5_DEPTH_ROUTER_LOOP_CONTROL_CE_WEIGHT",
            "STAGE5_DEPTH_ROUTER_HALT_TARGET_NLL_WEIGHT",
            "colab/run_stage5_curriculum_sft.py",
            "colab/run_stage5_benchmark_suite.py",
            "colab/assess_stage5_traced_sft.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_DEPTH_ROUTER_TRACE_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json"
            ),
            "STAGE5_DEPTH_ROUTER_DIRECT_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_lr1e6/summary.json"
            ),
            "STAGE5_DEPTH_ROUTER_RUN_ID": "stage5_depth_router_after_direct_preserve_scale64",
            "STAGE5_DEPTH_ROUTER_STEPS": "100",
            "STAGE5_DEPTH_ROUTER_LR": "5e-5",
            "STAGE5_DEPTH_ROUTER_LOOP_CONTROL_CE_WEIGHT": "4.0",
            "STAGE5_DEPTH_ROUTER_HALT_TARGET_NLL_WEIGHT": "5.0",
            "STAGE5_DEPTH_ROUTER_OPTIMIZER_MODULES": "halt",
            "STAGE5_DEPTH_ROUTER_ARC_EASY_LIMIT": "128",
            "STAGE5_DEPTH_ROUTER_ARC_CHALLENGE_LIMIT": "128",
            "STAGE5_DEPTH_ROUTER_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated",
            "STAGE5_DEPTH_ROUTER_DISCONNECT": "1",
        },
    },
    "depth_sweep_heldout": {
        "path": "colab/STAGE5_DEPTH_SWEEP_BENCHMARK_CELL.py",
        "markers": [
            "STAGE5_DEPTH_SWEEP_BENCHMARK_CELL_VERSION",
            "STAGE5_DEPTH_SWEEP_DRIVE_BACKUP",
            "STAGE5_DEPTH_SWEEP_LOOPS",
            "STAGE5_BENCHMARK_ARC_EASY_OFFSET",
            "STAGE5_BENCHMARK_ARC_CHALLENGE_OFFSET",
            "eval/analyze_depth_sweep.py",
            "stage5_depth_sweep_arc_loop1234",
            "Drive backup disabled; using GitHub as primary artifact store.",
        ],
        "env": {
            "STAGE5_DEPTH_SWEEP_LOOPS": "1,2,3",
            "STAGE5_DEPTH_SWEEP_ARC_EASY_OFFSET": "256",
            "STAGE5_DEPTH_SWEEP_ARC_EASY_LIMIT": "full",
            "STAGE5_DEPTH_SWEEP_ARC_CHALLENGE_OFFSET": "256",
            "STAGE5_DEPTH_SWEEP_ARC_CHALLENGE_LIMIT": "full",
            "STAGE5_DEPTH_SWEEP_DISCONNECT": "0",
            "STAGE5_DEPTH_SWEEP_DRIVE_BACKUP": "0",
        },
    },
    "model_viability_probe": {
        "path": "colab/STAGE5_MODEL_VIABILITY_PROBE_CELL.py",
        "markers": [
            "STAGE5_MODEL_VIABILITY_PROBE_CELL_VERSION",
            "model_viability_probe_v1",
            "STAGE5_MODEL_PROBE_MODEL_NAME",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "STAGE5_MODEL_PROBE_LAYER_SPLIT",
            "STAGE5_MODEL_PROBE_LOOPS",
            "STAGE5_MODEL_PROBE_SCORE_TARGETS",
            "colab/run_stage5_model_viability_probe.py",
            "tests/test_model_viability_probe.py",
            "tests/test_stage5_notebooks.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "model_viability_queue": {
        "path": "colab/STAGE5_MODEL_VIABILITY_QUEUE_CELL.py",
        "markers": [
            "STAGE5_MODEL_VIABILITY_QUEUE_CELL_VERSION",
            "model_viability_queue_v1",
            "STAGE5_MODEL_QUEUE_MODELS",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "STAGE5_MODEL_QUEUE_CHILD_PUSH",
            "colab/run_stage5_model_viability_queue.py",
            "tests/test_model_viability_probe.py",
            "tests/test_stage5_notebooks.py",
            "runtime.unassign",
        ],
        "env": {},
    },
}

def secret(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None

if TARGET not in TARGETS:
    raise AssertionError(f"Unknown TARGET={TARGET!r}; expected one of {sorted(TARGETS)}")

GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."


def github_json(url):
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


ref_payload = github_json(
    f"https://api.github.com/repos/{REPO}/git/ref/heads/{REF}?cache_bust={int(time.time())}"
)
RESOLVED_REF = ((ref_payload.get("object") or {}).get("sha") or REF).strip()

selected = TARGETS[TARGET]
if SOURCE_SUMMARY_OVERRIDE:
    os.environ["STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_SURFACE_ALIGN_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_ARC_MIX_CHAIN_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_ARC_MIX_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
else:
    # Avoid accidentally pinning a new session to an old target-specific source
    # summary. The safe-continue launcher will follow config/stage5_current_source_summary.txt.
    os.environ.pop("STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_SURFACE_ALIGN_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_ARC_MIX_CHAIN_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_ARC_MIX_SOURCE_SUMMARY", None)
for key, value in selected["env"].items():
    # Target configs are defaults. Planner/user-supplied env must win so chained
    # actions can pass repaired checkpoints, benchmark summaries, and run IDs.
    os.environ.setdefault(key, value)
if TARGET == "depth_sweep_heldout":
    if os.environ.get("STAGE5_DEPTH_SWEEP_RUN_ID_LOCKED", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "y",
    }:
        os.environ["STAGE5_DEPTH_SWEEP_RUN_ID"] = time.strftime(
            "stage5_depth_sweep_arc_heldout_tail_loop123_%Y%m%d_%H%M%S"
        )
    else:
        os.environ.setdefault(
            "STAGE5_DEPTH_SWEEP_RUN_ID",
            time.strftime("stage5_depth_sweep_arc_heldout_tail_loop123_%Y%m%d_%H%M%S"),
        )
os.environ.setdefault("STAGE5_SAFE_CONTINUE_DISCONNECT", "1")

launcher_path = selected["path"]
url = f"https://api.github.com/repos/{REPO}/contents/{launcher_path}?ref={RESOLVED_REF}&cache_bust={int(time.time())}"
payload = github_json(url)

code = base64.b64decode(payload["content"]).decode("utf-8")
missing = [marker for marker in selected["markers"] if marker not in code]
assert not missing, f"Fetched launcher is missing expected safety markers: {missing}"

print(
    f"bootstrap_version={BOOTSTRAP_VERSION} resolved_ref={RESOLVED_REF} target={TARGET}",
    flush=True,
)
print(f"Fetched {launcher_path} from {REPO}@{REF} ({RESOLVED_REF[:12]}) sha={payload.get('sha')} target={TARGET}", flush=True)
exec(compile(code, launcher_path, "exec"))
