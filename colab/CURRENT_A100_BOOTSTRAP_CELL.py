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
#   "capability_ladder_mcq_probe" - bounded Qwen 0.5B/1.5B/3B ARC MCQ scoring for depth-label data.
#   "capability_ladder_7b_mcq_probe" - high-memory Qwen 0.5B/1.5B/3B/7B ARC MCQ scoring for depth-4 data.
#   "capability_ladder_7b_trace_chain" - high-memory 7B ladder probe followed by trace-job build.
#   "capability_ladder_trace_jobs_cpu" - CPU-only trace-job build from latest capability ladder probe.
#   "capability_ladder_trace_responses_cpu" - CPU/network provider responses for trace jobs.
#   "capability_ladder_trace_collect_cpu" - CPU-only trace-response collection into gated SFT data.
#   "capability_ladder_trace_response_collect_cpu" - CPU/network provider responses then immediate collection.
#   "capability_ladder_local_hf_trace_collect" - GPU local-HF responses then immediate collection.
#   "capability_ladder_local_hf_trace_sft" - GPU local-HF traces, collection, then bounded recurrent SFT.
#   "traced_capability_ladder_sft" - GPU Phase 1 SFT from the latest gate-ready traced capability ladder.
#   "direct_preservation_probe" - bounded max_loops=1 base-preservation training probe.
#   "depth_sweep_heldout" - L4/T4 held-out ARC tail loop-depth sweep for routing validation.
#   "model_viability_probe" - no-training Qwen model scale probe; defaults to 1.5B and is env-configurable for 3B+.
#   "model_viability_queue" - queued no-training Qwen 3B/7B probes with VRAM-aware skipping.
TARGET = os.environ.get("STAGE5_CURRENT_A100_TARGET", "preflight")
SOURCE_SUMMARY_OVERRIDE = os.environ.get("STAGE5_CURRENT_A100_SOURCE_SUMMARY", "").strip()

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
    "capability_ladder_mcq_probe": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL_VERSION",
            "capability_ladder_mcq_probe",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "STAGE5_CAPABILITY_LADDER_MODEL_LADDER",
            "STAGE5_CAPABILITY_LADDER_ARC_LIMIT",
            "content_question_only",
            "colab/run_stage5_capability_ladder_mcq_probe.py",
            "tests/test_stage5_capability_ladder_mcq_probe.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_CAPABILITY_LADDER_MODEL_LADDER": "qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3",
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
        "env": {},
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
            "STAGE5_CURRICULUM_USE_LEARNED_LOOP_CONTROL",
            "STAGE5_CURRICULUM_LOOP_CONTROL_CE_WEIGHT",
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
    os.environ["STAGE5_ARC_MIX_CHAIN_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_ARC_MIX_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
else:
    # Avoid accidentally pinning a new session to an old target-specific source
    # summary. The safe-continue launcher will follow config/stage5_current_source_summary.txt.
    os.environ.pop("STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_ARC_MIX_CHAIN_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_ARC_MIX_SOURCE_SUMMARY", None)
for key, value in selected["env"].items():
    os.environ[key] = value
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
