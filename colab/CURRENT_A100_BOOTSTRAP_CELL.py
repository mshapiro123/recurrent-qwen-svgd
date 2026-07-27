import base64, json, os, subprocess, time, urllib.request
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"
BOOTSTRAP_VERSION = "sha_resolved_nested_fetch_v3_short_sha"

# Safe default: verify Drive/checkpoint visibility on a CPU/cheap runtime.
# Other options:
#   "programmatic_curriculum_cpu" - generate/publish the synthetic direct/deep curriculum gate on CPU.
#   "claim_curriculum_scaleup_cpu" - build/fill the provider-backed claim-sized direct/deep curriculum shard on CPU.
#   "master_sequence_status" - cheap CPU readout of current phase, pointer, and next target.
#   "safe_continue_dry_run" - fetch safe-continue but do not spend GPU.
#   "safe_continue_execute" - fetch safe-continue and opt in to the guarded paid action.
#   "arc_challenge_mcq_debias_confirm" - bounded no-training cyclic MCQ confirmation on ARC-Challenge.
#   "debiased_benchmark_suite" - bounded ARC-Easy/Challenge/GPQA-lite benchmark with debiased MCQ scoring.
#   "depth_balanced_benchmark" - balanced ARC content/cyclic benchmark for learned-depth checkpoints.
#   "arc_mix_offset_confirm" - bounded ARC-Easy/Challenge offset-256 confirmation for the latest ARC-mix checkpoint.
#   "arc_mix_offset_then_depth_chain" - offset confirmation, then learned-depth ARC-mix SFT only if confirmed.
#   "arc_mix_depth_routing_probe" - bounded learned-depth ARC-mix SFT probe from the latest recovered checkpoint.
#   "effective_pathways_diagnostic" - bounded deterministic recurrent pathway-collapse diagnostic.
#   "candidate_conversion_diagnostic" - bounded particle-noise candidate conversion with correctness-split pathways.
#   "reentry_drift_diagnostic" - bounded read-only recurrent loop-closure drift diagnostic.
#   "reentry_norm_diagnostic" - bounded eval-only loop re-entry RMS normalization comparison.
#   "reentry_norm_recover_only" - publish a completed Stage 2 re-entry norm run from Drive without rerunning eval.
#   "reentry_repair_smoke" - bounded trainable bridge/re-entry repair smoke.
#   "reentry_spectral_repair_smoke" - same Stage 3 smoke with spectral re-entry adapter mode.
#   "reentry_recovery_training" - gated recovery SFT after re-entry repair passes.
#   "reentry_tail_damper_recovery_training" - recovery SFT with fixed strength-1.0 tail damper held constant.
#   "reentry_tail_damper_capacity_lora32_training" - same fixed-damper recovery arm with recurrent LoRA rank 32.
#   "reentry_capacity_localization_rank64" - rank-64 fixed-damper capacity localization arm.
#   "unfreeze_recurrent_curriculum" - merge recovered LoRA, unfreeze recurrent block, train with Muon and loop curriculum.
#   "prelude_path_development" - corrected re-injection unfreeze with boosted bridge-prelude gradient and ablation.
#   "reentry_tail_damper_recovery_readout_only" - finish fixed-damper readout for an already completed recovery SFT.
#   "depth_signal_confirmation" - recovery SFT, then expanded hard-content benchmark with open hard fallback.
#   "capability_ladder_mcq_probe" - bounded Qwen 0.5B/1.5B/3B ARC MCQ cyclic-permutation MCQ diagnostic for depth-label data.
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
#   "forced_depth_diagnostic" - no-training ARC-Challenge forced-depth diagnostic.
#   "rescue_detectability_gate" - CPU-only rescue-direction agreement against a permutation null.
#   "rescue_predictability_analysis" - CPU-only precursor asking whether deeper-loop rescue is predictable.
#   "rescue_selector_transfer" - CPU-only held-out rescue selector trade curve with spectral/probe diagnostics.
#   "tail_convergence_selector" - GPU hidden-state probe for cross-loop tail convergence selector features.
#   "heldout_router_validation" - no-training forced-depth router transfer reality test.
#   "latent_criticality_probe" - prompt-state criticality/Jacobian probe from a completed router-validation run.
#   "reentry_covariance_check" - read-only covariance gate before directional re-entry adapter changes.
#   "reentry_tail_diagnostic" - tail-resolved re-entry inflation/rotation and harmed/rescued diagnostic.
#   "reentry_tail_damper_sweep" - eval-only tail damper stability/signal tradeoff sweep.
#   "reentry_tail_damper_heldout" - held-out ARC offset confirmation of tail damper tradeoff.
#   "reentry_tail_damper_powered_arc_train" - powered ARC-Challenge train confirmation using 0/0.5/1.0 strengths.
#   "direct_preservation_probe" - bounded max_loops=1 base-preservation training probe.
#   "depth_sweep_heldout" - L4/T4 held-out ARC tail loop-depth sweep for routing validation.
#   "synthetic_depth_task" - L4/T4 iterated-function staircase test for whether recurrence supplies sequential depth.
#   "synthetic_depth_primitive_curve" - L4/T4 Phase 1 depth-1 primitive curve over N=8,12,16.
#   "synthetic_depth_staged_staircase" - L4/T4 Phase 2 target-loop staircase from the N=16 primitive checkpoint.
#   "synthetic_depth_chain_supervision" - L4/T4 train-split diagnostic plus per-loop intermediate-chain supervision.
#   "synthetic_depth_split_bridge_microtest" - L4/T4 split-bridge true prelude-LR micro-fit.
#   "synthetic_depth_chain_scaled_corrected" - L4/T4 active-label full-symbol chain run at N=16.
#   "depth_extrapolation_eval" - forward-only depth 5/6 extrapolation from the positive synthetic chain checkpoint.
#   "synthetic_probe_battery" - forward-only state probe grid plus loop-index probe for the positive synthetic chain checkpoint.
#   "chain_anneal_to_outcome" - persistence test: anneal active chain labels away while keeping outcome supervision.
#   "post_anneal_extended_readouts" - eval-only depth 1-8 and norm-standardized clock probes from the annealed checkpoint.
#   "chain_continuation_attribution" - same-budget continued chain supervision control plus depth 1-8 extrapolation.
#   "chain_continuation_probe_readout" - probe battery on the chain-continuation checkpoint.
#   "regression_battery_loop1_current" - loop-1 AI2 ARC non-inferiority battery before route comparison.
#   "depth_support_route_comparison" - train support depth 1-6 and score on frozen depth 1-10 rows.
#   "depth_support_ladder8" - train support depth 1-8 and score on frozen depth 1-14 rows.
#   "support8_probe_readout" - eval-only envelope/clock/probe readout on the support-8 checkpoint over frozen depth 1-14 rows.
#   "support8_dose_arm" - continue support-8 for +2000 same-curriculum steps and rescore locked frozen depth 1-14 gates.
#   "same_reader_final_symbol" - release-gate final-symbol scoring with the same full-symbol reader used by active labels.
#   "support6_seed_replication" - two added support-6 route seeds for replication-band evidence.
#   "support6_replication_receipts" - CPU-only canonical frontier rescore and config-diff receipts for support-6 seeds.
#   "support6_dosed_seed_resolution" - continue failed support-6 replicate seeds for a fixed-dose resolution.
#   "scorer_equivalence_receipt" - tiny GPU fast-vs-slow active-label scorer equivalence receipt.
#   "synthetic_release_receipts" - CPU-only dashboard of synthetic-line release receipts and missing guardrails.
#   "n24_support12_rung" - final N-24 support-12 synthetic rung with locked gates and canary policy.
#   "phase_a_surpass_prereg" - publish the same-reader Phase-A surpass comparison preregistration.
#   "phase_a_surpass_receipt" - publish the paired Phase-A result, checkpoint hashes, and figure.
#   "phase_a_dense_full" - full-model AdamW dense B/C or D controls on locked synthetic rows.
#   "phase_a_checkpoint_comparison" - eval-only 2k-vs-4k comparison for dense B/C/D checkpoints.
#   "splice_injection_diagnostic" - inference-only hidden-state splice test for state-driven iteration vs shortcut.
#   "gradient_path_audit" - read-only per-loop gradient matrix plus finite-difference bridge check.
#   "model_viability_probe" - no-training Qwen model scale probe; defaults to 1.5B and is env-configurable for 3B+.
#   "model_viability_queue" - queued no-training Qwen 3B/7B probes with VRAM-aware skipping.
#   "natural_surface_prepare_cpu" - CPU-only verbal relay/pointer transfer dataset prep and manifest receipt.
#   "natural_surface_transfer_rung0" - GPU frozen natural-surface baseline, then verbal rung-zero SFT.
#   "phase_g_experiment1" - deterministic injective/abductive gates plus matched-K answer sampling.
#   "phase_g_alpha" - frozen-substrate guided stochastic transition KL sweep and exact coverage gate.
#   "phase_g_multitarget_control" - locked repeated-prompt posterior-control A0 gate.
#   "phase_g_forced_injection_probe" - eval-only A0 posterior-residual causal magnitude probe.
#   "oracle_interface_probe" - terminal additive-versus-FiLM oracle re-entry capacity probe.
#   "oracle_intrablock_control" - parameter-matched layerwise oracle-control localization.
#   "phase_g_injective_curriculum_recovery" - continue the fixed-boundary injective checkpoint with a 2-to-8 loop curriculum.
#   "phase_g_curriculum_autopsy" - read-only train/held-out loop matrix and curriculum-construction audit.
#   "inverse_composition_staircase" - matched forward/inverse-table staircase with weighted loop-dose gates.
#   "inverse_table_rebase_caps3_4" - continue the green inverse-table control through caps 3 and 4.
#   "inverse_table_cap3_rehearsal" - repair cap-3 retention with exact-epoch forward rehearsal.
#   "inverse_rendered_width_gate" - test deterministic validity on the exact multimodal inverse rendering.
#   "inverse_rendered_n24_continuation" - one bounded deterministic tune for the inverse-rendered N24 validity gate.
#   "phase_g_n24_calibration_gate" - arbitrary-function deterministic substrate gate before G-alpha.
#   "multichannel_bridge_precursor" - bounded eval-only M1/M2 pilot for the query-head bridge battery.
#   "multichannel_bridge_precursor_replication" - one bounded backward-recovery M1/M2 replication using the locked N24 pilot receipt.
#   "multichannel_bridge_precursor_full" - explicit full M1/M2/M3 battery after pilot review.
#   "peft_ponder_closure" - corrected-loop frozen-LoRA ladder plus halting-only Ponder phase.
#   "adapter_budget_arm_e" - matched Arm-A R16 LoRA plus repaired-bridge depth profile.
#   "adapter_parity_e3a" - Arm E zero-shot relay/pointer transfer.
#   "adapter_parity_e2" - Arm E outcome-only persistence.
#   "adapter_parity_e4" - E2-gated Arm E inverse retention.
#   "adapter_verbal_transference_e3b" - matched installed-vs-fresh R16 verbal transfer.
#   "adapter_verbal_transference_e3b_salvage" - eval-only receipt after the Arm S guardrail stop.
#   "paper1_closure_receipts" - CPU-safe Paper 1 evidence compiler and Drive backup.
#   "depth_selector_bounded_assessment" - frozen N24 supervised-depth and Ponder-outcome selector closure.
#   "wall_clock_latency_descriptive" - eval-only five-arm Paper One batch-1 latency receipt.
#   "paper2_d0_oracle_router_audit" - CPU-only exact oracle ceiling and cached-signal audit; no checkpoint load.
#   "paper2_d0_router_probe" - read-only L4 Prelude and per-loop deployable router probes.
#   "paper2_d0_floor_calibration" - forced-depth 1-6 floor against cached 7B/14B labels; L4-safe.
#   "paper2_d0_teacher_cache" - registered 7B/14B single-pass teacher caches; A100/H100 only.
#   "paper2_d0_prelaunch" - CPU-only target-policy and teacher-demand receipts; required before training.
#   "paper2_d0_train_eval" - registered 4,000-step D0 pilot and complete final battery; A100/H100.
#   "paper2_d1_causal_allocation_audit" - read-only A100 D0 decomposition and D1 utility-label dry run.
#   "paper2_d0_prelock_publish_resume" - publish completed Drive-backed D0 lock receipts without inference.
#   "paper2_d0_prelock" - authorized density probe, corpus freeze, and preregistration lock only.
TARGET = os.environ.get("STAGE5_CURRENT_A100_TARGET", "preflight")
SOURCE_SUMMARY_OVERRIDE = os.environ.get("STAGE5_CURRENT_A100_SOURCE_SUMMARY", "").strip()
PREFER_LOCAL_HEAD = os.environ.get("STAGE5_BOOTSTRAP_PREFER_LOCAL_HEAD", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}

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
    "paper2_d1_causal_allocation_audit": {
        "path": "colab/STAGE5_PAPER2_D1_CAUSAL_ALLOCATION_AUDIT_CELL.py",
        "markers": [
            "STAGE5_PAPER2_D1_CAUSAL_ALLOCATION_AUDIT_VERSION",
            "paper2_d1_causal_allocation_audit_v1",
            "read-only A100 post-D0 audit no optimizer no backward no checkpoint writes",
            "exact replay equivalence to banked D0 A100 anchors",
            "source-row grouped five-fold cross-fit seed 20260727",
            "D1 continue only when next loop helps stop on hurts or neutral",
            "teacher top1 top2 margin unavailable no teacher reload",
            "100000-position label-train forced-depth-4 dry run",
            "colab/run_stage5_paper2_d1_causal_allocation_audit.py",
        ],
        "env": {},
    },
    "paper2_d0_router_probe": {
        "path": "colab/STAGE5_PAPER2_D0_ROUTER_PROBE_CELL.py",
        "markers": [
            "STAGE5_PAPER2_D0_ROUTER_PROBE_VERSION",
            "paper2_d0_router_probe_v3_hardware_sensitivity",
            "read-only L4 feature extraction no model optimizer no model training",
            "colab/run_stage5_paper2_d0_router_probe.py",
        ],
        "env": {},
    },
    "paper2_d0_oracle_router_audit": {
        "path": "colab/STAGE5_PAPER2_D0_ORACLE_ROUTER_AUDIT_CELL.py",
        "markers": [
            "STAGE5_PAPER2_D0_ORACLE_ROUTER_AUDIT_VERSION",
            "paper2_d0_oracle_router_audit_v1",
            "read-only calibration receipt no checkpoint no optimizer no training",
            "colab/run_stage5_paper2_d0_oracle_router_audit.py",
        ],
        "env": {},
    },
    "paper2_d0_prelaunch": {
        "path": "colab/STAGE5_PAPER2_D0_PRELAUNCH_CELL.py",
        "markers": [
            "STAGE5_PAPER2_D0_PRELAUNCH_VERSION",
            "paper2_d0_prelaunch_v1",
            "read-only prelaunch post-processing no model no optimizer no evaluation partition",
            "authenticated figure-review addendum",
            "binned target policy receipt before training",
            "teacher demand uses each teachers own rejection population",
            "Drive mount retry with explicit Pharma Initiatives account authorization",
            "colab/run_stage5_paper2_d0_prelaunch.py",
        ],
        "env": {},
    },
    "paper2_d0_train_eval": {
        "path": "colab/STAGE5_PAPER2_D0_TRAIN_EVAL_CELL.py",
        "markers": [
            "STAGE5_PAPER2_D0_TRAIN_EVAL_VERSION",
            "paper2_d0_train_eval_v4_mixed_rehearsal",
            "locked 4000-step 70/30 D0 pilot",
            "final-step EMA primary",
            "blocked outcomes exit 2 with tables written",
            "prelaunch receipts required before optimizer construction",
            "frozen q1-q4 binned target table",
            "deterministic fp32 argmax lowest token id ties counted",
            "teacher shift uses each teachers own rejection population",
            "Drive mount retry with explicit Pharma Initiatives account authorization",
            "preflight-only pass before model or optimizer construction",
            "evaluation partition restored only after training",
            "mixed rehearsal mirrors T1 control-active and mechanism-only semantics",
            "colab/run_stage5_paper2_d0_train_eval.py",
        ],
        "env": {},
    },
    "paper2_d0_floor_calibration": {
        "path": "colab/STAGE5_PAPER2_D0_FLOOR_CALIBRATION_CELL.py",
        "markers": [
            "STAGE5_PAPER2_D0_FLOOR_CALIBRATION_VERSION",
            "paper2_d0_floor_calibration_v2_diagnostic",
            "floor calibration only no optimizer no training",
            "colab/run_stage5_paper2_d0_floor_calibration.py",
        ],
        "env": {},
    },
    "paper2_d0_teacher_cache": {
        "path": "colab/STAGE5_PAPER2_D0_TEACHER_CACHE_CELL.py",
        "markers": [
            "STAGE5_PAPER2_D0_TEACHER_CACHE_VERSION",
            "paper2_d0_teacher_cache_v2_diagnostic",
            "minimum_vram_mib=35000",
            "labeling proper only no optimizer no training",
            "tests/test_speculative_depth_d0_postlock.py",
            "colab/run_stage5_paper2_d0_teacher_cache.py",
        ],
        "env": {},
    },
    "paper2_d0_prelock_publish_resume": {
        "path": "colab/STAGE5_PAPER2_D0_PRELOCK_PUBLISH_RESUME_CELL.py",
        "markers": [
            "STAGE5_PAPER2_D0_PRELOCK_PUBLISH_RESUME_VERSION",
            "paper2_d0_prelock_publish_resume_v1",
            "restore completed receipts from Drive",
            "verify every frozen private artifact hash",
            "no model inference no teacher labeling no optimizer no training",
            "git add force aggregate ignored receipts only",
            "colab/run_stage5_paper2_d0_prelock_publish_resume.py",
        ],
        "env": {},
    },
    "paper2_d0_prelock": {
        "path": "colab/STAGE5_PAPER2_D0_PRELOCK_CELL.py",
        "markers": [
            "STAGE5_PAPER2_D0_PRELOCK_CELL_VERSION",
            "paper2_d0_prelock_density_and_hash_v2_stack_smol",
            "Draft 7 authenticated governing hash",
            "CC-MAIN-2025-26 pinned FineWeb-Edu dump",
            "the-stack-smol direct Hugging Face content Stack v1 lineage",
            "no AWS dependency no Software Heritage raw API",
            "pinned Stack revision must equal current main",
            "raw source is tokenized never executed and remains private",
            "seed-1 raw checkpoint SHA required",
            "density probe only before lock",
            "no labeling proper no 14B forward no optimizer no training",
            "post-lock launcher must be created after lock commit",
            "tests/test_speculative_depth_d0_spec.py",
            "colab/run_stage5_paper2_d0_prelock.py",
        ],
        "env": {},
    },
    "wall_clock_latency_descriptive": {
        "path": "colab/STAGE5_WALL_CLOCK_LATENCY_CELL.py",
        "markers": [
            "STAGE5_WALL_CLOCK_LATENCY_CELL_VERSION",
            "wall_clock_latency_descriptive",
            "tests/test_wall_clock_latency.py",
            "colab/run_stage5_wall_clock_latency.py",
            "single hardware configuration, batch size 1, registered evaluation paths",
            "drive.mount",
        ],
        "env": {},
    },
    "paper2_phase_t1_p0": {
        "path": "colab/STAGE5_PAPER2_PHASE_T1_P0_CELL.py",
        "markers": [
            "STAGE5_PAPER2_PHASE_T1_P0_CELL_VERSION",
            "paper2_internal_token_t1_p0_v1",
            "P0 pilot only registered T1 remains locked",
            "seed 9999 1500 steps checkpoints 500 1000 1500",
            "dedicated 256 row pilot slice never enters a registered set",
            "exact 70 percent control 30 percent mechanism rehearsal",
            "no silent sweep extension when both recalls miss 0.60",
            "exact normalized trie multi-token candidate scoring",
            "complete ten-cell calibration grid before coefficient lock",
            "Phase A letter symbols and loop-target alignment preflight",
            "tests/test_internal_think_token_t1.py",
            "colab/run_stage5_paper2_phase_t1_p0.py",
        ],
        "env": {},
    },
    "paper2_t1_lite": {
        "path": "colab/STAGE5_PAPER2_T1_LITE_CELL.py",
        "markers": [
            "STAGE5_PAPER2_T1_LITE_CELL_VERSION",
            "paper2_t1_lite_locked_v1",
            "locked_before_training commit 44459f30",
            "pretraining manifest amendment finalized 8ea5ce64",
            "10500 staged steps boundaries 500 2500 6500 8500",
            "gate4 exact 4608 forced stop plus 1024 forced continue",
            "final-step EMA primary raw secondary",
            "D0 and C track remain unauthorized",
            "tests/test_internal_think_token_t1_lite.py",
            "colab/run_stage5_paper2_t1_lite.py",
        ],
        "env": {},
    },
    "paper2_t1_lite_r": {
        "path": "colab/STAGE5_PAPER2_T1_LITE_R_CELL.py",
        "markers": [
            "STAGE5_PAPER2_T1_LITE_R_CELL_VERSION",
            "paper2_t1_lite_r_locked_v2_canonical_hash",
            "locked before launcher commit ae2793ac",
            "seed 1 raw final-step primary",
            "continuous EMA and stage-reset EMA passive shadows",
            "atomic hashed stage states 500 2500 6500 8500 10500",
            "gate4 exact 4608 forced stop plus 1024 forced continue",
            "D0 build-only no labeling GPU no training",
            "C track design-stage RG-12 unauthorized",
            "tests/test_internal_think_token_t1_r_spec.py",
            "colab/run_stage5_paper2_t1_lite_r.py",
        ],
        "env": {},
    },
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
    "master_sequence_status": {
        "path": "colab/STAGE5_MASTER_SEQUENCE_STATUS_CELL.py",
        "markers": [
            "STAGE5_MASTER_SEQUENCE_STATUS_CELL_VERSION",
            "master_sequence_status_v1",
            "MASTER_SEQUENCE_STATUS",
            "colab/print_current_stage5_action.py",
            "colab/review_stage5_reentry.py",
            "colab/review_stage5_recovery.py",
            "colab/review_stage5_phase1_gate.py",
            "colab/review_stage5_recovery_curriculum.py",
            "colab/plan_stage5_curriculum_scaleup.py",
            "Phase 1 Gate Review",
            "Stage 4 Recovery Curriculum Readiness",
            "Claim-Sized Curriculum Scale-Up Plan",
            "NEXT_COLAB_SEQUENCE excerpt",
            "STAGE5_MASTER_SEQUENCE_STATUS_DISCONNECT",
            "runtime.unassign",
        ],
        "env": {},
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
    "claim_curriculum_scaleup_cpu": {
        "path": "colab/CURRICULUM_ARTIFACT_PIPELINE_CELL.py",
        "markers": [
            "data/curriculum/claim_direct_deep_001",
            "STAGE5_CURRICULUM_RUN_PROVIDER_RESPONSES",
            "STAGE5_CURRICULUM_PROVIDER_LIMIT",
            "STAGE5_CURRICULUM_MODEL_MAP_JSON",
            "STAGE5_CURRICULUM_OPUS_MODEL",
            "STAGE5_CURRICULUM_GLM_MODEL",
            "STAGE5_CURRICULUM_WEAK_REFERENCE_MODEL",
            "resolve_model_map",
            "MIN_POSITIVE_ROWS = 2000",
            'MIN_MODE_ROWS = "direct=1000,deep_narrow=1000"',
            "MIN_TARGET_LOOP_ROWS",
            "target_loop_requirements",
            "phase_order_warning",
            '"math,science"',
            '"1,2,5,9"',
            '"122"',
            "RUN_PROVIDER_RESPONSES",
            "training/run_curriculum_pipeline_from_artifacts.py",
            "training/run_curriculum_job_responses.py",
            "training/check_curriculum_sft_gate.py",
            "REFUSE_GPU_RUNTIME",
            "Refusing to run CPU/API curriculum pipeline",
            "runtime.unassign",
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
    "t1_lite_ema_audit": {
        "path": "colab/STAGE5_T1_LITE_EMA_AUDIT_CELL.py",
        "markers": [
            "STAGE5_T1_LITE_EMA_AUDIT_CELL_VERSION",
            "registered_negative verdict immutable posthoc read-only",
            "raw EMA stage boundaries interpolation group swaps",
            "control rows recurrent block bridge localization",
            "no training no optimizer step no checkpoint mutation",
            "missing stage checkpoints reported as partial evidence",
            "tests/test_t1_lite_ema_audit.py",
        ],
        "env": {
            "STAGE5_T1_EMA_AUDIT_DTYPE": "bfloat16",
            "STAGE5_T1_EMA_AUDIT_BATCH_SIZE": "8",
            "STAGE5_T1_EMA_AUDIT_DISCONNECT": "0",
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
            "arc_easy,arc_challenge,gpqa_lite",
            "colab/run_stage5_benchmark_suite.py",
            "colab/assess_stage5_benchmark_suite.py",
            "spectral_source_health_override",
            "stage4_benchmark_source_gate=spectral_health_override",
            "tests/test_stage5_benchmark_suite.py",
            "tests/test_stage5_benchmark_assessment.py",
            "STAGE5_DEBIASED_BENCHMARK_DISCONNECT",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_DEBIASED_BENCHMARKS": "arc_easy,arc_challenge,gpqa_lite",
            "STAGE5_DEBIASED_ARC_EASY_LIMIT": "128",
            "STAGE5_DEBIASED_ARC_CHALLENGE_LIMIT": "128",
            "STAGE5_DEBIASED_GPQA_LIMIT": "16",
            "STAGE5_DEBIASED_USE_LEARNED_LOOP_CONTROL": "1",
            "STAGE5_DEBIASED_BENCHMARK_DISCONNECT": "1",
        },
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
            "effective_pathways_checkpoint_source=",
            'phase_gate.get("checkpoint")',
            "eval/eval_effective_pathways.py",
            "tests/test_pathway_diversity.py",
            "colab.master_sequence_gate",
            "STAGE5_ALLOW_PRE_PHASE1_BREADTH",
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
            "STAGE5_REENTRY_NORM_SEEDS": "0",
            "STAGE5_REENTRY_NORM_K": "4",
            "STAGE5_REENTRY_NORM_MAX_NEW_TOKENS": "80",
            "STAGE5_REENTRY_NORM_LIMIT": "8",
            "STAGE5_REENTRY_NORM_DISCONNECT": "1",
        },
    },
    "reentry_norm_recover_only": {
        "path": "colab/STAGE5_REENTRY_NORM_RECOVER_CELL.py",
        "markers": [
            "STAGE5_REENTRY_NORM_RECOVER_CELL_VERSION",
            "stage5_reentry_norm_recover_v1",
            "STAGE5_REENTRY_NORM_RECOVER_SOURCE",
            "stage5_reentry_norm_*",
            "No complete stage5_reentry_norm_* artifact found on Drive",
            "colab/assess_stage5_reentry.py",
            "colab/review_stage5_reentry.py",
            "Recover Stage 5 re-entry norm",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_REENTRY_NORM_RECOVER_DISCONNECT": "1",
        },
    },
    "reentry_repair_smoke": {
        "path": "colab/STAGE5_REENTRY_REPAIR_SMOKE_CELL.py",
        "markers": [
            "STAGE5_REENTRY_REPAIR_SMOKE_CELL_VERSION",
            "stage5_reentry_repair_smoke_v2_spectral_optional",
            "bridge_gate_override",
            "bridge_reset_identity",
            "reentry_rescale_mode",
            "reentry_adapter_mode",
            "STAGE5_REENTRY_REPAIR_ADAPTER_MODE",
            "training/train_phase1_ponder.py",
            "eval/eval_reentry_drift.py",
            "Loop-1 Preservation",
            "loop1_preservation",
            "parse_train_log_metrics",
            "train_log_metrics",
            "existing_train_log_metrics",
            "resume_retrain=train_phase1_ponder",
            "require_gpu_runtime",
            "Stage 3 re-entry repair smoke requires an attached GPU runtime",
            "Training Smoke Metrics",
            "STAGE5_REENTRY_REPAIR_REQUIRE_NORM_PASS",
            "STAGE5_REENTRY_REPAIR_ALLOW_FALLBACK_CHECKPOINT",
            "Stage 3 repair smoke requires a checkpoint from the passed Stage 2 norm assessment",
            "stage2_norm_assessment",
            "current_pointer_norm_assessment_candidates",
            "stage2_norm_assessment_source=current_pointer",
            "incremental_backup",
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
            "STAGE5_REENTRY_REPAIR_OPTIMIZER_MODULES": "bridge,reentry,halt",
            "STAGE5_REENTRY_REPAIR_DISCONNECT": "1",
        },
    },
    "reentry_spectral_repair_smoke": {
        "path": "colab/STAGE5_REENTRY_REPAIR_SMOKE_CELL.py",
        "markers": [
            "STAGE5_REENTRY_REPAIR_SMOKE_CELL_VERSION",
            "stage5_reentry_repair_smoke_v2_spectral_optional",
            "bridge_gate_override",
            "bridge_reset_identity",
            "reentry_rescale_mode",
            "reentry_adapter_mode",
            "STAGE5_REENTRY_REPAIR_ADAPTER_MODE",
            "training/train_phase1_ponder.py",
            "eval/eval_reentry_drift.py",
            "Loop-1 Preservation",
            "loop1_preservation",
            "parse_train_log_metrics",
            "train_log_metrics",
            "existing_train_log_metrics",
            "resume_retrain=train_phase1_ponder",
            "require_gpu_runtime",
            "Stage 3 re-entry repair smoke requires an attached GPU runtime",
            "Training Smoke Metrics",
            "STAGE5_REENTRY_REPAIR_REQUIRE_NORM_PASS",
            "STAGE5_REENTRY_REPAIR_ALLOW_FALLBACK_CHECKPOINT",
            "Stage 3 repair smoke requires a checkpoint from the passed Stage 2 norm assessment",
            "stage2_norm_assessment",
            "current_pointer_norm_assessment_candidates",
            "stage2_norm_assessment_source=current_pointer",
            "incremental_backup",
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
            "STAGE5_REENTRY_REPAIR_OPTIMIZER_MODULES": "bridge,reentry,halt",
            "STAGE5_REENTRY_REPAIR_ADAPTER_MODE": "spectral",
            "STAGE5_REENTRY_REPAIR_DISCONNECT": "1",
        },
    },
    "reentry_recovery_training": {
        "path": "colab/STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py",
        "markers": [
            "STAGE5_REENTRY_RECOVERY_CELL_VERSION",
            "reentry_recovery_training_v5_fixed_tail_damper",
            "STAGE5_REENTRY_RECOVERY_REPAIR_ASSESSMENT",
            "run_bounded_recovery_training_with_reentry_repair",
            "stage5_reentry_recovery_training",
            "write_reentry_recovery_wrapper_summary",
            "repair_assessment_recovery_block_reason",
            "current_pointer_repair_assessment_candidates",
            "stage3_repair_assessment_source=current_pointer",
            "stage5_reentry_repair_smoke",
            "STAGE5_CURRICULUM_RESUME_FROM",
            "STAGE5_CURRICULUM_USE_LEARNED_LOOP_CONTROL",
            "STAGE5_CURRICULUM_OPTIMIZER_MODULES",
            "colab/run_stage5_curriculum_sft.py",
            "tests/test_stage5_curriculum_sft.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_REENTRY_RECOVERY_STEPS": "75",
            "STAGE5_REENTRY_RECOVERY_LR": "5e-6",
            "STAGE5_REENTRY_RECOVERY_OPTIMIZER_MODULES": "bridge,reentry,halt,lora",
            "STAGE5_REENTRY_RECOVERY_DISCONNECT": "1",
        },
    },
    "reentry_tail_damper_recovery_training": {
        "path": "colab/STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py",
        "markers": [
            "STAGE5_REENTRY_RECOVERY_CELL_VERSION",
            "reentry_recovery_training_v5_fixed_tail_damper",
            "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_SOURCE_SUMMARY",
            "STAGE5_CURRICULUM_REENTRY_TAIL_DAMPER_PATH",
            "STAGE5_CURRICULUM_REENTRY_TAIL_DAMPER_STRENGTH",
            "fixed_tail_damper_depth_readout",
            "rescued_vs_loop1",
            "harmed_vs_loop1",
            "eval/eval_tail_damper_depth_sweep.py",
            "colab/run_stage5_curriculum_sft.py",
            "tests/test_stage5_curriculum_sft.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_SOURCE_SUMMARY": (
                "outputs/stage5/"
                "stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/"
                "summary.json"
            ),
            "STAGE5_REENTRY_RECOVERY_REENTRY_TAIL_DAMPER_STRENGTH": "1.0",
            "STAGE5_REENTRY_RECOVERY_STEPS": "100",
            "STAGE5_REENTRY_RECOVERY_LR": "5e-6",
            "STAGE5_REENTRY_RECOVERY_OPTIMIZER_MODULES": "bridge,reentry,halt,lora",
            "STAGE5_REENTRY_RECOVERY_REENTRY_RESCALE_MODE": "entry_rms",
            "STAGE5_REENTRY_RECOVERY_REENTRY_ADAPTER_MODE": "spectral",
            "STAGE5_REENTRY_RECOVERY_REQUIRE_TARGET_LOOP_GRADIENT": "0",
            "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_READOUT_STRENGTHS": "0,1.0",
            "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_READOUT_LIMIT": "512",
            "STAGE5_REENTRY_RECOVERY_DISCONNECT": "0",
        },
    },
    "reentry_tail_damper_capacity_lora32_training": {
        "path": "colab/STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py",
        "markers": [
            "STAGE5_REENTRY_RECOVERY_CELL_VERSION",
            "reentry_recovery_training_v5_fixed_tail_damper",
            "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_SOURCE_SUMMARY",
            "STAGE5_CURRICULUM_REENTRY_TAIL_DAMPER_PATH",
            "STAGE5_CURRICULUM_REENTRY_TAIL_DAMPER_STRENGTH",
            "STAGE5_REENTRY_RECOVERY_LORA_RANK",
            "STAGE5_CURRICULUM_LORA_RANK",
            "fixed_tail_damper_depth_readout",
            "rescued_vs_loop1",
            "harmed_vs_loop1",
            "colab/run_stage5_curriculum_sft.py",
            "tests/test_stage5_curriculum_sft.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_SOURCE_SUMMARY": (
                "outputs/stage5/"
                "stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/"
                "summary.json"
            ),
            "STAGE5_REENTRY_RECOVERY_REENTRY_TAIL_DAMPER_STRENGTH": "1.0",
            "STAGE5_REENTRY_RECOVERY_STEPS": "100",
            "STAGE5_REENTRY_RECOVERY_LR": "5e-6",
            "STAGE5_REENTRY_RECOVERY_OPTIMIZER_MODULES": "bridge,reentry,halt,lora",
            "STAGE5_REENTRY_RECOVERY_LORA_RANK": "32",
            "STAGE5_REENTRY_RECOVERY_LORA_ALPHA": "64",
            "STAGE5_REENTRY_RECOVERY_REENTRY_RESCALE_MODE": "entry_rms",
            "STAGE5_REENTRY_RECOVERY_REENTRY_ADAPTER_MODE": "spectral",
            "STAGE5_REENTRY_RECOVERY_REQUIRE_TARGET_LOOP_GRADIENT": "0",
            "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_READOUT_STRENGTHS": "0,1.0",
            "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_READOUT_LIMIT": "512",
            "STAGE5_REENTRY_RECOVERY_DISCONNECT": "0",
        },
    },
    "reentry_capacity_localization_rank64": {
        "path": "colab/STAGE5_CAPACITY_LOCALIZATION_CELL.py",
        "markers": [
            "STAGE5_CAPACITY_LOCALIZATION_CELL_VERSION",
            "capacity_localization_v1",
            "STAGE5_CAPACITY_LOCALIZATION_MOUNT_DRIVE_FIRST",
            "STAGE5_CAPACITY_LOCALIZATION_RANKS",
            "trainable_parameter_ledger",
            "stage5_current_capacity_localization_summary",
            "STAGE5_REENTRY_RECOVERY_LORA_RANK",
            "fixed_tail_damper_depth_readout",
            "stage5_reentry_recovery_20260627_190155",
            "colab/STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_CAPACITY_LOCALIZATION_RANKS": "64",
            "STAGE5_CAPACITY_LOCALIZATION_BASELINE_SUMMARIES": (
                "outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json"
            ),
            "STAGE5_CAPACITY_LOCALIZATION_TAIL_DAMPER_SOURCE_SUMMARY": (
                "outputs/stage5/"
                "stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/"
                "summary.json"
            ),
            "STAGE5_CAPACITY_LOCALIZATION_TRACE_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json"
            ),
            "STAGE5_CAPACITY_LOCALIZATION_STEPS": "100",
            "STAGE5_CAPACITY_LOCALIZATION_LR": "5e-6",
            "STAGE5_CAPACITY_LOCALIZATION_DISCONNECT": "0",
        },
    },
    "unfreeze_recurrent_curriculum": {
        "path": "colab/STAGE5_UNFREEZE_RECURRENT_CURRICULUM_CELL.py",
        "markers": [
            "STAGE5_UNFREEZE_RECURRENT_CURRICULUM_CELL_VERSION",
            "unfreeze_recurrent_curriculum_v1",
            "training/train_unfrozen_recurrent.py",
            "merge_lora_before_unfreeze",
            "require_lora_loaded_before_merge",
            "STAGE5_UNFREEZE_SOURCE_SUMMARY",
            "STAGE5_UNFREEZE_MAX_STEPS",
            "STAGE5_BENCHMARK_LORA_RANK",
            "STAGE5_UNFREEZE_DISCONNECT",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_UNFREEZE_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json"
            ),
            "STAGE5_UNFREEZE_MAX_STEPS": "50",
            "STAGE5_UNFREEZE_END_LOOP": "8",
            "STAGE5_UNFREEZE_DISCONNECT": "0",
        },
    },
    "prelude_path_development": {
        "path": "colab/STAGE5_UNFREEZE_RECURRENT_CURRICULUM_CELL.py",
        "markers": [
            "STAGE5_UNFREEZE_RECURRENT_CURRICULUM_CELL_VERSION",
            "unfreeze_recurrent_curriculum_v1",
            "training/train_unfrozen_recurrent.py",
            "merge_lora_before_unfreeze",
            "require_lora_loaded_before_merge",
            "STAGE5_UNFREEZE_BRIDGE_PRELUDE_GRAD_MULTIPLIER",
            "STAGE5_UNFREEZE_RUN_PRELUDE_ABLATION",
            "eval/eval_prelude_ablation.py",
            "prelude_ablation_summary",
            "save_every",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_UNFREEZE_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json"
            ),
            "STAGE5_UNFREEZE_RUN_ID": "stage5_prelude_path_development",
            "STAGE5_UNFREEZE_MAX_STEPS": "300",
            "STAGE5_UNFREEZE_SAVE_EVERY": "50",
            "STAGE5_UNFREEZE_END_LOOP": "8",
            "STAGE5_UNFREEZE_LOG_EVERY": "10",
            "STAGE5_UNFREEZE_BRIDGE_PRELUDE_GRAD_MULTIPLIER": "10.0",
            "STAGE5_UNFREEZE_RUN_PRELUDE_ABLATION": "1",
            "STAGE5_UNFREEZE_PRELUDE_ABLATION_LIMIT": "8",
            "STAGE5_UNFREEZE_DISCONNECT": "0",
        },
    },
    "reentry_tail_damper_recovery_readout_only": {
        "path": "colab/STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py",
        "markers": [
            "STAGE5_REENTRY_RECOVERY_CELL_VERSION",
            "reentry_recovery_training_v5_fixed_tail_damper",
            "STAGE5_REENTRY_RECOVERY_READOUT_ONLY",
            "readout_only=true",
            "STAGE5_REENTRY_RECOVERY_CHILD_SUMMARY",
            "fixed_tail_damper_depth_readout",
            "rescued_vs_loop1",
            "harmed_vs_loop1",
            "eval/eval_tail_damper_depth_sweep.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_REENTRY_RECOVERY_READOUT_ONLY": "1",
            "STAGE5_REENTRY_RECOVERY_CHILD_SUMMARY": (
                "outputs/stage5/stage5_reentry_recovery_20260627_131940_curriculum_sft/summary.json"
            ),
            "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_SOURCE_SUMMARY": (
                "outputs/stage5/"
                "stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/"
                "summary.json"
            ),
            "STAGE5_REENTRY_RECOVERY_REENTRY_TAIL_DAMPER_STRENGTH": "1.0",
            "STAGE5_REENTRY_RECOVERY_RUN_ID": "stage5_reentry_recovery_20260627_131940_readout",
            "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_READOUT_STRENGTHS": "0,1.0",
            "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_READOUT_LIMIT": "512",
            "STAGE5_REENTRY_RECOVERY_DISCONNECT": "0",
        },
    },
    "depth_signal_confirmation": {
        "path": "colab/STAGE5_DEPTH_SIGNAL_CONFIRMATION_CELL.py",
        "markers": [
            "STAGE5_DEPTH_SIGNAL_CONFIRMATION_CELL_VERSION",
            "depth_signal_confirmation_v1",
            "Stage 4: depth-routing recovery",
            "Stage 5: depth-signal benchmark",
            "STAGE5_REENTRY_RECOVERY_HALT_TARGET_NLL_WEIGHT",
            "STAGE5_REENTRY_RECOVERY_LOOP_CONTROL_CE_WEIGHT",
            "STAGE5_REENTRY_RECOVERY_REENTRY_ADAPTER_MODE",
            "STAGE5_DEBIASED_BENCHMARK_SUITE_PROFILE",
            "depth_signal_confirmation",
            "open_hard_arc_challenge",
            "STAGE5_DEBIASED_ASSESS_REQUIRED_BENCHMARKS",
            "STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py",
            "STAGE5_DEBIASED_BENCHMARK_SUITE_CELL.py",
            "depth_signal_confirmation_complete=true",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_REENTRY_RECOVERY_STEPS": "100",
            "STAGE5_REENTRY_RECOVERY_HALT_TARGET_NLL_WEIGHT": "6.0",
            "STAGE5_REENTRY_RECOVERY_LOOP_CONTROL_CE_WEIGHT": "5.0",
            "STAGE5_REENTRY_RECOVERY_REENTRY_ADAPTER_MODE": "spectral",
            "STAGE5_DEBIASED_BENCHMARK_SUITE_PROFILE": "depth_signal_confirmation",
            "STAGE5_DEBIASED_BENCHMARKS": "arc_easy,arc_challenge,open_hard_arc_challenge",
            "STAGE5_DEBIASED_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated",
            "STAGE5_DEBIASED_ARC_EASY_LIMIT": "128",
            "STAGE5_DEBIASED_ARC_CHALLENGE_LIMIT": "256",
            "STAGE5_DEBIASED_OPEN_HARD_ARC_CHALLENGE_LIMIT": "256",
            "STAGE5_DEBIASED_ASSESS_REQUIRED_BENCHMARKS": "arc_challenge,open_hard_arc_challenge",
            "STAGE5_DEPTH_SIGNAL_CONFIRMATION_DISCONNECT": "1",
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
            "candidate_conversion_checkpoint_source=",
            'phase_gate.get("checkpoint")',
            "eval/eval_best_of_k_jsonl.py",
            "tests/test_eval_best_of_k_generation.py",
            "colab.master_sequence_gate",
            "STAGE5_ALLOW_PRE_PHASE1_BREADTH",
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
            "stage5_current_source_summary.txt",
            "dense_mcq_source_pointer",
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
            "STAGE5_DENSE_MCQ_RUN_ID": "stage5_dense_mcq_trace_sft_control_current",
            "STAGE5_DENSE_MCQ_BENCHMARKS": "arc_easy,arc_challenge",
            "STAGE5_DENSE_MCQ_ARC_EASY_LIMIT": "256",
            "STAGE5_DENSE_MCQ_ARC_CHALLENGE_LIMIT": "256",
            "STAGE5_DENSE_MCQ_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated",
            "STAGE5_DENSE_MCQ_AGGREGATES": "mean",
            "STAGE5_DENSE_MCQ_COMMIT_CHECKPOINT": "0",
            "STAGE5_DENSE_MCQ_PUSH": "1",
            "STAGE5_DENSE_MCQ_DISCONNECT": "1",
        },
    },
    "traced_sft_competence_preserving_pipeline": {
        "path": "colab/STAGE5_COMPETENCE_PRESERVING_PIPELINE_CELL.py",
        "markers": [
            "STAGE5_COMPETENCE_PRESERVING_PIPELINE_CELL_VERSION",
            "competence_preserving_pipeline_v2",
            "traced_sft_competence_preserving_pipeline",
            "STAGE5_COMPETENCE_SOURCE_SUMMARY",
            "STAGE5_COMPETENCE_MOUNT_DRIVE_FIRST",
            "FORCE_DRIVE_REMOUNT",
            "drive.mount",
            "stage5_debiased_benchmark_assessment_20260625_121302",
            "colab/run_stage5_competence_preserving_pipeline.py",
            "tests/test_stage5_competence_preserving_pipeline.py",
            "tests/test_stage5_balanced_arc_mix_gate.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_COMPETENCE_SOURCE_SUMMARY": (
                "outputs/stage5/stage5_debiased_benchmark_assessment_20260625_121302/summary.json"
            ),
            "STAGE5_COMPETENCE_PIPELINE_RUN_ID": (
                "stage5_competence_recovery_20260625_from_reentry"
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
    "forced_depth_diagnostic": {
        "path": "colab/STAGE5_FORCED_DEPTH_DIAGNOSTIC_CELL.py",
        "markers": [
            "STAGE5_FORCED_DEPTH_DIAGNOSTIC_CELL_VERSION",
            "forced_depth_arc_v1",
            "STAGE5_BENCHMARK_FORCED_LOOP_COUNT",
            "STAGE5_FORCED_DEPTH_SOURCE_SUMMARY",
            "forced_depth_requested_source_summary",
            "checkpoint_bearing_source_summary",
            "STAGE5_FORCED_DEPTH_LORA_RANK",
            "STAGE5_BENCHMARK_LORA_RANK",
            "forced_depth_lora_rank",
            "require_cuda_runtime",
            "Forced-depth diagnostic requires an attached GPU runtime",
            "content_question_only,cyclic_label_aggregated",
            "eval/analyze_depth_sweep.py",
            "--score_target",
            "cyclic_label_aggregated",
            "tests/test_eval_mcq_loop_diagnostics.py",
            "tests/test_stage5_benchmark_suite.py",
            "tests/test_analyze_depth_sweep.py",
            "tests/test_stage5_benchmark_assessment.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_FORCED_DEPTH_LOOPS": "1,2,3,4,8",
            "STAGE5_FORCED_DEPTH_BENCHMARKS": "arc_challenge",
            "STAGE5_FORCED_DEPTH_ARC_CHALLENGE_LIMIT": "256",
            "STAGE5_FORCED_DEPTH_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated",
            "STAGE5_FORCED_DEPTH_DRIVE_BACKUP": "0",
            "STAGE5_FORCED_DEPTH_DISCONNECT": "0",
        },
    },
    "deterministic_final_gate": {
        "path": "colab/STAGE5_DETERMINISTIC_FINAL_GATE_CELL.py",
        "markers": [
            "STAGE5_DETERMINISTIC_FINAL_GATE_CELL_VERSION",
            "deterministic_final_gate_v2_nested_selector",
            "eval/evaluate_rescue_detectability.py",
            "eval/evaluate_rescue_selector_kfold.py",
            "cyclic_label_aggregated",
            "permutation_mean",
            "closed_at_detectability_gate",
            "selector_transfer_passed",
            "nested_outer_fold_train_only",
            "STAGE5_FINAL_GATE_RESUME_EXISTING",
            "STAGE5_BENCHMARK_BASE_REUSE_RUN_ID",
            "STAGE5_FINAL_GATE_OPEN_HARD_ARC_CHALLENGE_SPLIT",
            "STAGE5_BENCHMARK_FORCED_LOOP_COUNT",
            "STAGE5_BENCHMARK_LORA_RANK",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_FINAL_GATE_DISCOVERY_SWEEP": "outputs/stage5/stage5_prelude_forced_depth_heldout_arc_loop1248/summary.json",
            "STAGE5_FINAL_GATE_SOURCE_SUMMARY": "outputs/stage5/stage5_prelude_path_development/summary.json",
            "STAGE5_FINAL_GATE_BENCHMARKS": "arc_easy,arc_challenge,open_hard_arc_challenge",
            "STAGE5_FINAL_GATE_LOOPS": "1,2,4,8",
            "STAGE5_FINAL_GATE_ARC_EASY_LIMIT": "all",
            "STAGE5_FINAL_GATE_ARC_CHALLENGE_LIMIT": "all",
            "STAGE5_FINAL_GATE_OPEN_HARD_ARC_CHALLENGE_LIMIT": "256",
            "STAGE5_FINAL_GATE_OPEN_HARD_ARC_CHALLENGE_SPLIT": "test",
            "STAGE5_FINAL_GATE_KFOLD_FOLDS": "5",
            "STAGE5_FINAL_GATE_KFOLD_INNER_FOLDS": "4",
            "STAGE5_FINAL_GATE_KFOLD_POLICY_LABELS": "zero_harm,harm_budget_1",
            "STAGE5_FINAL_GATE_RESUME_EXISTING": "1",
            "STAGE5_FINAL_GATE_DISCONNECT": "0",
        },
    },
    "rescue_predictability_analysis": {
        "path": "colab/STAGE5_RESCUE_PREDICTABILITY_CELL.py",
        "markers": [
            "STAGE5_RESCUE_PREDICTABILITY_CELL_VERSION",
            "rescue_predictability_precursor_v1",
            "STAGE5_RESCUE_PREDICTABILITY_SWEEP_SUMMARY",
            "eval/analyze_rescue_predictability.py",
            "stage5_current_rescue_predictability_summary",
            "tests/test_analyze_rescue_predictability.py",
            "oriented AUC",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_RESCUE_PREDICTABILITY_SCORE_TARGET": "content_question_only",
            "STAGE5_RESCUE_PREDICTABILITY_AGGREGATE": "mean",
            "STAGE5_RESCUE_PREDICTABILITY_DISCONNECT": "1",
        },
    },
    "rescue_detectability_gate": {
        "path": "colab/STAGE5_RESCUE_DETECTABILITY_CELL.py",
        "markers": [
            "STAGE5_RESCUE_DETECTABILITY_CELL_VERSION",
            "rescue_detectability_gate_v1",
            "STAGE5_RESCUE_DETECTABILITY_SWEEP_SUMMARY",
            "eval/evaluate_rescue_detectability.py",
            "observed_minus_null_p95",
            "stage5_current_rescue_detectability_summary",
            "tests/test_evaluate_rescue_detectability.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_RESCUE_DETECTABILITY_SCORE_TARGET": "content_question_only",
            "STAGE5_RESCUE_DETECTABILITY_AGGREGATE": "mean",
            "STAGE5_RESCUE_DETECTABILITY_SHRINKAGES": "0.1,1,10",
            "STAGE5_RESCUE_DETECTABILITY_REPEATS": "64",
            "STAGE5_RESCUE_DETECTABILITY_PERMUTATIONS": "128",
            "STAGE5_RESCUE_DETECTABILITY_DISCONNECT": "1",
        },
    },
    "rescue_selector_transfer": {
        "path": "colab/STAGE5_RESCUE_SELECTOR_TRANSFER_CELL.py",
        "markers": [
            "STAGE5_RESCUE_SELECTOR_TRANSFER_CELL_VERSION",
            "rescue_selector_transfer_v1",
            "STAGE5_RESCUE_SELECTOR_DISCOVERY_SWEEP_SUMMARY",
            "STAGE5_RESCUE_SELECTOR_HELDOUT_SWEEP_SUMMARY",
            "eval/evaluate_rescue_selector_transfer.py",
            "transferred_curve_summary",
            "stage5_current_rescue_selector_transfer_summary",
            "tests/test_evaluate_rescue_selector_transfer.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_RESCUE_SELECTOR_SCORE_TARGET": "content_question_only",
            "STAGE5_RESCUE_SELECTOR_AGGREGATE": "mean",
            "STAGE5_RESCUE_SELECTOR_TRANSFER_DISCONNECT": "1",
        },
    },
    "tail_convergence_selector": {
        "path": "colab/STAGE5_TAIL_CONVERGENCE_SELECTOR_CELL.py",
        "markers": [
            "STAGE5_TAIL_CONVERGENCE_SELECTOR_CELL_VERSION",
            "tail_convergence_selector_v1",
            "eval/evaluate_tail_convergence_selector.py",
            "tail_deceleration_12_minus_23",
            "stage5_current_tail_convergence_selector_summary",
            "tests/test_tail_convergence_selector.py",
            "restore_checkpoint",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_TAIL_CONVERGENCE_SELECTOR_SCORE_TARGET": "content_question_only",
            "STAGE5_TAIL_CONVERGENCE_SELECTOR_AGGREGATE": "mean",
            "STAGE5_TAIL_CONVERGENCE_N_TAIL": "7",
            "STAGE5_TAIL_CONVERGENCE_DROP_TOP": "1",
            "STAGE5_TAIL_CONVERGENCE_SELECTOR_DISCONNECT": "1",
        },
    },
    "heldout_router_validation": {
        "path": "colab/STAGE5_HELDOUT_ROUTER_VALIDATION_CELL.py",
        "markers": [
            "STAGE5_HELDOUT_ROUTER_VALIDATION_CELL_VERSION",
            "heldout_router_validation_v1",
            "STAGE5_HELDOUT_ROUTER_DISCOVERY_SUMMARY",
            "STAGE5_BENCHMARK_FORCED_LOOP_COUNT",
            "eval/evaluate_depth_router_transfer.py",
            "eval/eval_latent_criticality.py",
            "content_question_only,cyclic_label_aggregated",
            "open_hard_arc_challenge",
            "tests/test_evaluate_depth_router_transfer.py",
            "latent_criticality",
            "router_transfer_content_question_only",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_HELDOUT_ROUTER_LOOPS": "1,2,3",
            "STAGE5_HELDOUT_ROUTER_BENCHMARKS": "arc_easy,arc_challenge,open_hard_arc_challenge",
            "STAGE5_HELDOUT_ROUTER_ARC_EASY_OFFSET": "256",
            "STAGE5_HELDOUT_ROUTER_ARC_EASY_LIMIT": "128",
            "STAGE5_HELDOUT_ROUTER_ARC_CHALLENGE_OFFSET": "256",
            "STAGE5_HELDOUT_ROUTER_ARC_CHALLENGE_LIMIT": "128",
            "STAGE5_HELDOUT_ROUTER_OPEN_HARD_ARC_CHALLENGE_SPLIT": "test",
            "STAGE5_HELDOUT_ROUTER_OPEN_HARD_ARC_CHALLENGE_OFFSET": "0",
            "STAGE5_HELDOUT_ROUTER_OPEN_HARD_ARC_CHALLENGE_LIMIT": "128",
            "STAGE5_HELDOUT_ROUTER_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated",
            "STAGE5_HELDOUT_ROUTER_MIN_ORACLE_CAPTURE": "0.2",
            "STAGE5_HELDOUT_ROUTER_RUN_LATENT_CRITICALITY": "1",
            "STAGE5_LATENT_CRITICALITY_MAX_EXAMPLES_PER_BENCHMARK": "64",
            "STAGE5_LATENT_CRITICALITY_JACOBIAN_EXAMPLES_PER_BENCHMARK": "8",
            "STAGE5_LATENT_CRITICALITY_JACOBIAN_RANDOM_PROBES": "1",
            "STAGE5_LATENT_CRITICALITY_JACOBIAN_EPSILON": "0.02",
            "STAGE5_HELDOUT_ROUTER_DRIVE_BACKUP": "0",
            "STAGE5_HELDOUT_ROUTER_DISCONNECT": "0",
        },
    },
    "latent_criticality_probe": {
        "path": "colab/STAGE5_LATENT_CRITICALITY_CELL.py",
        "markers": [
            "STAGE5_LATENT_CRITICALITY_CELL_VERSION",
            "latent_criticality_probe_v1",
            "STAGE5_LATENT_CRITICALITY_SOURCE_SUMMARY",
            "eval/eval_latent_criticality.py",
            "STAGE5_LATENT_CRITICALITY_JACOBIAN_EXAMPLES_PER_BENCHMARK",
            "restored_latent_criticality_checkpoint",
            "finite_difference_random_gain",
            "tests/test_eval_latent_criticality.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_LATENT_CRITICALITY_MAX_EXAMPLES_PER_BENCHMARK": "64",
            "STAGE5_LATENT_CRITICALITY_JACOBIAN_EXAMPLES_PER_BENCHMARK": "8",
            "STAGE5_LATENT_CRITICALITY_JACOBIAN_RANDOM_PROBES": "1",
            "STAGE5_LATENT_CRITICALITY_JACOBIAN_EPSILON": "0.02",
            "STAGE5_LATENT_CRITICALITY_DISCONNECT": "0",
        },
    },
    "reentry_covariance_check": {
        "path": "colab/STAGE5_REENTRY_COVARIANCE_CHECK_CELL.py",
        "markers": [
            "STAGE5_REENTRY_COVARIANCE_CHECK_CELL_VERSION",
            "reentry_covariance_prebuild_v1",
            "STAGE5_REENTRY_COVARIANCE_SOURCE_SUMMARY",
            "eval/eval_reentry_covariance_check.py",
            "covariance_match_check",
            "directional_prebuild_gate",
            "restored_reentry_covariance_checkpoint",
            "tests/test_eval_reentry_covariance_check.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_REENTRY_COVARIANCE_PROMPTS": "eval/smoke_exact_tasks_v2.jsonl",
            "STAGE5_REENTRY_COVARIANCE_LIMIT": "14",
            "STAGE5_REENTRY_COVARIANCE_SUBSPACE_RANK": "8",
            "STAGE5_REENTRY_COVARIANCE_ANALYSIS_RANK": "8",
            "STAGE5_REENTRY_COVARIANCE_DISCONNECT": "0",
        },
    },
    "reentry_tail_diagnostic": {
        "path": "colab/STAGE5_REENTRY_TAIL_DIAGNOSTIC_CELL.py",
        "markers": [
            "STAGE5_REENTRY_TAIL_DIAGNOSTIC_CELL_VERSION",
            "reentry_tail_resolved_v1",
            "STAGE5_REENTRY_TAIL_SOURCE_SUMMARY",
            "eval/eval_reentry_tail_diagnostic.py",
            "tail_decomposition",
            "harmed_rescued_tail_readout",
            "restored_reentry_tail_checkpoint",
            "tests/test_eval_reentry_tail_diagnostic.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_REENTRY_TAIL_ARC_LIMIT": "256",
            "STAGE5_REENTRY_TAIL_LOOP_COUNTS": "1,2,3,4,8",
            "STAGE5_REENTRY_TAIL_N": "7",
            "STAGE5_REENTRY_TAIL_DISCONNECT": "0",
        },
    },
    "reentry_tail_damper_sweep": {
        "path": "colab/STAGE5_REENTRY_TAIL_DAMPER_SWEEP_CELL.py",
        "markers": [
            "STAGE5_REENTRY_TAIL_DAMPER_SWEEP_CELL_VERSION",
            "tail_damper_tradeoff_v1",
            "energy_oracle_tradeoff",
            "eval/eval_tail_damper_depth_sweep.py",
            "tests/test_reentry_tail_damper.py",
            "tests/test_eval_tail_damper_depth_sweep.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_TAIL_DAMPER_ARC_LIMIT": "256",
            "STAGE5_TAIL_DAMPER_STRENGTHS": "0,0.25,0.5,0.75,1.0",
            "STAGE5_TAIL_DAMPER_SCORE_LOOPS": "1,2,3",
            "STAGE5_TAIL_DAMPER_TAIL_LOOPS": "1,2,3,4,8",
            "STAGE5_TAIL_DAMPER_N": "7",
            "STAGE5_TAIL_DAMPER_DISCONNECT": "0",
        },
    },
    "reentry_tail_damper_heldout": {
        "path": "colab/STAGE5_REENTRY_TAIL_DAMPER_SWEEP_CELL.py",
        "markers": [
            "STAGE5_REENTRY_TAIL_DAMPER_SWEEP_CELL_VERSION",
            "tail_damper_tradeoff_v1",
            "energy_oracle_tradeoff",
            "eval/eval_tail_damper_depth_sweep.py",
            "tests/test_reentry_tail_damper.py",
            "tests/test_eval_tail_damper_depth_sweep.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_TAIL_DAMPER_ARC_OFFSET": "256",
            "STAGE5_TAIL_DAMPER_ARC_LIMIT": "256",
            "STAGE5_TAIL_DAMPER_STRENGTHS": "0,0.25,0.5,0.75,1.0",
            "STAGE5_TAIL_DAMPER_SCORE_LOOPS": "1,2,3",
            "STAGE5_TAIL_DAMPER_TAIL_LOOPS": "1,2,3,4,8",
            "STAGE5_TAIL_DAMPER_N": "7",
            "STAGE5_TAIL_DAMPER_DISCONNECT": "0",
        },
    },
    "reentry_tail_damper_powered_arc_train": {
        "path": "colab/STAGE5_REENTRY_TAIL_DAMPER_SWEEP_CELL.py",
        "markers": [
            "STAGE5_REENTRY_TAIL_DAMPER_SWEEP_CELL_VERSION",
            "tail_damper_tradeoff_v1",
            "energy_oracle_tradeoff",
            "eval/eval_tail_damper_depth_sweep.py",
            "tests/test_reentry_tail_damper.py",
            "tests/test_eval_tail_damper_depth_sweep.py",
            "runtime.unassign",
        ],
        "env": {
            "STAGE5_TAIL_DAMPER_ARC_CONFIG": "ARC-Challenge",
            "STAGE5_TAIL_DAMPER_ARC_SPLIT": "train",
            "STAGE5_TAIL_DAMPER_ARC_OFFSET": "0",
            "STAGE5_TAIL_DAMPER_ARC_LIMIT": "512",
            "STAGE5_TAIL_DAMPER_STRENGTHS": "0,0.5,1.0",
            "STAGE5_TAIL_DAMPER_SCORE_LOOPS": "1,2,3",
            "STAGE5_TAIL_DAMPER_TAIL_LOOPS": "1,2,3,4,8",
            "STAGE5_TAIL_DAMPER_N": "7",
            "STAGE5_TAIL_DAMPER_SCORE_TARGET": "option_text",
            "STAGE5_TAIL_DAMPER_DISCONNECT": "0",
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
    "synthetic_depth_task": {
        "path": "colab/STAGE5_SYNTHETIC_DEPTH_TASK_CELL.py",
        "markers": [
            "STAGE5_SYNTHETIC_DEPTH_TASK_CELL_VERSION",
            "synthetic_depth_task_v2_mcq_aligned",
            "training/generate_synthetic_depth_task.py",
            "eval/eval_synthetic_depth_matrix.py",
            "distinct_prefix_length_depth_plus_one",
            "frontier_strictly_expands",
            "STAGE5_SYNTH_DEPTH_MAX_DEPTH",
            "STAGE5_SYNTH_DEPTH_ROWS_PER_DEPTH",
            "STAGE5_SYNTH_DEPTH_MAX_STEPS",
            "STAGE5_SYNTH_DEPTH_TRAIN_FORMAT",
            "STAGE5_SYNTH_DEPTH_RUN_BASE_EVAL",
            "train_mcq_option_text_sft.jsonl",
            "tests/test_synthetic_depth_task.py",
            "tests/test_eval_synthetic_depth_matrix.py",
        ],
        "env": {
            "STAGE5_SYNTH_DEPTH_MAX_DEPTH": "8",
            "STAGE5_SYNTH_DEPTH_MAX_LOOPS": "8",
            "STAGE5_SYNTH_DEPTH_ROWS_PER_DEPTH": "24",
            "STAGE5_SYNTH_DEPTH_MAX_STEPS": "25",
            "STAGE5_SYNTH_DEPTH_TRAIN_FORMAT": "free_answer",
            "STAGE5_SYNTH_DEPTH_RUN_BASE_EVAL": "0",
            "STAGE5_SYNTH_DEPTH_DISCONNECT": "0",
        },
    },
    "synthetic_depth_primitive_curve": {
        "path": "colab/STAGE5_SYNTHETIC_DEPTH_PRIMITIVE_CURVE_CELL.py",
        "markers": [
            "STAGE5_SYNTHETIC_DEPTH_PRIMITIVE_CURVE_CELL_VERSION",
            "synthetic_depth_primitive_curve_v1",
            "Phase 1 changes only N and keeps max_depth=1",
            "STAGE5_SYNTH_PRIMITIVE_N_VALUES",
            "STAGE5_SYNTH_PRIMITIVE_ROWS_PER_DEPTH",
            "STAGE5_SYNTH_PRIMITIVE_MAX_STEPS",
            "STAGE5_SYNTH_PRIMITIVE_BACKUP_CHECKPOINTS_TO_DRIVE",
            "colab/summarize_synthetic_depth_primitive_curve.py",
            "tests/test_synthetic_depth_primitive_curve.py",
            "primitive_accuracy_bar",
        ],
        "env": {
            "STAGE5_SYNTH_PRIMITIVE_N_VALUES": "8,12,16",
            "STAGE5_SYNTH_PRIMITIVE_ROWS_PER_DEPTH": "256",
            "STAGE5_SYNTH_PRIMITIVE_MAX_STEPS": "500",
            "STAGE5_SYNTH_PRIMITIVE_BACKUP_CHECKPOINTS_TO_DRIVE": "0",
            "STAGE5_SYNTH_PRIMITIVE_DISCONNECT": "0",
        },
    },
    "synthetic_depth_staged_staircase": {
        "path": "colab/STAGE5_SYNTHETIC_DEPTH_STAGED_STAIRCASE_CELL.py",
        "markers": [
            "STAGE5_SYNTHETIC_DEPTH_STAGED_STAIRCASE_CELL_VERSION",
            "synthetic_depth_staged_staircase_v1",
            "Phase 2 resumes from primitive N=16 and uses loop_loss_mode=target",
            "STAGE5_SYNTH_STAIRCASE_PRIMITIVE_CURVE_SUMMARY",
            "stage_depth_le2_finished",
            "train_depth_le2_mcq_option_text_sft.jsonl",
            "train_depth_le4_mcq_option_text_sft.jsonl",
            "\"loop_loss_mode\": \"target\"",
            "tests/test_recurrent_wrapper_tiny.py::test_target_loop_loss_mode_uses_requested_loop_on_tiny_model",
        ],
        "env": {
            "STAGE5_SYNTH_STAIRCASE_N_SYMBOLS": "16",
            "STAGE5_SYNTH_STAIRCASE_MAX_DEPTH": "4",
            "STAGE5_SYNTH_STAIRCASE_ROWS_PER_DEPTH": "256",
            "STAGE5_SYNTH_STAIRCASE_STAGE12_STEPS": "500",
            "STAGE5_SYNTH_STAIRCASE_STAGE1234_STEPS": "1000",
            "STAGE5_SYNTH_STAIRCASE_EVAL_LOOPS": "1,2,3,4",
            "STAGE5_SYNTH_STAIRCASE_BACKUP_CHECKPOINTS_TO_DRIVE": "1",
            "STAGE5_SYNTH_STAIRCASE_DISCONNECT": "0",
        },
    },
    "synthetic_depth_chain_supervision": {
        "path": "colab/STAGE5_SYNTHETIC_DEPTH_CHAIN_SUPERVISION_CELL.py",
        "markers": [
            "STAGE5_SYNTHETIC_DEPTH_CHAIN_SUPERVISION_CELL_VERSION",
            "synthetic_depth_chain_supervision_v1",
            "Phase B uses loop_loss_mode=per_loop_labels and train_chain_label_sft",
            "STAGE5_SYNTH_CHAIN_RUN_AFTER_TRAIN_DIAGNOSTIC",
            "phase_a_failed_checkpoint_train_split",
            "chain_depth_le2_finished",
            "train_chain_label_depth_le2_sft.jsonl",
            "train_chain_label_depth_le4_sft.jsonl",
            "\"loop_loss_mode\": \"per_loop_labels\"",
            "tests/test_recurrent_wrapper_tiny.py::test_per_loop_label_loss_mode_uses_active_intermediate_labels_on_tiny_model",
        ],
        "env": {
            "STAGE5_SYNTH_CHAIN_SOURCE_SUMMARY": "outputs/stage5/stage5_synthetic_depth_staged_staircase_20260701_180040/summary.json",
            "STAGE5_SYNTH_CHAIN_RUN_AFTER_TRAIN_DIAGNOSTIC": "auto",
            "STAGE5_SYNTH_CHAIN_STAGE12_STEPS": "500",
            "STAGE5_SYNTH_CHAIN_STAGE1234_STEPS": "1000",
            "STAGE5_SYNTH_CHAIN_BRIDGE_PRELUDE_GRAD_MULTIPLIER": "8.0",
            "STAGE5_SYNTH_CHAIN_EVAL_LOOPS": "1,2,3,4",
            "STAGE5_SYNTH_CHAIN_BACKUP_CHECKPOINTS_TO_DRIVE": "1",
            "STAGE5_SYNTH_CHAIN_DISCONNECT": "0",
        },
    },
    "synthetic_depth_split_bridge_microtest": {
        "path": "colab/STAGE5_SPLIT_BRIDGE_MICROTEST_CELL.py",
        "markers": [
            "STAGE5_SPLIT_BRIDGE_MICROTEST_CELL_VERSION",
            "split_bridge_true_lr_microtest_v1",
            "bridge_projection_mode=split",
            "true bridge_prelude_lr_multiplier param group",
            "STAGE5_SPLIT_MICRO_PRELUDE_LR_MULTIPLIER",
            "STAGE5_SPLIT_MICRO_STAGE12_STEPS",
            "STAGE5_SPLIT_MICRO_STAGE1234_STEPS",
            "tests/test_bridge.py",
            "tests/test_train_unfrozen_recurrent.py",
            "tests/test_eval_synthetic_depth_matrix.py",
        ],
        "env": {
            "STAGE5_SPLIT_MICRO_N_SYMBOLS": "8",
            "STAGE5_SPLIT_MICRO_MAX_DEPTH": "4",
            "STAGE5_SPLIT_MICRO_ROWS_PER_DEPTH": "4",
            "STAGE5_SPLIT_MICRO_DTYPE": "float32",
            "STAGE5_SPLIT_MICRO_PRELUDE_LR_MULTIPLIER": "10.0",
            "STAGE5_SPLIT_MICRO_STAGE12_STEPS": "2000",
            "STAGE5_SPLIT_MICRO_STAGE1234_STEPS": "4000",
            "STAGE5_SPLIT_MICRO_BACKUP_CHECKPOINTS_TO_DRIVE": "1",
            "STAGE5_SPLIT_MICRO_DISCONNECT": "0",
        },
    },
    "synthetic_depth_chain_scaled_corrected": {
        "path": "colab/STAGE5_CHAIN_SCALED_CORRECTED_CELL.py",
        "markers": [
            "STAGE5_CHAIN_SCALED_CORRECTED_CELL_VERSION",
            "chain_scaled_corrected_v1",
            "active-label evaluator scores f^k(x) for k <= depth",
            "full-symbol chain SFT avoids MCQ label bottleneck",
            "eval/eval_synthetic_depth_active_labels.py",
            "colab/run_stage5_chain_scaled_corrected.py",
            "STAGE5_CHAIN_CORRECTED_STAGE12_STEPS",
            "STAGE5_CHAIN_CORRECTED_STAGE1234_STEPS",
            "STAGE5_CHAIN_CORRECTED_PRELUDE_LR_MULTIPLIER",
            "tests/test_eval_synthetic_depth_active_labels.py",
            "tests/test_stage5_chain_scaled_corrected.py",
        ],
        "env": {
            "STAGE5_CHAIN_CORRECTED_N_SYMBOLS": "16",
            "STAGE5_CHAIN_CORRECTED_MAX_DEPTH": "4",
            "STAGE5_CHAIN_CORRECTED_ROWS_PER_DEPTH": "256",
            "STAGE5_CHAIN_CORRECTED_HELDOUT_ROWS_PER_DEPTH": "64",
            "STAGE5_CHAIN_CORRECTED_TRAIN_EVAL_ROWS_PER_DEPTH": "64",
            "STAGE5_CHAIN_CORRECTED_DTYPE": "bfloat16",
            "STAGE5_CHAIN_CORRECTED_PRELUDE_LR_MULTIPLIER": "10.0",
            "STAGE5_CHAIN_CORRECTED_STAGE12_STEPS": "2000",
            "STAGE5_CHAIN_CORRECTED_STAGE1234_STEPS": "4000",
            "STAGE5_CHAIN_CORRECTED_BACKUP_CHECKPOINTS_TO_DRIVE": "1",
            "STAGE5_CHAIN_CORRECTED_DISCONNECT": "0",
        },
    },
    "peft_ponder_closure": {
        "path": "colab/STAGE5_PEFT_PONDER_CLOSURE_CELL.py",
        "markers": [
            "STAGE5_PEFT_PONDER_CLOSURE_CELL_VERSION",
            "peft_ponder_closure_v2",
            "frozen_lora",
            "controller_only",
            "reject_muon",
            "bridge_prelude_lr_multiplier",
            "require_frozen_base_hash",
            "base_capability_canary_64",
            "canary_baseline_gate",
            "pinned_checkout",
            "eval/eval_peft_identity.py",
            "eval/eval_ponder_depth.py",
            "colab/run_stage5_peft_ponder_closure.py",
            "tests/test_peft_ponder_closure.py",
            "tests/test_stage5_peft_ponder_closure.py",
        ],
        "env": {
            "STAGE5_PEFT_DTYPE": "bfloat16",
            "STAGE5_PEFT_LR": "1e-5",
            "STAGE5_PONDER_LR": "1e-4",
            "STAGE5_PONDER_BETA": "0.02",
            "STAGE5_PONDER_TARGET_NLL_WEIGHT": "0.1",
            "STAGE5_PEFT_PONDER_DISCONNECT": "0",
        },
    },
    "adapter_budget_arm_e": {
        "path": "colab/STAGE5_ADAPTER_BUDGET_ARM_CELL.py",
        "markers": [
            "STAGE5_ADAPTER_BUDGET_ARM_CELL_VERSION",
            "adapter_budget_arm_e_v1",
            "fresh_base_qwen_surgery",
            "same_reader_final_rows.jsonl",
            "pretrained_base_hash_unchanged",
            "track_loop_dose",
            "adapter_budget_depth_profile",
            "tests/test_adapter_budget_arm.py",
            "colab/run_stage5_adapter_budget_arm.py",
        ],
        "env": {
            "STAGE5_ADAPTER_BUDGET_DTYPE": "bfloat16",
            "STAGE5_ADAPTER_BUDGET_DISCONNECT": "0",
        },
    },
    "adapter_parity_e3a": {
        "path": "colab/STAGE5_ADAPTER_PARITY_BATTERY_CELL.py",
        "markers": [
            "STAGE5_ADAPTER_PARITY_BATTERY_CELL_VERSION",
            "adapter_parity_battery_v1",
            "stage5_adapter_parity_e3a",
            "tests/test_adapter_parity_battery.py",
            "bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839",
        ],
        "env": {"STAGE5_ADAPTER_PARITY_DISCONNECT": "0"},
    },
    "adapter_parity_e2": {
        "path": "colab/STAGE5_ADAPTER_PARITY_BATTERY_CELL.py",
        "markers": [
            "STAGE5_ADAPTER_PARITY_BATTERY_CELL_VERSION",
            "adapter_parity_battery_v1",
            "stage5_adapter_parity_e2",
            "tests/test_adapter_parity_battery.py",
            "bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839",
        ],
        "env": {"STAGE5_ADAPTER_PARITY_DISCONNECT": "0"},
    },
    "adapter_parity_e4": {
        "path": "colab/STAGE5_ADAPTER_PARITY_BATTERY_CELL.py",
        "markers": [
            "STAGE5_ADAPTER_PARITY_BATTERY_CELL_VERSION",
            "adapter_parity_battery_v1",
            "stage5_adapter_parity_e4",
            "tests/test_adapter_parity_battery.py",
            "bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839",
        ],
        "env": {"STAGE5_ADAPTER_PARITY_DISCONNECT": "0"},
    },
    "adapter_verbal_transference_e3b": {
        "path": "colab/STAGE5_ADAPTER_PARITY_BATTERY_CELL.py",
        "markers": [
            "STAGE5_ADAPTER_PARITY_BATTERY_CELL_VERSION",
            "stage5_adapter_verbal_transference_e3b_20260720",
            "tests/test_adapter_verbal_transference.py",
            "bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839",
        ],
        "env": {"STAGE5_ADAPTER_PARITY_DISCONNECT": "0"},
    },
    "adapter_verbal_transference_e3b_salvage": {
        "path": "colab/STAGE5_ADAPTER_PARITY_BATTERY_CELL.py",
        "markers": [
            "STAGE5_ADAPTER_PARITY_BATTERY_CELL_VERSION",
            "stage5_adapter_verbal_transference_e3b_20260720",
            "evaluation_only_after_preregistered_guardrail_stop",
            "tests/test_adapter_verbal_transference.py",
            "bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839",
        ],
        "env": {"STAGE5_ADAPTER_PARITY_DISCONNECT": "0"},
    },
    "paper1_closure_receipts": {
        "path": "colab/STAGE5_PAPER1_CLOSURE_RECEIPTS_CELL.py",
        "markers": [
            "STAGE5_PAPER1_CLOSURE_RECEIPTS_CELL_VERSION",
            "paper1_closure_receipts_v1",
            "PAPER1_EXPERIMENTAL_CLOSURE_RECEIPTS_20260718",
            "tests/test_paper1_closure_receipts.py",
            "Bonferroni",
            "manuscript prose was not edited",
            "colab/build_paper1_closure_receipts.py",
        ],
        "env": {},
    },
    "depth_selector_bounded_assessment": {
        "path": "colab/STAGE5_DEPTH_SELECTOR_CELL.py",
        "markers": [
            "STAGE5_DEPTH_SELECTOR_CELL_VERSION",
            "depth_selector_bounded_v1",
            "N24_KEEPER_SHA256",
            "frozen_parameter_hash",
            "S1_supervised_depth_reading",
            "S2_ponder_outcome",
            "canary_exemption",
            "pinned_checkout",
            "colab/run_stage5_depth_selector_bounded.py",
            "tests/test_depth_selector_bounded.py",
            "tests/test_stage5_depth_selector_bounded.py",
        ],
        "env": {
            "STAGE5_DEPTH_SELECTOR_DTYPE": "bfloat16",
            "STAGE5_DEPTH_SELECTOR_STEPS": "2000",
            "STAGE5_DEPTH_SELECTOR_BATCH_SIZE": "8",
            "STAGE5_DEPTH_SELECTOR_EXTRACTION_BATCH": "8",
            "STAGE5_DEPTH_SELECTOR_S1_LR": "1e-3",
            "STAGE5_DEPTH_SELECTOR_S2_LR": "1e-3",
            "STAGE5_DEPTH_SELECTOR_DISCONNECT": "0",
        },
    },
    "depth_extrapolation_eval": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "depth_extrapolation_eval",
            "eval/eval_synthetic_depth_artifact_check.py",
            "colab/run_stage5_depth_extrapolation_eval.py",
            "STAGE5_EXTRAP_DEPTHS",
            "STAGE5_EXTRAP_MAX_LOOPS",
            "tests/test_stage5_chain_consolidation.py",
        ],
        "env": {
            "STAGE5_EXTRAP_CHECKPOINT": "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
            "STAGE5_EXTRAP_N_SYMBOLS": "16",
            "STAGE5_EXTRAP_DEPTHS": "1,2,3,4,5,6",
            "STAGE5_EXTRAP_ROWS_PER_DEPTH": "64",
            "STAGE5_EXTRAP_MAX_LOOPS": "6",
            "STAGE5_EXTRAP_FORCE_LOOPS": "1",
            "STAGE5_EXTRAP_DTYPE": "bfloat16",
            "STAGE5_EXTRAP_DISCONNECT": "0",
        },
    },
    "synthetic_probe_battery": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "chain_continuation_probe_readout",
            "synthetic_probe_battery",
            "eval/eval_synthetic_depth_probe.py",
            "loop_index_probe",
            "colab/run_stage5_synthetic_probe_battery.py",
            "tests/test_eval_synthetic_depth_probe.py",
            "tests/test_stage5_chain_consolidation.py",
        ],
        "env": {
            "STAGE5_PROBE_CHECKPOINT": "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
            "STAGE5_PROBE_N_SYMBOLS": "16",
            "STAGE5_PROBE_MAX_DEPTH": "6",
            "STAGE5_PROBE_ROWS_PER_DEPTH": "64",
            "STAGE5_PROBE_LOOP_COUNTS": "1,2,3,4,5,6",
            "STAGE5_PROBE_TARGET_STEPS": "0,1,2,3,4,5,6",
            "STAGE5_PROBE_FEATURE_TRANSFORMS": "raw,unit_norm",
            "STAGE5_PROBE_DISCONNECT": "0",
        },
    },
    "post_anneal_readouts": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "post_anneal_readouts",
            "eval/analyze_synthetic_reader_alignment.py",
            "colab/run_stage5_post_anneal_readouts.py",
            "STAGE5_POST_ANNEAL_SOURCE_SUMMARY",
            "STAGE5_EXTRAP_CHECKPOINT",
            "STAGE5_PROBE_CHECKPOINT",
            "router_leak_exclusion",
            "state_envelope",
            "tests/test_analyze_synthetic_reader_alignment.py",
            "tests/test_stage5_chain_consolidation.py",
        ],
        "env": {
            "STAGE5_POST_ANNEAL_SOURCE_SUMMARY": "outputs/stage5/stage5_chain_anneal_20260703_160250/summary.json",
            "STAGE5_POST_ANNEAL_N_SYMBOLS": "16",
            "STAGE5_POST_ANNEAL_EXTRAP_DEPTHS": "1,2,3,4,5,6",
            "STAGE5_POST_ANNEAL_MAX_LOOPS": "6",
            "STAGE5_POST_ANNEAL_PROBE_MAX_DEPTH": "6",
            "STAGE5_POST_ANNEAL_ROWS_PER_DEPTH": "64",
            "STAGE5_POST_ANNEAL_PROBE_LOOP_COUNTS": "1,2,3,4,5,6",
            "STAGE5_POST_ANNEAL_PROBE_TARGET_STEPS": "0,1,2,3,4,5,6",
            "STAGE5_PROBE_FEATURE_TRANSFORMS": "raw,unit_norm",
            "STAGE5_POST_ANNEAL_DTYPE": "bfloat16",
            "STAGE5_POST_ANNEAL_DISCONNECT": "0",
        },
    },
    "post_anneal_extended_readouts": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "post_anneal_readouts",
            "eval/analyze_synthetic_reader_alignment.py",
            "colab/run_stage5_post_anneal_readouts.py",
            "STAGE5_POST_ANNEAL_SOURCE_SUMMARY",
            "STAGE5_PROBE_FEATURE_TRANSFORMS",
            "router_leak_exclusion",
            "state_envelope",
            "tests/test_stage5_chain_consolidation.py",
        ],
        "env": {
            "STAGE5_POST_ANNEAL_SOURCE_SUMMARY": "outputs/stage5/stage5_chain_anneal_20260703_160250/summary.json",
            "STAGE5_POST_ANNEAL_N_SYMBOLS": "16",
            "STAGE5_POST_ANNEAL_EXTRAP_DEPTHS": "1,2,3,4,5,6,7,8",
            "STAGE5_POST_ANNEAL_MAX_LOOPS": "8",
            "STAGE5_POST_ANNEAL_PROBE_MAX_DEPTH": "8",
            "STAGE5_POST_ANNEAL_ROWS_PER_DEPTH": "128",
            "STAGE5_POST_ANNEAL_PROBE_LOOP_COUNTS": "1,2,3,4,5,6,7,8",
            "STAGE5_POST_ANNEAL_PROBE_TARGET_STEPS": "0,1,2,3,4,5,6,7,8",
            "STAGE5_PROBE_FEATURE_TRANSFORMS": "raw,unit_norm",
            "STAGE5_POST_ANNEAL_DTYPE": "bfloat16",
            "STAGE5_POST_ANNEAL_DISCONNECT": "0",
        },
    },
    "chain_anneal_to_outcome": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "chain_anneal_to_outcome",
            "loop_loss_mode='annealed_chain_to_outcome'",
            "colab/run_stage5_chain_anneal_to_outcome.py",
            "STAGE5_ANNEAL_TOTAL_STEPS",
            "STAGE5_ANNEAL_PRELUDE_LR_MULT",
            "tests/test_recurrent_wrapper_tiny.py::test_annealed_chain_to_outcome_loss_mixes_chain_and_target_ce_on_tiny_model",
        ],
        "env": {
            "STAGE5_ANNEAL_INIT_CHECKPOINT": "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
            "STAGE5_ANNEAL_N_SYMBOLS": "16",
            "STAGE5_ANNEAL_MAX_DEPTH": "4",
            "STAGE5_ANNEAL_ROWS_PER_DEPTH": "256",
            "STAGE5_ANNEAL_HELDOUT_ROWS_PER_DEPTH": "64",
            "STAGE5_ANNEAL_TOTAL_STEPS": "2000",
            "STAGE5_ANNEAL_HOLD_FRAC": "0.5",
            "STAGE5_ANNEAL_PRELUDE_LR_MULT": "10.0",
            "STAGE5_ANNEAL_BACKUP_CHECKPOINTS_TO_DRIVE": "1",
            "STAGE5_ANNEAL_DISCONNECT": "0",
        },
    },
    "chain_continuation_attribution": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "chain_continuation_attribution",
            "colab/run_stage5_chain_continuation_attribution.py",
            "STAGE5_CHAIN_CONTINUATION_EXTRAP_DEPTHS",
            "STAGE5_ANNEAL_LOOP_LOSS_MODE",
            "per_loop_labels",
            "tests/test_stage5_chain_consolidation.py",
        ],
        "env": {
            "STAGE5_CHAIN_CONTINUATION_INIT_CHECKPOINT": "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
            "STAGE5_CHAIN_CONTINUATION_N_SYMBOLS": "16",
            "STAGE5_CHAIN_CONTINUATION_MAX_DEPTH": "4",
            "STAGE5_CHAIN_CONTINUATION_ROWS_PER_DEPTH": "256",
            "STAGE5_CHAIN_CONTINUATION_HELDOUT_ROWS_PER_DEPTH": "64",
            "STAGE5_CHAIN_CONTINUATION_TOTAL_STEPS": "2000",
            "STAGE5_CHAIN_CONTINUATION_SAVE_MID_FRAC": "0.5",
            "STAGE5_CHAIN_CONTINUATION_PRELUDE_LR_MULT": "10.0",
            "STAGE5_CHAIN_CONTINUATION_EXTRAP_DEPTHS": "1,2,3,4,5,6,7,8",
            "STAGE5_CHAIN_CONTINUATION_EXTRAP_ROWS_PER_DEPTH": "128",
            "STAGE5_CHAIN_CONTINUATION_EXTRAP_MAX_LOOPS": "8",
            "STAGE5_CHAIN_CONTINUATION_BACKUP_CHECKPOINTS_TO_DRIVE": "1",
            "STAGE5_CHAIN_CONTINUATION_DTYPE": "bfloat16",
            "STAGE5_CHAIN_CONTINUATION_DISCONNECT": "0",
        },
    },
    "chain_continuation_probe_readout": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "synthetic_probe_battery",
            "colab/run_stage5_synthetic_probe_battery.py",
            "STAGE5_PROBE_CHECKPOINT",
            "STAGE5_PROBE_FEATURE_TRANSFORMS",
            "loop_index_probe",
            "state_envelope",
            "tests/test_stage5_chain_consolidation.py",
        ],
        "env": {
            "STAGE5_PROBE_CHECKPOINT": "outputs/stage5/stage5_chain_continuation_attribution_20260704_163056/summary.json",
            "STAGE5_PROBE_RUN_ID": "stage5_chain_continuation_probe_readout",
            "STAGE5_PROBE_N_SYMBOLS": "16",
            "STAGE5_PROBE_MAX_DEPTH": "10",
            "STAGE5_PROBE_ROWS_PER_DEPTH": "128",
            "STAGE5_PROBE_LOOP_COUNTS": "1,2,3,4,5,6,7,8,9,10",
            "STAGE5_PROBE_TARGET_STEPS": "0,1,2,3,4,5,6,7,8,9,10",
            "STAGE5_PROBE_FEATURE_TRANSFORMS": "raw,unit_norm",
            "STAGE5_PROBE_DTYPE": "bfloat16",
            "STAGE5_PROBE_DISCONNECT": "0",
        },
    },
    "depth_support_route_comparison": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "depth_support_route_comparison",
            "colab/run_stage5_depth_support_route_comparison.py",
            "STAGE5_ROUTE_FROZEN_EVAL_ID",
            "STAGE5_ROUTE_TRAIN_MAX_DEPTH",
            "STAGE5_ROUTE_EVAL_MAX_DEPTH",
            "SELECTION_MIN_CORRECT",
            "NONREGRESSION_FLOORS",
            "tests/test_stage5_chain_consolidation.py",
        ],
        "env": {
            "STAGE5_ROUTE_INIT_CHECKPOINT": "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
            "STAGE5_ROUTE_N_SYMBOLS": "16",
            "STAGE5_ROUTE_TRAIN_MAX_DEPTH": "6",
            "STAGE5_ROUTE_EVAL_MAX_DEPTH": "10",
            "STAGE5_ROUTE_ROWS_PER_DEPTH": "256",
            "STAGE5_ROUTE_FROZEN_ROWS_PER_DEPTH": "128",
            "STAGE5_ROUTE_TOTAL_STEPS": "2000",
            "STAGE5_ROUTE_PRELUDE_LR_MULT": "10.0",
            "STAGE5_ROUTE_FROZEN_EVAL_ID": "stage5_synthetic_depth_frozen_eval_v1",
            "STAGE5_ROUTE_BACKUP_CHECKPOINTS_TO_DRIVE": "1",
            "STAGE5_ROUTE_DTYPE": "bfloat16",
            "STAGE5_ROUTE_DISCONNECT": "0",
        },
    },
    "depth_support_ladder8": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "depth_support_ladder8",
            "colab/run_stage5_depth_support_ladder.py",
            "STAGE5_LADDER_FROZEN_EVAL_ID",
            "STAGE5_LADDER_TRAIN_MAX_DEPTH",
            "STAGE5_LADDER_EVAL_MAX_DEPTH",
            "STRONG_SCALING_MIN_CORRECT = 91",
            "ASYMPTOTE_REJECTION_MIN_CORRECT = 79",
            "CHANCE_REJECTION_MIN_CORRECT = 14",
            "NONREGRESSION_FLOORS = {\"1\": 0.93",
            "tests/test_stage5_chain_consolidation.py",
        ],
        "env": {
            "STAGE5_LADDER_INIT_CHECKPOINT": "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
            "STAGE5_LADDER_ROUTE_SOURCE_SUMMARY": "outputs/stage5/stage5_depth_support_route_20260705_124320/summary.json",
            "STAGE5_LADDER_N_SYMBOLS": "16",
            "STAGE5_LADDER_TRAIN_MAX_DEPTH": "8",
            "STAGE5_LADDER_EVAL_MAX_DEPTH": "14",
            "STAGE5_LADDER_ROWS_PER_DEPTH": "256",
            "STAGE5_LADDER_FROZEN_ROWS_PER_DEPTH": "128",
            "STAGE5_LADDER_STEPS": "2000",
            "STAGE5_LADDER_PRELUDE_LR_MULT": "10.0",
            "STAGE5_LADDER_FROZEN_EVAL_ID": "stage5_synthetic_depth_frozen_eval_v2_depth14",
            "STAGE5_LADDER_BASE_FROZEN_EVAL_ID": "stage5_synthetic_depth_frozen_eval_v1",
            "STAGE5_LADDER_BACKUP_CHECKPOINTS_TO_DRIVE": "1",
            "STAGE5_LADDER_DTYPE": "bfloat16",
            "STAGE5_LADDER_DISCONNECT": "0",
        },
    },
    "support8_probe_readout": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "support8_probe_readout",
            "colab/run_stage5_support8_probe_readout.py",
            "STAGE5_SUPPORT8_SOURCE_SUMMARY",
            "STAGE5_SUPPORT8_PROBE_LOOP_COUNTS",
            "STAGE5_SUPPORT8_PROBE_TARGET_STEPS",
            "STAGE5_SUPPORT8_PROBE_FEATURE_TRANSFORMS",
            "state_envelope",
            "loop_index_probe",
            "router_leak_exclusion",
            "tests/test_eval_synthetic_depth_probe.py",
            "tests/test_stage5_chain_consolidation.py",
        ],
        "env": {
            "STAGE5_SUPPORT8_SOURCE_SUMMARY": "outputs/stage5/stage5_depth_support_ladder8_20260705_204923/summary.json",
            "STAGE5_SUPPORT8_PROBE_LOOP_COUNTS": "1,2,3,4,5,6,7,8,9,10,11,12,13,14",
            "STAGE5_SUPPORT8_PROBE_TARGET_STEPS": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14",
            "STAGE5_SUPPORT8_PROBE_FEATURE_TRANSFORMS": "raw,unit_norm,rms_norm",
            "STAGE5_SUPPORT8_PROBE_DTYPE": "bfloat16",
            "STAGE5_SUPPORT8_PROBE_DISCONNECT": "0",
        },
    },
    "support8_dose_arm": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "support8_dose_arm",
            "colab/run_stage5_support8_dose_arm.py",
            "STAGE5_DOSE_SOURCE_SUMMARY",
            "STAGE5_DOSE_STEPS",
            "soft_depth10_min_correct",
            "soft_depth11_min_correct",
            "STRONG_SCALING_MIN_CORRECT = 91",
            "ASYMPTOTE_REJECTION_MIN_CORRECT = 79",
            "CHANCE_REJECTION_MIN_CORRECT = 14",
            "tests/test_stage5_chain_consolidation.py",
        ],
        "env": {
            "STAGE5_DOSE_SOURCE_SUMMARY": "outputs/stage5/stage5_depth_support_ladder8_20260705_204923/summary.json",
            "STAGE5_DOSE_MAX_DEPTH": "8",
            "STAGE5_DOSE_ROWS_PER_DEPTH": "256",
            "STAGE5_DOSE_STEPS": "2000",
            "STAGE5_DOSE_SEED": "20260705",
            "STAGE5_DOSE_PRELUDE_LR_MULT": "10.0",
            "STAGE5_DOSE_BACKUP_CHECKPOINTS_TO_DRIVE": "1",
            "STAGE5_DOSE_DTYPE": "bfloat16",
            "STAGE5_DOSE_DISCONNECT": "0",
        },
    },
    "same_reader_final_symbol": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "same_reader_final_symbol",
            "colab/run_stage5_same_reader_final_symbol.py",
            "eval/eval_synthetic_depth_final_symbol.py",
            "STAGE5_SAME_READER_SOURCE_SUMMARY",
            "full-symbol argmax",
            "tests/test_eval_synthetic_depth_final_symbol.py",
        ],
        "env": {
            "STAGE5_SAME_READER_SOURCE_SUMMARY": "outputs/stage5/stage5_support8_dose_arm_20260706_153028/summary.json",
            "STAGE5_SAME_READER_MAX_LOOPS": "14",
            "STAGE5_SAME_READER_DTYPE": "bfloat16",
            "STAGE5_SAME_READER_DISCONNECT": "0",
        },
    },
    "n24_same_reader_receipt": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "n24_same_reader_receipt",
            "colab/run_stage5_same_reader_final_symbol.py",
            "STAGE5_SAME_READER_EXPECT_IDENTITY_WITH_ACTIVE",
            "same_reader_active_identity_check",
            "stage5_n24_same_reader_final_symbol_current",
        ],
        "env": {
            "STAGE5_SAME_READER_RUN_ID": "stage5_n24_same_reader_final_symbol_current",
            "STAGE5_SAME_READER_SOURCE_SUMMARY": "outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json",
            "STAGE5_SAME_READER_DATA_JSONL": "outputs/stage5/stage5_synthetic_depth_frozen_eval_v3_depth22_n24/data/test_chain_mcq.jsonl",
            "STAGE5_SAME_READER_MAX_LOOPS": "22",
            "STAGE5_SAME_READER_EXPECT_IDENTITY_WITH_ACTIVE": "1",
            "STAGE5_SAME_READER_IDENTITY_TOLERANCE": "0.000001",
            "STAGE5_SAME_READER_DTYPE": "bfloat16",
            "STAGE5_SAME_READER_DISCONNECT": "0",
        },
    },
    "support6_seed_replication": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "support6_seed_replication",
            "colab/run_stage5_support6_seed_replication.py",
            "STAGE5_SUPPORT6_REPLICATION_SEEDS",
            "STAGE5_ROUTE_TRAIN_SEED",
            "tests/test_stage5_support6_seed_replication.py",
        ],
        "env": {
            "STAGE5_SUPPORT6_REPLICATION_SEEDS": "20260716,20260726",
            "STAGE5_SUPPORT6_REPLICATION_STEPS": "2000",
            "STAGE5_SUPPORT6_REPLICATION_ROWS_PER_DEPTH": "256",
            "STAGE5_SUPPORT6_REPLICATION_DISCONNECT": "0",
        },
    },
    "support6_replication_receipts": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "support6_replication_receipts",
            "colab/run_stage5_support6_replication_receipts.py",
            "bar_crossing_frontier",
            "tests/test_stage5_support6_seed_replication.py",
        ],
        "env": {
            "STAGE5_SUPPORT6_RECEIPTS_DISCONNECT": "0",
        },
    },
    "support6_dosed_seed_resolution": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "support6_dosed_seed_resolution",
            "colab/run_stage5_support6_dosed_seed_resolution.py",
            "STAGE5_SUPPORT6_DOSED_RECEIPT_SUMMARY",
            "STAGE5_SUPPORT6_DOSED_STEPS",
            "bar_crossing_frontier",
            "tests/test_stage5_support6_seed_replication.py",
        ],
        "env": {
            "STAGE5_SUPPORT6_DOSED_RECEIPT_SUMMARY": "outputs/stage5/stage5_support6_replication_receipts_20260708_003055/summary.json",
            "STAGE5_SUPPORT6_DOSED_STEPS": "2000",
            "STAGE5_SUPPORT6_DOSED_ROWS_PER_DEPTH": "256",
            "STAGE5_SUPPORT6_DOSED_DTYPE": "bfloat16",
            "STAGE5_SUPPORT6_DOSED_DISCONNECT": "0",
        },
    },
    "support6_seed26_plateau_test": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "support6_seed26_plateau_test",
            "colab/run_stage5_support6_seed26_plateau.py",
            "STAGE5_SEED26_PLATEAU_SOURCE_SUMMARY",
            "PLATEAU_MIN_GAIN",
            "seed26_unified",
            "seed26_plateau",
        ],
        "env": {
            "STAGE5_SEED26_PLATEAU_SOURCE_SUMMARY": "outputs/stage5/stage5_support6_dosed_seed_resolution_20260708_004504_seed_20260726_dose2000/summary.json",
            "STAGE5_SEED26_PLATEAU_STEPS": "2000",
            "STAGE5_SEED26_PLATEAU_SEED": "20260726",
            "STAGE5_SEED26_PLATEAU_DTYPE": "bfloat16",
            "STAGE5_SEED26_PLATEAU_DISCONNECT": "0",
        },
    },
    "scorer_equivalence_receipt": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "scorer_equivalence_receipt",
            "colab/run_stage5_scorer_equivalence_receipt.py",
            "eval/check_synthetic_active_label_scorer_equivalence.py",
            "force_slow_candidate_score",
            "tests/test_eval_synthetic_depth_active_labels.py",
        ],
        "env": {
            "STAGE5_SCORER_EQUIV_SOURCE_SUMMARY": "outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json",
            "STAGE5_SCORER_EQUIV_MAX_ROWS": "2",
            "STAGE5_SCORER_EQUIV_LOOP_COUNTS": "1,2,12,22",
            "STAGE5_SCORER_EQUIV_DTYPE": "bfloat16",
            "STAGE5_SCORER_EQUIV_DISCONNECT": "0",
        },
    },
    "synthetic_release_receipts": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "synthetic_release_receipts",
            "colab/run_stage5_synthetic_release_receipts.py",
            "stage5_synthetic_release_receipts",
            "STAGE5_RELEASE_RECEIPTS_PUBLISH",
            "support6_dosed_seed_resolution",
            "scorer_equivalence_receipt",
        ],
        "env": {
            "STAGE5_RELEASE_RECEIPTS_DISCONNECT": "0",
        },
    },
    "n24_support12_rung": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "n24_support12_rung",
            "colab/run_stage5_n24_support12_rung.py",
            "STAGE5_N24_FROZEN_EVAL_ID",
            "STAGE5_N24_EVAL_CHECKPOINTS",
            "STAGE5_RUNG_CANARY_HARD_STOP",
            "N24_STRONG_SCALING_MIN_CORRECT",
            "N24_CHANCE_REJECTION_MIN_CORRECT",
            "tests/test_stage5_n24_rung.py",
        ],
        "env": {
            "STAGE5_N24_INIT_CHECKPOINT": "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
            "STAGE5_N24_FROZEN_EVAL_ID": "stage5_synthetic_depth_frozen_eval_v3_depth22_n24",
            "STAGE5_N24_EVAL_CHECKPOINTS": "2000,4000,6000",
            "STAGE5_N24_STEPS": "6000",
            "STAGE5_N24_PRELUDE_LR_MULT": "10.0",
            "STAGE5_RUNG_CANARY_EVERY": "1000",
            "STAGE5_RUNG_CANARY_HARD_STOP": "1",
            "STAGE5_N24_DTYPE": "bfloat16",
            "STAGE5_N24_DISCONNECT": "0",
        },
    },
    "phase_a_surpass_prereg": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "phase_a_surpass_prereg",
            "colab/run_stage5_phase_a_surpass_plan.py",
            "STAGE5_PHASE_A_PLAN_RUN_ID",
            "tests/test_stage5_phase_a_surpass.py",
            "same-reader final-symbol metric",
        ],
        "env": {
            "STAGE5_PHASE_A_PLAN_DISCONNECT": "0",
        },
    },
    "phase_a_surpass_receipt": {
        "path": "colab/STAGE5_PHASE_A_SURPASS_RECEIPT_CELL.py",
        "markers": [
            "STAGE5_PHASE_A_SURPASS_RECEIPT_CELL_VERSION",
            "phase_a_surpass_receipt",
            "colab/run_stage5_phase_a_surpass_receipt.py",
            "tests/test_stage5_phase_a_surpass_receipt.py",
            "exact_paired_sign_mcnemar",
            "accepted_returncodes={0, 2}",
        ],
        "env": {
            "STAGE5_PHASE_A_RECEIPT_RUN_ID": "stage5_phase_a_surpass_receipt_20260714",
            "STAGE5_PHASE_A_RECEIPT_DISCONNECT": "0",
        },
    },
    "permutation_zero_shot_baseline": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "permutation_zero_shot_baseline",
            "colab/run_stage5_permutation_zero_shot.py",
            "stage5_synthetic_depth_permutation_eval_set",
            "STAGE5_PERM_PARITY_TOLERANCE",
            "--permutation",
        ],
        "env": {
            "STAGE5_PERM_SOURCE_SUMMARY": "outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json",
            "STAGE5_PERM_EVAL_ID": "stage5_synthetic_depth_permutation_eval_v1_n24_depth12",
            "STAGE5_PERM_N_SYMBOLS": "24",
            "STAGE5_PERM_MAX_DEPTH": "12",
            "STAGE5_PERM_ROWS_PER_DEPTH": "128",
            "STAGE5_PERM_PARITY_TOLERANCE": "0.05",
            "STAGE5_PERM_DTYPE": "bfloat16",
            "STAGE5_PERM_DISCONNECT": "0",
        },
    },
    "natural_surface_prepare_cpu": {
        "path": "colab/STAGE5_NATURAL_SURFACE_PREPARE_CELL.py",
        "markers": [
            "STAGE5_NATURAL_SURFACE_PREPARE_CELL_VERSION",
            "natural_surface_prepare_cpu",
            "stage5_natural_surface_transfer_dataset",
            "training/generate_natural_surface_transfer.py",
            "colab/run_stage5_natural_surface_prepare.py",
            "relay_test_chain_mcq",
            "pointer_test_chain_mcq",
            "rung0_train_mix_chain_symbol_sft",
            "STAGE5_NATURAL_VERIFY_TOKENIZER",
            "value_prefix=name:",
        ],
        "env": {
            "STAGE5_NATURAL_VERIFY_TOKENIZER": "1",
        },
    },
    "natural_surface_transfer_rung0": {
        "path": "colab/STAGE5_NATURAL_SURFACE_TRANSFER_CELL.py",
        "markers": [
            "STAGE5_NATURAL_SURFACE_TRANSFER_CELL_VERSION",
            "natural_surface_transfer_rung0",
            "frozen_natural_surface_baseline",
            "verbal_rung_zero",
            "Experiment 0",
            "Experiment 1",
            "colab/run_stage5_natural_surface_transfer.py",
            "training/train_unfrozen_recurrent.py",
            "tests/test_causal_dataset_loop_targets.py",
            "eval/eval_synthetic_depth_active_labels.py",
            "relay_test_chain_mcq",
            "pointer_test_chain_mcq",
            "synthetic_rehearsal_chain_symbol_sft",
            "rung0_train_mix_chain_symbol_sft",
            "stage5_natural_surface_transfer_20260708_230229",
            "stage5_n24_support12_rung_20260707_140139",
            "STAGE5_NATURAL_TRANSFER_RUN_TRAIN",
            "STAGE5_NATURAL_TRANSFER_INIT_SOURCE_SUMMARY",
            "STAGE5_NATURAL_TRANSFER_EXPECTED_INIT_RUN_ID",
            "STAGE5_NATURAL_TRANSFER_EXPECTED_INIT_STEP",
            "STAGE5_NATURAL_TRANSFER_EXPECTED_INIT_SHA256",
            "STAGE5_NATURAL_TRANSFER_DATA_SUMMARY",
            "STAGE5_NATURAL_TRANSFER_TRAIN_STEPS",
            "STAGE5_NATURAL_TRANSFER_REUSE_EXISTING",
            "value_prefix=name:",
            "value_prefix=letter:",
        ],
        "env": {
            "STAGE5_NATURAL_TRANSFER_DATA_SUMMARY": "outputs/stage5/stage5_natural_surface_transfer_20260708_230229/summary.json",
            "STAGE5_NATURAL_TRANSFER_INIT_SOURCE_SUMMARY": "outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json",
            "STAGE5_NATURAL_TRANSFER_EXPECTED_INIT_RUN_ID": "stage5_n24_support12_rung_20260707_140139",
            "STAGE5_NATURAL_TRANSFER_EXPECTED_INIT_STEP": "6000",
            "STAGE5_NATURAL_TRANSFER_RUN_TRAIN": "1",
            "STAGE5_NATURAL_TRANSFER_TRAIN_STEPS": "8000",
            "STAGE5_NATURAL_TRANSFER_EVAL_MAX_DEPTH": "12",
            "STAGE5_NATURAL_TRANSFER_TRAIN_MAX_DEPTH": "8",
            "STAGE5_NATURAL_TRANSFER_DTYPE": "bfloat16",
            "STAGE5_NATURAL_TRANSFER_KEEP_FULL_ACTIVE_ROWS": "0",
            "STAGE5_NATURAL_TRANSFER_REUSE_EXISTING": "1",
            "STAGE5_NATURAL_TRANSFER_BACKUP_CHECKPOINTS_TO_DRIVE": "1",
            "STAGE5_NATURAL_TRANSFER_DISCONNECT": "0",
        },
    },
    "natural_surface_receipts": {
        "path": "colab/STAGE5_NATURAL_SURFACE_RECEIPTS_CELL.py",
        "markers": [
            "STAGE5_NATURAL_SURFACE_RECEIPTS_CELL_VERSION",
            "natural_surface_receipts",
            "pointer_holdout",
            "untouched_relay_d13_16",
            "untouched_pointer_d13_16",
            "robust_baton_default_d1_12",
            "robust_relay_unseen_names_d1_12",
            "robust_pointer_unseen_names_d1_12",
            "paired_relay_pointer_mcnemar",
            "eval/eval_synthetic_depth_final_symbol.py",
            "colab/run_stage5_natural_surface_receipts.py",
        ],
        "env": {
            "STAGE5_NATURAL_RECEIPTS_RUN_EVALS": "1",
            "STAGE5_NATURAL_RECEIPTS_RUN_FULL_SYNTHETIC": "1",
            "STAGE5_NATURAL_RECEIPTS_RUN_SAME_READER": "1",
            "STAGE5_NATURAL_RECEIPTS_DTYPE": "bfloat16",
            "STAGE5_NATURAL_RECEIPTS_CHECKPOINTS": "frozen_n24,step_2000,step_4000,step_6000",
            "STAGE5_NATURAL_RECEIPTS_DISCONNECT": "0",
        },
    },
    "natural_surface_receipts_resume": {
        "path": "colab/STAGE5_NATURAL_SURFACE_RECEIPTS_RESUME_CELL.py",
        "markers": [
            "STAGE5_NATURAL_SURFACE_RECEIPTS_RESUME_CELL_VERSION",
            "natural_surface_receipts_resume",
            "STAGE5_NATURAL_RECEIPTS_RESUME_RUN_ID",
            "stage5_natural_surface_receipts_20260709_210151",
            "resume_skip_active",
            "colab/run_stage5_natural_surface_receipts_resume.py",
            "tests/test_stage5_natural_surface_receipts_resume.py",
        ],
        "env": {
            "STAGE5_NATURAL_RECEIPTS_RESUME_RUN_ID": "stage5_natural_surface_receipts_20260709_210151",
            "STAGE5_NATURAL_RECEIPTS_RUN_FULL_SYNTHETIC": "1",
            "STAGE5_NATURAL_RECEIPTS_RUN_SAME_READER": "1",
            "STAGE5_NATURAL_RECEIPTS_DTYPE": "bfloat16",
            "STAGE5_NATURAL_RECEIPTS_CHECKPOINTS": "frozen_n24,step_2000,step_4000,step_6000",
            "STAGE5_NATURAL_RECEIPTS_DISCONNECT": "0",
        },
    },
    "natural_surface_followups_2_4": {
        "path": "colab/STAGE5_NATURAL_SURFACE_FOLLOWUPS_CELL.py",
        "markers": [
            "STAGE5_NATURAL_SURFACE_FOLLOWUPS_CELL_VERSION",
            "natural_surface_followups_2_4",
            "CORRECTED_HELDOUT_SINGLE_TOKEN_NAMES",
            "robust_relay_fronted_d1_12",
            "eval/eval_synthetic_depth_probe.py",
            "colab/run_stage5_natural_surface_followups.py",
            "colab/run_stage5_natural_surface_replication_dose.py",
            "1000,1500,2000,2500,3000,4000,6000",
            "untouched_depth_13_16_opened",
        ],
        "env": {
            "STAGE5_NATURAL_FOLLOWUP_RUN_ID": "stage5_natural_surface_followups_2_3_20260710",
            "STAGE5_NATURAL_FOLLOWUP_CHECKPOINTS": "frozen_n24,step_2000,step_4000,step_6000",
            "STAGE5_NATURAL_FOLLOWUP_DTYPE": "bfloat16",
            "STAGE5_NATURAL_FOLLOWUP_PROBE_PERMUTATIONS": "20",
            "STAGE5_NATURAL_REPLICATION_RUN_ID": "stage5_natural_surface_replication_dose_seed931337_20260710",
            "STAGE5_NATURAL_REPLICATION_SEED": "931337",
            "STAGE5_NATURAL_REPLICATION_SAVE_STEPS": "1000,1500,2000,2500,3000,4000,6000",
            "STAGE5_NATURAL_REPLICATION_TRAIN_STEPS": "6000",
            "STAGE5_NATURAL_REPLICATION_DTYPE": "bfloat16",
            "STAGE5_NATURAL_REPLICATION_BACKUP_CHECKPOINTS_TO_DRIVE": "1",
            "STAGE5_NATURAL_FOLLOWUP_DISCONNECT": "0",
        },
    },
    "natural_surface_replication_dose": {
        "path": "colab/STAGE5_NATURAL_SURFACE_REPLICATION_DOSE_CELL.py",
        "markers": [
            "STAGE5_NATURAL_SURFACE_REPLICATION_DOSE_CELL_VERSION",
            "natural_surface_replication_dose",
            "STAGE5_NATURAL_REPLICATION_SAVE_STEPS",
            "STAGE5_NATURAL_REPLICATION_SEED",
            "1000,1500,2000,2500,3000,4000,6000",
            "colab/run_stage5_natural_surface_replication_dose.py",
            "tail_peaks_before_final",
            "original landed frozen relay/pointer/synthetic rows",
        ],
        "env": {
            "STAGE5_NATURAL_REPLICATION_SEED": "931337",
            "STAGE5_NATURAL_REPLICATION_SAVE_STEPS": "1000,1500,2000,2500,3000,4000,6000",
            "STAGE5_NATURAL_REPLICATION_TRAIN_STEPS": "8000",
            "STAGE5_NATURAL_REPLICATION_DTYPE": "bfloat16",
            "STAGE5_NATURAL_REPLICATION_BACKUP_CHECKPOINTS_TO_DRIVE": "1",
            "STAGE5_NATURAL_REPLICATION_DISCONNECT": "0",
        },
    },
    "phase_g_experiment1": {
        "path": "colab/STAGE5_PHASE_G_EXPERIMENT1_CELL.py",
        "markers": [
            "STAGE5_PHASE_G_EXPERIMENT1_CELL_VERSION",
            "phase_g_experiment1",
            "colab/run_stage5_phase_g_experiment1.py",
            "eval/eval_abductive_coverage.py",
            "eval/eval_synthetic_diagonal_guardrail.py",
            "STAGE5_PHASE_G_EXP1_MAX_STEPS",
            "deterministic controls; latent, learned halting, LPRM, and SVGD disabled",
        ],
        "env": {
            "STAGE5_PHASE_G_EXP1_RUN_ID": "stage5_phase_g_experiment1_fixed_boundary_20260712",
            "STAGE5_PHASE_G_EXP1_MAX_STEPS": "1000",
            "STAGE5_PHASE_G_EXP1_DATA_SEED": "1104729",
            "STAGE5_PHASE_G_EXP1_DTYPE": "bfloat16",
            "STAGE5_PHASE_G_EXP1_DISCONNECT": "0",
        },
    },
    "phase_g_alpha": {
        "path": "colab/STAGE5_PHASE_G_ALPHA_CELL.py",
        "markers": [
            "STAGE5_PHASE_G_ALPHA_CELL_VERSION",
            "training/train_phase_g_alpha.py",
            "eval/eval_phase_g_alpha.py",
            "phase_g_prior_head",
            "phase_g_posterior_head",
            "phase_g_injection_scale",
            "STAGE5_PHASE_G_ALPHA_KL_SWEEP",
            "STAGE5_PHASE_G_ALPHA_TRAJECTORY_MICROBATCH_SIZE",
            "STAGE5_PHASE_G_ALPHA_CHECKPOINT_EVERY",
            "progress_backup_path",
            "blocked exit",
        ],
        "env": {
            "STAGE5_PHASE_G_ALPHA_KL_SWEEP": "0.0001,0.001,0.01",
            "STAGE5_PHASE_G_ALPHA_STEPS": "1000",
            "STAGE5_PHASE_G_ALPHA_TRAJECTORY_MICROBATCH_SIZE": "0",
            "STAGE5_PHASE_G_ALPHA_CHECKPOINT_EVERY": "100",
            "STAGE5_PHASE_G_ALPHA_DISCONNECT": "0",
        },
    },
    "phase_g_multitarget_control": {
        "path": "colab/STAGE5_PHASE_G_MULTITARGET_CONTROL_CELL.py",
        "markers": [
            "STAGE5_PHASE_G_MULTITARGET_CONTROL_CELL_VERSION",
            "docs/STAGE5_PHASE_G_A0_MARGIN_LOCK_20260718.json",
            "base_problem_uniform",
            "kl_0p001",
            "kl_0p0001_confirmation",
            "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_TARGET_RATE=0.60",
            "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_PRIOR_TARGET_LIFT=0.15",
            "STAGE5_PHASE_G_MULTITARGET_MIN_TEACHER_SWITCHING_GROUPS=24",
            "tests/test_score_phase_g_posterior_control.py",
        ],
        "env": {
            "STAGE5_PHASE_G_MULTITARGET_STEPS": "1000",
            "STAGE5_PHASE_G_MULTITARGET_KL": "0.001",
            "STAGE5_PHASE_G_MULTITARGET_SEED": "20260718",
            "STAGE5_PHASE_G_MULTITARGET_CHECKPOINT_EVERY": "100",
            "STAGE5_PHASE_G_MULTITARGET_DISCONNECT": "0",
        },
    },
    "phase_g_forced_injection_probe": {
        "path": "colab/STAGE5_PHASE_G_FORCED_INJECTION_PROBE_CELL.py",
        "markers": [
            "STAGE5_PHASE_G_FORCED_INJECTION_PROBE_CELL_VERSION",
            "eval/eval_phase_g_forced_injection.py",
            "STAGE5_PHASE_G_FORCED_INJECTION_FACTORS=1,3,10,30,100",
            "CHANNEL-EXISTS switching >=16/32 K1 validity >0.50",
            "NO-CHANNEL switching <8/32",
            "factor_1_exact_equivalence",
            "frozen_lineage_unchanged",
            "tests/test_phase_g_forced_injection.py",
        ],
        "env": {
            "STAGE5_PHASE_G_FORCED_INJECTION_FACTORS": "1,3,10,30,100",
            "STAGE5_PHASE_G_FORCED_INJECTION_DTYPE": "bfloat16",
            "STAGE5_PHASE_G_FORCED_INJECTION_DISCONNECT": "0",
        },
    },
    "oracle_interface_probe": {
        "path": "colab/STAGE5_ORACLE_INTERFACE_PROBE_CELL.py",
        "markers": [
            "STAGE5_ORACLE_INTERFACE_PROBE_CELL_VERSION",
            "training/train_oracle_interface_probe.py",
            "eval/eval_oracle_interface_probe.py",
            "additive film parameter-matched",
            "nondefault_branch_control >=0.85",
            "overall_transition_control >=0.90",
            "transition_legality >=0.95 terminal_validity >=0.71",
            "zeroed_conditioning_identity frozen_keeper_lineage",
            "106 rows 32 groups 305 transitions",
            "no KL no coverage no selector no particles no SVGD",
            "tests/test_oracle_reentry_conditioner.py",
            "tests/test_oracle_interface_probe.py",
        ],
        "env": {
            "STAGE5_ORACLE_INTERFACE_STEPS": "1500",
            "STAGE5_ORACLE_INTERFACE_SEED": "20260718",
            "STAGE5_ORACLE_INTERFACE_BOTTLENECK_DIM": "256",
            "STAGE5_ORACLE_INTERFACE_DTYPE": "bfloat16",
            "STAGE5_ORACLE_INTERFACE_DISCONNECT": "0",
        },
    },
    "oracle_train_readout": {
        "path": "colab/STAGE5_ORACLE_TRAIN_READOUT_CELL.py",
        "markers": [
            "STAGE5_ORACLE_TRAIN_READOUT_CELL_VERSION",
            "posthoc diagnostic only no training no parameter mutation",
            "registered BOTH_FAIL verdict immutable",
            "seed 20260722 matched 106 rows 32 groups 305 transitions",
            "full 1899 rows 512 groups 5617 transitions",
            "fit >=0.85 no-fit <=0.25 partial between",
            "tests/test_oracle_train_readout_spec.py",
        ],
        "env": {
            "STAGE5_ORACLE_TRAIN_READOUT_DTYPE": "bfloat16",
            "STAGE5_ORACLE_TRAIN_READOUT_DISCONNECT": "0",
        },
    },
    "paper2_phase_t0_preflight": {
        "path": "colab/STAGE5_PAPER2_PHASE_T0_CELL.py",
        "markers": [
            "STAGE5_PAPER2_PHASE_T0_CELL_VERSION",
            "tokenizer collision exactly three rows tie policy",
            "visible generation masks all three control logits",
            "one loop identity max abs diff below 1e-3",
            "requested executed selected loop counts agree under forcing",
            "no training no checkpoint written",
            "tests/test_internal_think_token_runtime.py",
        ],
        "env": {
            "STAGE5_PAPER2_T0_DTYPE": "bfloat16",
            "STAGE5_PAPER2_T0_DISCONNECT": "0",
        },
    },
    "coconut_composite_preflight": {
        "path": "colab/STAGE5_COCONUT_COMPOSITE_PREFLIGHT_CELL.py",
        "markers": [
            "STAGE5_COCONUT_COMPOSITE_PREFLIGHT_CELL_VERSION",
            "no training RG-12 remains unrun",
            "H times L feedback and H plus one times L total applications",
            "full recompute reference sliced cache L1 only",
            "finite difference cache checkpointing bfloat16 equivalence",
            "frozen adapter backbone gradient transparent",
            "tests/test_coconut_composite.py",
        ],
        "env": {
            "STAGE5_COCONUT_PREFLIGHT_DISCONNECT": "0",
        },
    },
    "coconut_composite_numerics": {
        "path": "colab/STAGE5_COCONUT_COMPOSITE_NUMERICS_CELL.py",
        "markers": [
            "STAGE5_COCONUT_COMPOSITE_NUMERICS_CELL_VERSION",
            "coconut_composite_numerics_v1",
            "recompute only sliced cache retired",
            "fixed-weight fixed-direction epsilon stability sweep",
            "original 10 percent derivative criterion unchanged",
            "fp32 full bf16 and fp32-master bf16-autocast fixed prompts",
            "per-example gradient cosine threshold 0.99 unchanged",
            "no training no checkpoint RG-12 unauthorized",
            "tests/test_coconut_composite_numerics.py",
            "colab/run_stage5_coconut_composite_numerics.py",
        ],
        "env": {
            "STAGE5_COCONUT_NUMERICS_DISCONNECT": "0",
        },
    },
    "oracle_intrablock_control": {
        "path": "colab/STAGE5_ORACLE_INTRABLOCK_CONTROL_CELL.py",
        "markers": [
            "STAGE5_ORACLE_INTRABLOCK_CONTROL_CELL_VERSION",
            "parameter-matched shared layerwise FiLM",
            "only_variable command_access_location",
            "nondefault_branch_control >=0.85",
            "overall_transition_control >=0.90",
            "transition_legality >=0.95 terminal_validity >=0.71",
            "zeroed_conditioning_identity frozen_keeper_lineage",
            "106 rows 32 groups 305 transitions",
            "no KL no coverage no selector no particles no SVGD",
            "tests/test_oracle_intrablock_control_spec.py",
        ],
        "env": {
            "STAGE5_ORACLE_INTRABLOCK_STEPS": "1500",
            "STAGE5_ORACLE_INTRABLOCK_SEED": "20260718",
            "STAGE5_ORACLE_INTRABLOCK_BOTTLENECK_DIM": "256",
            "STAGE5_ORACLE_INTRABLOCK_DTYPE": "bfloat16",
            "STAGE5_ORACLE_INTRABLOCK_DISCONNECT": "0",
        },
    },
    "phase_a_dense_full": {
        "path": "colab/STAGE5_PHASE_A_DENSE_FULL_CELL.py",
        "markers": [
            "STAGE5_PHASE_A_DENSE_FULL_CELL_VERSION",
            "phase_a_dense_full",
            "adamw_full_fp32_state",
            "STAGE5_PHASE_A_DENSE_ARMS",
            "tests/test_stage5_phase_a_dense_full.py",
            "tests/test_train_dense_full.py",
            "eval/eval_synthetic_depth_dense.py",
        ],
        "env": {
            "STAGE5_PHASE_A_DENSE_ARMS": "B,C",
            "STAGE5_PHASE_A_DENSE_RUN_ID": "stage5_phase_a_dense_full_bc_20260713",
            "STAGE5_PHASE_A_DENSE_LR": "2e-6",
            "STAGE5_PHASE_A_DENSE_SEED": "931337",
            "STAGE5_PHASE_A_DENSE_DISCONNECT": "0",
        },
    },
    "phase_a_checkpoint_comparison": {
        "path": "colab/STAGE5_PHASE_A_CHECKPOINT_COMPARISON_CELL.py",
        "markers": [
            "STAGE5_PHASE_A_CHECKPOINT_COMPARISON_CELL_VERSION",
            "phase_a_checkpoint_comparison",
            "tests/test_stage5_phase_a_checkpoint_comparison.py",
            "STAGE5_PHASE_A_COMPARISON_RUN_ID",
            "paired_rows.jsonl",
        ],
        "env": {
            "STAGE5_PHASE_A_COMPARISON_RUN_ID": "stage5_phase_a_checkpoint_comparison_20260713",
            "STAGE5_PHASE_A_COMPARISON_DISCONNECT": "0",
        },
    },
    "phase_g_injective_curriculum_recovery": {
        "path": "colab/STAGE5_PHASE_G_EXPERIMENT1_CELL.py",
        "markers": [
            "STAGE5_PHASE_G_EXPERIMENT1_CELL_VERSION",
            "phase_g_experiment1",
            "colab/run_stage5_phase_g_experiment1.py",
            "eval/eval_abductive_coverage.py",
            "STAGE5_PHASE_G_EXP1_MAX_STEPS",
            "deterministic controls; latent, learned halting, LPRM, and SVGD disabled",
        ],
        "env": {
            "STAGE5_PHASE_G_EXP1_RUN_ID": "stage5_phase_g_injective_curriculum_recovery_20260712",
            "STAGE5_PHASE_G_EXP1_MAX_STEPS": "2000",
            "STAGE5_PHASE_G_EXP1_DATA_SEED": "1104729",
            "STAGE5_PHASE_G_EXP1_DTYPE": "bfloat16",
            "STAGE5_PHASE_G_EXP1_INIT_CHECKPOINT": "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_phase_g_experiment1_fixed_boundary_20260712/injective_control/unfrozen_recurrent_step_1000.pt",
            "STAGE5_PHASE_G_EXP1_INIT_SHA256": "0d6cf119bd66290a2c85686bf58fdc6f9363109c8fdae0ea625f32d13409a1a6",
            "STAGE5_PHASE_G_EXP1_CURRICULUM_ENABLED": "1",
            "STAGE5_PHASE_G_EXP1_CURRICULUM_START": "2",
            "STAGE5_PHASE_G_EXP1_CURRICULUM_END": "8",
            "STAGE5_PHASE_G_EXP1_CURRICULUM_RAMP_COMPUTE": "1",
            "STAGE5_PHASE_G_EXP1_GATE_SAMPLE_COUNTS": "1",
            "STAGE5_PHASE_G_EXP1_DISCONNECT": "0",
        },
    },
    "phase_g_curriculum_autopsy": {
        "path": "colab/STAGE5_PHASE_G_CURRICULUM_AUTOPSY_CELL.py",
        "markers": [
            "STAGE5_PHASE_G_CURRICULUM_AUTOPSY_CELL_VERSION",
            "phase_g_curriculum_autopsy",
            "eval/eval_abductive_curriculum_autopsy.py",
            "uniform_expected_coverage",
            "inverse_table_control_prepared_not_trained",
            "next_training_disabled_pending_strategy_review",
        ],
        "env": {
            "STAGE5_PHASE_G_AUTOPSY_RUN_ID": "stage5_phase_g_curriculum_autopsy_20260712",
            "STAGE5_PHASE_G_AUTOPSY_ROWS_PER_DEPTH": "16",
            "STAGE5_PHASE_G_AUTOPSY_STATE_QUERY_EXAMPLES": "8",
            "STAGE5_PHASE_G_AUTOPSY_DTYPE": "bfloat16",
            "STAGE5_PHASE_G_AUTOPSY_DISCONNECT": "0",
        },
    },
    "inverse_composition_staircase": {
        "path": "colab/STAGE5_INVERSE_COMPOSITION_STAIRCASE_CELL.py",
        "markers": [
            "STAGE5_INVERSE_COMPOSITION_STAIRCASE_CELL_VERSION",
            "inverse_composition_staircase",
            "weighted_per_loop_labels",
            "gradient_accumulation_steps",
            "active_weighted_labels_per_loop",
            "tests/test_stage5_inverse_composition_staircase.py",
            "eval/eval_abductive_staircase.py",
            "phase_g_alpha_remains_closed",
        ],
        "env": {
            "STAGE5_STAIRCASE_RUN_ID": "stage5_inverse_composition_staircase_20260713",
            "STAGE5_STAIRCASE_DTYPE": "bfloat16",
            "STAGE5_STAIRCASE_PROBE_PERMUTATIONS": "100",
            "STAGE5_STAIRCASE_DISCONNECT": "0",
        },
    },
    "inverse_table_rebase_caps3_4": {
        "path": "colab/STAGE5_INVERSE_TABLE_REBASE_CELL.py",
        "markers": [
            "STAGE5_INVERSE_TABLE_REBASE_CELL_VERSION",
            "inverse_table_rebase_caps3_4",
            "SOURCE_CAP2_SHA256",
            "tests/test_stage5_inverse_table_rebase.py",
            "rebase_caps3_4_green_pending_review",
            "accepted_returncodes={0, 2}",
        ],
        "env": {
            "STAGE5_REBASE_RUN_ID": "stage5_inverse_table_rebase_caps3_4_20260713",
            "STAGE5_REBASE_SOURCE_SUMMARY": "outputs/stage5/stage5_inverse_composition_staircase_20260713/summary.json",
            "STAGE5_STAIRCASE_DTYPE": "bfloat16",
            "STAGE5_REBASE_DISCONNECT": "0",
        },
    },
    "inverse_table_cap3_rehearsal": {
        "force_env": True,
        "path": "colab/STAGE5_INVERSE_TABLE_REHEARSAL_CELL.py",
        "markers": [
            "STAGE5_INVERSE_TABLE_REHEARSAL_CELL_VERSION",
            "inverse_table_cap3_rehearsal",
            "colab/run_stage5_inverse_table_rehearsal.py",
            "tests/test_stage5_inverse_table_rehearsal.py",
            "row_specific_forward_loops",
            "accepted_returncodes={0, 2}",
        ],
        "env": {
            "STAGE5_REHEARSAL_RUN_ID": "stage5_inverse_table_cap3_rehearsal_20260714",
            "STAGE5_STAIRCASE_DTYPE": "bfloat16",
            "STAGE5_REHEARSAL_DISCONNECT": "0",
        },
    },
    "inverse_rehearsal_checkpoint_pareto": {
        "force_env": True,
        "path": "colab/STAGE5_INVERSE_REHEARSAL_CHECKPOINT_PARETO_CELL.py",
        "markers": [
            "STAGE5_INVERSE_REHEARSAL_CHECKPOINT_PARETO_CELL_VERSION",
            "inverse_rehearsal_checkpoint_pareto",
            "colab/run_stage5_inverse_rehearsal_checkpoint_pareto.py",
            "tests/test_stage5_inverse_rehearsal_checkpoint_pareto.py",
            "candidate_requires_fresh_confirmation",
        ],
        "env": {
            "STAGE5_REHEARSAL_PARETO_STEPS": "100,200,300,334",
            "STAGE5_STAIRCASE_DTYPE": "bfloat16",
            "STAGE5_REHEARSAL_PARETO_DISCONNECT": "1",
        },
    },
    "inverse_rendered_width_gate": {
        "force_env": True,
        "path": "colab/STAGE5_INVERSE_RENDERED_WIDTH_GATE_CELL.py",
        "markers": [
            "STAGE5_INVERSE_RENDERED_WIDTH_GATE_CELL_VERSION",
            "inverse_rendered_width_gate",
            "colab/run_stage5_inverse_rendered_width_gate.py",
            "tests/test_stage5_inverse_rendered_width_gate.py",
            "data/phase_g_alpha_inverse_rendered",
            "accepted_returncodes={0, 2}",
        ],
        "env": {
            "STAGE5_INVERSE_RENDERED_RUN_ID": "stage5_inverse_rendered_width_gate_20260714",
            "STAGE5_INVERSE_RENDERED_DTYPE": "bfloat16",
            "STAGE5_INVERSE_RENDERED_DISCONNECT": "0",
        },
    },
    "inverse_rendered_width_gate_rehearsal": {
        "force_env": True,
        "path": "colab/STAGE5_INVERSE_RENDERED_WIDTH_GATE_CELL.py",
        "markers": [
            "STAGE5_INVERSE_RENDERED_WIDTH_GATE_CELL_VERSION",
            "inverse_rendered_width_gate",
            "colab/run_stage5_inverse_rendered_width_gate.py",
            "tests/test_stage5_inverse_rendered_width_gate.py",
            "data/phase_g_alpha_inverse_rendered",
            "accepted_returncodes={0, 2}",
        ],
        "env": {
            "STAGE5_INVERSE_RENDERED_RUN_ID": "stage5_inverse_rendered_width_gate_rehearsal_20260715",
            "STAGE5_INVERSE_RENDERED_SOURCE_SUMMARY": "outputs/stage5/stage5_inverse_table_cap3_rehearsal_20260714/summary.json",
            "STAGE5_INVERSE_RENDERED_DTYPE": "bfloat16",
            "STAGE5_INVERSE_RENDERED_DISCONNECT": "0",
        },
    },
    "inverse_rendered_n24_continuation": {
        "force_env": True,
        "path": "colab/STAGE5_INVERSE_RENDERED_CONTINUATION_CELL.py",
        "markers": [
            "STAGE5_INVERSE_RENDERED_CONTINUATION_CELL_VERSION",
            "inverse_rendered_n24_continuation",
            "colab/run_stage5_inverse_rendered_continuation.py",
            "tests/test_stage5_inverse_rendered_continuation.py",
            "forward_rehearsal_fraction",
            "bounded_tune_review_required",
        ],
        "env": {
            "STAGE5_INVERSE_RENDERED_CONTINUATION_RUN_ID": "stage5_inverse_rendered_n24_continuation_20260715",
            "STAGE5_INVERSE_RENDERED_CONTINUATION_DTYPE": "bfloat16",
            "STAGE5_INVERSE_RENDERED_CONTINUATION_DISCONNECT": "1",
        },
    },
    "phase_g_n24_calibration_gate": {
        "path": "colab/STAGE5_PHASE_G_N24_CALIBRATION_CELL.py",
        "markers": [
            "STAGE5_PHASE_G_N24_CALIBRATION_CELL_VERSION",
            "phase_g_n24_calibration_gate",
            "data/phase_g_alpha/calibration_n24.jsonl",
            "test_split_opened",
            "eval/eval_abductive_coverage.py",
            "run_one_bounded_deterministic_arbitrary_table_continuation",
        ],
        "env": {
            "STAGE5_PHASE_G_N24_GATE_RUN_ID": "stage5_phase_g_n24_calibration_20260712",
            "STAGE5_PHASE_G_N24_GATE_SOURCE": "outputs/stage5/stage5_phase_g_experiment1_fixed_boundary_20260712/summary.json",
            "STAGE5_PHASE_G_N24_GATE_DTYPE": "bfloat16",
            "STAGE5_PHASE_G_N24_GATE_DISCONNECT": "0",
        },
    },
    "multichannel_bridge_precursor": {
        "force_env": True,
        "path": "colab/STAGE5_MULTICHANNEL_BRIDGE_PRECURSOR_CELL.py",
        "markers": [
            "STAGE5_MULTICHANNEL_BRIDGE_PRECURSOR_CELL_VERSION",
            "eval/eval_multichannel_bridge_precursor.py",
            "tests/test_multichannel_bridge_precursor.py",
            "prelude_ablation_basis",
            "random_orthogonal_partitions",
        ],
        "env": {
            "STAGE5_MULTICHANNEL_RUN_ID": "stage5_multichannel_bridge_precursor_pilot_20260714",
            "STAGE5_MULTICHANNEL_MODE": "pilot",
            "STAGE5_MULTICHANNEL_CONDITIONS": "n24_step6000",
            "STAGE5_MULTICHANNEL_ROWS_PER_DEPTH": "1",
            "STAGE5_MULTICHANNEL_RANDOM_DRAWS": "20",
            "STAGE5_MULTICHANNEL_M3_BATCH_SIZE": "8",
            "STAGE5_MULTICHANNEL_DYNAMICS_ROW_TIMEOUT_SECONDS": "600",
            "STAGE5_MULTICHANNEL_M3_PASS_TIMEOUT_SECONDS": "1800",
            "STAGE5_MULTICHANNEL_LIVENESS_TIMEOUT_SECONDS": "900",
            "STAGE5_MULTICHANNEL_DTYPE": "bfloat16",
            "STAGE5_MULTICHANNEL_DISCONNECT": "0",
        },
    },
    "multichannel_bridge_precursor_replication": {
        "force_env": True,
        "path": "colab/STAGE5_MULTICHANNEL_BRIDGE_PRECURSOR_CELL.py",
        "markers": [
            "STAGE5_MULTICHANNEL_BRIDGE_PRECURSOR_CELL_VERSION",
            "eval/eval_multichannel_bridge_precursor.py",
            "tests/test_multichannel_bridge_precursor.py",
            "prelude_ablation_basis",
            "random_orthogonal_partitions",
            "STAGE5_MULTICHANNEL_SEED_SUMMARY",
        ],
        "env": {
            "STAGE5_MULTICHANNEL_RUN_ID": "stage5_multichannel_bridge_precursor_replication_20260714",
            "STAGE5_MULTICHANNEL_MODE": "pilot",
            "STAGE5_MULTICHANNEL_CONDITIONS": "backward_recovery",
            "STAGE5_MULTICHANNEL_SEED_SUMMARY": "outputs/stage5/stage5_multichannel_bridge_precursor_pilot_20260714/summary.json",
            "STAGE5_MULTICHANNEL_ROWS_PER_DEPTH": "1",
            "STAGE5_MULTICHANNEL_RANDOM_DRAWS": "20",
            "STAGE5_MULTICHANNEL_M3_BATCH_SIZE": "8",
            "STAGE5_MULTICHANNEL_DYNAMICS_ROW_TIMEOUT_SECONDS": "600",
            "STAGE5_MULTICHANNEL_M3_PASS_TIMEOUT_SECONDS": "1800",
            "STAGE5_MULTICHANNEL_LIVENESS_TIMEOUT_SECONDS": "900",
            "STAGE5_MULTICHANNEL_DTYPE": "bfloat16",
            "STAGE5_MULTICHANNEL_DISCONNECT": "0",
        },
    },
    "multichannel_bridge_precursor_full": {
        "force_env": True,
        "path": "colab/STAGE5_MULTICHANNEL_BRIDGE_PRECURSOR_CELL.py",
        "markers": [
            "STAGE5_MULTICHANNEL_BRIDGE_PRECURSOR_CELL_VERSION",
            "eval/eval_multichannel_bridge_precursor.py",
            "tests/test_multichannel_bridge_precursor.py",
            "prelude_ablation_basis",
            "random_orthogonal_partitions",
            "STAGE5_MULTICHANNEL_MODE",
        ],
        "env": {
            "STAGE5_MULTICHANNEL_RUN_ID": "stage5_multichannel_bridge_precursor_full_20260714",
            "STAGE5_MULTICHANNEL_MODE": "full",
            "STAGE5_MULTICHANNEL_CONDITIONS": "n24_step6000,natural_surface_keeper,backward_fixed_boundary,backward_recovery",
            "STAGE5_MULTICHANNEL_ROWS_PER_DEPTH": "64",
            "STAGE5_MULTICHANNEL_RANDOM_DRAWS": "20",
            "STAGE5_MULTICHANNEL_M3_BATCH_SIZE": "8",
            "STAGE5_MULTICHANNEL_DYNAMICS_ROW_TIMEOUT_SECONDS": "600",
            "STAGE5_MULTICHANNEL_M3_PASS_TIMEOUT_SECONDS": "1800",
            "STAGE5_MULTICHANNEL_LIVENESS_TIMEOUT_SECONDS": "900",
            "STAGE5_MULTICHANNEL_DTYPE": "bfloat16",
            "STAGE5_MULTICHANNEL_DISCONNECT": "0",
        },
    },
    "part1_closeout_pivot_session": {
        "force_env": True,
        "path": "colab/STAGE5_PART1_CLOSEOUT_PIVOT_CELL.py",
        "markers": [
            "STAGE5_PART1_CLOSEOUT_PIVOT_CELL_VERSION",
            "training/continuation_policy.py",
            "disposable_measurement",
            "training/loop_position_transfer_task.py",
            "training/branching_relations_task.py",
            "eval/eval_branching_relations.py",
        ],
        "env": {
            "STAGE5_PART1_PIVOT_RUN_ID": "stage5_part1_closeout_pivot_20260715",
            "STAGE5_PART1_PIVOT_DTYPE": "bfloat16",
            "STAGE5_PART1_PIVOT_DISCONNECT": "0",
        },
    },
    "splice_injection_diagnostic": {
        "path": "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py",
        "markers": [
            "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION",
            "splice_injection_diagnostic",
            "colab/run_stage5_splice_injection.py",
            "eval/eval_synthetic_depth_splice.py",
            "STAGE5_SPLICE_SOURCE_SUMMARY",
            "STAGE5_SPLICE_POINTS",
            "source_orbit_fraction_j1_to_j3",
            "source_state_continuation",
            "lawful_fraction_j1_to_j3",
            "prompt_position_shortcut",
            "tests/test_eval_synthetic_depth_splice.py",
        ],
        "env": {
            "STAGE5_SPLICE_SOURCE_SUMMARY": "outputs/stage5/stage5_depth_support_route_20260705_124320/summary.json",
            "STAGE5_SPLICE_TARGET_DEPTH": "8",
            "STAGE5_SPLICE_POINTS": "2,4",
            "STAGE5_SPLICE_MAX_LOOPS": "8",
            "STAGE5_SPLICE_N_PAIRS": "128",
            "STAGE5_SPLICE_DTYPE": "bfloat16",
            "STAGE5_SPLICE_DISCONNECT": "0",
        },
    },
    "regression_battery_loop1_current": {
        "path": "colab/STAGE5_REGRESSION_BATTERY_CELL.py",
        "markers": [
            "STAGE5_REGRESSION_BATTERY_CELL_VERSION",
            "regression_battery_ai2_arc_v1",
            "STAGE5_REGRESSION_SOURCE_SUMMARIES",
            "STAGE5_REGRESSION_ARC_SPLIT",
            "STAGE5_BENCHMARK_ARC_EASY_SPLIT",
            "STAGE5_BENCHMARK_ARC_CHALLENGE_SPLIT",
            "forced loop 1",
            "AI2 ARC, not ARC-AGI",
            "drive.mount",
            "eval/assess_regression_battery.py",
            "colab/run_stage5_regression_battery.py",
            "tests/test_regression_battery.py",
        ],
        "env": {
            "STAGE5_REGRESSION_CURRENT_SOURCE_SUMMARY": "outputs/stage5/stage5_chain_continuation_attribution_20260704_163056/summary.json",
            "STAGE5_REGRESSION_BATTERY_RUN_ID": "stage5_regression_battery_loop1_current",
            "STAGE5_REGRESSION_ARC_SPLIT": "all",
            "STAGE5_REGRESSION_ARC_EASY_LIMIT": "all",
            "STAGE5_REGRESSION_ARC_CHALLENGE_LIMIT": "all",
            "STAGE5_REGRESSION_MARGIN": "0.03",
            "STAGE5_REGRESSION_YELLOW_MARGIN": "0.015",
            "STAGE5_REGRESSION_PUSH": "1",
            "STAGE5_REGRESSION_DISCONNECT": "0",
        },
    },
    "lineage_regression_battery": {
        "path": "colab/STAGE5_REGRESSION_BATTERY_CELL.py",
        "markers": [
            "STAGE5_REGRESSION_BATTERY_CELL_VERSION",
            "regression_battery_ai2_arc_v1",
            "STAGE5_REGRESSION_SOURCE_SUMMARIES",
            "forced loop 1",
            "AI2 ARC, not ARC-AGI",
            "tier1_canary_status",
            "hellaswag_winogrande_lambada_status",
        ],
        "env": {
            "STAGE5_REGRESSION_SOURCE_SUMMARIES": "outputs/stage5/stage5_reentry_recovery_20260625_154210/summary.json,outputs/stage5/stage5_depth_support_route_20260705_124320/summary.json,outputs/stage5/stage5_support8_dose_arm_20260706_153028/summary.json,outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json",
            "STAGE5_REGRESSION_BATTERY_RUN_ID": "stage5_lineage_regression_battery_current",
            "STAGE5_REGRESSION_ARC_SPLIT": "all",
            "STAGE5_REGRESSION_ARC_EASY_LIMIT": "all",
            "STAGE5_REGRESSION_ARC_CHALLENGE_LIMIT": "all",
            "STAGE5_REGRESSION_MARGIN": "0.03",
            "STAGE5_REGRESSION_YELLOW_MARGIN": "0.015",
            "STAGE5_REGRESSION_PUSH": "1",
            "STAGE5_REGRESSION_DISCONNECT": "0",
        },
    },
    "gradient_path_audit": {
        "path": "colab/STAGE5_GRADIENT_PATH_AUDIT_CELL.py",
        "markers": [
            "STAGE5_GRADIENT_PATH_AUDIT_CELL_VERSION",
            "gradient_path_audit_v1",
            "read-only gradient matrix plus finite_difference_bridge_prelude",
            "STAGE5_GRADIENT_PATH_AUDIT_SOURCE_SUMMARY",
            "eval/eval_gradient_path_audit.py",
            "tests/test_eval_gradient_path_audit.py",
            "tests/test_stage5_notebooks.py::test_gradient_path_audit_target_is_wired_and_guarded",
            "CoherenceAccumulator",
            "multiplier_consumption_check",
            "Record Stage 5 gradient-path audit",
        ],
        "env": {
            "STAGE5_GRADIENT_PATH_AUDIT_SOURCE_SUMMARY": "outputs/stage5/stage5_synthetic_depth_chain_supervision_20260701_201715/summary.json",
            "STAGE5_GRADIENT_PATH_AUDIT_STAGE_NAME": "",
            "STAGE5_GRADIENT_PATH_AUDIT_MAX_LOOPS": "auto",
            "STAGE5_GRADIENT_PATH_AUDIT_MIN_ACTIVE_LOOP_LABELS": "auto",
            "STAGE5_GRADIENT_PATH_AUDIT_NUM_ROWS": "48",
            "STAGE5_GRADIENT_PATH_AUDIT_DEPTHS": "1,2,3,4",
            "STAGE5_GRADIENT_PATH_AUDIT_CROSS_LOOP_FD": "2:4",
            "STAGE5_GRADIENT_PATH_AUDIT_MATCH_TRAIN_PRECISION": "1",
            "STAGE5_GRADIENT_PATH_AUDIT_FD_EPSILON": "0.01",
            "STAGE5_GRADIENT_PATH_AUDIT_DISCONNECT": "0",
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


def is_commit_sha(value):
    return len(value) == 40 and all(character in "0123456789abcdefABCDEF" for character in value)


def is_abbreviated_commit_sha(value):
    return 7 <= len(value) < 40 and all(character in "0123456789abcdefABCDEF" for character in value)


if is_commit_sha(REF):
    RESOLVED_REF = REF.lower()
elif is_abbreviated_commit_sha(REF):
    # GitHub's branch-ref endpoint treats a short SHA as a branch name and
    # returns 404. Resolve abbreviated commits through the commits endpoint.
    commit_payload = github_json(
        f"https://api.github.com/repos/{REPO}/commits/{REF}?cache_bust={int(time.time())}"
    )
    RESOLVED_REF = (commit_payload.get("sha") or REF).strip().lower()
else:
    ref_payload = github_json(
        f"https://api.github.com/repos/{REPO}/git/ref/heads/{REF}?cache_bust={int(time.time())}"
    )
    RESOLVED_REF = ((ref_payload.get("object") or {}).get("sha") or REF).strip()
if PREFER_LOCAL_HEAD:
    try:
        local_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        local_head = ""
    if local_head and local_head != RESOLVED_REF:
        print(
            "GitHub ref resolution differs from local checkout; using local HEAD "
            f"{local_head[:12]} instead of {RESOLVED_REF[:12]}.",
            flush=True,
        )
        RESOLVED_REF = local_head

selected = TARGETS[TARGET]
if SOURCE_SUMMARY_OVERRIDE:
    os.environ["STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_SURFACE_ALIGN_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_ARC_MIX_CHAIN_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_ARC_MIX_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_DEBIASED_BENCHMARK_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_DENSE_MCQ_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_COMPETENCE_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_REENTRY_REPAIR_NORM_ASSESSMENT"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_REENTRY_RECOVERY_REPAIR_ASSESSMENT"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_FORCED_DEPTH_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_HELDOUT_ROUTER_DISCOVERY_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_LATENT_CRITICALITY_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_REENTRY_COVARIANCE_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_REENTRY_TAIL_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_TAIL_DAMPER_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
else:
    # Avoid accidentally pinning a new session to an old target-specific source
    # summary. The safe-continue launcher will follow config/stage5_current_source_summary.txt.
    os.environ.pop("STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_SURFACE_ALIGN_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_ARC_MIX_CHAIN_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_ARC_MIX_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_DEBIASED_BENCHMARK_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_DENSE_MCQ_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY", None)
    os.environ.pop("STAGE5_COMPETENCE_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_REENTRY_REPAIR_NORM_ASSESSMENT", None)
    os.environ.pop("STAGE5_REENTRY_RECOVERY_REPAIR_ASSESSMENT", None)
    os.environ.pop("STAGE5_FORCED_DEPTH_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_HELDOUT_ROUTER_DISCOVERY_SUMMARY", None)
    os.environ.pop("STAGE5_LATENT_CRITICALITY_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_REENTRY_COVARIANCE_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_REENTRY_TAIL_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_TAIL_DAMPER_SOURCE_SUMMARY", None)
for key, value in selected["env"].items():
    # Most target configs are defaults so a planner can deliberately override
    # them. A bounded pilot is different: stale settings from a previous full
    # battery would silently turn it back into the expensive job it replaces.
    if selected.get("force_env", False):
        os.environ[key] = value
    else:
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
