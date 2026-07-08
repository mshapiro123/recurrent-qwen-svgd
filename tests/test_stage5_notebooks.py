from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def notebook_payload(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


CURRENT_STAGE5_QUEUE = [
    "master_sequence_status",
    "traced_sft_competence_preserving_pipeline",
    "review_stage5_competence_pipeline.py",
    "debiased_benchmark_suite",
    "dense_mcq_trace_sft_control",
]
CURRENT_STAGE5_ACTION_QUEUE = CURRENT_STAGE5_QUEUE[1:]


def assert_ordered_queue(text: str) -> None:
    positions = []
    for target in CURRENT_STAGE5_ACTION_QUEUE:
        index = text.find(target)
        assert index >= 0, target
        positions.append(index)
    assert positions == sorted(positions)


def fenced_python_block(path: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    start = text.index("```python\n") + len("```python\n")
    end = text.index("\n```", start)
    return text[start:end].strip() + "\n"


def test_current_bootstrap_target_markers_exist_in_launcher_files() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    targets = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TARGETS":
                    targets = ast.literal_eval(node.value)
                    break
    assert isinstance(targets, dict)

    failures = []
    for name, config in targets.items():
        launcher = ROOT / config["path"]
        body = launcher.read_text(encoding="utf-8")
        missing = [marker for marker in config.get("markers", []) if marker not in body]
        if missing:
            failures.append((name, config["path"], missing))

    assert failures == []


def test_current_bootstrap_preserves_planner_supplied_target_env() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert 'os.environ.setdefault(key, value)' in text
    assert 'os.environ[key] = value' not in text
    assert "Planner/user-supplied env must win" in text


def test_current_bootstrap_exposes_capacity_localization_rank64_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "reentry_capacity_localization_rank64" in text
    assert "STAGE5_CAPACITY_LOCALIZATION_RANKS" in text
    assert '"STAGE5_CAPACITY_LOCALIZATION_RANKS": "64"' in text
    assert "stage5_current_capacity_localization_summary" in text
    assert "stage5_reentry_recovery_20260627_190155" in text


def test_current_bootstrap_exposes_regression_battery_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "regression_battery_loop1_current" in text
    assert "colab/STAGE5_REGRESSION_BATTERY_CELL.py" in text
    assert "colab/run_stage5_regression_battery.py" in text
    assert "eval/assess_regression_battery.py" in text
    assert "AI2 ARC, not ARC-AGI" in text
    assert '"STAGE5_REGRESSION_ARC_SPLIT": "all"' in text
    assert '"STAGE5_REGRESSION_BATTERY_RUN_ID": "stage5_regression_battery_loop1_current"' in text


def test_single_a100_runbook_uses_current_bootstrap_target_queue() -> None:
    payload = notebook_payload("colab/00_single_a100_runbook.ipynb")
    text = "\n".join(str(cell.get("source", "")) for cell in payload.get("cells", []))

    assert_ordered_queue(text)
    assert "KEEP_RUNTIME_OPEN" in text
    assert "STAGE5_REENTRY_REPAIR_DISCONNECT" in text
    assert "STAGE5_REENTRY_RECOVERY_DISCONNECT" in text
    assert "STAGE5_DEBIASED_BENCHMARK_DISCONNECT" in text
    assert "STAGE5_DENSE_MCQ_DISCONNECT" in text
    assert "STAGE5_CURRICULUM_PIPELINE_DISCONNECT" in text
    assert "claim_curriculum_scaleup_cpu" in text
    assert "Parallel CPU/API Curriculum Scale-Up" in text
    assert "should not be treated as a GPU gate" in text
    assert "Optional Scale Probe - Information Only" in text
    assert "model_viability_probe" in text
    assert "model_viability_queue" in text
    assert "This does not replace the current deterministic competence gate" in text
    assert "Phase 2 Breadth Diagnostics - Gated" in text
    assert "effective_pathways_diagnostic" in text
    assert "candidate_conversion_diagnostic" in text
    assert "Do not run until deterministic recurrence passes against base" in text
    assert "CPU Review - Competence Pipeline" in text
    assert "checkpoint restore preflight prints `ok`" in text
    assert "Do not run until deterministic recurrence passes against base" in text
    assert "sha_resolved_nested_fetch_v3" in text
    assert "api.github.com/repos/{REPO}/contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py" in text
    assert "exec(compile(code" in text
    assert "exec(open(" not in text
    assert "colab/run_stage5_colab_continue.py" not in text
    assert payload["cells"][1]["cell_type"] == "code"


def test_current_user_facing_colab_queue_stays_in_sync() -> None:
    paths = [
        "README.md",
        "colab/CURRENT_A100_ACTION.md",
        "colab/NEXT_COLAB_SEQUENCE.md",
        "colab/STAGED_NOTEBOOKS.md",
    ]
    for path in paths:
        assert_ordered_queue((ROOT / path).read_text(encoding="utf-8"))


def test_stage_launcher_uses_current_bootstrap_target_queue() -> None:
    payload = notebook_payload("colab/00_stage_launcher.ipynb")
    text = "\n".join(str(cell.get("source", "")) for cell in payload.get("cells", []))

    assert "reentry_repair_smoke" in text
    assert "STAGE5_CURRENT_A100_TARGET" in text
    assert "sha_resolved_nested_fetch_v3" in text
    assert "api.github.com/repos/{REPO}/contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py" in text
    assert "exec(compile(code" in text
    assert "exec(open(" not in text
    assert "colab/run_stage5_colab_continue.py" not in text
    assert payload["cells"][0]["cell_type"] == "markdown"
    assert payload["cells"][1]["cell_type"] == "code"


def test_direct_preservation_precheck_notebook_is_fresh_runtime_safe() -> None:
    payload = notebook_payload("colab/10_stage5_direct_preservation_precheck.ipynb")
    text = "\n".join(str(cell.get("source", "")) for cell in payload.get("cells", []))

    assert "traced_sft_direct_preservation_precheck" in text
    assert "STAGE5_DIRECT_PRESERVE_PRECHECK_ONLY" in text
    assert "direct_route_precheck_needs_training" in text
    assert "sha_resolved_nested_fetch_v3" in text
    assert "api.github.com/repos/mshapiro123/recurrent-qwen-svgd" in text
    assert "GH_TOKEN" in text
    assert "GITHUB_TOKEN" in text
    assert "HF_TOKEN" in text
    assert "HUGGINGFACE_HUB_TOKEN" in text
    assert "STAGE5_DIRECT_PRESERVE_DRIVE_BACKUP" in text
    assert "Cache-Control" in text
    assert "exec(compile(code" in text
    assert "exec(open(" not in text
    assert "capability_ladder_local_hf_trace_sft_scale64" not in text
    assert payload["cells"][0]["cell_type"] == "markdown"
    assert payload["cells"][1]["cell_type"] == "code"
    assert payload["metadata"]["accelerator"] == "GPU"


def test_direct_preservation_g4_auto_notebook_runs_bounded_probe_not_scale64() -> None:
    payload = notebook_payload("colab/11_stage5_direct_preservation_g4_auto.ipynb")
    text = "\n".join(str(cell.get("source", "")) for cell in payload.get("cells", []))

    assert "traced_sft_direct_preservation_probe" in text
    assert "STAGE5_DIRECT_PRESERVE_SWEEP" in text
    assert "STAGE5_DIRECT_PRESERVE_CHAIN_CONFIRM" in text
    assert "STAGE5_DIRECT_PRESERVE_CHAIN_DEPTH_ROUTER" in text
    assert "STAGE5_DIRECT_PRESERVE_DRIVE_BACKUP" in text
    assert "direct_route_precheck_needs_training" in text
    assert "api.github.com/repos/mshapiro123/recurrent-qwen-svgd" in text
    assert "GH_TOKEN" in text
    assert "HF_TOKEN" in text
    assert "exec(compile(code" in text
    assert "exec(open(" not in text
    assert "capability_ladder_local_hf_trace_sft_scale64" not in text
    assert payload["cells"][0]["cell_type"] == "markdown"
    assert payload["cells"][1]["cell_type"] == "code"
    assert payload["metadata"]["accelerator"] == "GPU"


def test_full_arc_assessment_notebook_is_single_purpose() -> None:
    payload = notebook_payload("colab/07_stage5_full_arc_assessment.ipynb")
    text = "\n".join(str(cell.get("source", "")) for cell in payload.get("cells", []))

    assert "colab/run_stage5_full_assessment_once.py" in text
    assert "STAGE5_FULL_ASSESS_AUTO_DISCONNECT" in text
    assert "stage5_arc_mix_recovery_once_20260622_003331/summary.json" in text
    assert "colab/check_stage5_a100_go_no_go.py" in text
    assert "GO_NO_GO_RUN_ID" in text
    assert "A100 go/no-go blocked full ARC assessment" in text
    assert "disconnect_runtime(\"full assessment notebook failed\")" in text
    assert "colab/run_stage5_colab_continue.py" not in text
    assert "colab/run_stage5_reasoning_dataset_pipeline.py" not in text
    assert payload["cells"][0]["cell_type"] == "markdown"
    assert payload["cells"][1]["cell_type"] == "code"


def test_arc_mix_recovery_cell_is_single_purpose() -> None:
    text = (ROOT / "colab/STAGE5_ARC_MIX_RECOVERY_CELL.md").read_text(encoding="utf-8")
    plain = (ROOT / "colab/STAGE5_ARC_MIX_RECOVERY_CELL.py").read_text(encoding="utf-8")

    assert "colab/run_stage5_arc_mix_recovery_once.py" in text
    assert "colab/run_stage5_arc_mix_recovery_once.py" in plain
    assert "colab/check_stage5_a100_go_no_go.py" in text
    assert "colab/check_stage5_a100_go_no_go.py" in plain
    assert "STAGE5_ARC_MIX_ONCE_AUTO_DISCONNECT" in text
    assert "STAGE5_ARC_MIX_ONCE_AUTO_DISCONNECT" in plain
    assert "stage5_full_assessment_once_20260622_005522/summary.json" in text
    assert "stage5_full_assessment_once_20260622_005522/summary.json" in plain
    assert "arc_mix_response_w01_lr2e6" in text
    assert "arc_mix_response_w01_lr2e6" in plain
    assert "drive.mount" in text
    assert "drive.mount" in plain
    assert "GO_NO_GO_RUN_ID" in text
    assert "GO_NO_GO_RUN_ID" in plain
    assert "A100 go/no-go blocked ARC-mix recovery" in text
    assert "A100 go/no-go blocked ARC-mix recovery" in plain
    assert 'go_decision.get("spend_class") != "single_arc_mix_proxy"' in text
    assert 'go_decision.get("spend_class") != "single_arc_mix_proxy"' in plain
    assert "disconnect_runtime(\"ARC-mix recovery cell failed\")" in text
    assert "disconnect_runtime(\"ARC-mix recovery cell failed\")" in plain
    assert "colab/run_stage5_full_assessment_once.py" not in text
    assert "colab/run_stage5_full_assessment_once.py" not in plain
    assert "colab/run_stage5_colab_continue.py" not in text
    assert "colab/run_stage5_colab_continue.py" not in plain
    assert "STAGE5_ARC_MIX_RECOVERY_CELL.py" in text


def test_arc_mix_recovery_plain_cell_matches_markdown_code() -> None:
    markdown_cell = fenced_python_block("colab/STAGE5_ARC_MIX_RECOVERY_CELL.md")
    plain_cell = (ROOT / "colab/STAGE5_ARC_MIX_RECOVERY_CELL.py").read_text(encoding="utf-8")

    assert plain_cell == markdown_cell


def test_arc_mix_bootstrap_cell_fetches_recovery_cell_safely() -> None:
    text = (ROOT / "colab/STAGE5_ARC_MIX_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    plain = (ROOT / "colab/STAGE5_ARC_MIX_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_ARC_MIX_RECOVERY_CELL.py" in text
    assert "STAGE5_ARC_MIX_RECOVERY_CELL.py" in plain
    assert "api.github.com/repos" in plain
    assert "GH_TOKEN" in plain
    assert "GITHUB_TOKEN" in plain
    assert "required_markers" in plain
    assert "colab/run_stage5_arc_mix_recovery_once.py" in plain
    assert "colab/check_stage5_a100_go_no_go.py" in plain
    assert "arc_mix_response_w01_lr2e6" in plain
    assert "STAGE5_ARC_MIX_ONCE_AUTO_DISCONNECT" in plain
    assert "exec(compile(code" in plain
    assert "colab/run_stage5_full_assessment_once.py" not in plain
    assert "colab/run_stage5_colab_continue.py" not in plain


def test_arc_mix_bootstrap_plain_cell_matches_markdown_code() -> None:
    markdown_cell = fenced_python_block("colab/STAGE5_ARC_MIX_BOOTSTRAP_CELL.md")
    plain_cell = (ROOT / "colab/STAGE5_ARC_MIX_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert plain_cell == markdown_cell


def test_arc_mix_recovery_notebook_is_single_purpose() -> None:
    payload = notebook_payload("colab/09_stage5_arc_mix_recovery_once.ipynb")
    text = "\n".join(str(cell.get("source", "")) for cell in payload.get("cells", []))

    assert "colab/run_stage5_arc_mix_recovery_once.py" in text
    assert "colab/check_stage5_a100_go_no_go.py" in text
    assert "STAGE5_ARC_MIX_ONCE_AUTO_DISCONNECT" in text
    assert "stage5_full_assessment_once_20260622_005522/summary.json" in text
    assert "arc_mix_response_w01_lr2e6" in text
    assert "drive.mount" in text
    assert "GO_NO_GO_RUN_ID" in text
    assert "A100 go/no-go blocked ARC-mix recovery" in text
    assert 'go_decision.get("spend_class") != "single_arc_mix_proxy"' in text
    assert "disconnect_runtime(\"ARC-mix recovery notebook failed\")" in text
    assert "colab/run_stage5_full_assessment_once.py" not in text
    assert "colab/run_stage5_colab_continue.py" not in text
    assert payload["cells"][0]["cell_type"] == "markdown"
    assert payload["cells"][1]["cell_type"] == "code"
    assert payload["metadata"]["colab"]["gpuType"] == "A100"


def test_current_a100_action_points_to_competence_pipeline() -> None:
    text = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

    assert "docs/PROGRAM_TRACK_MASTER_SEQUENCE.md" in text
    assert "docs/EXPERIMENT_LOG.md" in text
    assert "traced_sft_competence_preserving_pipeline" in text
    assert "CURRENT_STAGE5_FRESH_LAUNCHER_CELL.py" in text
    assert "checkpoint_restore_preflight=ok" in text
    assert "review_stage5_competence_pipeline.py" in text
    assert "debiased_benchmark_suite" in text
    assert "dense_mcq_trace_sft_control" in text
    assert "checkpoint restore preflight does not print `ok`" in text
    assert "Phase 2/SVGD" in text
    assert "particles/SVGD" not in text or "deferred" in text
    assert "reentry_repair_smoke -> reentry_recovery_training" not in text
    assert "stage2_norm / entry_rms_safe_for_smoke" not in text
    assert "bridge_gate=0.0" not in text
    assert "STAGE5_CURRENT_A100_TARGET=safe_continue_execute" not in text
    assert "traced_sft_score_alignment_repair" not in text
    assert "capability_ladder_local_hf_trace_sft_scale64" not in text
    assert "run_stage5_routing_repair.py" not in text


def test_current_bootstrap_exposes_model_viability_queue_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert '"model_viability_queue"' in text
    assert "colab/STAGE5_MODEL_VIABILITY_QUEUE_CELL.py" in text
    assert "Qwen/Qwen2.5-3B-Instruct" in text
    assert "Qwen/Qwen2.5-7B-Instruct" in text
    assert "colab/run_stage5_model_viability_queue.py" in text


def test_current_bootstrap_exposes_forced_depth_diagnostic_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_FORCED_DEPTH_DIAGNOSTIC_CELL.py").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert '"forced_depth_diagnostic"' in payload
        assert "colab/STAGE5_FORCED_DEPTH_DIAGNOSTIC_CELL.py" in payload
        assert "STAGE5_FORCED_DEPTH_SOURCE_SUMMARY" in payload
        assert "STAGE5_BENCHMARK_FORCED_LOOP_COUNT" in payload
        assert "STAGE5_FORCED_DEPTH_LORA_RANK" in payload
        assert "STAGE5_BENCHMARK_LORA_RANK" in payload
        assert "forced_depth_lora_rank" in payload
        assert "forced_depth_requested_source_summary" in payload
        assert "checkpoint_bearing_source_summary" in payload
        assert "require_cuda_runtime" in payload
        assert "Forced-depth diagnostic requires an attached GPU runtime" in payload
    assert "STAGE5_FORCED_DEPTH_DIAGNOSTIC_CELL_VERSION" in cell
    assert "forced_depth_arc_v1" in cell
    assert "content_question_only,cyclic_label_aggregated" in cell
    assert "eval/analyze_depth_sweep.py" in cell
    assert "checkpoint_bearing_source_summary" in cell
    assert 'kind == "stage5_forced_depth_diagnostic"' in cell
    assert "require_cuda_runtime" in cell
    assert "forward_max_loops = max(loops)" in cell
    assert '"STAGE5_BENCHMARK_MAX_LOOPS": str(forward_max_loops)' in cell
    assert "ensure_drive_for_checkpoint_restore" in cell
    assert 'drive.mount("/content/drive", force_remount=FORCE_DRIVE_REMOUNT)' in cell
    assert 'source_payload.get("kind") == "stage5_unfreeze_recurrent_curriculum"' in cell
    assert '"STAGE5_BENCHMARK_LORA_RANK": lora_rank' in cell


def test_current_bootstrap_exposes_deterministic_final_gate_target() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_DETERMINISTIC_FINAL_GATE_CELL.py").read_text(encoding="utf-8")

    for text in (bootstrap, bootstrap_md):
        assert "deterministic_final_gate" in text
        assert "colab/STAGE5_DETERMINISTIC_FINAL_GATE_CELL.py" in text
        assert "STAGE5_FINAL_GATE_DISCOVERY_SWEEP" in text
        assert "STAGE5_FINAL_GATE_SOURCE_SUMMARY" in text
        assert "STAGE5_FINAL_GATE_OPEN_HARD_ARC_CHALLENGE_SPLIT" in text
        assert "STAGE5_FINAL_GATE_RESUME_EXISTING" in text
        assert "STAGE5_BENCHMARK_BASE_REUSE_RUN_ID" in text
        assert "eval/evaluate_rescue_selector_kfold.py" in text
    assert "STAGE5_DETERMINISTIC_FINAL_GATE_CELL_VERSION" in cell
    assert "deterministic_final_gate_v2_nested_selector" in cell
    assert "nested_outer_fold_train_only" in cell
    assert "paired_comparisons" in cell
    assert "correct_delta_recurrent_vs_base" in cell
    assert "closed_at_detectability_gate" in cell
    assert "selector_transfer_passed" in cell
    assert "STAGE5_BENCHMARK_FORCED_LOOP_COUNT" in cell


def test_current_bootstrap_exposes_rescue_predictability_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_RESCUE_PREDICTABILITY_CELL.py").read_text(encoding="utf-8")
    analyzer = (ROOT / "eval/analyze_rescue_predictability.py").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert '"rescue_predictability_analysis"' in payload
        assert "colab/STAGE5_RESCUE_PREDICTABILITY_CELL.py" in payload
        assert "STAGE5_RESCUE_PREDICTABILITY_SWEEP_SUMMARY" in payload
        assert "eval/analyze_rescue_predictability.py" in payload
    assert "STAGE5_RESCUE_PREDICTABILITY_CELL_VERSION" in cell
    assert "rescue_predictability_precursor_v1" in cell
    assert "stage5_current_rescue_predictability_summary" in cell
    assert "oriented AUC" in cell
    assert "binary_gate_top10" in analyzer


def test_current_bootstrap_exposes_rescue_detectability_gate_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_RESCUE_DETECTABILITY_CELL.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "eval/evaluate_rescue_detectability.py").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert '"rescue_detectability_gate"' in payload
        assert "colab/STAGE5_RESCUE_DETECTABILITY_CELL.py" in payload
        assert "STAGE5_RESCUE_DETECTABILITY_SWEEP_SUMMARY" in payload
        assert "observed_minus_null_p95" in payload
    assert "STAGE5_RESCUE_DETECTABILITY_CELL_VERSION" in cell
    assert "rescue_detectability_gate_v1" in cell
    assert "eval/evaluate_rescue_detectability.py" in cell
    assert "stage5_current_rescue_detectability_summary" in cell
    assert "diverse_probe_detectability" in evaluator
    assert "train_supervised_probes" in evaluator


def test_current_bootstrap_exposes_rescue_selector_transfer_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_RESCUE_SELECTOR_TRANSFER_CELL.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "eval/evaluate_rescue_selector_transfer.py").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert '"rescue_selector_transfer"' in payload
        assert "colab/STAGE5_RESCUE_SELECTOR_TRANSFER_CELL.py" in payload
        assert "STAGE5_RESCUE_SELECTOR_DISCOVERY_SWEEP_SUMMARY" in payload
        assert "STAGE5_RESCUE_SELECTOR_HELDOUT_SWEEP_SUMMARY" in payload
        assert "eval/evaluate_rescue_selector_transfer.py" in payload
        assert "stage5_current_rescue_selector_transfer_summary" in payload
    assert "STAGE5_RESCUE_SELECTOR_TRANSFER_CELL_VERSION" in cell
    assert "rescue_selector_transfer_v1" in cell
    assert "transferred_curve_summary" in cell
    assert "runtime.unassign" in cell
    assert "spectral_localization" in evaluator
    assert "diverse_probe_detectability" in evaluator
    assert "regularized_whitened_rescue_score" in evaluator


def test_current_bootstrap_exposes_tail_convergence_selector_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_TAIL_CONVERGENCE_SELECTOR_CELL.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "eval/evaluate_tail_convergence_selector.py").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert '"tail_convergence_selector"' in payload
        assert "colab/STAGE5_TAIL_CONVERGENCE_SELECTOR_CELL.py" in payload
        assert "eval/evaluate_tail_convergence_selector.py" in payload
        assert "tail_deceleration_12_minus_23" in payload
        assert "stage5_current_tail_convergence_selector_summary" in payload
    assert "STAGE5_TAIL_CONVERGENCE_SELECTOR_CELL_VERSION" in cell
    assert "tail_convergence_selector_v1" in cell
    assert "restore_checkpoint" in cell
    assert "runtime.unassign" in cell
    assert "tail_convergence_features" in evaluator
    assert "principal_subspace_rotation" in evaluator
    assert "tail_probe_detectability" in evaluator


def test_current_bootstrap_exposes_heldout_router_validation_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_HELDOUT_ROUTER_VALIDATION_CELL.py").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert '"heldout_router_validation"' in payload
        assert "colab/STAGE5_HELDOUT_ROUTER_VALIDATION_CELL.py" in payload
        assert "STAGE5_HELDOUT_ROUTER_DISCOVERY_SUMMARY" in payload
        assert "eval/evaluate_depth_router_transfer.py" in payload
        assert "eval/eval_latent_criticality.py" in payload
        assert "open_hard_arc_challenge" in payload
        assert "STAGE5_LATENT_CRITICALITY_JACOBIAN_EXAMPLES_PER_BENCHMARK" in payload
    assert "STAGE5_HELDOUT_ROUTER_VALIDATION_CELL_VERSION" in cell
    assert "heldout_router_validation_v1" in cell
    assert "router_transfer_content_question_only" in cell
    assert "eval/eval_latent_criticality.py" in cell
    assert "latent_criticality" in cell
    assert "STAGE5_BENCHMARK_FORCED_LOOP_COUNT" in cell


def test_current_bootstrap_exposes_latent_criticality_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_LATENT_CRITICALITY_CELL.py").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert '"latent_criticality_probe"' in payload
        assert "colab/STAGE5_LATENT_CRITICALITY_CELL.py" in payload
        assert "STAGE5_LATENT_CRITICALITY_SOURCE_SUMMARY" in payload
        assert "eval/eval_latent_criticality.py" in payload
        assert "STAGE5_LATENT_CRITICALITY_JACOBIAN_EXAMPLES_PER_BENCHMARK" in payload
        assert "restored_latent_criticality_checkpoint" in payload
    assert "STAGE5_LATENT_CRITICALITY_CELL_VERSION" in cell
    assert "latent_criticality_probe_v1" in cell
    assert "finite_difference_random_gain" in (ROOT / "eval/eval_latent_criticality.py").read_text(encoding="utf-8")


def test_current_bootstrap_exposes_reentry_covariance_check_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_REENTRY_COVARIANCE_CHECK_CELL.py").read_text(encoding="utf-8")
    eval_script = (ROOT / "eval/eval_reentry_covariance_check.py").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert '"reentry_covariance_check"' in payload
        assert "colab/STAGE5_REENTRY_COVARIANCE_CHECK_CELL.py" in payload
        assert "STAGE5_REENTRY_COVARIANCE_SOURCE_SUMMARY" in payload
        assert "eval/eval_reentry_covariance_check.py" in payload
        assert "covariance_match_check" in payload
        assert "directional_prebuild_gate" in payload
    assert "STAGE5_REENTRY_COVARIANCE_CHECK_CELL_VERSION" in cell
    assert "reentry_covariance_prebuild_v1" in cell
    assert "restored_reentry_covariance_checkpoint" in cell
    assert "tests/test_eval_reentry_covariance_check.py" in cell
    assert "covariance_match_check" in eval_script
    assert "general_linear_directional_adapter" in eval_script


def test_current_bootstrap_exposes_reentry_tail_diagnostic_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_REENTRY_TAIL_DIAGNOSTIC_CELL.py").read_text(encoding="utf-8")
    eval_script = (ROOT / "eval/eval_reentry_tail_diagnostic.py").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert '"reentry_tail_diagnostic"' in payload
        assert "colab/STAGE5_REENTRY_TAIL_DIAGNOSTIC_CELL.py" in payload
        assert "STAGE5_REENTRY_TAIL_SOURCE_SUMMARY" in payload
        assert "eval/eval_reentry_tail_diagnostic.py" in payload
        assert "tail_decomposition" in payload
        assert "harmed_rescued_tail_readout" in payload
    assert "STAGE5_REENTRY_TAIL_DIAGNOSTIC_CELL_VERSION" in cell
    assert "reentry_tail_resolved_v1" in cell
    assert "restored_reentry_tail_checkpoint" in cell
    assert "tests/test_eval_reentry_tail_diagnostic.py" in cell
    assert "tail_decomposition" in eval_script
    assert "harmed_minus_rescued" in eval_script


def test_current_bootstrap_exposes_reentry_tail_damper_sweep_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_REENTRY_TAIL_DAMPER_SWEEP_CELL.py").read_text(encoding="utf-8")
    eval_script = (ROOT / "eval/eval_tail_damper_depth_sweep.py").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert '"reentry_tail_damper_sweep"' in payload
        assert "colab/STAGE5_REENTRY_TAIL_DAMPER_SWEEP_CELL.py" in payload
        assert "STAGE5_TAIL_DAMPER_SOURCE_SUMMARY" in payload
        assert "eval/eval_tail_damper_depth_sweep.py" in payload
        assert "energy_oracle_tradeoff" in payload
    assert "STAGE5_REENTRY_TAIL_DAMPER_SWEEP_CELL_VERSION" in cell
    assert "tail_damper_tradeoff_v1" in cell
    assert "restored_tail_damper_checkpoint" in cell
    assert "tests/test_eval_tail_damper_depth_sweep.py" in cell
    assert "oracle_gap_vs_loop1" in eval_script
    assert "loop8 tail ratio" in eval_script


def test_current_bootstrap_exposes_reentry_tail_damper_heldout_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_REENTRY_TAIL_DAMPER_SWEEP_CELL.py").read_text(encoding="utf-8")
    eval_script = (ROOT / "eval/eval_tail_damper_depth_sweep.py").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert '"reentry_tail_damper_heldout"' in payload
        assert '"STAGE5_TAIL_DAMPER_ARC_OFFSET": "256"' in payload
        assert '"STAGE5_TAIL_DAMPER_ARC_LIMIT": "256"' in payload
        assert "colab/STAGE5_REENTRY_TAIL_DAMPER_SWEEP_CELL.py" in payload
    assert "stage5_reentry_tail_damper_sweep_" in cell
    assert "safe_slug(arc_config)" in cell
    assert "safe_slug(arc_split)" in cell
    assert "--source_summary" in cell
    assert "--arc_config" in cell
    assert "--arc_split" in cell
    assert "--score_target" in cell
    assert "source_summary" in eval_script


def test_current_bootstrap_exposes_reentry_tail_damper_powered_arc_train_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_REENTRY_TAIL_DAMPER_SWEEP_CELL.py").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert '"reentry_tail_damper_powered_arc_train"' in payload
        assert '"STAGE5_TAIL_DAMPER_ARC_CONFIG": "ARC-Challenge"' in payload
        assert '"STAGE5_TAIL_DAMPER_ARC_SPLIT": "train"' in payload
        assert '"STAGE5_TAIL_DAMPER_ARC_LIMIT": "512"' in payload
        assert '"STAGE5_TAIL_DAMPER_STRENGTHS": "0,0.5,1.0"' in payload
        assert '"STAGE5_TAIL_DAMPER_SCORE_TARGET": "option_text"' in payload
        assert "powered ARC-Challenge train confirmation" in payload
    assert "tail_damper_arc_config" in cell
    assert "tail_damper_score_target" in cell


def test_current_bootstrap_source_summary_override_fans_out_to_benchmark_and_control() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert "SOURCE_SUMMARY_OVERRIDE" in payload
        assert "STAGE5_DEBIASED_BENCHMARK_SOURCE_SUMMARY" in payload
        assert "STAGE5_DENSE_MCQ_SOURCE_SUMMARY" in payload
        assert "STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY" in payload
        assert "STAGE5_COMPETENCE_SOURCE_SUMMARY" in payload
        assert "STAGE5_HELDOUT_ROUTER_DISCOVERY_SUMMARY" in payload
        assert "STAGE5_LATENT_CRITICALITY_SOURCE_SUMMARY" in payload
        assert "STAGE5_REENTRY_COVARIANCE_SOURCE_SUMMARY" in payload
        assert "STAGE5_REENTRY_TAIL_SOURCE_SUMMARY" in payload
        assert "STAGE5_TAIL_DAMPER_SOURCE_SUMMARY" in payload
        assert 'os.environ["STAGE5_DEBIASED_BENCHMARK_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE' in payload
        assert 'os.environ["STAGE5_DENSE_MCQ_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE' in payload
        assert 'os.environ["STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE' in payload
        assert 'os.environ["STAGE5_COMPETENCE_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE' in payload
        assert 'os.environ["STAGE5_HELDOUT_ROUTER_DISCOVERY_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE' in payload
        assert 'os.environ["STAGE5_LATENT_CRITICALITY_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE' in payload
        assert 'os.environ["STAGE5_REENTRY_COVARIANCE_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE' in payload
        assert 'os.environ["STAGE5_REENTRY_TAIL_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE' in payload
        assert 'os.environ["STAGE5_TAIL_DAMPER_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE' in payload
        assert 'os.environ.pop("STAGE5_DEBIASED_BENCHMARK_SOURCE_SUMMARY", None)' in payload
        assert 'os.environ.pop("STAGE5_DENSE_MCQ_SOURCE_SUMMARY", None)' in payload
        assert 'os.environ.pop("STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY", None)' in payload
        assert 'os.environ.pop("STAGE5_COMPETENCE_SOURCE_SUMMARY", None)' in payload
        assert 'os.environ.pop("STAGE5_HELDOUT_ROUTER_DISCOVERY_SUMMARY", None)' in payload
        assert 'os.environ.pop("STAGE5_LATENT_CRITICALITY_SOURCE_SUMMARY", None)' in payload
        assert 'os.environ.pop("STAGE5_REENTRY_COVARIANCE_SOURCE_SUMMARY", None)' in payload
        assert 'os.environ.pop("STAGE5_REENTRY_TAIL_SOURCE_SUMMARY", None)' in payload
        assert 'os.environ.pop("STAGE5_TAIL_DAMPER_SOURCE_SUMMARY", None)' in payload


def test_current_bootstrap_can_explicitly_prefer_local_head_to_stale_ref_resolution() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    markdown = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")

    for payload in (text, markdown):
        assert "STAGE5_BOOTSTRAP_PREFER_LOCAL_HEAD" in payload
        assert 'os.environ.get("STAGE5_BOOTSTRAP_PREFER_LOCAL_HEAD", "0")' in payload
        assert "git\", \"rev-parse\", \"HEAD" in payload
        assert "using local HEAD" in payload
        assert "RESOLVED_REF = local_head" in payload


def test_current_bootstrap_exposes_depth_router_after_direct_preserve_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_CELL.py").read_text(encoding="utf-8")

    assert '"traced_sft_depth_router_after_direct_preserve"' in text
    assert "colab/STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_CELL.py" in text
    assert "stage5_traced_sft_direct_preservation_20260623_scale64_lr1e6/summary.json" in text
    assert "stage5_latest_direct_preservation_summary.txt" in text
    assert "STAGE5_DEPTH_ROUTER_LOOP_CONTROL_CE_WEIGHT" in text
    assert "STAGE5_DEPTH_ROUTER_HALT_TARGET_NLL_WEIGHT" in text
    assert "colab/run_stage5_curriculum_sft.py" in text
    assert "colab/run_stage5_benchmark_suite.py" in text
    assert "colab/assess_stage5_traced_sft.py" in text
    assert "STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_CELL_VERSION" in cell
    assert "No passed direct-preservation summary found" in cell
    assert "STAGE5_CURRICULUM_ALLOW_NO_DRIVE_BACKUP" in cell
    assert "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL" in cell


def test_runbooks_prefer_guarded_current_action_over_legacy_autopilot() -> None:
    arc_plan = (ROOT / "docs/ARC_AGI_PROGRESS_PLAN.md").read_text(encoding="utf-8")
    staged = (ROOT / "colab/STAGED_NOTEBOOKS.md").read_text(encoding="utf-8")

    assert "prefer the maintained\ncurrent-action path" in arc_plan
    assert "colab/CURRENT_A100_BOOTSTRAP_CELL.py" in arc_plan
    assert "legacy\nARC-AGI-specific branch runner" in arc_plan
    assert "For overnight runs, prefer `colab/run_stage5_arc_agi_autopilot.py`" not in arc_plan
    assert "STAGE5_CURRENT_A100_TARGET" in staged
    assert "traced_sft_competence_preserving_pipeline" in staged
    assert "review_stage5_competence_pipeline.py" in staged
    assert "debiased_benchmark_suite" in staged
    assert "dense_mcq_trace_sft_control" in staged
    assert "particles/SVGD" in staged
    assert "If a historical notebook conflicts with `CURRENT_A100_ACTION.md`" in staged


def test_safe_continue_cell_defaults_to_dry_run_and_guarded_action() -> None:
    text = (ROOT / "colab/STAGE5_SAFE_CONTINUE_CELL.md").read_text(encoding="utf-8")
    plain = (ROOT / "colab/STAGE5_SAFE_CONTINUE_CELL.py").read_text(encoding="utf-8")

    assert 'RUN_A100_ACTION = env_bool("STAGE5_SAFE_CONTINUE_RUN_A100_ACTION", False)' in text
    assert 'RUN_A100_ACTION = env_bool("STAGE5_SAFE_CONTINUE_RUN_A100_ACTION", False)' in plain
    assert 'DISCONNECT_RUNTIME_WHEN_DONE = env_bool("STAGE5_SAFE_CONTINUE_DISCONNECT", True)' in text
    assert 'DISCONNECT_RUNTIME_WHEN_DONE = env_bool("STAGE5_SAFE_CONTINUE_DISCONNECT", True)' in plain
    assert 'MAX_NEXT_ACTIONS = int(os.environ.get("STAGE5_SAFE_CONTINUE_MAX_ACTIONS", "1"))' in text
    assert 'MAX_NEXT_ACTIONS = int(os.environ.get("STAGE5_SAFE_CONTINUE_MAX_ACTIONS", "1"))' in plain
    assert 'ALLOW_REPEAT_NEXT_ACTION = env_bool("STAGE5_SAFE_CONTINUE_ALLOW_REPEAT", False)' in text
    assert 'ALLOW_REPEAT_NEXT_ACTION = env_bool("STAGE5_SAFE_CONTINUE_ALLOW_REPEAT", False)' in plain
    assert "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY" in text
    assert "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY" in plain
    assert "SOURCE_SUMMARY_OVERRIDE" in text
    assert "SOURCE_SUMMARY_OVERRIDE" in plain
    assert 'pointer = ROOT / "config" / "stage5_current_source_summary.txt"' in text
    assert 'pointer = ROOT / "config" / "stage5_current_source_summary.txt"' in plain
    assert "Using current source summary pointer" in text
    assert "Using current source summary pointer" in plain
    assert "colab/check_stage5_a100_go_no_go.py" in text
    assert "colab/check_stage5_a100_go_no_go.py" in plain
    assert "colab/run_stage5_next_action.py" in text
    assert "colab/run_stage5_next_action.py" in plain
    assert 'env["STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE"] = "1" if execute_action else "0"' in text
    assert 'env["STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE"] = "1" if execute_action else "0"' in plain
    assert "STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS" in text
    assert 'env["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] = str(MAX_NEXT_ACTIONS)' in text
    assert 'env["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] = str(MAX_NEXT_ACTIONS)' in plain
    assert 'env["STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT"] = "1" if ALLOW_REPEAT_NEXT_ACTION else "0"' in text
    assert 'env["STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT"] = "1" if ALLOW_REPEAT_NEXT_ACTION else "0"' in plain
    assert "Dry run complete" in text
    assert "STAGE5_SAFE_CONTINUE_DISCONNECT" in text
    assert "runtime.unassign()" in text
    assert "GO_NO_GO_RUN_ID" in text
    assert "Skipping requirements install because no paid action will execute." in text
    assert "execute_action = bool(RUN_A100_ACTION and go_allowed)" in text
    assert "DRY_RUN_GREEN" in text
    assert "DRY_RUN_RED" in text
    assert "a100_checkpoint_preflight" in text
    assert "a100_input_preflight" in text
    assert "curriculum input artifacts are not visible" in text
    assert "STAGE5_CURRENT_A100_TARGET=programmatic_curriculum_cpu" in text
    assert "tests/test_stage5_routing_repair.py" in text
    assert "tests/test_stage5_balanced_arc_mix_gate.py" in text
    assert "tests/test_curriculum_sft_gate.py" in text
    assert "tests/test_stage5_curriculum_sft.py" in text
    assert "tests/test_curriculum_pipeline_from_artifacts.py" in text
    assert "tests/test_curriculum_jsonl.py" in text
    assert "stage5_routing_diagnostic_20260622_041706/summary.json" in text
    assert "mount_drive_for_paid_action" in text
    assert "STAGE5_SAFE_CONTINUE_PREFER_TRAINING_SOURCE" in text
    assert "latest_training_source_summary" in text
    assert "stage5_capability_ladder_trace_collection" in text
    assert 'drive.mount("/content/drive", force_remount=True)' in text
    assert "Mounting Google Drive so checkpoint artifacts can be restored." in text
    assert text.index("if RUN_A100_ACTION and PREFER_TRAINING_SOURCE:\n    mount_drive_for_paid_action()") < text.index(
        '"colab/check_stage5_a100_go_no_go.py"'
    )
    assert "colab/run_stage5_full_assessment_once.py" not in text


def test_safe_continue_plain_cell_matches_markdown_code() -> None:
    markdown_cell = fenced_python_block("colab/STAGE5_SAFE_CONTINUE_CELL.md")
    plain_cell = (ROOT / "colab/STAGE5_SAFE_CONTINUE_CELL.py").read_text(encoding="utf-8")

    assert plain_cell == markdown_cell


def test_safe_continue_notebook_defaults_to_dry_run_and_guarded_action() -> None:
    payload = notebook_payload("colab/08_stage5_safe_continue.ipynb")
    text = "\n".join(str(cell.get("source", "")) for cell in payload.get("cells", []))

    assert 'RUN_A100_ACTION = env_bool("STAGE5_SAFE_CONTINUE_RUN_A100_ACTION", False)' in text
    assert "colab/check_stage5_a100_go_no_go.py" in text
    assert "colab/run_stage5_next_action.py" in text
    assert "STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE" in text
    assert "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY" in text
    assert "Using current source summary pointer" in text
    assert 'pointer = ROOT / "config" / "stage5_current_source_summary.txt"' in text
    assert "Dry run complete" in text
    assert 'DISCONNECT_RUNTIME_WHEN_DONE = env_bool("STAGE5_SAFE_CONTINUE_DISCONNECT", True)' in text
    assert 'MAX_NEXT_ACTIONS = int(os.environ.get("STAGE5_SAFE_CONTINUE_MAX_ACTIONS", "1"))' in text
    assert 'ALLOW_REPEAT_NEXT_ACTION = env_bool("STAGE5_SAFE_CONTINUE_ALLOW_REPEAT", False)' in text
    assert "runtime.unassign()" in text
    assert "GO_NO_GO_RUN_ID" in text
    assert "Skipping requirements install because no paid action will execute." in text
    assert "tests/test_curriculum_sft_gate.py" in text
    assert "DRY_RUN_GREEN" in text
    assert "DRY_RUN_RED" in text
    assert "a100_checkpoint_preflight" in text
    assert "a100_input_preflight" in text
    assert "curriculum input artifacts are not visible" in text
    assert "STAGE5_CURRENT_A100_TARGET=programmatic_curriculum_cpu" in text
    assert "tests/test_stage5_curriculum_sft.py" in text
    assert "tests/test_curriculum_pipeline_from_artifacts.py" in text
    assert "tests/test_curriculum_jsonl.py" in text
    assert "execute_action = bool(RUN_A100_ACTION and go_allowed)" in text
    assert 'env["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] = str(MAX_NEXT_ACTIONS)' in text
    assert 'env["STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT"] = "1" if ALLOW_REPEAT_NEXT_ACTION else "0"' in text
    assert "tests/test_stage5_routing_repair.py" in text
    assert "tests/test_stage5_balanced_arc_mix_gate.py" in text
    assert "stage5_routing_diagnostic_20260622_041706/summary.json" in text
    assert "mount_drive_for_paid_action" in text
    assert 'drive.mount("/content/drive", force_remount=True)' in text
    assert "Mounting Google Drive so checkpoint artifacts can be restored." in text
    assert text.index("if RUN_A100_ACTION:\\n") < text.index(
        '"colab/check_stage5_a100_go_no_go.py"'
    )
    assert "colab/run_stage5_full_assessment_once.py" not in text
    assert payload["cells"][0]["cell_type"] == "markdown"
    assert payload["cells"][1]["cell_type"] == "code"


def test_current_a100_bootstrap_fetches_only_current_plain_cells() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    plain = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert 'TARGET = os.environ.get("STAGE5_CURRENT_A100_TARGET", "preflight")' in plain
    assert "STAGE5_CURRENT_A100_SOURCE_SUMMARY" in plain
    assert 'os.environ.pop("STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY", None)' in plain
    assert 'os.environ.pop("STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY", None)' in plain
    assert "colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.py" in plain
    assert "colab/STAGE5_SAFE_CONTINUE_CELL.py" in plain
    assert '"programmatic_curriculum_cpu"' in plain
    assert "colab/STAGE5_PROGRAMMATIC_CURRICULUM_CELL.py" in plain
    assert '"arc_challenge_mcq_debias_confirm"' in plain
    assert "colab/STAGE5_ARC_CHALLENGE_MCQ_DEBIAS_CELL.py" in plain
    assert "STAGE5_ARC_CHALLENGE_MCQ_DEBIAS_CELL_VERSION" in plain
    assert "STAGE5_MCQ_DEBIAS_QUIET_EVAL" in plain
    assert "STAGE5_MCQ_DEBIAS_RESUME_EXISTING" in plain
    assert "STAGE5_MCQ_DEBIAS_PUSH" in plain
    assert "colab/assess_stage5_mcq_debias_pair.py" in plain
    assert "colab/apply_stage5_mcq_scoring_policy.py" in plain
    assert "training/run_programmatic_curriculum_pipeline.py" in plain
    assert "colab/publish_stage5_curriculum_gate.py" in plain
    assert "Refusing to run CPU-only programmatic curriculum generation" in plain
    assert '"safe_continue_execute"' in plain
    assert '"master_sequence_status"' in plain
    assert "colab/STAGE5_MASTER_SEQUENCE_STATUS_CELL.py" in plain
    assert "STAGE5_MASTER_SEQUENCE_STATUS_CELL_VERSION" in plain
    assert "colab/print_current_stage5_action.py" in plain
    assert "colab/review_stage5_reentry.py" in plain
    assert '"STAGE5_SAFE_CONTINUE_RUN_A100_ACTION": "1"' in plain
    assert '"STAGE5_SAFE_CONTINUE_PREFER_TRAINING_SOURCE": "1"' in plain
    assert '"STAGE5_SAFE_CONTINUE_RUN_A100_ACTION": "0"' in plain
    assert "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY" in plain
    assert "checkpoint_preflight" in plain
    assert "next_action_guard" in plain
    assert "mount_drive_for_paid_action" in plain
    assert "api.github.com/repos" in plain
    assert "git/ref/heads" in plain
    assert "RESOLVED_REF" in plain
    assert "BOOTSTRAP_VERSION" in plain
    assert "sha_resolved_nested_fetch_v3" in plain
    assert "PROGRAMMATIC_CURRICULUM_CELL_VERSION" in plain
    assert "shutil.which" in plain
    assert "nvidia-smi" in plain
    assert "FileNotFoundError" in plain
    assert "OSError" in plain
    assert "cache_bust" in plain
    assert '"Cache-Control": "no-cache"' in plain
    assert "GH_TOKEN" in plain
    assert "GITHUB_TOKEN" in plain
    assert "required_markers" not in plain
    assert "colab/STAGE5_ARC_MIX_RECOVERY_CELL.py" not in plain
    assert "preflight" in text
    assert "programmatic_curriculum_cpu" in text
    assert "safe_continue_execute" in text
    assert "arc_challenge_mcq_debias_confirm" in text
    assert "arc_mix_depth_routing_probe" in text
    assert '"arc_mix_depth_routing_probe"' in plain
    assert "STAGE5_ARC_MIX_DEPTH_ROUTING_CELL.py" in plain
    assert "STAGE5_ARC_MIX_DEPTH_ROUTING_CELL_VERSION" in plain
    assert "STAGE5_ARC_MIX_USE_LEARNED_LOOP_CONTROL" in plain
    assert "capability_ladder_mcq_probe" in text
    assert "capability_ladder_trace_jobs_cpu" in text
    assert "cyclic-permutation MCQ diagnostic" in text
    assert "STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL.py" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL_VERSION" in plain
    assert "training/build_capability_ladder_trace_jobs.py" in plain
    assert "capability_ladder_trace_responses_cpu" in text
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL.py" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL_VERSION" in plain
    assert "capability_ladder_trace_collect_cpu" in text
    assert "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL.py" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL_VERSION" in plain
    assert "capability_ladder_7b_trace_chain" in text
    assert '"capability_ladder_7b_trace_chain"' in plain
    assert "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_CELL.py" in plain
    assert "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_CELL_VERSION" in plain
    assert "traced_capability_ladder_sft" in text
    assert '"traced_capability_ladder_sft"' in plain
    assert "STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL.py" in plain
    assert "STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL_VERSION" in plain
    assert "model_viability_probe" in text
    assert '"model_viability_probe"' in plain
    assert "STAGE5_MODEL_VIABILITY_PROBE_CELL.py" in plain
    assert "STAGE5_MODEL_VIABILITY_PROBE_CELL_VERSION" in plain
    assert "colab/run_stage5_model_viability_probe.py" in plain
    assert "STAGE5_MODEL_PROBE_MODEL_NAME" in plain


def test_current_a100_bootstrap_plain_cell_matches_markdown_code() -> None:
    markdown_cell = fenced_python_block("colab/CURRENT_A100_BOOTSTRAP_CELL.md")
    plain_cell = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert plain_cell == markdown_cell


def test_master_sequence_status_cell_matches_markdown_code() -> None:
    text = (ROOT / "colab/STAGE5_MASTER_SEQUENCE_STATUS_CELL.md").read_text(encoding="utf-8")
    plain = (ROOT / "colab/STAGE5_MASTER_SEQUENCE_STATUS_CELL.py").read_text(encoding="utf-8")
    markdown_cell = fenced_python_block("colab/STAGE5_MASTER_SEQUENCE_STATUS_CELL.md")

    assert plain == markdown_cell
    assert "STAGE5_MASTER_SEQUENCE_STATUS_CELL_VERSION" in plain
    assert "master_sequence_status_v1" in plain
    assert "MASTER_SEQUENCE_STATUS" in plain
    assert "colab/print_current_stage5_action.py" in plain
    assert "colab/review_stage5_reentry.py" in plain
    assert "colab/review_stage5_recovery.py" in plain
    assert "colab/review_stage5_phase1_gate.py" in plain
    assert "colab/review_stage5_recovery_curriculum.py" in plain
    assert "colab/plan_stage5_curriculum_scaleup.py" in plain
    assert "Stage 4 Recovery Review" in plain
    assert "Phase 1 Gate Review" in plain
    assert "Claim-Sized Curriculum Scale-Up Plan" in plain
    assert "Phase 1 Gate Review" in (
        ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py"
    ).read_text(encoding="utf-8")
    assert "Stage 4 Recovery Curriculum Readiness" in (
        ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py"
    ).read_text(encoding="utf-8")
    assert "Claim-Sized Curriculum Scale-Up Plan" in (
        ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py"
    ).read_text(encoding="utf-8")
    assert "NEXT_COLAB_SEQUENCE excerpt" in plain


def test_model_viability_probe_cell_is_generic_and_matches_markdown_code() -> None:
    text = (ROOT / "colab/STAGE5_MODEL_VIABILITY_PROBE_CELL.md").read_text(encoding="utf-8")
    plain = (ROOT / "colab/STAGE5_MODEL_VIABILITY_PROBE_CELL.py").read_text(encoding="utf-8")
    markdown_cell = fenced_python_block("colab/STAGE5_MODEL_VIABILITY_PROBE_CELL.md")

    assert plain == markdown_cell
    assert "STAGE5_MODEL_VIABILITY_PROBE_CELL_VERSION" in plain
    assert "STAGE5_MODEL_PROBE_MODEL_NAME" in plain
    assert "Qwen/Qwen2.5-1.5B-Instruct" in plain
    assert "STAGE5_MODEL_PROBE_LAYER_SPLIT" in plain
    assert "STAGE5_MODEL_PROBE_LOOPS" in plain
    assert "STAGE5_MODEL_PROBE_SCORE_TARGETS" in plain
    assert "colab/run_stage5_model_viability_probe.py" in plain
    assert "tests/test_model_viability_probe.py" in plain
    assert "runtime.unassign()" in plain
    assert "Qwen 3B" in text
    assert "larger compatible Qwen checkpoint" in text


def test_arc_mix_depth_routing_cell_is_single_purpose() -> None:
    plain = (ROOT / "colab/STAGE5_ARC_MIX_DEPTH_ROUTING_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_ARC_MIX_DEPTH_ROUTING_CELL_VERSION" in plain
    assert "colab/run_stage5_balanced_arc_mix_gate.py" in plain
    assert "stage5_content_arcmix_qonly_optiontext_arc256_check_20260623_123424/summary.json" in plain
    assert "STAGE5_ARC_MIX_USE_LEARNED_LOOP_CONTROL" in plain
    assert "STAGE5_ARC_MIX_EVAL_USE_LEARNED_LOOP_CONTROL" in plain
    assert "STAGE5_ARC_MIX_LOOP_CONTROL_CE_WEIGHT" in plain
    assert "STAGE5_ARC_MIX_HALT_TARGET_NLL_WEIGHT" in plain
    assert "target_loop_count ARC-Easy=1 ARC-Challenge=3" in plain
    assert "question_only" in plain
    assert "option_text" in plain
    assert "tests/test_stage5_balanced_arc_mix_gate.py" in plain
    assert "runtime.unassign" in plain
    assert "run_stage5_benchmark_suite.py" not in plain
    assert "sample_latents" not in plain


def test_arc_challenge_mcq_debias_cell_is_bounded_and_pushes_summary() -> None:
    plain = (ROOT / "colab/STAGE5_ARC_CHALLENGE_MCQ_DEBIAS_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_ARC_CHALLENGE_MCQ_DEBIAS_CELL_VERSION" in plain
    assert 'env["STAGE5_MCQ_DEBIAS_ARC_CONFIG"] = "ARC-Challenge"' in plain
    assert 'env["STAGE5_MCQ_DEBIAS_QUIET_EVAL"] = "1"' in plain
    assert 'env["STAGE5_MCQ_DEBIAS_RESUME_EXISTING"] = "1"' in plain
    assert 'env["STAGE5_MCQ_DEBIAS_PUSH"] = "1"' in plain
    assert "colab/run_stage5_mcq_debias_diagnostic.py" in plain
    assert "colab/assess_stage5_mcq_debias_pair.py" in plain
    assert "colab/apply_stage5_mcq_scoring_policy.py" in plain
    assert "STAGE5_MCQ_DEBIAS_ARC_CHALLENGE_SUMMARY" in plain
    assert "policy_summary:" in plain
    assert "tests/test_mcq_debias.py" in plain
    assert "tests/test_stage5_next_plan.py" in plain
    assert "runtime.unassign()" in plain


def test_debiased_benchmark_suite_cell_is_bounded_and_policy_compliant() -> None:
    plain = (ROOT / "colab/STAGE5_DEBIASED_BENCHMARK_SUITE_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_DEBIASED_BENCHMARK_SUITE_CELL_VERSION" in plain
    assert "STAGE5_DEBIASED_MOUNT_DRIVE_FIRST" in plain
    assert "STAGE5_DEBIASED_USE_LEARNED_LOOP_CONTROL" in plain
    assert "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL" in plain
    assert '"STAGE5_BENCHMARKS"] = os.environ.get(' in plain
    assert '"STAGE5_DEBIASED_BENCHMARKS",' in plain
    assert '"arc_easy,arc_challenge,gpqa_lite",' in plain
    assert '"STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT"] = os.environ.get("STAGE5_DEBIASED_ARC_CHALLENGE_LIMIT", "128")' in plain
    assert '"STAGE5_BENCHMARK_GPQA_LIMIT"] = os.environ.get("STAGE5_DEBIASED_GPQA_LIMIT", "16")' in plain
    assert '"STAGE5_BENCHMARK_SCORE_TARGETS"] = os.environ.get(' in plain
    assert '"STAGE5_DEBIASED_SCORE_TARGETS",' in plain
    assert '"label,content_question_only,cyclic_label_aggregated",' in plain
    assert '"STAGE5_BENCHMARK_ASSESS_SCORE_TARGET"] = "cyclic_label_aggregated"' in plain
    assert '"STAGE5_BENCHMARK_ASSESS_AGGREGATE"] = "permutation_mean"' in plain
    assert '"STAGE5_DEBIASED_USE_LEARNED_LOOP_CONTROL",\n        "1",' in plain
    assert "benchmark_source_summary" in plain
    assert "validate_stage4_benchmark_source" in plain
    assert "stage5_reentry_recovery_training" in plain
    assert "post_reentry_health_checks.status" in plain
    assert "spectral_source_health_override" in plain
    assert "stage4_benchmark_source_gate=spectral_health_override" in plain
    assert "spectral_source_health_override" in bootstrap
    assert "stage4_benchmark_source_gate=passed" in plain
    assert "colab/run_stage5_benchmark_suite.py" in plain
    assert "colab/assess_stage5_benchmark_suite.py" in plain
    assert "tests/test_stage5_benchmark_suite.py" in plain
    assert "tests/test_stage5_benchmark_assessment.py" in plain
    assert "STAGE5_DEBIASED_BENCHMARK_DISCONNECT" in plain
    assert "Leaving Colab runtime attached because STAGE5_DEBIASED_BENCHMARK_DISCONNECT=0." in plain
    assert "runtime.unassign()" in plain
    assert "debiased_benchmark_suite" in bootstrap
    assert "STAGE5_DEBIASED_BENCHMARK_SUITE_CELL.py" in bootstrap
    assert '"STAGE5_DEBIASED_BENCHMARKS": "arc_easy,arc_challenge,gpqa_lite"' in bootstrap
    assert '"STAGE5_DEBIASED_ARC_EASY_LIMIT": "128"' in bootstrap
    assert '"STAGE5_DEBIASED_USE_LEARNED_LOOP_CONTROL": "1"' in bootstrap
    assert '"STAGE5_DEBIASED_BENCHMARK_DISCONNECT": "1"' in bootstrap


def test_depth_balanced_benchmark_target_uses_learned_loop_control() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert '"depth_balanced_benchmark"' in bootstrap
    assert '"STAGE5_DEBIASED_BENCHMARKS": "arc_easy,arc_challenge"' in bootstrap
    assert '"STAGE5_DEBIASED_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated"' in bootstrap
    assert '"STAGE5_DEBIASED_USE_LEARNED_LOOP_CONTROL": "1"' in bootstrap
    assert '"STAGE5_DEBIASED_MOUNT_DRIVE_FIRST": "0"' in bootstrap
    assert '"STAGE5_DEBIASED_ARC_EASY_LIMIT": "512"' in bootstrap
    assert '"STAGE5_DEBIASED_ARC_CHALLENGE_LIMIT": "512"' in bootstrap


def test_depth_signal_confirmation_target_chains_recovery_and_hard_content_benchmark() -> None:
    plain = (ROOT / "colab/STAGE5_DEPTH_SIGNAL_CONFIRMATION_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_DEPTH_SIGNAL_CONFIRMATION_CELL_VERSION" in plain
    assert "Stage 4: depth-routing recovery" in plain
    assert "Stage 5: depth-signal benchmark" in plain
    assert "STAGE5_REENTRY_RECOVERY_DISCONNECT" in plain
    assert "STAGE5_DEBIASED_BENCHMARK_DISCONNECT" in plain
    assert "open_hard_arc_challenge" in plain
    assert "STAGE5_DEBIASED_ASSESS_REQUIRED_BENCHMARKS" in plain
    assert "depth_signal_confirmation_complete=true" in plain
    assert '"depth_signal_confirmation"' in bootstrap
    assert '"STAGE5_DEBIASED_BENCHMARKS": "arc_easy,arc_challenge,open_hard_arc_challenge"' in bootstrap
    assert '"STAGE5_DEBIASED_ASSESS_REQUIRED_BENCHMARKS": "arc_challenge,open_hard_arc_challenge"' in bootstrap


def test_capability_ladder_mcq_probe_cell_is_bounded_and_depth_ladder_focused() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL_VERSION" in plain
    assert "Qwen/Qwen2.5-0.5B-Instruct" in plain
    assert "Qwen/Qwen2.5-1.5B-Instruct" in plain
    assert "Qwen/Qwen2.5-3B-Instruct" in plain
    assert '"STAGE5_CAPABILITY_LADDER_ARC_LIMIT",\n        "48"' in plain
    assert '"STAGE5_CAPABILITY_LADDER_SCORE_MODE",\n        "content_question_only"' in plain
    assert '"STAGE5_CAPABILITY_LADDER_BACKUP_DRIVE", "0"' in plain
    assert "Drive backup disabled for capability-ladder probe" in plain
    assert "qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3" in plain
    assert "STAGE5_CAPABILITY_LADDER_MIN_TARGET_LOOP_ROWS" in plain
    assert "1=1,2=1,3=1" in plain
    assert "colab/run_stage5_capability_ladder_mcq_probe.py" in plain
    assert "tests/test_stage5_capability_ladder_mcq_probe.py" in plain
    assert "runtime.unassign()" in plain
    assert "capability_ladder_mcq_probe" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL.py" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_MODEL_LADDER" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_MIN_TARGET_LOOP_ROWS" in bootstrap


def test_capability_ladder_trace_jobs_cell_is_cpu_only_and_depth_ladder_focused() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL_VERSION" in plain
    assert "capability_ladder_trace_jobs_cpu" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_ALLOW_GPU" in plain
    assert "colab/run_stage5_capability_ladder_trace_jobs.py" in plain
    assert "tests/test_capability_ladder_trace_jobs.py" in plain
    assert "runtime.unassign()" in plain
    assert "capability_ladder_trace_jobs_cpu" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL.py" in bootstrap


def test_capability_ladder_7b_trace_chain_cell_runs_probe_then_trace_jobs() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_CELL_VERSION" in plain
    assert "capability_ladder_7b_trace_chain" in plain
    assert "Qwen/Qwen2.5-7B-Instruct" in plain
    assert "qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3,qwen_7b:4" in plain
    assert "STAGE5_CAPABILITY_LADDER_MIN_TARGET_LOOP_ROWS" in plain
    assert "1=1,2=1,3=1,4=1" in plain
    assert "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_ARC_LIMIT" in plain
    assert "colab/run_stage5_capability_ladder_mcq_probe.py" in plain
    assert "colab/run_stage5_capability_ladder_trace_jobs.py" in plain
    assert "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_RUN_PROVIDER" in plain
    assert "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_RUN_SFT" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER" in plain
    assert "colab/run_stage5_capability_ladder_trace_responses.py" in plain
    assert "colab/run_stage5_capability_ladder_trace_collect.py" in plain
    assert "stage5_capability_ladder_trace_collection" in plain
    assert "colab/run_stage5_curriculum_sft.py" in plain
    assert "STAGE5_CURRICULUM_MIN_TARGET_LOOP_ROWS" in plain
    assert "min_target_loop_rows" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_ALLOW_GPU" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_REFUSE_GPU" in plain
    assert '"pull"' in plain
    assert '"--rebase"' in plain
    assert "runtime.unassign()" in plain
    assert "capability_ladder_7b_trace_chain" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_CELL.py" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_MIN_TARGET_LOOP_ROWS" in bootstrap


def test_capability_ladder_trace_collect_cell_is_cpu_only_and_response_driven() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL_VERSION" in plain
    assert "capability_ladder_trace_collect_cpu" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_ALLOW_GPU" in plain
    assert "colab/run_stage5_capability_ladder_trace_collect.py" in plain
    assert "tests/test_stage5_capability_ladder_trace_collect_runner.py" in plain
    assert "runtime.unassign()" in plain
    assert "capability_ladder_trace_collect_cpu" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL.py" in bootstrap


def test_capability_ladder_trace_responses_cell_is_cpu_only_and_provider_opt_in() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL_VERSION" in plain
    assert "capability_ladder_trace_responses_cpu" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_GPU" in plain
    assert "colab/run_stage5_capability_ladder_trace_responses.py" in plain
    assert "training/run_curriculum_job_responses.py" in bootstrap
    assert "tests/test_stage5_capability_ladder_trace_responses_runner.py" in plain
    assert "runtime.unassign()" in plain
    assert "capability_ladder_trace_responses_cpu" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL.py" in bootstrap


def test_capability_ladder_trace_response_collect_cell_runs_both_steps() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_CELL_VERSION" in plain
    assert "capability_ladder_trace_response_collect_cpu" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_ALLOW_GPU" in plain
    assert "colab/run_stage5_capability_ladder_trace_responses.py" in plain
    assert "colab/run_stage5_capability_ladder_trace_collect.py" in plain
    assert "tests/test_stage5_capability_ladder_trace_responses_runner.py" in plain
    assert "tests/test_stage5_capability_ladder_trace_collect_runner.py" in plain
    assert "runtime.unassign()" in plain
    assert "capability_ladder_trace_response_collect_cpu" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_CELL.py" in bootstrap


def test_stage5_chain_commit_messages_skip_ci() -> None:
    trace_responses = (ROOT / "colab/run_stage5_capability_ladder_trace_responses.py").read_text(encoding="utf-8")
    trace_collect = (ROOT / "colab/run_stage5_capability_ladder_trace_collect.py").read_text(encoding="utf-8")
    curriculum_sft = (ROOT / "colab/run_stage5_curriculum_sft.py").read_text(encoding="utf-8")
    benchmark = (ROOT / "colab/run_stage5_benchmark_suite.py").read_text(encoding="utf-8")

    assert "Record capability-ladder trace responses [skip ci]" in trace_responses
    assert "Record capability-ladder trace collection [skip ci]" in trace_collect
    assert "Record Stage 5 curriculum SFT {RUN_ID} [skip ci]" in curriculum_sft
    assert "Record Stage 5 benchmark suite {RUN_ID} [skip ci]" in benchmark


def test_capability_ladder_local_hf_trace_collect_target_is_bootstrapped() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")

    assert "hf_local" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_LOCAL_HF" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_MODEL_NAME" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_STUDENT_LINEAGE" in plain
    assert "capability_ladder_local_hf_trace_collect" in bootstrap
    assert "Qwen/Qwen2.5-7B-Instruct" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_STUDENT_LINEAGE" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT" in bootstrap
    assert "capability_ladder_local_hf_trace_collect" in bootstrap_md


def test_capability_ladder_local_hf_trace_sft_target_is_bootstrapped() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")

    assert "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_CELL_VERSION" in plain
    assert "capability_ladder_local_hf_trace_sft" in plain
    assert "Qwen/Qwen2.5-7B-Instruct" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_LOCAL_HF" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_STUDENT_LINEAGE" in plain
    assert "colab/run_stage5_capability_ladder_trace_responses.py" in plain
    assert "colab/run_stage5_capability_ladder_trace_collect.py" in plain
    assert "stage5_capability_ladder_trace_collection" in plain
    assert "colab/run_stage5_curriculum_sft.py" in plain
    assert "colab/run_stage5_benchmark_suite.py" in plain
    assert "colab/assess_stage5_traced_sft.py" in plain
    assert "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_RUN_ASSESSMENT" in plain
    assert "local_hf_trace_resume_preflight" in plain
    assert "STAGE5_CURRICULUM_USE_LEARNED_LOOP_CONTROL" in plain
    assert "STAGE5_CURRICULUM_LOOP_CONTROL_CE_WEIGHT" in plain
    assert "STAGE5_CURRICULUM_HALT_TARGET_NLL_WEIGHT" in plain
    assert "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_RUN_BENCHMARK" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_MIN_POSITIVE_ROWS" in plain
    assert "STAGE5_TRACED_CAPABILITY_SFT_MIN_TRACE_ROWS" in plain
    assert "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_MIN_VRAM_MB" in plain
    assert "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_SKIP_VRAM_CHECK" in plain
    assert "local_hf_trace_vram_preflight" in plain
    assert "stage5_local_hf_trace_sft_failure" in plain
    assert "failure_summary:" in plain
    assert "set_stage(\"local_hf_trace_responses\")" in plain
    assert "cyclic_label_aggregated" in plain
    assert "runtime.unassign()" in plain
    assert "capability_ladder_local_hf_trace_sft" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_CELL.py" in bootstrap
    assert "colab/assess_stage5_traced_sft.py" in bootstrap
    assert "capability_ladder_local_hf_trace_sft" in bootstrap_md


def test_capability_ladder_local_hf_trace_sft_scale64_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")

    assert "capability_ladder_local_hf_trace_sft_scale64" in bootstrap
    assert "outputs/stage5/stage5_capability_ladder_trace_jobs_20260623_150116/summary.json" in bootstrap
    assert '"STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_ID": (' in bootstrap
    assert "stage5_capability_ladder_trace_responses_20260623_191545" in bootstrap
    assert '"STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT": "64"' in bootstrap
    assert '"STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RESUME": "1"' in bootstrap
    assert '"STAGE5_TRACED_CAPABILITY_SFT_MIN_TRACE_ROWS": "48"' in bootstrap
    assert "stage5_local_hf_traced_capability_sft_20260623_191843/phase1/phase1_step_150.pt" in bootstrap
    assert '"STAGE5_TRACED_CAPABILITY_SFT_PHASE1_STEPS": "200"' in bootstrap
    assert '"STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_BENCHMARKS": "arc_easy,arc_challenge"' in bootstrap
    assert '"STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_ARC_EASY_LIMIT": "128"' in bootstrap
    assert '"STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_ARC_CHALLENGE_LIMIT": "128"' in bootstrap
    assert "colab/assess_stage5_traced_sft.py" in bootstrap
    assert "capability_ladder_local_hf_trace_sft_scale64" in bootstrap_md


def test_traced_sft_scale64_benchmark_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    plain = (ROOT / "colab/STAGE5_TRACED_SFT_BENCHMARK_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_TRACED_SFT_BENCHMARK_CELL_VERSION" in plain
    assert "traced_sft_benchmark_v1" in plain
    assert "stage5_local_hf_traced_capability_sft_20260623_194543" in plain
    assert "STAGE5_BENCHMARK_SOURCE_SUMMARY" in plain
    assert "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL" in plain
    assert "stage5_traced_sft_benchmark_failure" in plain
    assert "set_stage(\"benchmark_suite\")" in plain
    assert "failure_summary:" in plain
    assert "colab/run_stage5_benchmark_suite.py" in plain
    assert "colab/assess_stage5_traced_sft.py" in plain
    assert "traced_sft_scale64_benchmark" in bootstrap
    assert "STAGE5_TRACED_SFT_BENCHMARK_CELL.py" in bootstrap
    assert "STAGE5_ALLOW_STALE_SCALE64_BENCHMARK" in bootstrap
    assert '"traced_sft_scale64_benchmark is complete; rerouting to "' in bootstrap
    assert '"traced_sft_direct_preservation_probe. Set "' in bootstrap
    assert "traced_sft_scale64_benchmark" in bootstrap_md
    assert "STAGE5_ALLOW_STALE_SCALE64_BENCHMARK" in bootstrap_md
    assert '"traced_sft_scale64_benchmark is complete; rerouting to "' in bootstrap_md
    assert '"traced_sft_direct_preservation_probe. Set "' in bootstrap_md


def test_traced_sft_direct_preservation_probe_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    direct_cell = (ROOT / "colab/STAGE5_DIRECT_PRESERVATION_PROBE_CELL.py").read_text(encoding="utf-8")
    direct_runner = (ROOT / "colab/run_stage5_direct_preservation_probe.py").read_text(encoding="utf-8")

    assert "traced_sft_direct_preservation_probe" in bootstrap
    assert "traced_sft_direct_preservation_precheck" in bootstrap
    assert "traced_sft_direct_preservation_recover_only" in bootstrap
    assert "STAGE5_DIRECT_PRESERVE_PROMPT_STYLE" in bootstrap
    assert '"question_only"' in bootstrap
    assert "STAGE5_DIRECT_PRESERVE_SCORE_TARGET" in bootstrap
    assert '"option_text"' in bootstrap
    assert "STAGE5_DIRECT_PRESERVE_SWEEP" in bootstrap
    assert "baseline:lr=5e-7,steps=75,distill=1.0" in bootstrap
    assert "lr2e6_distill2:lr=2e-6,steps=100,distill=2.0" in bootstrap
    assert "STAGE5_DIRECT_PRESERVE_PRECHECK_ONLY" in bootstrap
    assert "stage5_traced_sft_direct_preservation_precheck_20260623_scale64" in bootstrap
    assert "stage5_local_hf_traced_sft_scale64_assessment_20260623_202446" in bootstrap
    assert "stage5_traced_sft_assessment_20260623_195134_reassessed" not in bootstrap
    assert "traced_sft_direct_preservation_probe" in bootstrap_md
    assert "traced_sft_direct_preservation_precheck" in bootstrap_md
    assert "traced_sft_direct_preservation_recover_only" in bootstrap_md
    assert "stage5_latest_direct_preservation_summary.txt" in direct_cell
    assert "stage5_current_source_summary.txt" in direct_cell
    assert "stage5_direct_preservation_probe_failure" in direct_cell
    assert "direct_route_attempt_failed" in direct_cell
    assert "attempt_failure_summary" in direct_cell
    assert "direct_route_precheck_needs_training" in direct_cell
    assert "STAGE5_DIRECT_PRESERVE_PRECHECK_ONLY" in direct_cell
    assert "STAGE5_DIRECT_PRESERVE_PROMPT_STYLE" in direct_cell
    assert "STAGE5_DIRECT_PRESERVE_SCORE_TARGET" in direct_cell
    assert "STAGE5_DIRECT_PRESERVE_RESUME_EXISTING" in direct_cell
    assert "STAGE5_DIRECT_PRESERVE_RESUME_ONLY" in direct_cell
    assert "direct_preservation_resume_existing" in direct_cell
    assert "direct_preservation_resume_missing" in direct_cell
    assert "Record Stage 5 direct preservation probe {run_dir.name}" in direct_cell
    assert "STAGE5_DIRECT_PRESERVE_CHAIN_CONFIRM" in direct_cell
    assert "STAGE5_DIRECT_PRESERVE_CHAIN_DEPTH_ROUTER" in direct_cell
    assert "parse_sweep_spec" in direct_cell
    assert "direct_preservation_attempts" in direct_cell
    assert "staged_selected_checkpoint" in direct_cell
    assert "maybe_chain_depth_router" in direct_cell
    assert "colab/STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_CELL.py" in direct_cell
    assert "direct_preservation_confirmation_assessment" in direct_cell
    assert "colab/assess_stage5_benchmark_suite.py" in direct_cell
    assert "Record Stage 5 direct preservation probe" in direct_cell
    assert "[skip ci]" in direct_cell
    assert "def redact" in direct_cell
    assert 'replace(HF_TOKEN or "", "****")' not in direct_cell
    assert "STAGE5_DIRECT_PRESERVE_CHAIN_CONFIRM" in bootstrap
    assert '"STAGE5_DIRECT_PRESERVE_CHAIN_CONFIRM": "1"' in bootstrap
    assert "STAGE5_DIRECT_PRESERVE_CHAIN_DEPTH_ROUTER" in bootstrap
    assert '"STAGE5_DIRECT_PRESERVE_CHAIN_DEPTH_ROUTER": "1"' in bootstrap
    assert "content_question_only,cyclic_label_aggregated" in bootstrap
    assert "direct_route_precheck_needs_training" in direct_runner
    assert "STAGE5_DIRECT_PRESERVE_PRECHECK_ONLY" in direct_runner
    assert '"precheck_only": PRECHECK_ONLY' in direct_runner


def test_traced_sft_direct_preservation_confirm_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    confirm_cell = (ROOT / "colab/STAGE5_DIRECT_PRESERVATION_CONFIRM_CELL.py").read_text(encoding="utf-8")

    assert "traced_sft_direct_preservation_confirm" in bootstrap
    assert "traced_sft_direct_preservation_confirm" in bootstrap_md
    assert "STAGE5_DIRECT_PRESERVATION_CONFIRM_CELL_VERSION" in confirm_cell
    assert "direct_preservation_confirm_v2" in confirm_cell
    assert "stage5_latest_direct_preservation_summary.txt" in confirm_cell
    assert "STAGE5_DIRECT_CONFIRM_SOURCE_SUMMARY" in confirm_cell
    assert "STAGE5_BENCHMARK_CHECKPOINT" in confirm_cell
    assert "content_question_only,cyclic_label_aggregated" in confirm_cell
    assert "STAGE5_BENCHMARK_PUSH" in confirm_cell
    assert "colab/run_stage5_benchmark_suite.py" in confirm_cell
    assert "colab/assess_stage5_benchmark_suite.py" in confirm_cell
    assert "STAGE5_BENCHMARK_ASSESS_SCORE_TARGET" in confirm_cell
    assert "content_question_only" in confirm_cell
    assert "colab/assess_stage5_benchmark_suite.py" in bootstrap
    assert '"STAGE5_CURRENT_A100_TARGET", "preflight"' in bootstrap


def test_traced_sft_surface_alignment_repair_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL.py").read_text(encoding="utf-8")

    assert "traced_sft_surface_alignment_repair" in bootstrap
    assert "traced_sft_surface_alignment_repair" in bootstrap_md
    assert "colab/STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL.py" in bootstrap
    assert "eval/analyze_mcq_order_sensitivity.py" in bootstrap
    assert "eval/analyze_mcq_order_sensitivity.py" in bootstrap_md
    assert "training/prepare_mcq_conditional_invariance_jsonl.py" in bootstrap
    assert "training/prepare_mcq_conditional_invariance_jsonl.py" in bootstrap_md
    assert "STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL_VERSION" in cell
    assert "surface_alignment_repair_v1" in cell
    assert "STAGE5_SURFACE_ALIGN_SOURCE_SUMMARY" in cell
    assert "stage5_traced_sft_direct_preservation_20260623_scale64_confirm" in cell
    assert "eval/analyze_mcq_order_sensitivity.py" in cell
    assert "eval/analyze_mcq_surface_mismatch.py" in cell
    assert "training/prepare_mcq_conditional_invariance_jsonl.py" in cell
    assert "training/prepare_mcq_surface_alignment_jsonl.py" in cell
    assert "training/prepare_mcq_score_alignment_jsonl.py" in cell
    assert "training/train_phase1_mcq_score_align.py" in cell
    assert "STAGE5_SURFACE_ALIGN_TRAINER=score_ce" in cell
    assert "colab/assess_stage5_surface_repair.py" in cell
    assert "colab/run_stage5_surface_alignment_repair.py" in cell
    assert "tests/test_prepare_mcq_surface_alignment_jsonl.py" in cell
    assert "tests/test_prepare_mcq_conditional_invariance_jsonl.py" in cell
    assert "tests/test_analyze_mcq_order_sensitivity.py" in cell
    assert "tests/test_stage5_surface_alignment_repair.py" in cell
    assert "tests/test_stage5_surface_repair_assessment.py" in cell
    assert "runtime.unassign" in cell


def test_traced_sft_score_alignment_repair_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")

    assert "traced_sft_score_alignment_repair" in bootstrap
    assert "traced_sft_score_alignment_repair" in bootstrap_md
    assert '"STAGE5_SURFACE_ALIGN_TRAINER": "score_ce"' in bootstrap
    assert "training/prepare_mcq_score_alignment_jsonl.py" in bootstrap
    assert "training/train_phase1_mcq_score_align.py" in bootstrap
    assert "stage5_score_alignment_repair_content_route_20260624" in bootstrap
    assert "STAGE5_SURFACE_ALIGN_SCORE_DISTILL_WEIGHT" in bootstrap


def test_dense_mcq_trace_sft_control_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_DENSE_MCQ_TRACE_SFT_CONTROL_CELL.py").read_text(encoding="utf-8")

    assert "dense_mcq_trace_sft_control" in bootstrap
    assert "dense_mcq_trace_sft_control" in bootstrap_md
    assert "colab/STAGE5_DENSE_MCQ_TRACE_SFT_CONTROL_CELL.py" in bootstrap
    assert "STAGE5_DENSE_MCQ_TRACE_SFT_CONTROL_CELL_VERSION" in cell
    assert "dense_mcq_trace_sft_control_v1" in cell
    assert "CURRENT_SOURCE_POINTER" in cell
    assert "dense_mcq_source_pointer" in cell
    assert "stage5_current_source_summary.txt" in bootstrap
    assert "stage5_dense_mcq_trace_sft_control_current" in bootstrap
    assert "validate_dense_control_source_gate" in (ROOT / "colab/run_stage5_mcq_dense_sft_control.py").read_text(
        encoding="utf-8"
    )
    assert "STAGE5_DENSE_MCQ_ALLOW_UNPASSED_BENCHMARK" in (
        ROOT / "colab/run_stage5_mcq_dense_sft_control.py"
    ).read_text(encoding="utf-8")
    assert "dense_control_source_gate=passed_benchmark_assessment" in (
        ROOT / "colab/run_stage5_mcq_dense_sft_control.py"
    ).read_text(encoding="utf-8")
    assert "training/train_dense_lora.py" in cell
    assert "eval/eval_mcq.py --mode base --checkpoint" in cell
    assert "colab/run_stage5_mcq_dense_sft_control.py" in cell
    assert "colab/assess_stage5_mcq_recipe_control.py" in cell
    assert "tests/test_eval_mcq_dense_lora.py" in cell
    assert "tests/test_stage5_mcq_dense_sft_control.py" in cell
    assert "tests/test_stage5_mcq_recipe_control_assessment.py" in cell
    assert "STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY" in bootstrap
    assert "runtime.unassign" in cell


def test_traced_sft_competence_preserving_pipeline_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_COMPETENCE_PRESERVING_PIPELINE_CELL.py").read_text(encoding="utf-8")

    assert "traced_sft_competence_preserving_pipeline" in bootstrap
    assert "traced_sft_competence_preserving_pipeline" in bootstrap_md
    assert "colab/STAGE5_COMPETENCE_PRESERVING_PIPELINE_CELL.py" in bootstrap
    assert "STAGE5_COMPETENCE_PRESERVING_PIPELINE_CELL_VERSION" in cell
    assert "competence_preserving_pipeline_v2" in cell
    assert "print_pipeline_artifacts" in cell
    assert "STAGE5_COMPETENCE_SOURCE_SUMMARY" in cell
    assert "STAGE5_COMPETENCE_MOUNT_DRIVE_FIRST" in cell
    assert "FORCE_DRIVE_REMOUNT" in cell
    assert 'drive.mount("/content/drive", force_remount=FORCE_DRIVE_REMOUNT)' in cell
    assert "stage5_debiased_benchmark_assessment_20260625_121302" in cell
    assert "stage5_debiased_benchmark_assessment_20260625_121302" in bootstrap
    assert "colab/run_stage5_competence_preserving_pipeline.py" in cell
    assert "tests/test_stage5_competence_preserving_pipeline.py" in cell
    assert "tests/test_stage5_balanced_arc_mix_gate.py" in cell
    assert "runtime.unassign" in cell


def test_current_stage5_fresh_launcher_cell_is_self_contained_for_blank_colab() -> None:
    cell = (ROOT / "colab/CURRENT_STAGE5_FRESH_LAUNCHER_CELL.py").read_text(encoding="utf-8")

    assert "CURRENT_STAGE5_FRESH_LAUNCHER_VERSION" in cell
    assert "fresh_launcher_v2" in cell
    assert "GH_TOKEN" in cell
    assert "GITHUB_TOKEN" in cell
    assert "HF_TOKEN" in cell
    assert "HUGGINGFACE_HUB_TOKEN" in cell
    assert "git\", \"-C\", str(ROOT), \"reset\", \"--hard\", \"origin/main" in cell
    assert 'drive.mount("/content/drive"' in cell
    assert "STAGE5_BOOTSTRAP_PREFER_LOCAL_HEAD" in cell
    assert "STAGE5_COMPETENCE_MOUNT_DRIVE_FIRST" in cell
    assert 'os.environ.setdefault("STAGE5_COMPETENCE_MOUNT_DRIVE_FIRST", "1")' in cell
    assert "competence_preserving_pipeline_v2" in cell
    assert "print_pipeline_artifacts" in cell
    assert "traced_sft_competence_preserving_pipeline" in cell
    assert "stage5_debiased_benchmark_assessment_20260625_121302" in cell
    assert "stage5_competence_recovery_from_reentry_benchmark" in cell
    assert "exec(compile(bootstrap" in cell


def test_traced_capability_ladder_sft_cell_derives_training_env_from_collection() -> None:
    plain = (ROOT / "colab/STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL_VERSION" in plain
    assert "traced_capability_ladder_sft" in plain
    assert "stage5_capability_ladder_trace_collection" in plain
    assert "trace_curriculum_gate_ready" in plain
    assert "STAGE5_TRACED_CAPABILITY_SFT_SOURCE_SUMMARY" in plain
    assert "STAGE5_TRACED_CAPABILITY_SFT_PHASE1_STEPS" in plain
    assert "STAGE5_TRACED_CAPABILITY_SFT_MODEL_NAME" in plain
    assert "STAGE5_CURRICULUM_WORK_DIR" in plain
    assert "STAGE5_CURRICULUM_SUMMARY_JSON" in plain
    assert "STAGE5_CURRICULUM_MIN_POSITIVE_ROWS" in plain
    assert "STAGE5_CURRICULUM_MIN_MODE_ROWS" in plain
    assert "STAGE5_CURRICULUM_INPUT_BACKUP_DIR" in plain
    assert "colab/run_stage5_curriculum_sft.py" in plain
    assert "tests/test_stage5_curriculum_sft.py" in plain
    assert "tests/test_curriculum_sft_gate.py" in plain
    assert "runtime.unassign()" in plain
    assert "traced_capability_ladder_sft" in bootstrap
    assert "STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL.py" in bootstrap


def test_drive_checkpoint_preflight_cell_is_cpu_only_and_single_purpose() -> None:
    text = (ROOT / "colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.md").read_text(encoding="utf-8")
    code = fenced_python_block("colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.md")
    plain = (ROOT / "colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.py").read_text(encoding="utf-8")

    assert "Run this in a CPU or cheap GPU Colab runtime before attaching an A100/H100." in text
    assert "GH_TOKEN" in code
    assert "GH_TOKEN" in plain
    assert "GITHUB_TOKEN" in code
    assert "HF_TOKEN" in code
    assert "HUGGINGFACE_HUB_TOKEN" in code
    assert "STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY" in text
    assert "STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY" in code
    assert "STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY" in plain
    assert "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY" in code
    assert "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY" in plain
    assert "SOURCE_SUMMARY_OVERRIDE" in code
    assert "SOURCE_SUMMARY_OVERRIDE" in plain
    assert 'pointer = ROOT / "config" / "stage5_current_source_summary.txt"' in code
    assert 'pointer = ROOT / "config" / "stage5_current_source_summary.txt"' in plain
    assert "Using current source summary pointer" in code
    assert "Using current source summary pointer" in plain
    assert "drive.mount(\"/content/drive\", force_remount=True)" in code
    assert "stage5_routing_diagnostic_20260622_041706/summary.json" in code
    assert 'GO_NO_GO_RUN_ID = "stage5_drive_checkpoint_preflight"' in code
    assert '"colab/check_stage5_a100_go_no_go.py"' in code
    assert '"colab/run_stage5_next_action.py"' in code
    assert '"STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE"] = "0"' in code
    assert "next_action_guard" in code
    assert '"--source-summary"' in code
    assert "checkpoint_preflight" in code
    assert "PREFLIGHT_GREEN" in code
    assert "both guarded dry-runs are allowed" in code
    assert "PREFLIGHT_RED" in code
    assert "PREFLIGHT_BLOCKED" in code
    assert "summary[\"decision\"]" in code
    assert "runtime.unassign()" in code
    assert '["git", "pull", "--ff-only", "origin", "main"]' in code
    assert "pip install" not in code
    assert "colab/run_stage5_routing_repair.py" not in code
    assert "colab/run_stage5_full_assessment_once.py" not in code
    assert "training/train_phase" not in code


def test_drive_checkpoint_preflight_plain_cell_matches_markdown_code() -> None:
    markdown_cell = fenced_python_block("colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.md")
    plain_cell = (ROOT / "colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.py").read_text(encoding="utf-8")

    assert plain_cell == markdown_cell


def test_curriculum_artifact_pipeline_cell_defaults_to_no_provider_spend() -> None:
    text = (ROOT / "colab/CURRICULUM_ARTIFACT_PIPELINE_CELL.md").read_text(encoding="utf-8")
    plain = (ROOT / "colab/CURRICULUM_ARTIFACT_PIPELINE_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert "claim_curriculum_scaleup_cpu" in bootstrap
    assert "data/curriculum/claim_direct_deep_001" in text
    assert "data/curriculum/claim_direct_deep_001" in plain
    assert "STAGE5_CURRICULUM_RUN_PROVIDER_RESPONSES" in text
    assert "STAGE5_CURRICULUM_RUN_PROVIDER_RESPONSES" in plain
    assert "STAGE5_CURRICULUM_PROVIDER_LIMIT" in text
    assert "STAGE5_CURRICULUM_PROVIDER_LIMIT" in plain
    assert "STAGE5_CURRICULUM_MODEL_MAP_JSON" in text
    assert "STAGE5_CURRICULUM_MODEL_MAP_JSON" in plain
    assert "STAGE5_CURRICULUM_OPUS_MODEL" in text
    assert "STAGE5_CURRICULUM_OPUS_MODEL" in plain
    assert "STAGE5_CURRICULUM_GLM_MODEL" in plain
    assert "STAGE5_CURRICULUM_WEAK_REFERENCE_MODEL" in plain
    assert "resolve_model_map" in plain
    assert "DEFAULT_MODEL_MAP" in plain
    assert "PROVIDER_LIMIT_RAW" in plain
    assert "MIN_POSITIVE_ROWS = 2000" in text
    assert "MIN_POSITIVE_ROWS = 2000" in plain
    assert 'MIN_MODE_ROWS = "direct=1000,deep_narrow=1000"' in text
    assert 'MIN_MODE_ROWS = "direct=1000,deep_narrow=1000"' in plain
    assert '"math,science"' in plain
    assert '"1,2,5,9"' in plain
    assert '"122"' in plain
    assert "--min_mode_rows" in plain
    assert "str(MIN_POSITIVE_ROWS)" in plain
    assert '"--min_natural_agree"' in plain
    assert '"--min_distinct_agree"' in plain
    assert "training/run_curriculum_pipeline_from_artifacts.py" in plain
    assert "training/run_curriculum_job_responses.py" in plain
    assert '"--command",\n            PROVIDER_COMMAND,\n            "--model_map_json",' in plain
    assert "pending_reference_attempt_responses" in plain
    assert "pending_responses" in plain
    assert "usable_job_ids" in plain
    assert "response pair pending:" in plain
    assert "curriculum_readiness.json" in text
    assert "curriculum_readiness.json" in plain
    assert "write_readiness_report" in plain
    assert "next_safe_action" in plain
    assert "model_map_configured" in plain
    assert "MIN_TARGET_LOOP_ROWS" in plain
    assert "STAGE5_CURRICULUM_MIN_TARGET_LOOP_ROWS" in plain
    assert "1=1000,3=500,4=500" in plain
    assert "target_loop_requirements" in plain
    assert "phase_order_warning" in plain
    assert "This CPU/API curriculum work prepares Phase 1/Stage 4 data in parallel" in plain
    assert "response_lines < job_lines" in plain
    assert "runtime.unassign()" in plain
    assert "replace-with-opus-compatible-model-id" in plain
    assert "Fill MODEL_MAP with concrete provider model ids" in plain
    assert "REFUSE_GPU_RUNTIME = True" in plain
    assert "ALLOW_GPU_RUNTIME_FOR_CPU_API_CELL = False" in plain
    assert "attached_gpu_names" in plain
    assert "refuse_gpu_runtime_for_cpu_api_work()" in plain
    assert "Refusing to run CPU/API curriculum pipeline on an attached GPU runtime" in plain


def test_curriculum_artifact_pipeline_plain_cell_matches_markdown_code() -> None:
    markdown_cell = fenced_python_block("colab/CURRICULUM_ARTIFACT_PIPELINE_CELL.md")
    plain_cell = (ROOT / "colab/CURRICULUM_ARTIFACT_PIPELINE_CELL.py").read_text(encoding="utf-8")

    assert plain_cell == markdown_cell


def test_curriculum_sft_cell_defaults_to_direct_deep_objective() -> None:
    text = (ROOT / "colab/STAGE5_CURRICULUM_SFT_CELL.md").read_text(encoding="utf-8")
    plain = (ROOT / "colab/STAGE5_CURRICULUM_SFT_CELL.py").read_text(encoding="utf-8")
    runner = (ROOT / "colab/run_stage5_curriculum_sft.py").read_text(encoding="utf-8")

    assert 'WORK_DIR = "data/curriculum/programmatic_direct_deep_001"' in text
    assert 'WORK_DIR = "data/curriculum/programmatic_direct_deep_001"' in plain
    assert 'MIN_POSITIVE_ROWS = "2000"' in text
    assert 'MIN_POSITIVE_ROWS = "2000"' in plain
    assert 'MIN_MODE_ROWS = "direct=1000,deep_narrow=1000"' in text
    assert 'MIN_MODE_ROWS = "direct=1000,deep_narrow=1000"' in plain
    assert "stale tiny shard" in text
    assert "STAGE5_CURRICULUM_MIN_MODE_ROWS" in plain
    assert "STAGE5_CURRICULUM_LAYER_SPLIT" in runner
    assert "STAGE5_RECURRENT_LAYER_SPLIT" in runner
    assert '"layer_split": LAYER_SPLIT' in runner
    assert "LAYER_SPLIT," in runner


def test_programmatic_curriculum_cell_is_cpu_safe_and_matches_markdown_code() -> None:
    text = (ROOT / "colab/STAGE5_PROGRAMMATIC_CURRICULUM_CELL.md").read_text(encoding="utf-8")
    plain = (ROOT / "colab/STAGE5_PROGRAMMATIC_CURRICULUM_CELL.py").read_text(encoding="utf-8")
    markdown_cell = fenced_python_block("colab/STAGE5_PROGRAMMATIC_CURRICULUM_CELL.md")

    assert plain == markdown_cell
    assert 'WORK_DIR = "data/curriculum/programmatic_direct_deep_001"' in plain
    assert 'MIN_MODE_ROWS = f"direct={NUM_DIRECT},deep_narrow={NUM_DEEP_NARROW}"' in plain
    assert "training/run_programmatic_curriculum_pipeline.py" in plain
    assert "training/check_curriculum_sft_gate.py" in plain
    assert "colab/publish_stage5_curriculum_gate.py" in plain
    assert "REQUIRE_DRIVE_BACKUP_FOR_PUBLISH = True" in plain
    assert "PUBLISH_GATE_TO_GITHUB = True" in plain
    assert "stage5_current_source_summary" in text
    assert "stage5_current_source_summary" in plain
    assert "REFUSE_GPU_RUNTIME = True" in plain
    assert "ALLOW_GPU_RUNTIME_FOR_CPU_WORK = False" in plain
    assert "PROGRAMMATIC_CURRICULUM_CELL_VERSION" in plain
    assert 'shutil.which("nvidia-smi")' in plain
    assert "FileNotFoundError" in plain
    assert "OSError" in plain
    assert "Refusing to run CPU-only programmatic curriculum generation" in plain
    assert "zero-provider" in text
    assert "STAGE5_CURRENT_A100_TARGET=reentry_repair_smoke" in plain
    assert "STAGE5_CURRENT_A100_TARGET=safe_continue_execute" not in plain


def test_reentry_repair_smoke_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_REENTRY_REPAIR_SMOKE_CELL.py").read_text(encoding="utf-8")

    assert "reentry_repair_smoke" in bootstrap
    assert "reentry_repair_smoke" in bootstrap_md
    assert "colab/STAGE5_REENTRY_REPAIR_SMOKE_CELL.py" in bootstrap
    assert "train_log_metrics" in bootstrap
    assert "train_log_metrics" in bootstrap_md
    assert "existing_train_log_metrics" in bootstrap
    assert "existing_train_log_metrics" in bootstrap_md
    assert "resume_retrain=train_phase1_ponder" in bootstrap
    assert "resume_retrain=train_phase1_ponder" in bootstrap_md
    assert "require_gpu_runtime" in bootstrap
    assert "require_gpu_runtime" in bootstrap_md
    assert "STAGE5_REENTRY_REPAIR_SMOKE_CELL_VERSION" in cell
    assert "stage5_reentry_repair_smoke_v2_spectral_optional" in cell
    assert "reentry_spectral_repair_smoke" in bootstrap
    assert "reentry_spectral_repair_smoke" in bootstrap_md
    assert "STAGE5_REENTRY_REPAIR_ADAPTER_MODE" in bootstrap
    assert "STAGE5_REENTRY_REPAIR_ADAPTER_MODE" in bootstrap_md
    assert "STAGE5_REENTRY_REPAIR_ADAPTER_MODE" in cell
    assert "reentry_adapter_mode" in cell
    assert "STAGE5_RECURRENT_LAYER_SPLIT" in cell
    assert "STAGE5_REENTRY_REPAIR_LAYER_SPLIT" in cell
    assert '"layer_split": LAYER_SPLIT' in cell
    assert '"--model_name"' in cell
    assert '"--split"' in cell
    assert "attached_gpu_names" in cell
    assert "require_gpu_runtime" in cell
    assert "Stage 3 re-entry repair smoke requires an attached GPU runtime" in cell
    assert "bridge_gate_override" in cell
    assert "bridge_reset_identity" in cell
    assert "reentry_rescale_mode" in cell
    assert "use_reentry_adapter" in cell
    assert "bridge,reentry,halt" in cell
    assert "training/train_phase1_ponder.py" in cell
    assert "eval/eval_reentry_drift.py" in cell
    assert "Loop-1 Preservation" in cell
    assert "loop1_preservation" in cell
    assert "loop1_preservation_tasks" in cell
    assert "parse_train_log_metrics" in cell
    assert "train_log_metrics" in cell
    assert "Training Smoke Metrics" in cell
    assert "target_loop_abs_error" in cell
    assert "halting_target_nll" in cell
    assert "bridge_gate_active" in (ROOT / "colab/assess_stage5_reentry.py").read_text(encoding="utf-8")
    assert "STAGE5_REENTRY_REPAIR_REQUIRE_NORM_PASS" in cell
    assert "STAGE5_REENTRY_REPAIR_ALLOW_FALLBACK_CHECKPOINT" in cell
    assert "STAGE5_REENTRY_REPAIR_ALLOW_FALLBACK_CHECKPOINT" in bootstrap
    assert "STAGE5_REENTRY_REPAIR_ALLOW_FALLBACK_CHECKPOINT" in bootstrap_md
    assert "Stage 3 repair smoke requires a checkpoint from the passed Stage 2 norm assessment" in cell
    assert "Stage 3 repair smoke requires a checkpoint from the passed Stage 2 norm assessment" in bootstrap
    assert "Stage 3 repair smoke requires a checkpoint from the passed Stage 2 norm assessment" in bootstrap_md
    assert "stage2_norm_assessment" in cell
    assert "current_pointer_norm_assessment_candidates" in bootstrap
    assert "current_pointer_norm_assessment_candidates" in bootstrap_md
    assert "current_pointer_norm_assessment_candidates" in cell
    assert "stage2_norm_assessment_source=current_pointer" in bootstrap
    assert "stage2_norm_assessment_source=current_pointer" in bootstrap_md
    assert "stage2_norm_assessment_source=current_pointer" in cell
    assert 'payload.get("kind") == "stage5_reentry_norm_eval_only"' in cell
    assert '"checkpoint": checkpoint' in cell
    assert "checkpoint_override or norm_checkpoint or DEFAULT_CHECKPOINT" not in cell
    assert "DEFAULT_CHECKPOINT" not in cell
    assert "FALLBACK_CHECKPOINTS" not in cell
    assert "checkpoint_from_norm = bool(norm_checkpoint and not checkpoint_override)" in cell
    assert '"checkpoint_source": checkpoint_source' in cell
    assert '"checkpoint_from_stage2_norm": checkpoint_from_norm' in cell
    assert '"stage2_norm_checkpoint": norm_checkpoint' in cell
    assert '"checkpoint_override_used": bool(checkpoint_override)' in cell
    assert "Checkpoint source:" in cell
    assert "Checkpoint from Stage 2 norm assessment:" in cell
    assert 'allow_fallback=checkpoint_source == "explicit_fallback"' in cell
    assert "reentry_repair_checkpoint_source=" in cell
    assert 'candidate.with_name("reentry_assessment.json")' in cell
    assert "incremental_backup" in cell
    assert "restore_incremental_backup" in cell
    assert "STAGE5_REENTRY_REPAIR_RESTORE_INCREMENTAL_BACKUP" in cell
    assert "restored_repair_incremental_backup" in cell
    assert "LEGACY_DRIVE_ROOT" in cell
    assert 'LEGACY_DRIVE_ROOT / "outputs" / "stage5"' in cell
    assert "existing_train_log_metrics" in cell
    assert "resume_retrain=train_phase1_ponder" in cell
    assert "resume_skip=train_phase1_ponder" in cell
    assert "resume_cache_mismatch=reentry_repair_smoke_config_changed" in cell
    assert "cache_compatible" in cell
    assert "force=not cache_compatible" in cell
    assert "colab/assess_stage5_reentry.py" in cell
    assert "publishable_artifact_paths" in cell
    assert "update_current_source_summary" in cell
    assert 'git", "add", "-f", str(out_dir.relative_to(ROOT))' not in cell
    assert "STAGE5_REENTRY_REPAIR_PUSH" in cell
    assert 'run(["git", "push", "origin", "main"], check=False)' in cell
    assert 'run(["git", "pull", "--rebase", "--autostash", "origin", "main"])' in cell
    assert 'run(["git", "push", "origin", "main"])' in cell
    assert "Readout Pause" in cell
    assert "runtime.unassign" in cell


def test_reentry_norm_recover_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_REENTRY_NORM_RECOVER_CELL.py").read_text(encoding="utf-8")

    assert "reentry_norm_recover_only" in bootstrap
    assert "reentry_norm_recover_only" in bootstrap_md
    assert "colab/STAGE5_REENTRY_NORM_RECOVER_CELL.py" in bootstrap
    assert "STAGE5_REENTRY_NORM_RECOVER_CELL_VERSION" in cell
    assert "stage5_reentry_norm_recover_v1" in cell
    assert "STAGE5_REENTRY_NORM_RECOVER_SOURCE" in cell
    assert "stage5_reentry_norm_*" in cell
    assert "No complete stage5_reentry_norm_* artifact found on Drive" in cell
    assert "Visible incomplete stage5_reentry_norm_* candidates" in cell
    assert "stage2_missing_reasons" in cell
    assert "ensure_summary" in cell
    assert "rebuilt_summary" in cell
    assert "reentry_norm_recover_utils" in cell
    assert "colab/assess_stage5_reentry.py" in cell
    assert "colab/review_stage5_reentry.py" in cell
    assert "Recover Stage 5 re-entry norm" in cell
    assert "publishable_artifact_paths" in cell
    assert "update_current_source_summary" in cell
    assert 'git", "add", "-f", str(out_dir.relative_to(ROOT))' not in cell
    assert "runtime.unassign" in cell


def test_reentry_recovery_training_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py").read_text(encoding="utf-8")

    assert "reentry_recovery_training" in bootstrap
    assert "reentry_recovery_training" in bootstrap_md
    assert "colab/STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py" in bootstrap
    assert "STAGE5_REENTRY_RECOVERY_CELL_VERSION" in cell
    assert "reentry_recovery_training_v5_fixed_tail_damper" in cell
    assert "STAGE5_REENTRY_RECOVERY_REPAIR_ASSESSMENT" in cell
    assert 'candidate.with_name("reentry_assessment.json")' in cell
    assert "current_pointer_repair_assessment_candidates" in bootstrap
    assert "current_pointer_repair_assessment_candidates" in bootstrap_md
    assert "current_pointer_repair_assessment_candidates" in cell
    assert "stage3_repair_assessment_source=current_pointer" in bootstrap
    assert "stage3_repair_assessment_source=current_pointer" in bootstrap_md
    assert "stage3_repair_assessment_source=current_pointer" in cell
    assert 'payload.get("kind") == "stage5_reentry_repair_smoke"' in cell
    assert "run_bounded_recovery_training_with_reentry_repair" in cell
    assert "repair_assessment_recovery_block_reason" in cell
    assert '"metrics": assessment.get("metrics", {})' in cell
    assert "STAGE5_CURRICULUM_RESUME_FROM" in cell
    assert "STAGE5_CURRICULUM_LAYER_SPLIT" in cell
    assert "STAGE5_REENTRY_RECOVERY_LAYER_SPLIT" in cell
    assert "STAGE5_CURRICULUM_USE_LEARNED_LOOP_CONTROL" in cell
    assert "STAGE5_CURRICULUM_OPTIMIZER_MODULES" in cell
    assert "STAGE5_CURRICULUM_LORA_RANK" in cell
    assert "STAGE5_REENTRY_RECOVERY_LORA_RANK" in cell
    assert "lora_rank" in cell
    assert "STAGE5_CURRICULUM_REENTRY_RESCALE_MODE" in cell
    assert "STAGE5_REENTRY_RECOVERY_REENTRY_RESCALE_MODE" in cell
    assert '"entry_rms"' in cell
    assert "STAGE5_CURRICULUM_USE_REENTRY_ADAPTER" in cell
    assert "DRIVE_ARTIFACT_ROOT / \"outputs\" / \"stage5\"" in cell
    assert "LEGACY_DRIVE_ROOT / \"outputs\" / \"stage5\"" in cell
    assert "colab.review_stage5_recovery_curriculum" in cell
    assert "_resolve_trace_collection_summary(" in cell
    assert "colab.reentry_recovery_config" in cell
    assert "attached_gpu_names" in cell
    assert "require_gpu_runtime" in cell
    assert "stage4_gpu_runtime=" in cell
    assert "Attach an L4/T4/A100/H100 GPU runtime before running Stage 4 recovery training." in cell
    assert '"STAGE5_CURRICULUM_SFT_COMMIT_CHECKPOINTS": os.environ.get(' in cell
    assert '"STAGE5_REENTRY_RECOVERY_COMMIT_CHECKPOINTS",' in cell
    assert "sys.path.insert(0, root_str)" in cell
    assert 'parts.append(f"{loop}=1")' not in cell
    assert "colab/run_stage5_curriculum_sft.py" in cell
    assert "write_reentry_recovery_wrapper_summary" in cell
    assert "run_post_reentry_health_probe" in cell
    assert "post_reentry_health_checks" in cell
    assert "eval/eval_reentry_drift.py" in cell
    assert '"kind": "stage5_reentry_recovery_training"' in cell
    assert "publish_reentry_recovery_wrapper" in cell
    assert "debiased_benchmark_suite" in cell
    assert "tests/test_stage5_curriculum_sft.py" in cell
    assert "runtime.unassign" in cell


def test_reentry_tail_damper_recovery_training_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py").read_text(encoding="utf-8")

    assert "reentry_tail_damper_recovery_training" in bootstrap
    assert "reentry_tail_damper_recovery_training" in bootstrap_md
    assert "reentry_tail_damper_capacity_lora32_training" in bootstrap
    assert "reentry_tail_damper_capacity_lora32_training" in bootstrap_md
    assert "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_SOURCE_SUMMARY" in bootstrap
    assert "stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857" in bootstrap
    assert "STAGE5_REENTRY_RECOVERY_REENTRY_TAIL_DAMPER_STRENGTH" in bootstrap
    assert '"STAGE5_REENTRY_RECOVERY_REENTRY_TAIL_DAMPER_STRENGTH": "1.0"' in bootstrap
    assert '"STAGE5_REENTRY_RECOVERY_LORA_RANK": "32"' in bootstrap
    assert '"STAGE5_REENTRY_RECOVERY_LORA_ALPHA": "64"' in bootstrap
    assert "STAGE5_CURRICULUM_REENTRY_TAIL_DAMPER_PATH" in cell
    assert "STAGE5_CURRICULUM_REENTRY_TAIL_DAMPER_STRENGTH" in cell
    assert "fixed_tail_damper_depth_readout" in cell
    assert "rescued_vs_loop1" in bootstrap
    assert "harmed_vs_loop1" in bootstrap
    assert "eval/eval_tail_damper_depth_sweep.py" in cell


def test_reentry_tail_damper_recovery_readout_only_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py").read_text(encoding="utf-8")

    assert "reentry_tail_damper_recovery_readout_only" in bootstrap
    assert "reentry_tail_damper_recovery_readout_only" in bootstrap_md
    assert "STAGE5_REENTRY_RECOVERY_READOUT_ONLY" in bootstrap
    assert "stage5_reentry_recovery_20260627_131940_curriculum_sft/summary.json" in bootstrap
    assert "stage5_reentry_recovery_20260627_131940_readout" in bootstrap
    assert "readout_only=true" in cell
    assert "readout_only_child_summary_path" in cell
    assert "skipping curriculum SFT" in cell


def test_current_bootstrap_exposes_unfreeze_recurrent_curriculum_target() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_UNFREEZE_RECURRENT_CURRICULUM_CELL.py").read_text(encoding="utf-8")
    trainer = (ROOT / "training/train_unfrozen_recurrent.py").read_text(encoding="utf-8")

    for text in (bootstrap, bootstrap_md):
        assert "unfreeze_recurrent_curriculum" in text
        assert "STAGE5_UNFREEZE_SOURCE_SUMMARY" in text
        assert "STAGE5_UNFREEZE_MAX_STEPS" in text
        assert "STAGE5_BENCHMARK_LORA_RANK" in text
        assert "require_lora_loaded_before_merge" in text
        assert "stage5_reentry_recovery_20260627_190155/summary.json" in text
    assert "STAGE5_UNFREEZE_RECURRENT_CURRICULUM_CELL_VERSION" in cell
    assert "unfreeze_recurrent_curriculum_v1" in cell
    assert "training/train_unfrozen_recurrent.py" in cell
    assert "merge_lora_before_unfreeze" in cell
    assert "require_lora_loaded_before_merge" in cell
    assert '"rank": resume_lora_rank' in cell
    assert '"auto"' in cell
    assert "STAGE5_BENCHMARK_LORA_RANK=0" in cell
    assert "lora_rank" in cell
    assert "eval/eval_reentry_drift.py" in cell
    assert "publishable_artifact_paths" in cell
    assert "unfrozen_recurrent_step_" in cell
    assert "merge_lora_adapters" in trainer
    assert "reentry_tail_damper_path=None" in trainer
    assert "use_reentry_adapter=False" in trainer


def test_current_bootstrap_exposes_prelude_path_development_target() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_UNFREEZE_RECURRENT_CURRICULUM_CELL.py").read_text(encoding="utf-8")
    trainer = (ROOT / "training/train_unfrozen_recurrent.py").read_text(encoding="utf-8")
    ablation = (ROOT / "eval/eval_prelude_ablation.py").read_text(encoding="utf-8")

    for text in (bootstrap, bootstrap_md):
        assert "prelude_path_development" in text
        assert "STAGE5_UNFREEZE_BRIDGE_PRELUDE_GRAD_MULTIPLIER" in text
        assert "STAGE5_UNFREEZE_RUN_PRELUDE_ABLATION" in text
        assert "stage5_prelude_path_development" in text
        assert '"STAGE5_UNFREEZE_MAX_STEPS": "300"' in text
        assert '"STAGE5_UNFREEZE_SAVE_EVERY": "50"' in text
    assert "eval/eval_prelude_ablation.py" in cell
    assert "prelude_ablation_summary" in cell
    assert "bridge_prelude_grad_multiplier" in trainer
    assert "apply_bridge_prelude_grad_multiplier" in trainer
    assert "prelude_variant" in ablation
    assert "shuffled" in ablation


def test_debiased_benchmark_suite_threads_fixed_tail_damper() -> None:
    cell = (ROOT / "colab/STAGE5_DEBIASED_BENCHMARK_SUITE_CELL.py").read_text(encoding="utf-8")
    runner = (ROOT / "colab/run_stage5_benchmark_suite.py").read_text(encoding="utf-8")

    assert "fixed_tail_damper_env" in cell
    assert "STAGE5_DEBIASED_USE_FIXED_TAIL_DAMPER" in cell
    assert "benchmark_fixed_tail_damper=" in cell
    assert "STAGE5_BENCHMARK_REENTRY_TAIL_DAMPER_PATH" in cell
    assert "STAGE5_BENCHMARK_REENTRY_TAIL_DAMPER_STRENGTH" in cell
    assert "restore_artifact_from_drive" in runner
    assert "restored_benchmark_artifact=" in runner
    assert "resolved_reentry_tail_damper=" in runner


def test_reentry_drift_and_norm_targets_run_self_assessment() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    drift_cell = (ROOT / "colab/STAGE5_REENTRY_DRIFT_CELL.py").read_text(encoding="utf-8")
    norm_cell = (ROOT / "colab/STAGE5_REENTRY_NORM_CELL.py").read_text(encoding="utf-8")

    assert "reentry_drift_diagnostic" in bootstrap
    assert "reentry_norm_diagnostic" in bootstrap
    assert "colab/assess_stage5_reentry.py" in bootstrap
    assert "colab/assess_stage5_reentry.py" in bootstrap_md
    assert "run_reentry_assessment" in drift_cell
    assert "run_reentry_assessment" in norm_cell
    assert "reentry_assessment.json" in drift_cell
    assert "reentry_assessment.md" in drift_cell
    assert "reentry_assessment.json" in norm_cell
    assert "reentry_assessment.md" in norm_cell
    assert "STAGE5_REENTRY_NORM_CANDIDATE_TASK_LIMIT" in norm_cell
    assert "candidate_conversion_is_complete" in norm_cell
    assert "incomplete_candidate_conversion" in norm_cell
    assert "incremental_backup" in norm_cell
    assert "restore_incremental_backup" in norm_cell
    assert "STAGE5_REENTRY_NORM_RESTORE_INCREMENTAL_BACKUP" in norm_cell
    assert "restored_incremental_backup" in norm_cell
    assert "resume_skip=candidate_conversion" in norm_cell
    assert "publishable_artifact_paths" in drift_cell
    assert "publishable_artifact_paths" in norm_cell
    assert "update_current_source_summary" in drift_cell
    assert "update_current_source_summary" in norm_cell
    assert 'git", "add", "-f", str(out_dir.relative_to(ROOT))' not in drift_cell
    assert 'git", "add", "-f", str(out_dir.relative_to(ROOT))' not in norm_cell


def test_phase2_breadth_targets_require_master_sequence_gate() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    effective_cell = (ROOT / "colab/STAGE5_EFFECTIVE_PATHWAYS_CELL.py").read_text(encoding="utf-8")
    candidate_cell = (ROOT / "colab/STAGE5_CANDIDATE_CONVERSION_CELL.py").read_text(encoding="utf-8")

    for target in ["effective_pathways_diagnostic", "candidate_conversion_diagnostic"]:
        assert target in bootstrap
        assert target in bootstrap_md
    for text in [bootstrap, bootstrap_md, effective_cell, candidate_cell]:
        assert "colab.master_sequence_gate" in text
        assert "STAGE5_ALLOW_PRE_PHASE1_BREADTH" in text
    assert "require_phase1_depth_gate_for_breadth" in effective_cell
    assert "require_phase1_depth_gate_for_breadth" in candidate_cell
    assert "master_sequence_phase2_gate" in effective_cell
    assert "master_sequence_phase2_gate" in candidate_cell
    assert "effective_pathways_checkpoint_source=" in effective_cell
    assert "candidate_conversion_checkpoint_source=" in candidate_cell
    assert "effective_pathways_checkpoint_source=" in bootstrap
    assert "candidate_conversion_checkpoint_source=" in bootstrap
    assert "effective_pathways_checkpoint_source=" in bootstrap_md
    assert "candidate_conversion_checkpoint_source=" in bootstrap_md
    assert 'phase_gate.get("checkpoint")' in effective_cell
    assert 'phase_gate.get("checkpoint")' in candidate_cell
    assert 'phase_gate.get("checkpoint")' in bootstrap
    assert 'phase_gate.get("checkpoint")' in bootstrap_md
    assert "STAGE5_EFFECTIVE_PATHWAYS_CHECKPOINT" in effective_cell
    assert "STAGE5_CANDIDATE_CONVERSION_CHECKPOINT" in candidate_cell
    stale_checkpoint = "stage5_content_arcmix_qonly_optiontext_20260623_121707"
    assert stale_checkpoint not in effective_cell
    assert stale_checkpoint not in candidate_cell


def test_stage5_sft_launchers_do_not_force_checkpoint_commits() -> None:
    launcher_paths = [
        ROOT / "colab/STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py",
        ROOT / "colab/STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_CELL.py",
        ROOT / "colab/STAGE5_DEPTH_ROUTER_AFTER_DIRECT_PRESERVE_CELL.py",
        ROOT / "colab/STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL.py",
        ROOT / "colab/run_stage5_halt_target_repair.py",
    ]
    for path in launcher_paths:
        text = path.read_text(encoding="utf-8")
        assert '"STAGE5_CURRICULUM_SFT_COMMIT_CHECKPOINTS": "1"' not in text
        assert "STAGE5_CURRICULUM_SFT_COMMIT_CHECKPOINTS" in text


def test_current_colab_docs_link_master_sequence_and_future_gates() -> None:
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")
    runbook = (ROOT / "docs/STAGE5_REENTRY_STAGE3_STAGE4_RUNBOOK.md").read_text(encoding="utf-8")
    master = (ROOT / "docs/PROGRAM_TRACK_MASTER_SEQUENCE.md").read_text(encoding="utf-8")
    next_sequence = (ROOT / "colab/NEXT_COLAB_SEQUENCE.md").read_text(encoding="utf-8")
    staged = (ROOT / "colab/STAGED_NOTEBOOKS.md").read_text(encoding="utf-8")

    assert "docs/PROGRAM_TRACK_MASTER_SEQUENCE.md" in current_action
    assert "docs/EXPERIMENT_LOG.md" in current_action
    assert "traced_sft_competence_preserving_pipeline" in current_action
    assert "review_stage5_competence_pipeline.py" in current_action
    assert "CURRENT_A100_ACTION.md" in next_sequence
    assert "claim_curriculum_scaleup_cpu" in staged
    assert "CPU/API data-prep target" in staged
    assert "not a GPU gate" in staged
    assert "model_viability_probe" in staged
    assert "model_viability_queue" in staged
    assert "effective_pathways_diagnostic" in staged
    assert "candidate_conversion_diagnostic" in staged
    assert "reentry_repair_smoke" in next_sequence
    assert "reentry_recovery_training" in next_sequence
    assert "Completed Context" in next_sequence
    assert "PROGRAM_TRACK_MASTER_SEQUENCE.md" in runbook
    assert "STAGE5_REENTRY_REPAIR_NORM_ASSESSMENT" in (
        ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py"
    ).read_text(encoding="utf-8")
    assert "STAGE5_REENTRY_RECOVERY_REPAIR_ASSESSMENT" in (
        ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py"
    ).read_text(encoding="utf-8")
    assert "same curriculum" in next_sequence
    assert "deferred until the deterministic Phase 1 gate passes" in staged
    assert "Phase 0, loop-closure re-entry" in master
    assert "Phase 1, depth" in master
    assert "Phase 2, breadth and multistability" in master
    assert "Phase 3, particles, SVGD, and the selector" in master
    assert "These were never parallel workstreams" in master
    assert "The binding uncertainty has shifted from architectural to empirical" in master
    assert "STAGE5_CURRENT_A100_TARGET=reentry_norm_diagnostic" in runbook
    assert "STAGE5_CURRENT_A100_TARGET=reentry_repair_smoke" in runbook
    assert "STAGE5_CURRENT_A100_TARGET=reentry_recovery_training" in runbook
    assert "bridge gate remains active" in runbook
    assert "bridge_gate_active=true" in runbook
    assert "bridge_proj,reentry,halt" in runbook
    assert "entry_rms" in runbook
    assert "fix_loop1_preservation_eval_before_recovery_training" in runbook
    assert "STAGE5_CURRICULUM_MIN_TARGET_LOOP_ROWS" in runbook
    assert "1=48,2=16,4=8" in runbook
    assert "1=1,2=1,4=1" in runbook


def test_stage5_control_ledger_names_current_reentry_target() -> None:
    log = (ROOT / "docs/EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    ledger = log.split("## 2026-06-24: Stage 5 Control Ledger", 1)[1]
    current_action = ledger.split("### Decisions", 1)[0]

    assert "Target: `reentry_repair_smoke`" in current_action
    assert "`bridge_gate` stayed active" in current_action or "bridge_gate_active" in current_action
    assert "stage5_reentry_norm_20260625_013527/summary.json" in current_action
    assert "reentry_repair_smoke -> reentry_recovery_training" in current_action
    assert "debiased_benchmark_suite -> dense_mcq_trace_sft_control" in current_action
    assert "Target: `traced_sft_score_alignment_repair`" not in current_action


def test_synthetic_depth_task_target_is_wired_and_guarded() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_SYNTHETIC_DEPTH_TASK_CELL.py").read_text(encoding="utf-8")
    spec = (ROOT / "docs/SYNTHETIC_DEPTH_TASK.md").read_text(encoding="utf-8")

    for text in (bootstrap, bootstrap_md):
        assert "synthetic_depth_task" in text
        assert "colab/STAGE5_SYNTHETIC_DEPTH_TASK_CELL.py" in text
        assert "training/generate_synthetic_depth_task.py" in text
        assert "eval/eval_synthetic_depth_matrix.py" in text
        assert "distinct_prefix_length_depth_plus_one" in text
        assert "frontier_strictly_expands" in text

    assert "STAGE5_SYNTHETIC_DEPTH_TASK_CELL_VERSION" in cell
    assert "synthetic_depth_task_v2_mcq_aligned" in cell
    assert "tests/test_synthetic_depth_task.py" in cell
    assert "tests/test_eval_synthetic_depth_matrix.py" in cell
    assert "STAGE5_SYNTH_DEPTH_MAX_STEPS" in cell
    assert "STAGE5_SYNTH_DEPTH_ROWS_PER_DEPTH" in cell
    assert "STAGE5_SYNTH_DEPTH_TRAIN_FORMAT" in cell
    assert "STAGE5_SYNTH_DEPTH_RUN_BASE_EVAL" in cell
    assert "train_mcq_option_text_sft.jsonl" in cell
    assert "frontier_strictly_expands" in cell
    assert 'run(["nvidia-smi"], cwd=Path("/content"), check=False)' in cell

    assert "A(d, k)" in spec
    assert "distinct orbit prefix" in spec
    assert "Launch target" in spec


def test_synthetic_depth_primitive_curve_target_is_wired_and_guarded() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_SYNTHETIC_DEPTH_PRIMITIVE_CURVE_CELL.py").read_text(encoding="utf-8")
    summarizer = (ROOT / "colab/summarize_synthetic_depth_primitive_curve.py").read_text(encoding="utf-8")

    for text in (bootstrap, bootstrap_md):
        assert "synthetic_depth_primitive_curve" in text
        assert "colab/STAGE5_SYNTHETIC_DEPTH_PRIMITIVE_CURVE_CELL.py" in text
        assert "STAGE5_SYNTH_PRIMITIVE_N_VALUES" in text
        assert "tests/test_synthetic_depth_primitive_curve.py" in text

    assert "STAGE5_SYNTHETIC_DEPTH_PRIMITIVE_CURVE_CELL_VERSION" in cell
    assert "synthetic_depth_primitive_curve_v1" in cell
    assert "Phase 1 changes only N and keeps max_depth=1" in cell
    assert "max_depth" in cell
    assert '"schedule": "linear"' in cell
    assert '"schedule": "constant"' not in cell
    assert "STAGE5_SYNTH_PRIMITIVE_BACKUP_CHECKPOINTS_TO_DRIVE" in cell
    assert "colab/summarize_synthetic_depth_primitive_curve.py" in cell
    assert "primitive_accuracy_bar" in cell
    assert "recommended_phase2_n_symbols" in summarizer


def test_synthetic_depth_staged_staircase_target_is_wired_and_guarded() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_SYNTHETIC_DEPTH_STAGED_STAIRCASE_CELL.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "models/recurrent_wrapper.py").read_text(encoding="utf-8")

    for text in (bootstrap, bootstrap_md):
        assert "synthetic_depth_staged_staircase" in text
        assert "colab/STAGE5_SYNTHETIC_DEPTH_STAGED_STAIRCASE_CELL.py" in text
        assert "STAGE5_SYNTH_STAIRCASE_STAGE12_STEPS" in text
        assert "STAGE5_SYNTH_STAIRCASE_STAGE1234_STEPS" in text
        assert "loop_loss_mode=target" in text

    assert "STAGE5_SYNTHETIC_DEPTH_STAGED_STAIRCASE_CELL_VERSION" in cell
    assert "synthetic_depth_staged_staircase_v1" in cell
    assert "restore_primitive_checkpoint" in cell
    assert "train_depth_le2_mcq_option_text_sft.jsonl" in cell
    assert "train_depth_le4_mcq_option_text_sft.jsonl" in cell
    assert '"loop_loss_mode": "target"' in cell
    assert "stage_depth_le2_finished" in cell
    assert "tests/test_recurrent_wrapper_tiny.py::test_target_loop_loss_mode_uses_requested_loop_on_tiny_model" in cell
    assert "loop_loss_mode: str = \"halting_weighted\"" in wrapper


def test_synthetic_depth_chain_supervision_target_is_wired_and_guarded() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_SYNTHETIC_DEPTH_CHAIN_SUPERVISION_CELL.py").read_text(encoding="utf-8")
    dataset = (ROOT / "training/synthetic_depth_task.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "models/recurrent_wrapper.py").read_text(encoding="utf-8")

    for text in (bootstrap, bootstrap_md):
        assert "synthetic_depth_chain_supervision" in text
        assert "colab/STAGE5_SYNTHETIC_DEPTH_CHAIN_SUPERVISION_CELL.py" in text
        assert "STAGE5_SYNTH_CHAIN_RUN_AFTER_TRAIN_DIAGNOSTIC" in text
        assert "loop_loss_mode=per_loop_labels" in text

    assert "STAGE5_SYNTHETIC_DEPTH_CHAIN_SUPERVISION_CELL_VERSION" in cell
    assert "synthetic_depth_chain_supervision_v1" in cell
    assert "phase_a_failed_checkpoint_train_split" in cell
    assert "train_chain_label_depth_le2_sft.jsonl" in cell
    assert "train_chain_label_depth_le4_sft.jsonl" in cell
    assert '"loop_loss_mode": "per_loop_labels"' in cell
    assert "chain_depth_le2_finished" in cell
    assert "tests/test_recurrent_wrapper_tiny.py::test_per_loop_label_loss_mode_uses_active_intermediate_labels_on_tiny_model" in cell
    assert "build_chain_label_sft_row" in dataset
    assert "chain_answer_by_loop" in dataset
    assert "loop_loss_mode == \"per_loop_labels\"" in wrapper


def test_split_bridge_microtest_target_is_wired_and_guarded() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_SPLIT_BRIDGE_MICROTEST_CELL.py").read_text(encoding="utf-8")
    bridge = (ROOT / "models/bridge.py").read_text(encoding="utf-8")
    trainer = (ROOT / "training/train_unfrozen_recurrent.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "eval/eval_synthetic_depth_matrix.py").read_text(encoding="utf-8")

    for text in (bootstrap, bootstrap_md):
        assert "synthetic_depth_split_bridge_microtest" in text
        assert "colab/STAGE5_SPLIT_BRIDGE_MICROTEST_CELL.py" in text
        assert "STAGE5_SPLIT_MICRO_PRELUDE_LR_MULTIPLIER" in text
        assert "STAGE5_SPLIT_MICRO_STAGE1234_STEPS" in text

    assert "STAGE5_SPLIT_BRIDGE_MICROTEST_CELL_VERSION" in cell
    assert "split_bridge_true_lr_microtest_v1" in cell
    assert "bridge_projection_mode=split" in cell
    assert "true bridge_prelude_lr_multiplier param group" in cell
    assert "split_chain_depth_le2" in cell
    assert "split_chain_depth_le4" in cell
    assert '"bridge_projection_mode": "split"' in cell
    assert '"bridge_prelude_lr_multiplier"' in cell
    assert "bridge_projection_mode" in evaluator
    assert "convert_to_split_projection" in bridge
    assert "bridge_prelude_optimizer_group_ok" in trainer


def test_chain_scaled_corrected_target_is_wired_and_guarded() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_CHAIN_SCALED_CORRECTED_CELL.py").read_text(encoding="utf-8")
    runner = (ROOT / "colab/run_stage5_chain_scaled_corrected.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "eval/eval_synthetic_depth_active_labels.py").read_text(encoding="utf-8")
    dataset = (ROOT / "training/synthetic_depth_task.py").read_text(encoding="utf-8")

    for text in (bootstrap, bootstrap_md):
        assert "synthetic_depth_chain_scaled_corrected" in text
        assert "colab/STAGE5_CHAIN_SCALED_CORRECTED_CELL.py" in text
        assert "STAGE5_CHAIN_CORRECTED_STAGE12_STEPS" in text
        assert "eval/eval_synthetic_depth_active_labels.py" in text

    assert "STAGE5_CHAIN_SCALED_CORRECTED_CELL_VERSION" in cell
    assert "chain_scaled_corrected_v1" in cell
    assert "active-label evaluator scores f^k(x) for k <= depth" in cell
    assert "full-symbol chain SFT avoids MCQ label bottleneck" in cell
    assert "tests/test_eval_synthetic_depth_active_labels.py" in cell
    assert "colab/run_stage5_chain_scaled_corrected.py" in cell
    assert "chain_scaled_corrected_depth_le2" in runner
    assert "chain_scaled_corrected_depth_le4" in runner
    assert "train_chain_symbol_depth_le4_sft.jsonl" in runner
    assert "--prediction_space" in runner
    assert "full_symbols" in runner
    assert "active_diagonal_min" in runner
    assert "bridge_prelude_lr_multiplier" in runner
    assert "stage5_split_bridge_microtest_20260702_154804/summary.json" in runner
    assert "stage5_synthetic_depth_primitive_curve_20260701_161524/summary.json" in runner
    assert "active_target_for_loop" in evaluator
    assert "above_diagonal_behavior" in evaluator
    assert "build_chain_symbol_sft_row" in dataset
    assert "chain_symbol_sft" in dataset


def test_chain_consolidation_targets_are_wired_and_guarded() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py").read_text(encoding="utf-8")
    extrap = (ROOT / "colab/run_stage5_depth_extrapolation_eval.py").read_text(encoding="utf-8")
    artifact = (ROOT / "eval/eval_synthetic_depth_artifact_check.py").read_text(encoding="utf-8")
    probe = (ROOT / "eval/eval_synthetic_depth_probe.py").read_text(encoding="utf-8")
    anneal = (ROOT / "colab/run_stage5_chain_anneal_to_outcome.py").read_text(encoding="utf-8")
    route = (ROOT / "colab/run_stage5_depth_support_route_comparison.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "models/recurrent_wrapper.py").read_text(encoding="utf-8")
    trainer = (ROOT / "training/train_unfrozen_recurrent.py").read_text(encoding="utf-8")

    for text in (bootstrap, bootstrap_md):
        assert "depth_extrapolation_eval" in text
        assert "synthetic_probe_battery" in text
        assert "chain_anneal_to_outcome" in text
        assert "post_anneal_extended_readouts" in text
        assert "chain_continuation_attribution" in text
        assert "chain_continuation_probe_readout" in text
        assert "depth_support_route_comparison" in text
        assert "depth_support_ladder8" in text
        assert "support8_probe_readout" in text
        assert "support8_dose_arm" in text
        assert "same_reader_final_symbol" in text
        assert "n24_same_reader_receipt" in text
        assert "support6_seed_replication" in text
        assert "support6_replication_receipts" in text
        assert "support6_dosed_seed_resolution" in text
        assert "support6_seed26_plateau_test" in text
        assert "synthetic_release_receipts" in text
        assert "n24_support12_rung" in text
        assert "phase_a_surpass_prereg" in text
        assert "permutation_zero_shot_baseline" in text
        assert "splice_injection_diagnostic" in text
        assert "colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py" in text
        assert "STAGE5_EXTRAP_MAX_LOOPS" in text
        assert "STAGE5_ANNEAL_TOTAL_STEPS" in text
        assert "STAGE5_CHAIN_CONTINUATION_EXTRAP_DEPTHS" in text
        assert "STAGE5_ROUTE_FROZEN_EVAL_ID" in text
        assert "STAGE5_LADDER_FROZEN_EVAL_ID" in text
        assert "STAGE5_SUPPORT8_SOURCE_SUMMARY" in text
        assert "STAGE5_DOSE_SOURCE_SUMMARY" in text
        assert "STAGE5_SAME_READER_SOURCE_SUMMARY" in text
        assert "STAGE5_SAME_READER_EXPECT_IDENTITY_WITH_ACTIVE" in text
        assert "STAGE5_SUPPORT6_REPLICATION_SEEDS" in text
        assert "STAGE5_SUPPORT6_DOSED_RECEIPT_SUMMARY" in text
        assert "STAGE5_SEED26_PLATEAU_SOURCE_SUMMARY" in text
        assert "STAGE5_RELEASE_RECEIPTS_PUBLISH" in text
        assert "STAGE5_N24_FROZEN_EVAL_ID" in text
        assert "STAGE5_RUNG_CANARY_HARD_STOP" in text
        assert "STAGE5_PHASE_A_PLAN_RUN_ID" in text
        assert "STRONG_SCALING_MIN_CORRECT = 91" in text
        assert "ASYMPTOTE_REJECTION_MIN_CORRECT = 79" in text
        assert "CHANCE_REJECTION_MIN_CORRECT = 14" in text
        assert "STAGE5_SPLICE_SOURCE_SUMMARY" in text
        assert "source_orbit_fraction_j1_to_j3" in text

    assert "STAGE5_CHAIN_CONSOLIDATION_CELL_VERSION" in cell
    assert "eval/eval_synthetic_depth_artifact_check.py" in cell
    assert "eval/eval_synthetic_depth_probe.py" in cell
    assert "eval/eval_synthetic_depth_splice.py" in cell
    assert "loop_loss_mode='annealed_chain_to_outcome'" in cell
    assert "chain_continuation_attribution" in cell
    assert "depth_support_route_comparison" in cell
    assert "depth_support_ladder8" in cell
    assert "support8_probe_readout" in cell
    assert "support8_dose_arm" in cell
    assert "same_reader_final_symbol" in cell
    assert "support6_seed_replication" in cell
    assert "support6_replication_receipts" in cell
    assert "support6_dosed_seed_resolution" in cell
    assert "synthetic_release_receipts" in cell
    assert "n24_support12_rung" in cell
    assert "phase_a_surpass_prereg" in cell
    assert "splice_injection_diagnostic" in cell
    assert "colab/run_stage5_depth_support_ladder.py" in cell
    assert "colab/run_stage5_support8_probe_readout.py" in cell
    assert "colab/run_stage5_support8_dose_arm.py" in cell
    assert "colab/run_stage5_same_reader_final_symbol.py" in cell
    assert "colab/run_stage5_support6_seed_replication.py" in cell
    assert "colab/run_stage5_support6_replication_receipts.py" in cell
    assert "colab/run_stage5_support6_dosed_seed_resolution.py" in cell
    assert "colab/run_stage5_support6_seed26_plateau.py" in cell
    assert "colab/run_stage5_synthetic_release_receipts.py" in cell
    assert "colab/run_stage5_n24_support12_rung.py" in cell
    assert "colab/run_stage5_phase_a_surpass_plan.py" in cell
    assert "colab/run_stage5_permutation_zero_shot.py" in cell
    assert "same_reader_active_identity_check" in cell
    assert "STAGE5_PERM_PARITY_TOLERANCE" in cell
    assert "soft_depth10_min_correct" in cell
    assert "soft_depth11_min_correct" in cell
    assert "N24_STRONG_SCALING_MIN_CORRECT" in cell
    assert "STRONG_SCALING_MIN_CORRECT = 91" in cell
    assert "ASYMPTOTE_REJECTION_MIN_CORRECT = 79" in cell
    assert "CHANCE_REJECTION_MIN_CORRECT = 14" in cell
    assert "colab/run_stage5_splice_injection.py" in cell
    assert "source_orbit_fraction_j1_to_j3" in cell
    assert "source_state_continuation" in cell
    assert "lawful_fraction_j1_to_j3" in cell
    assert "prompt_position_shortcut" in cell
    assert "pre_registered_bands" in extrap
    assert "bridge_forward_calls" in artifact
    assert "loop_index_probe" in probe
    assert "feature_transforms" in probe
    assert "permutation_p95" in probe
    assert "annealed_chain_to_outcome" in anneal
    assert "STAGE5_ANNEAL_LOOP_LOSS_MODE" in anneal
    assert "STAGE5_ANNEAL_PRELUDE_LR_MULT" in anneal
    assert "SELECTION_MIN_CORRECT" in route
    assert "NONREGRESSION_FLOORS" in route
    assert "stage5_synthetic_depth_frozen_eval_v1" in route
    ladder = (ROOT / "colab/run_stage5_depth_support_ladder.py").read_text(encoding="utf-8")
    assert "STRONG_SCALING_MIN_CORRECT" in ladder
    assert "DEFAULT_ROUTE_SOURCE_SUMMARY" in ladder
    assert "base_route_identity_check" in ladder
    assert "constructive_distinct_orbit_prefix_no_rejection" in ladder
    assert "stage5_synthetic_depth_frozen_eval_v2_depth14" in ladder
    assert "loop_loss_mode == \"annealed_chain_to_outcome\"" in wrapper
    assert "chain_label_weight" in trainer
    assert "trainable_parameter_norm_stats" in trainer


def test_natural_surface_prepare_target_is_wired_and_guarded() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_NATURAL_SURFACE_PREPARE_CELL.py").read_text(encoding="utf-8")
    runner = (ROOT / "colab/run_stage5_natural_surface_prepare.py").read_text(encoding="utf-8")
    generator = (ROOT / "training/generate_natural_surface_transfer.py").read_text(encoding="utf-8")
    dataset = (ROOT / "training/natural_surface_transfer.py").read_text(encoding="utf-8")
    active_eval = (ROOT / "eval/eval_synthetic_depth_active_labels.py").read_text(encoding="utf-8")

    for text in (bootstrap, bootstrap_md):
        assert "natural_surface_prepare_cpu" in text
        assert "colab/STAGE5_NATURAL_SURFACE_PREPARE_CELL.py" in text
        assert "STAGE5_NATURAL_VERIFY_TOKENIZER" in text
        assert "value_prefix=name:" in text

    assert "STAGE5_NATURAL_SURFACE_PREPARE_CELL_VERSION" in cell
    assert "natural_surface_prepare_v1" in cell
    assert "stage5_natural_surface_transfer_dataset" in cell
    assert "stage5_natural_surface_transfer_prepare" in runner
    assert "update_pointer=False" in runner
    assert "verify_single_token_names" in runner
    assert "training.natural_surface_transfer" in generator
    assert "main" in generator
    assert "verbalize_relay" in dataset
    assert "verbalize_pointer" in dataset
    assert "rung0_train_mix_chain_symbol_sft" in dataset
    assert "NAME_SYMBOLS" in active_eval
    assert 'prefix == "name:"' in active_eval


def test_gradient_path_audit_target_is_wired_and_guarded() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_GRADIENT_PATH_AUDIT_CELL.py").read_text(encoding="utf-8")
    audit = (ROOT / "eval/eval_gradient_path_audit.py").read_text(encoding="utf-8")

    for text in (bootstrap, bootstrap_md):
        assert "gradient_path_audit" in text
        assert "colab/STAGE5_GRADIENT_PATH_AUDIT_CELL.py" in text
        assert "STAGE5_GRADIENT_PATH_AUDIT_SOURCE_SUMMARY" in text
        assert "finite_difference_bridge_prelude" in text
        assert "STAGE5_GRADIENT_PATH_AUDIT_NUM_ROWS" in text
        assert "STAGE5_GRADIENT_PATH_AUDIT_CROSS_LOOP_FD" in text

    assert "STAGE5_GRADIENT_PATH_AUDIT_CELL_VERSION" in cell
    assert "gradient_path_audit_v1" in cell
    assert "read-only gradient matrix plus finite_difference_bridge_prelude" in cell
    assert "eval/eval_gradient_path_audit.py" in cell
    assert "STAGE5_GRADIENT_PATH_AUDIT_MATCH_TRAIN_PRECISION" in cell
    assert "--cross_loop_fd" in cell
    assert "stage5_synthetic_depth_chain_supervision_20260701_201715/summary.json" in cell
    assert "Record Stage 5 gradient-path audit" in cell
    assert "interpret_gradient_signature" in audit
    assert "autograd_cut_suspected" in audit
    assert "structural_independence_or_decode_bypass_suspected" in audit
    assert "cross_loop_bridge_output_fd" in audit
    assert "target_validity_summary" in audit
    assert "CoherenceAccumulator" in audit
    assert "multiplier_consumption_check" in audit
