from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def notebook_payload(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


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


def test_single_a100_runbook_uses_colab_continue_wrapper() -> None:
    payload = notebook_payload("colab/00_single_a100_runbook.ipynb")
    text = "\n".join(str(cell.get("source", "")) for cell in payload.get("cells", []))

    assert "colab/run_stage5_colab_continue.py" in text
    assert "colab/run_stage5_reasoning_dataset_pipeline.py" in text
    assert "STAGE5_REASONING_DATASET_PIPELINE_EXECUTE_NEXT" in text
    assert "STAGE5_ARC_AGI_COLAB_CONTINUE_MAX_ACTIONS" in text
    assert "STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE" in text
    assert "claim" in text
    assert payload["cells"][3]["cell_type"] == "code"
    assert payload["cells"][5]["cell_type"] == "code"


def test_stage_launcher_uses_colab_continue_wrapper() -> None:
    payload = notebook_payload("colab/00_stage_launcher.ipynb")
    text = "\n".join(str(cell.get("source", "")) for cell in payload.get("cells", []))

    assert "colab/run_stage5_colab_continue.py" in text
    assert "STAGE5_ARC_AGI_COLAB_CONTINUE_MAX_ACTIONS" in text
    assert "STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE" in text
    assert "claim" in text
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


def test_current_a100_action_points_to_safe_continue_routing_repair() -> None:
    text = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

    assert "colab/CURRENT_A100_BOOTSTRAP_CELL.md" in text
    assert "colab/CURRENT_A100_BOOTSTRAP_CELL.py" in text
    assert "10_stage5_direct_preservation_precheck.ipynb" in text
    assert "11_stage5_direct_preservation_g4_auto.ipynb" in text
    assert "STAGE5_CURRENT_A100_TARGET=programmatic_curriculum_cpu" in text
    assert "colab/STAGE5_PROGRAMMATIC_CURRICULUM_CELL.py" in text
    assert "git/ref/heads/main" in text
    assert "resolved_ref" in text
    assert "Fetched stale bootstrap" in text
    assert "sha_resolved_nested_fetch_v3" in text
    assert "next_action_guard.allowed" in text
    assert "refuses attached GPU runtimes" in text
    assert "STAGE5_CURRENT_A100_TARGET=safe_continue_execute" in text
    assert "STAGE5_CURRENT_A100_SOURCE_SUMMARY" in text
    assert "## Fetch Doctor" in text
    assert "diagnostic-only cell" in text
    assert "colab/CURRENT_A100_BOOTSTRAP_CELL.py" in text
    assert "colab/STAGE5_PROGRAMMATIC_CURRICULUM_CELL.py" in text
    assert "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY" not in text
    assert "STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY" not in text
    assert "config/stage5_current_source_summary.txt" in text
    assert "colab/STAGE5_SAFE_CONTINUE_CELL.md" in text
    assert "colab/STAGE5_SAFE_CONTINUE_CELL.py" in text
    assert "colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.py" in text
    assert "stage5_routing_diagnostic_20260622_041706/summary.json" in text
    assert "run_stage5_routing_repair.py" in text
    assert "needs_direct_halting_repair" in text
    assert "ARC-Easy direct delta = -2" in text
    assert "ARC-Challenge direct delta = -3" in text
    assert "STAGE5_ARC_MIX_ARC_EASY_TARGET_LOOP=1" in text
    assert "STAGE5_ARC_MIX_ARC_CHALLENGE_TARGET_LOOP=2" in text
    assert "repair_proxy_lift" in text
    assert "direct rows stop regressing" in text
    assert "do **not** run GPQA" in text


def test_current_bootstrap_exposes_model_viability_queue_target() -> None:
    text = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert '"model_viability_queue"' in text
    assert "colab/STAGE5_MODEL_VIABILITY_QUEUE_CELL.py" in text
    assert "Qwen/Qwen2.5-3B-Instruct" in text
    assert "Qwen/Qwen2.5-7B-Instruct" in text
    assert "colab/run_stage5_model_viability_queue.py" in text


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
    assert "For unattended A100 time under the current low-credit policy" in staged
    assert "STAGE5_CURRENT_A100_TARGET=safe_continue_execute" in staged
    assert "older\n   `colab/run_stage5_arc_agi_autopilot.py` remains available only" in staged


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
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

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
    assert "Previous Paste-Anywhere ARC-Challenge Cell" in current_action
    assert '"STAGE5_CURRENT_A100_TARGET"] = "arc_challenge_mcq_debias_confirm"' in current_action


def test_debiased_benchmark_suite_cell_is_bounded_and_policy_compliant() -> None:
    plain = (ROOT / "colab/STAGE5_DEBIASED_BENCHMARK_SUITE_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

    assert "STAGE5_DEBIASED_BENCHMARK_SUITE_CELL_VERSION" in plain
    assert "STAGE5_DEBIASED_MOUNT_DRIVE_FIRST" in plain
    assert "STAGE5_DEBIASED_USE_LEARNED_LOOP_CONTROL" in plain
    assert "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL" in plain
    assert '"STAGE5_BENCHMARKS"] = os.environ.get("STAGE5_DEBIASED_BENCHMARKS", "arc_challenge,gpqa_lite")' in plain
    assert '"STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT"] = os.environ.get("STAGE5_DEBIASED_ARC_CHALLENGE_LIMIT", "128")' in plain
    assert '"STAGE5_BENCHMARK_GPQA_LIMIT"] = os.environ.get("STAGE5_DEBIASED_GPQA_LIMIT", "16")' in plain
    assert '"STAGE5_BENCHMARK_SCORE_TARGETS"] = os.environ.get(' in plain
    assert '"STAGE5_DEBIASED_SCORE_TARGETS",' in plain
    assert '"label,content_question_only,cyclic_label_aggregated",' in plain
    assert '"STAGE5_BENCHMARK_ASSESS_SCORE_TARGET"] = "cyclic_label_aggregated"' in plain
    assert '"STAGE5_BENCHMARK_ASSESS_AGGREGATE"] = "permutation_mean"' in plain
    assert "benchmark_source_summary" in plain
    assert "colab/run_stage5_benchmark_suite.py" in plain
    assert "colab/assess_stage5_benchmark_suite.py" in plain
    assert "tests/test_stage5_benchmark_suite.py" in plain
    assert "tests/test_stage5_benchmark_assessment.py" in plain
    assert "runtime.unassign()" in plain
    assert "debiased_benchmark_suite" in bootstrap
    assert "STAGE5_DEBIASED_BENCHMARK_SUITE_CELL.py" in bootstrap
    assert "Next Paste-Anywhere Debiased Benchmark Cell" in current_action
    assert '"STAGE5_CURRENT_A100_TARGET"] = "debiased_benchmark_suite"' in current_action


def test_depth_balanced_benchmark_target_uses_learned_loop_control() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert '"depth_balanced_benchmark"' in bootstrap
    assert '"STAGE5_DEBIASED_BENCHMARKS": "arc_easy,arc_challenge"' in bootstrap
    assert '"STAGE5_DEBIASED_SCORE_TARGETS": "content_question_only,cyclic_label_aggregated"' in bootstrap
    assert '"STAGE5_DEBIASED_USE_LEARNED_LOOP_CONTROL": "1"' in bootstrap
    assert '"STAGE5_DEBIASED_MOUNT_DRIVE_FIRST": "0"' in bootstrap
    assert '"STAGE5_DEBIASED_ARC_EASY_LIMIT": "512"' in bootstrap
    assert '"STAGE5_DEBIASED_ARC_CHALLENGE_LIMIT": "512"' in bootstrap


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
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

    assert "STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL_VERSION" in plain
    assert "capability_ladder_trace_jobs_cpu" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_ALLOW_GPU" in plain
    assert "colab/run_stage5_capability_ladder_trace_jobs.py" in plain
    assert "tests/test_capability_ladder_trace_jobs.py" in plain
    assert "runtime.unassign()" in plain
    assert "capability_ladder_trace_jobs_cpu" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL.py" in bootstrap
    assert "Next Paste-Anywhere Capability-Ladder Trace Jobs Cell" in current_action
    assert '"STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_trace_jobs_cpu"' in current_action


def test_capability_ladder_7b_trace_chain_cell_runs_probe_then_trace_jobs() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

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
    assert "STAGE5_CURRENT_A100_TARGET=capability_ladder_7b_trace_chain" in current_action


def test_capability_ladder_trace_collect_cell_is_cpu_only_and_response_driven() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

    assert "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL_VERSION" in plain
    assert "capability_ladder_trace_collect_cpu" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_ALLOW_GPU" in plain
    assert "colab/run_stage5_capability_ladder_trace_collect.py" in plain
    assert "tests/test_stage5_capability_ladder_trace_collect_runner.py" in plain
    assert "runtime.unassign()" in plain
    assert "capability_ladder_trace_collect_cpu" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL.py" in bootstrap
    assert "STAGE5_CURRENT_A100_TARGET=capability_ladder_trace_collect_cpu" in current_action


def test_capability_ladder_trace_responses_cell_is_cpu_only_and_provider_opt_in() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

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
    assert "Next Paste-Anywhere Capability-Ladder Trace Responses Cell" in current_action
    assert '"STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_trace_responses_cpu"' in current_action


def test_capability_ladder_trace_response_collect_cell_runs_both_steps() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

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
    assert "Next Paste-Anywhere Capability-Ladder Trace Response+Collection Cell" in current_action
    assert '"STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_trace_response_collect_cpu"' in current_action


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
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

    assert "hf_local" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_LOCAL_HF" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_MODEL_NAME" in plain
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_STUDENT_LINEAGE" in plain
    assert "capability_ladder_local_hf_trace_collect" in bootstrap
    assert "Qwen/Qwen2.5-7B-Instruct" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_STUDENT_LINEAGE" in bootstrap
    assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT" in bootstrap
    assert "capability_ladder_local_hf_trace_collect" in bootstrap_md
    assert "Next Paste-Anywhere Local-HF Trace Response+Collection Cell" in current_action
    assert '"STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_local_hf_trace_collect"' in current_action


def test_capability_ladder_local_hf_trace_sft_target_is_bootstrapped() -> None:
    plain = (ROOT / "colab/STAGE5_CAPABILITY_LADDER_LOCAL_HF_TRACE_SFT_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

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
    assert "Next Paste-Anywhere Local-HF Trace-to-SFT Chain Cell" in current_action
    assert '"STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_local_hf_trace_sft"' in current_action


def test_capability_ladder_local_hf_trace_sft_scale64_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

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
    assert "Short fresh-runtime launcher" in current_action
    assert 'git", "clone", repo_url' in current_action
    assert '"STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_local_hf_trace_sft_scale64"' in current_action


def test_traced_sft_scale64_benchmark_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")
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
    assert "MCQ score-level repair" in current_action
    assert "STAGE5_CURRENT_A100_TARGET=traced_sft_score_alignment_repair" in current_action
    assert "STAGE5_CURRENT_A100_TARGET=traced_sft_scale64_benchmark" in current_action


def test_traced_sft_direct_preservation_probe_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")
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
    assert "surface-alignment SFT repairs have now run" in current_action
    assert "STAGE5_CURRENT_A100_TARGET=traced_sft_score_alignment_repair" in current_action
    assert "ARC-Easy content:      recurrent 140/256 vs base 148/256, delta -8" in current_action
    assert "ARC-Challenge content: recurrent 86/256 vs base 87/256, delta -1" in current_action
    assert "ARC-Easy content delta: -7" in current_action
    assert "losses on permutation-disagreeing rows: 0/10" in current_action
    assert "losses rescued by cyclic aggregation: 8/10" in current_action
    assert "diagnose_content_route_scoring_or_prompt_alignment_before_more_distillation" in current_action
    assert "direct_route_loop1_matches_base_without_training" in current_action
    assert "direct_route_precheck_needs_training" in current_action
    assert "Only after the precheck says training is needed" in current_action
    assert "bounded\nstop-on-first-pass sweep" in current_action
    assert "learned-depth router continuation" in current_action
    assert "traced_sft_depth_router_after_direct_preserve" in current_action
    assert "before commit `d7682ec`" in current_action
    assert "Stale-safe fresh-runtime launcher" in current_action
    assert "api.github.com/repos/mshapiro123/recurrent-qwen-svgd" in current_action
    assert "STAGE5_SURFACE_ALIGNMENT_REPAIR_CELL_VERSION" in current_action
    assert '"STAGE5_CURRENT_A100_TARGET"] = "traced_sft_score_alignment_repair"' in current_action
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
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")
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
    assert "STAGE5_CURRENT_A100_TARGET=traced_sft_score_alignment_repair" in current_action
    assert "prioritize_content_cyclic_surface_alignment" in current_action


def test_traced_sft_score_alignment_repair_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

    assert "traced_sft_score_alignment_repair" in bootstrap
    assert "traced_sft_score_alignment_repair" in bootstrap_md
    assert '"STAGE5_SURFACE_ALIGN_TRAINER": "score_ce"' in bootstrap
    assert "training/prepare_mcq_score_alignment_jsonl.py" in bootstrap
    assert "training/train_phase1_mcq_score_align.py" in bootstrap
    assert "stage5_score_alignment_repair_content_route_20260624" in bootstrap
    assert "STAGE5_SURFACE_ALIGN_SCORE_DISTILL_WEIGHT" in bootstrap
    assert "STAGE5_CURRENT_A100_TARGET=traced_sft_score_alignment_repair" in current_action
    assert "ARC-Easy content repair delta: -2" in current_action
    assert "direct option-score CE" in current_action


def test_dense_mcq_trace_sft_control_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_DENSE_MCQ_TRACE_SFT_CONTROL_CELL.py").read_text(encoding="utf-8")

    assert "dense_mcq_trace_sft_control" in bootstrap
    assert "dense_mcq_trace_sft_control" in bootstrap_md
    assert "colab/STAGE5_DENSE_MCQ_TRACE_SFT_CONTROL_CELL.py" in bootstrap
    assert "STAGE5_DENSE_MCQ_TRACE_SFT_CONTROL_CELL_VERSION" in cell
    assert "dense_mcq_trace_sft_control_v1" in cell
    assert "STAGE5_DENSE_MCQ_SOURCE_SUMMARY" in cell
    assert "stage5_local_hf_traced_capability_sft_20260623_194543" in cell
    assert "training/train_dense_lora.py" in cell
    assert "eval/eval_mcq.py --mode base --checkpoint" in cell
    assert "colab/run_stage5_mcq_dense_sft_control.py" in cell
    assert "colab/assess_stage5_mcq_recipe_control.py" in cell
    assert "tests/test_eval_mcq_dense_lora.py" in cell
    assert "tests/test_stage5_mcq_dense_sft_control.py" in cell
    assert "tests/test_stage5_mcq_recipe_control_assessment.py" in cell
    assert "STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY" in cell
    assert "stage5_local_hf_traced_sft_scale64_benchmark_20260623_201923" in cell
    assert "runtime.unassign" in cell


def test_traced_sft_competence_preserving_pipeline_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_COMPETENCE_PRESERVING_PIPELINE_CELL.py").read_text(encoding="utf-8")

    assert "traced_sft_competence_preserving_pipeline" in bootstrap
    assert "traced_sft_competence_preserving_pipeline" in bootstrap_md
    assert "colab/STAGE5_COMPETENCE_PRESERVING_PIPELINE_CELL.py" in bootstrap
    assert "STAGE5_COMPETENCE_PRESERVING_PIPELINE_CELL_VERSION" in cell
    assert "competence_preserving_pipeline_v1" in cell
    assert "STAGE5_COMPETENCE_SOURCE_SUMMARY" in cell
    assert "stage5_traced_sft_direct_preservation_20260623_scale64_confirm_assessment" in cell
    assert "colab/run_stage5_competence_preserving_pipeline.py" in cell
    assert "tests/test_stage5_competence_preserving_pipeline.py" in cell
    assert "tests/test_stage5_balanced_arc_mix_gate.py" in cell
    assert "runtime.unassign" in cell


def test_traced_capability_ladder_sft_cell_derives_training_env_from_collection() -> None:
    plain = (ROOT / "colab/STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    current_action = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

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
    assert "STAGE5_CURRENT_A100_TARGET=traced_capability_ladder_sft" in current_action


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

    assert "RUN_PROVIDER_RESPONSES = False" in text
    assert "RUN_PROVIDER_RESPONSES = False" in plain
    assert "PROVIDER_LIMIT = 2" in text
    assert "PROVIDER_LIMIT = 2" in plain
    assert "MIN_POSITIVE_ROWS = 16" in text
    assert "MIN_POSITIVE_ROWS = 16" in plain
    assert 'MIN_MODE_ROWS = ""' in text
    assert 'MIN_MODE_ROWS = ""' in plain
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

    assert 'WORK_DIR = "data/curriculum/programmatic_direct_deep_001"' in text
    assert 'WORK_DIR = "data/curriculum/programmatic_direct_deep_001"' in plain
    assert 'MIN_POSITIVE_ROWS = "2000"' in text
    assert 'MIN_POSITIVE_ROWS = "2000"' in plain
    assert 'MIN_MODE_ROWS = "direct=1000,deep_narrow=1000"' in text
    assert 'MIN_MODE_ROWS = "direct=1000,deep_narrow=1000"' in plain
    assert "stale tiny shard" in text
    assert "STAGE5_CURRICULUM_MIN_MODE_ROWS" in plain


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


def test_reentry_repair_smoke_target_is_bootstrapped() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    bootstrap_md = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_REENTRY_REPAIR_SMOKE_CELL.py").read_text(encoding="utf-8")

    assert "reentry_repair_smoke" in bootstrap
    assert "reentry_repair_smoke" in bootstrap_md
    assert "colab/STAGE5_REENTRY_REPAIR_SMOKE_CELL.py" in bootstrap
    assert "STAGE5_REENTRY_REPAIR_SMOKE_CELL_VERSION" in cell
    assert "stage5_reentry_repair_smoke_v1_trainable" in cell
    assert "bridge_gate_override" in cell
    assert "bridge_reset_identity" in cell
    assert "reentry_rescale_mode" in cell
    assert "training/train_phase1_ponder.py" in cell
    assert "eval/eval_reentry_drift.py" in cell
    assert "colab/assess_stage5_reentry.py" in cell
    assert "Readout Pause" in cell
    assert "runtime.unassign" in cell


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
