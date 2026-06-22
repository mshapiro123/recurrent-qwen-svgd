from __future__ import annotations

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
    assert "STAGE5_CURRENT_A100_TARGET=safe_continue_execute" in text
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


def test_safe_continue_cell_defaults_to_dry_run_and_guarded_action() -> None:
    text = (ROOT / "colab/STAGE5_SAFE_CONTINUE_CELL.md").read_text(encoding="utf-8")
    plain = (ROOT / "colab/STAGE5_SAFE_CONTINUE_CELL.py").read_text(encoding="utf-8")

    assert 'RUN_A100_ACTION = env_bool("STAGE5_SAFE_CONTINUE_RUN_A100_ACTION", False)' in text
    assert 'RUN_A100_ACTION = env_bool("STAGE5_SAFE_CONTINUE_RUN_A100_ACTION", False)' in plain
    assert 'DISCONNECT_RUNTIME_WHEN_DONE = env_bool("STAGE5_SAFE_CONTINUE_DISCONNECT", True)' in text
    assert 'DISCONNECT_RUNTIME_WHEN_DONE = env_bool("STAGE5_SAFE_CONTINUE_DISCONNECT", True)' in plain
    assert "colab/check_stage5_a100_go_no_go.py" in text
    assert "colab/check_stage5_a100_go_no_go.py" in plain
    assert "colab/run_stage5_next_action.py" in text
    assert "colab/run_stage5_next_action.py" in plain
    assert 'env["STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE"] = "1" if execute_action else "0"' in text
    assert 'env["STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE"] = "1" if execute_action else "0"' in plain
    assert "STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS" in text
    assert "Dry run complete" in text
    assert "STAGE5_SAFE_CONTINUE_DISCONNECT" in text
    assert "runtime.unassign()" in text
    assert "GO_NO_GO_RUN_ID" in text
    assert "Skipping requirements install because no paid action will execute." in text
    assert "execute_action = bool(RUN_A100_ACTION and go_allowed)" in text
    assert "tests/test_stage5_routing_repair.py" in text
    assert "tests/test_stage5_balanced_arc_mix_gate.py" in text
    assert "tests/test_curriculum_sft_gate.py" in text
    assert "tests/test_stage5_curriculum_sft.py" in text
    assert "tests/test_curriculum_pipeline_from_artifacts.py" in text
    assert "tests/test_curriculum_jsonl.py" in text
    assert "stage5_routing_diagnostic_20260622_041706/summary.json" in text
    assert "mount_drive_for_paid_action" in text
    assert 'drive.mount("/content/drive", force_remount=True)' in text
    assert "Mounting Google Drive so checkpoint artifacts can be restored." in text
    assert text.index("if RUN_A100_ACTION:\n    mount_drive_for_paid_action()") < text.index(
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

    assert "RUN_A100_ACTION = False" in text
    assert "colab/check_stage5_a100_go_no_go.py" in text
    assert "colab/run_stage5_next_action.py" in text
    assert "STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE" in text
    assert "Dry run complete" in text
    assert "DISCONNECT_RUNTIME_WHEN_DONE = True" in text
    assert "runtime.unassign()" in text
    assert "GO_NO_GO_RUN_ID" in text
    assert "Skipping requirements install because no paid action will execute." in text
    assert "tests/test_curriculum_sft_gate.py" in text
    assert "tests/test_stage5_curriculum_sft.py" in text
    assert "tests/test_curriculum_pipeline_from_artifacts.py" in text
    assert "tests/test_curriculum_jsonl.py" in text
    assert "execute_action = bool(RUN_A100_ACTION and go_allowed)" in text
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
    assert "colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.py" in plain
    assert "colab/STAGE5_SAFE_CONTINUE_CELL.py" in plain
    assert '"safe_continue_execute"' in plain
    assert '"STAGE5_SAFE_CONTINUE_RUN_A100_ACTION": "1"' in plain
    assert '"STAGE5_SAFE_CONTINUE_RUN_A100_ACTION": "0"' in plain
    assert "checkpoint_preflight" in plain
    assert "mount_drive_for_paid_action" in plain
    assert "api.github.com/repos" in plain
    assert "GH_TOKEN" in plain
    assert "GITHUB_TOKEN" in plain
    assert "required_markers" not in plain
    assert "colab/STAGE5_ARC_MIX_RECOVERY_CELL.py" not in plain
    assert "preflight" in text
    assert "safe_continue_execute" in text


def test_current_a100_bootstrap_plain_cell_matches_markdown_code() -> None:
    markdown_cell = fenced_python_block("colab/CURRENT_A100_BOOTSTRAP_CELL.md")
    plain_cell = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert plain_cell == markdown_cell


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
    assert "drive.mount(\"/content/drive\", force_remount=True)" in code
    assert "stage5_routing_diagnostic_20260622_041706/summary.json" in code
    assert 'GO_NO_GO_RUN_ID = "stage5_drive_checkpoint_preflight"' in code
    assert '"colab/check_stage5_a100_go_no_go.py"' in code
    assert '"--source-summary"' in code
    assert "checkpoint_preflight" in code
    assert "summary[\"decision\"]" in code
    assert "runtime.unassign()" in code
    assert '["git", "pull", "--ff-only", "origin", "main"]' in code
    assert "pip install" not in code
    assert "colab/run_stage5_next_action.py" not in code
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

    assert 'MIN_MODE_ROWS = "direct=64,deep_narrow=64"' in text
    assert 'MIN_MODE_ROWS = "direct=64,deep_narrow=64"' in plain
    assert "width-only shards" in text
    assert "STAGE5_CURRICULUM_MIN_MODE_ROWS" in plain
