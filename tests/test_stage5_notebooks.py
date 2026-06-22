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


def test_current_a100_action_points_to_safe_continue_routing_diagnostic() -> None:
    text = (ROOT / "colab/CURRENT_A100_ACTION.md").read_text(encoding="utf-8")

    assert "colab/STAGE5_SAFE_CONTINUE_CELL.md" in text
    assert "stage5_arc_mix_recovery_once_20260622_030628/summary.json" in text
    assert "run_stage5_routing_diagnostic.py" in text
    assert "run_stage5_routing_repair.py" in text
    assert "needs_direct_halting_repair" in text
    assert "needs_deep_narrow_recovery" in text
    assert "stop_and_revise_objective" in text
    assert "do **not** extend ARC-mix training" in text
    assert "do **not** run GPQA" in text


def test_safe_continue_cell_defaults_to_dry_run_and_guarded_action() -> None:
    text = (ROOT / "colab/STAGE5_SAFE_CONTINUE_CELL.md").read_text(encoding="utf-8")

    assert "RUN_A100_ACTION = False" in text
    assert "colab/check_stage5_a100_go_no_go.py" in text
    assert "colab/run_stage5_next_action.py" in text
    assert 'env["STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE"] = "1" if execute_action else "0"' in text
    assert "STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS" in text
    assert "Dry run complete" in text
    assert "DISCONNECT_RUNTIME_WHEN_DONE = True" in text
    assert "runtime.unassign()" in text
    assert "GO_NO_GO_RUN_ID" in text
    assert "Skipping requirements install because no paid action will execute." in text
    assert "execute_action = bool(RUN_A100_ACTION and go_allowed)" in text
    assert "stage5_arc_mix_recovery_once_20260622_030628/summary.json" in text
    assert "mount_drive_for_paid_action" in text
    assert 'drive.mount("/content/drive", force_remount=True)' in text
    assert "Mounting Google Drive so checkpoint artifacts can be restored." in text
    assert "colab/run_stage5_full_assessment_once.py" not in text


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
    assert "execute_action = bool(RUN_A100_ACTION and go_allowed)" in text
    assert "stage5_arc_mix_recovery_once_20260622_030628/summary.json" in text
    assert "mount_drive_for_paid_action" in text
    assert 'drive.mount("/content/drive", force_remount=True)' in text
    assert "Mounting Google Drive so checkpoint artifacts can be restored." in text
    assert "colab/run_stage5_full_assessment_once.py" not in text
    assert payload["cells"][0]["cell_type"] == "markdown"
    assert payload["cells"][1]["cell_type"] == "code"
