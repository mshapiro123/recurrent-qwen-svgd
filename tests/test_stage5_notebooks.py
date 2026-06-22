from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def notebook_payload(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


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
    assert "colab/run_stage5_colab_continue.py" not in text
    assert "colab/run_stage5_reasoning_dataset_pipeline.py" not in text
    assert payload["cells"][0]["cell_type"] == "markdown"
    assert payload["cells"][1]["cell_type"] == "code"


def test_arc_mix_recovery_cell_is_single_purpose() -> None:
    text = (ROOT / "colab/STAGE5_ARC_MIX_RECOVERY_CELL.md").read_text(encoding="utf-8")

    assert "colab/run_stage5_arc_mix_recovery_once.py" in text
    assert "STAGE5_ARC_MIX_ONCE_AUTO_DISCONNECT" in text
    assert "stage5_recovery_full_assessment_current/summary.json" in text
    assert "arc_mix_response_w005_lr2e6" in text
    assert "colab/run_stage5_full_assessment_once.py" not in text
    assert "colab/run_stage5_colab_continue.py" not in text
