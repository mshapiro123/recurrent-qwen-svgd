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
