from __future__ import annotations

import json
from pathlib import Path

from training.run_paper2_phase2_a2_guardrail_sweep import run


ROOT = Path(__file__).resolve().parents[1]


def test_guardrail_sweep_validates_locked_step237_sources(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    result = run(root=ROOT, output=output)
    assert result["status"] == "complete_lock_valid"
    assert result["training_authorized_by_this_job"] is False
    assert result["gpu_used"] is False
    assert result["generator_contract"]["next_attempt"] == 238
    assert result["inventory_validation"]["valid"]
    assert len(result["source_checks"]) == 4
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete_lock_valid"
