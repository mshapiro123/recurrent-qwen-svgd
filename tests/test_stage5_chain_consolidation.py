from __future__ import annotations

import json
from pathlib import Path

from colab import run_stage5_depth_extrapolation_eval as extrap
from colab.stage5_chain_consolidation_utils import resolve_checkpoint_reference


def test_resolve_checkpoint_reference_restores_from_stage_summary(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "run_id": "run",
        "stages": [
            {
                "stage_name": "final",
                "checkpoint": str(checkpoint),
                "checkpoint_drive_backup": None,
            }
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr("colab.stage5_chain_consolidation_utils.ROOT", tmp_path)

    restored, metadata = resolve_checkpoint_reference(run_dir, tmp_path / "restored.pt")

    assert restored.read_bytes() == b"checkpoint"
    assert metadata["source_run_id"] == "run"
    assert metadata["source_stage_name"] == "final"


def test_extrapolation_classification_uses_conservative_band_and_bar() -> None:
    assert extrap.classify_depth(0.90, lower=0.831, bar=0.71) == "inside_or_above_conservative_band"
    assert extrap.classify_depth(0.80, lower=0.831, bar=0.71) == "partial_extrapolation_below_conservative_band"
    assert extrap.classify_depth(0.30, lower=0.831, bar=0.71) == "below_bar"


def test_chain_consolidation_cell_names_all_three_targets() -> None:
    text = Path("colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py").read_text(encoding="utf-8")

    assert "depth_extrapolation_eval" in text
    assert "synthetic_probe_battery" in text
    assert "chain_anneal_to_outcome" in text
    assert "colab/run_stage5_chain_anneal_to_outcome.py" in text
