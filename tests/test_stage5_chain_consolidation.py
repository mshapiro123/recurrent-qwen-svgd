from __future__ import annotations

import json
import runpy
from pathlib import Path

from colab import run_stage5_depth_extrapolation_eval as extrap
from colab.run_stage5_post_anneal_readouts import compact_extrapolation, compact_probe
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


def test_resolve_checkpoint_reference_restores_from_final_checkpoint_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"final checkpoint")
    summary = {
        "run_id": "anneal",
        "final_checkpoint": str(checkpoint),
        "final_checkpoint_drive_backup": None,
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr("colab.stage5_chain_consolidation_utils.ROOT", tmp_path)

    restored, metadata = resolve_checkpoint_reference(summary_path, tmp_path / "restored.pt")

    assert restored.read_bytes() == b"final checkpoint"
    assert metadata["source_run_id"] == "anneal"
    assert metadata["source_final_checkpoint"] == str(checkpoint)


def test_extrapolation_classification_uses_conservative_band_and_bar() -> None:
    assert extrap.classify_depth(0.90, lower=0.831, bar=0.71) == "inside_or_above_conservative_band"
    assert extrap.classify_depth(0.80, lower=0.831, bar=0.71) == "partial_extrapolation_below_conservative_band"
    assert extrap.classify_depth(0.30, lower=0.831, bar=0.71) == "below_bar"
    assert extrap.classify_depth(0.80, lower=None, bar=0.71) == "meets_bar_unbanded"


def test_chain_consolidation_cell_names_all_three_targets() -> None:
    text = Path("colab/STAGE5_CHAIN_CONSOLIDATION_CELL.py").read_text(encoding="utf-8")

    assert "depth_extrapolation_eval" in text
    assert "synthetic_probe_battery" in text
    assert "chain_anneal_to_outcome" in text
    assert "post_anneal_readouts" in text
    assert "post_anneal_extended_readouts" in text
    assert "chain_continuation_attribution" in text
    assert "colab/run_stage5_chain_anneal_to_outcome.py" in text
    assert "colab/run_stage5_post_anneal_readouts.py" in text
    assert "colab/run_stage5_chain_continuation_attribution.py" in text


def test_chain_consolidation_runners_import_when_executed_by_path() -> None:
    for path in [
        "colab/run_stage5_depth_extrapolation_eval.py",
        "colab/run_stage5_synthetic_probe_battery.py",
        "colab/run_stage5_chain_anneal_to_outcome.py",
        "colab/run_stage5_post_anneal_readouts.py",
        "colab/run_stage5_chain_continuation_attribution.py",
    ]:
        namespace = runpy.run_path(path, run_name="not_main")
        assert "main" in namespace


def test_post_anneal_readout_compacts_extrapolation_and_probe_fields() -> None:
    extrap_payload = {
        "run_id": "extrap",
        "status": "finished",
        "checkpoint": "ckpt.pt",
        "artifact_check": {"pass": True},
        "active_eval": {
            "active_diagonal": {"5": 0.5},
            "active_total": {"accuracy": 0.9},
            "above_diagonal": {"rates": {"iterate": 1.0}},
        },
        "extrapolation_read": {"5": {"observed": 0.5}},
    }
    probe_payload = {
        "run_id": "probe",
        "status": "finished",
        "checkpoint": "ckpt.pt",
        "probe_diagonal": {"5": 0.1},
        "probe": {
            "loop_index_probe": {"accuracy": 0.8},
            "depth_stratified_diagonal": {"5": {"5": {"accuracy": 0.1}}},
            "loop_index_deflation_curve": [{"rank": 1}],
            "state_envelope": {"late_loop_reconstruction_error": 0.2},
            "feature_transform_probes": {"unit_norm": {"loop_index_probe": {"accuracy": 0.3}}},
            "router_leak_exclusion": {"forced_loop_path_pass": True},
        },
    }

    assert compact_extrapolation(extrap_payload)["artifact_check_pass"] is True
    compacted_probe = compact_probe(probe_payload)
    assert compacted_probe["router_leak_exclusion"]["forced_loop_path_pass"] is True
    assert compacted_probe["state_envelope"]["late_loop_reconstruction_error"] == 0.2
    assert compacted_probe["feature_transform_probes"]["unit_norm"]["loop_index_probe"]["accuracy"] == 0.3
