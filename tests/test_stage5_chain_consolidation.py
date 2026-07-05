from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from colab import run_stage5_depth_extrapolation_eval as extrap
from colab import run_stage5_depth_support_ladder as ladder
from colab import run_stage5_depth_support_route_comparison as route
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
    assert "chain_continuation_probe_readout" in text
    assert "depth_support_route_comparison" in text
    assert "depth_support_ladder8" in text
    assert "splice_injection_diagnostic" in text
    assert "colab/run_stage5_chain_anneal_to_outcome.py" in text
    assert "colab/run_stage5_post_anneal_readouts.py" in text
    assert "colab/run_stage5_chain_continuation_attribution.py" in text
    assert "colab/run_stage5_depth_support_route_comparison.py" in text
    assert "colab/run_stage5_depth_support_ladder.py" in text
    assert "colab/run_stage5_splice_injection.py" in text


def test_chain_consolidation_runners_import_when_executed_by_path() -> None:
    for path in [
        "colab/run_stage5_depth_extrapolation_eval.py",
        "colab/run_stage5_synthetic_probe_battery.py",
        "colab/run_stage5_chain_anneal_to_outcome.py",
        "colab/run_stage5_post_anneal_readouts.py",
        "colab/run_stage5_chain_continuation_attribution.py",
        "colab/run_stage5_depth_support_route_comparison.py",
        "colab/run_stage5_depth_support_ladder.py",
        "colab/run_stage5_splice_injection.py",
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


def test_depth_route_scoring_locks_thresholds_and_nonregression() -> None:
    def cell(correct: int, total: int = 128) -> dict[str, float | int]:
        return {"correct": correct, "total": total, "accuracy": correct / total}

    active_summary = {
        "active_matrix": {
            "1": {"1": cell(128)},
            "2": {"2": cell(124)},
            "3": {"3": cell(124)},
            "4": {"4": cell(124)},
            "5": {"5": cell(110)},
            "6": {"6": cell(109)},
            "7": {"7": cell(52)},
            "8": {"8": cell(19)},
            "9": {"9": cell(14)},
            "10": {"10": cell(14)},
        }
    }

    score = route.score_route(active_summary, rows_per_depth=128)

    assert score["overall_pass"] is True
    assert score["locked_thresholds"]["selection_min_correct"]["7"] == 52
    assert score["locked_thresholds"]["selection_min_correct"]["8"] == 19
    assert score["locked_thresholds"]["nonregression_floors"]["6"] == 0.85


def test_depth_route_scoring_fails_depth6_nonregression() -> None:
    active_summary = {
        "active_matrix": {
            "1": {"1": {"correct": 128, "total": 128, "accuracy": 1.0}},
            "2": {"2": {"correct": 124, "total": 128, "accuracy": 0.96875}},
            "3": {"3": {"correct": 124, "total": 128, "accuracy": 0.96875}},
            "4": {"4": {"correct": 124, "total": 128, "accuracy": 0.96875}},
            "5": {"5": {"correct": 110, "total": 128, "accuracy": 0.859375}},
            "6": {"6": {"correct": 108, "total": 128, "accuracy": 0.84375}},
            "7": {"7": {"correct": 52, "total": 128, "accuracy": 0.40625}},
            "8": {"8": {"correct": 19, "total": 128, "accuracy": 0.1484375}},
            "9": {"9": {"correct": 14, "total": 128, "accuracy": 0.109375}},
            "10": {"10": {"correct": 14, "total": 128, "accuracy": 0.109375}},
        }
    }

    score = route.score_route(active_summary, rows_per_depth=128)

    assert score["nonregression"]["6"]["pass"] is False


def test_depth_support_ladder_scores_strong_scaling() -> None:
    def cell(correct: int, total: int = 128) -> dict[str, float | int]:
        return {"correct": correct, "total": total, "accuracy": correct / total}

    active_summary = {
        "active_matrix": {
            **{str(depth): {str(depth): cell(128 if depth <= 4 else 110)} for depth in range(1, 9)},
            "9": {"9": cell(80)},
            "10": {"10": cell(91)},
            "11": {"11": cell(91)},
            "12": {"12": cell(14)},
            "13": {"13": cell(14)},
            "14": {"14": cell(14)},
        }
    }

    score = ladder.score_ladder(active_summary)

    assert score["nonregression_pass"] is True
    assert score["strong_scaling_pass"] is True
    assert score["verdict"] == "strong_scaling"
    gates = ladder.locked_gate_summary()
    assert gates["strong_scaling_min_correct"] == 91
    assert gates["asymptote_rejection_min_correct"] == 79
    assert gates["chance_rejection_min_correct"] == 14
    assert gates["nonregression_floors"]["1"] == 0.93


def test_depth_support_ladder_detects_asymptote_rejection_without_strong_scaling() -> None:
    def cell(correct: int, total: int = 128) -> dict[str, float | int]:
        return {"correct": correct, "total": total, "accuracy": correct / total}

    active_summary = {
        "active_matrix": {
            **{str(depth): {str(depth): cell(128 if depth <= 4 else 110)} for depth in range(1, 9)},
            "9": {"9": cell(52)},
            "10": {"10": cell(79)},
            "11": {"11": cell(13)},
            "12": {"12": cell(13)},
            "13": {"13": cell(13)},
            "14": {"14": cell(13)},
        }
    }

    score = ladder.score_ladder(active_summary)

    assert score["asymptote_rejected"] is True
    assert score["strong_scaling_pass"] is False
    assert score["verdict"] == "asymptote_rejected_at_depth10"
    assert score["selection_pass"] is False
    assert score["overall_pass"] is False


def test_depth_support_ladder_manifest_detects_row_identity_mismatch() -> None:
    rows = [
        {"id": "r1", "depth": 1, "question": "a"},
        {"id": "r2", "depth": 2, "question": "b"},
    ]
    same = ladder.manifest_for_rows(list(rows))
    changed = ladder.manifest_for_rows([{**rows[0]}, {**rows[1], "question": "changed"}])

    assert same["row_id_sha256"] == changed["row_id_sha256"]
    assert same["row_sha256"] != changed["row_sha256"]

    with pytest.raises(RuntimeError, match="Frozen eval manifest mismatch"):
        ladder.assert_manifest_match(same, changed, label="test_chain_mcq")


def test_depth_support_ladder_refuses_to_regenerate_missing_base_frozen_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ladder, "ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="regeneration is intentionally forbidden"):
        ladder.ensure_base_frozen_eval_set(
            base_id="missing_frozen",
            n_symbols=16,
            rows_per_depth=128,
            seed="20260704",
            value_prefix="letter:",
        )
