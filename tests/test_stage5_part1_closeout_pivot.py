from __future__ import annotations

import ast
from pathlib import Path

import yaml

from colab.run_stage5_part1_closeout_pivot import (
    _trained_position_gate,
    interpret_loop_position,
)


ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_targets() -> dict:
    tree = ast.parse((ROOT / "colab" / "CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TARGETS" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("TARGETS not found")


def test_position_transfer_bands_are_preregistered_exactly() -> None:
    assert interpret_loop_position({"active_diagonal": {"3": 0.56, "4": 0.55}})["reading"] == (
        "substantially_position_invariant"
    )
    assert interpret_loop_position({"active_diagonal": {"3": 0.15, "4": 0.12}})["reading"] == (
        "per_position_installation_confirmed"
    )
    assert interpret_loop_position({"active_diagonal": {"3": 0.54, "4": 0.16}})["reading"] == "partial_transfer"


def test_trained_position_prerequisite_requires_both_positions() -> None:
    assert _trained_position_gate({"active_diagonal": {"1": 0.72, "2": 0.71}})["passed"] is True
    assert _trained_position_gate({"active_diagonal": {"1": 0.90, "2": 0.70}})["passed"] is False


def test_micro_config_declares_disposable_nonpromotable_lineage(tmp_path: Path) -> None:
    from colab.run_stage5_part1_closeout_pivot import _write_micro_config

    path = tmp_path / "config.yaml"
    _write_micro_config(path, checkpoint=tmp_path / "source.pt", output_dir=tmp_path / "out", seed=3)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["lineage_policy"]["regime"] == "disposable_measurement"
    assert config["lineage_policy"]["checkpoint_promotable"] is False
    assert config["lineage_policy"]["successor_source_allowed"] is False
    assert config["batch_size"] * config["gradient_accumulation_steps"] >= 8


def test_bootstrap_exposes_one_shared_closeout_pivot_target() -> None:
    target = _bootstrap_targets()["part1_closeout_pivot_session"]
    assert target["path"] == "colab/STAGE5_PART1_CLOSEOUT_PIVOT_CELL.py"
    assert target["env"]["STAGE5_PART1_PIVOT_DISCONNECT"] == "0"
    assert "disposable_measurement" in target["markers"]
