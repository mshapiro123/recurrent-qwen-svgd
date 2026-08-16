from __future__ import annotations

import json
from pathlib import Path

from analysis.build_paper2_phase3_p35_amplitude_surface import CEILINGS, build, condition_name
from eval.eval_paper2_phase3_p34_task_trajectory import resolve_evaluation_gate_ceiling


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_explicit_amplitude_authorization_does_not_widen_legacy_probe() -> None:
    receipts = [{"label": "p35", "evaluation_gate_ceiling": 0.02}]
    try:
        resolve_evaluation_gate_ceiling(receipts, 0.05)
    except ValueError:
        pass
    else:
        raise AssertionError("legacy fixed-ceiling scope widened")
    assert resolve_evaluation_gate_ceiling(
        receipts, 0.05, authorized_overrides=CEILINGS
    ) == (0.05, "score_only_fixed_ceiling_override")


def test_amplitude_builder_applies_replicated_safety_rule(tmp_path: Path) -> None:
    for seed in (0, 1):
        for ceiling in CEILINGS:
            name = condition_name(seed, ceiling)
            rows = [
                {
                    "item_id": f"row-{index}",
                    "panel_group": "floor" if index < 512 else "target",
                    "base_correct": index % 2 == 0,
                    "augmented_correct": index % 2 == 0,
                }
                for index in range(1024)
            ]
            write_jsonl(tmp_path / f"{name}.jsonl", rows)
            write_json(tmp_path / f"{name}_summary.json", {
                "panel_sha256": "panel", "evaluation_gate_ceiling": ceiling,
                "confirm_scored": False, "eval_e_scored": False,
                "optimizer_constructed": False, "optimizer_steps": 0,
            })
            write_json(tmp_path / f"{name}_audit.json", {
                "ceiling": ceiling, "optimizer_steps": 0,
                "audit": {
                    "collateral_chi": 0.0 if ceiling <= 0.08 or seed == 0 else 0.01,
                    "pi_dir": {"point": ceiling}, "pi_dep": {"point": ceiling},
                },
            })
    result = build(tmp_path, {
        "authority": {"drive_id": "authority"},
        "amplitude_surface": {"selection_rule": "rule", "previously_seen": [0.02, 0.08]},
    })
    assert result["selected_ceiling_under_preregistered_rule"] == 0.08
    assert result["scope"]["optimizer_steps"] == 0
