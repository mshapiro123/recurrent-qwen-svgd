from __future__ import annotations

import json
from pathlib import Path

from analysis.build_paper2_phase3_p34_fixed_ceiling_probe import build


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_reconstructs_paired_ceiling_effect(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "analysis.build_paper2_phase3_p34_fixed_ceiling_probe.REGISTERED_ENDPOINT",
        {0: {"ceiling": 0.08, "correct": 2}, 1: {"ceiling": 0.02, "correct": 1}},
    )
    for seed in (0, 1):
        for ceiling, suffix in ((0.02, "0p02"), (0.08, "0p08")):
            name = f"seed_{seed}_ceiling_{suffix}"
            augmented = [True, ceiling == 0.08]
            if seed == 1 and ceiling == 0.02:
                augmented = [True, False]
            rows = []
            for index in range(1_024):
                rows.append({
                    "item_id": f"row-{index}",
                    "battery": "gsm8k" if index % 2 else "mbpp",
                    "panel_group": "target" if index % 2 else "floor",
                    "base_correct": index == 0,
                    "augmented_correct": augmented[index] if index < 2 else False,
                })
            write_jsonl(tmp_path / f"{name}.jsonl", rows)
            write_json(tmp_path / f"{name}_summary.json", {
                "seed": seed,
                "evaluation_gate_ceiling": ceiling,
                "evaluation_gate_ceiling_source": "score_only_fixed_ceiling_override",
                "confirm_scored": False,
                "eval_e_scored": False,
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "panel_sha256": "panel",
            })
    summary, _inputs = build(tmp_path)
    assert summary["paired_ceiling_comparison"]["seed_0"]["net_correct_change"] == 1
    assert summary["scope"]["checkpoint_selection_barred"] is True
