from __future__ import annotations

import json
from pathlib import Path

from analysis.build_paper2_phase3_p35_depth_discrimination import build


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_depth_builder_separates_registered_and_clamped_marginals(tmp_path: Path) -> None:
    for seed in (0, 1):
        for k in range(1, 7):
            rows = [
                {
                    "item_id": f"row-{index}",
                    "battery": "gsm8k" if index < 512 else "mbpp",
                    "augmented_correct": index < 100 + k,
                }
                for index in range(1024)
            ]
            _write_jsonl(tmp_path / f"seed_{seed}_k_{k}.jsonl", rows)
            _write(tmp_path / f"seed_{seed}_k_{k}.json", {
                "flow_loops": k,
                "clamped_extension": k > 4,
                "evaluation_gate_ceiling": 0.02,
                "panel_sha256": "panel",
                "confirm_scored": False,
                "eval_e_scored": False,
                "optimizer_constructed": False,
                "optimizer_steps": 0,
            })
    result = build(tmp_path, {"authority": {"drive_id": "authority"}})
    assert result["registered_k4_marginal_positive_both_seeds"] is True
    assert result["cells"]["seed_0_k_4"]["scope"] == "registered"
    assert result["cells"]["seed_0_k_5"]["scope"] == "exploratory_clamped"
    assert result["marginal_improvement"]["seed_1_k_4_minus_k_3"]["pooled"]["net_rows"] == 1
    assert result["d1_archive_read"]["status"] == "folded_into_k_plus"
