from __future__ import annotations

import json
from pathlib import Path

from eval.adjudicate_paper2_stage2b_depth import adjudicate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _seed_tree(
    root: Path,
    seed: int,
    increments: tuple[float, float],
    *,
    heterogeneous_offsets: bool = False,
) -> Path:
    private = root / "private" / f"seed_{seed}"
    public = root / "receipts" / f"seed_{seed}"
    _write_json(
        public / "summary.json",
        {
            "seed": seed,
            "step": 5000,
            "status": "awaiting_step_5000_strategy_adjudication",
            "confirm_scored": False,
            "eval_e_scored": False,
        },
    )
    for look in (3, 4, 5):
        rows = []
        for index in range(8):
            base = float(index) * 0.01
            offset = (-0.1 if index < 4 else 0.1) if heterogeneous_offsets else 0.0
            first_transition = offset + increments[0] * look
            second_transition = offset + increments[1] * look
            rows.append(
                {
                    "seed": seed,
                    "look": look,
                    "item_id": f"row-{index}",
                    "per_loop_mean_teacher_token_margin": [
                        base,
                        base,
                        base + first_transition,
                        base + first_transition + second_transition,
                    ],
                }
            )
        _write_jsonl(private / f"dev2_margin_rows_look_{look}.jsonl", rows)
    _write_json(
        public / "look_5.json",
        {
            "confirm_scored": False,
            "eval_e_scored": False,
            "loop1_kl": 0.1,
            "dev1": {"safety": {"pass": True}},
            "finite_horizon": {"catastrophe": False},
            "r2_desk_read": {"loop1_kl": 0.1},
        },
    )
    return private


def test_adjudication_continues_when_one_seed_separates(tmp_path: Path) -> None:
    seed0 = _seed_tree(tmp_path, 0, (0.02, 0.03))
    seed1 = _seed_tree(tmp_path, 1, (-0.01, -0.01))
    receipt = adjudicate({0: seed0, 1: seed1})
    assert receipt["verdict"] == "continue_m4"
    assert receipt["seed_reads"]["0"]["separating"] is True
    assert receipt["trend_read"] is None
    assert receipt["sealed_partitions_remain_sealed"] is True


def test_adjudication_defers_once_for_positive_trends(tmp_path: Path) -> None:
    # At look five the rows straddle zero, so neither seed separates, while the
    # paired look-3--5 slopes are deterministically positive.
    seed0 = _seed_tree(tmp_path, 0, (0.001, 0.001), heterogeneous_offsets=True)
    seed1 = _seed_tree(tmp_path, 1, (0.001, 0.001), heterogeneous_offsets=True)
    receipt = adjudicate({0: seed0, 1: seed1})
    assert receipt["verdict"] == "defer_once_to_step_8000"
    assert receipt["trend_read"]["positive_trend"] is True
    assert receipt["confirm_scored"] is False
