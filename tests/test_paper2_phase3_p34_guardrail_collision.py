from __future__ import annotations

import json
from pathlib import Path

from eval.eval_paper2_phase3_p34_guardrail_collision import build_receipt


ROOT = Path(__file__).resolve().parents[1]


def test_nested_guardrail_receipt_reproduces_collision_and_repair() -> None:
    lock = json.loads(
        (ROOT / "training/paper2_phase3_p34_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = build_receipt(lock=lock, campaigns=2_000, seed=20260813)
    assert receipt["registered_collision"]["identical_predicates_at_two_consecutive_looks"]
    assert receipt["recommended_nested_rule"]["nested_consequence"]
    assert receipt["simulation"]["candidates"]["streak_4"]["drop_0.055"][
        "estimated_action_probability"
    ] >= 0.98
