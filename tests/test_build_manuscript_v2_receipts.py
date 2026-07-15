from __future__ import annotations

import math
from pathlib import Path

from eval.build_manuscript_v2_receipts import build_receipt, softmax


def test_softmax_is_stable_and_normalized() -> None:
    probabilities = softmax([1000.0, 1001.0, 999.0])
    assert math.isclose(sum(probabilities), 1.0)
    assert probabilities[1] == max(probabilities)


def test_receipt_crosses_depth_and_stratum() -> None:
    rows = [
        {
            "depth": 1,
            "prediction": "A",
            "reachable_set_size": 2,
            "reachable_set_stratum": "2",
            "scores": {"A": 2.0, "B": 1.0},
            "valid": True,
        },
        {
            "depth": 2,
            "prediction": "B",
            "reachable_set_size": 3,
            "reachable_set_stratum": "3-4",
            "scores": {"A": 1.0, "B": 2.0},
            "valid": False,
        },
    ]
    receipt = build_receipt(rows, source=Path("rows.jsonl"))
    assert receipt["overall"]["valid"] == 1
    assert receipt["by_depth_and_stratum"]["depth_1__stratum_2"]["validity"] == 1.0
    assert receipt["by_depth_and_stratum"]["depth_2__stratum_3-4"]["validity"] == 0.0
    assert receipt["reachable_set_size_distribution"]["2"]["rows"] == 1
