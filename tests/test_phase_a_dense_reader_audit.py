from __future__ import annotations

from eval.audit_phase_a_dense_reader import audit_arm, paired_binary


def test_audit_reports_first_response_correction() -> None:
    source = {
        "r1": {"id": "r1", "mapping": {"A": "B", "B": "C"}, "orbit": ["A", "B"]},
        "r2": {"id": "r2", "mapping": {"A": "B", "B": "C"}, "orbit": ["A", "B", "C"]},
    }
    rows = [
        {
            "id": "r1",
            "depth": 1,
            "target": "B",
            "prediction": "C",
            "correct": False,
            "continuation": " B\nAnswer: C",
        },
        {
            "id": "r2",
            "depth": 2,
            "target": "C",
            "prediction": "A",
            "correct": False,
            "continuation": " steps: B -> C answer: C answer: A",
        },
    ]

    result = audit_arm(rows, source, {"r1": True, "r2": True}, "mixed_test")

    assert result["registered_correct"] == 0
    assert result["corrected_correct"] == 2
    assert result["correct_delta"] == 2
    assert result["by_depth"]["1"]["corrected_correct"] == 1
    assert result["by_depth"]["2"]["corrected_correct"] == 1


def test_paired_binary_reports_checkpoint_extension_direction() -> None:
    result = paired_binary(
        {"a": True, "b": True, "c": False, "d": False},
        {"a": False, "b": True, "c": True, "d": False},
    )

    assert result["left_only"] == 1
    assert result["right_only"] == 1
    assert result["ties"] == 2
    assert result["net_correct"] == 0
    assert result["two_sided_p"] == 1.0
