from __future__ import annotations

from colab.run_stage5_inverse_rehearsal_attribution import compare_natural_canaries


def _summary(values: list[int]) -> dict:
    total = len(values) * 32
    correct = sum(values)
    return {
        "accuracy": correct / total,
        "by_depth": {
            str(index + 1): {"accuracy": value / 32, "correct": value, "total": 32}
            for index, value in enumerate(values)
        },
    }


def test_compare_natural_canaries_separates_inherited_and_incremental_damage() -> None:
    comparison = compare_natural_canaries(
        locked=_summary([32, 32]),
        source=_summary([30, 28]),
        repaired=_summary([31, 27]),
    )

    assert comparison["source_minus_locked"] == -6 / 64
    assert comparison["repaired_minus_source"] == 0.0
    assert comparison["repaired_minus_locked"] == -6 / 64
    assert comparison["by_depth"]["1"]["repaired_minus_source"] == 1 / 32
    assert comparison["by_depth"]["2"]["repaired_minus_source"] == -1 / 32
