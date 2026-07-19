from __future__ import annotations

import pytest

from eval.analyze_adapter_budget_segments import analyze_segments


def row(row_id: str, depth: int, hit: bool) -> dict:
    return {"id": row_id, "depth": depth, "same_reader_final_hit": hit}


def test_segment_analysis_reports_paired_crossover() -> None:
    arm_a = [
        row("d1_a", 1, True),
        row("d1_b", 1, False),
        row("d2_a", 2, True),
        row("d2_b", 2, True),
    ]
    arm_e = [
        row("d1_a", 1, True),
        row("d1_b", 1, True),
        row("d2_a", 2, False),
        row("d2_b", 2, False),
    ]

    result = analyze_segments(
        arm_a,
        arm_e,
        segments={"trained": (1,), "tail": (2,), "all": (1, 2)},
    )

    assert result["analysis_status"] == "post_hoc_localization_not_preregistered_gate"
    assert result["segments"]["trained"]["accuracy_delta"] == pytest.approx(0.5)
    assert result["segments"]["trained"]["paired"]["helped"] == 1
    assert result["segments"]["tail"]["accuracy_delta"] == pytest.approx(-1.0)
    assert result["segments"]["tail"]["paired"]["hurt"] == 2
    assert result["segments"]["all"]["paired"]["net_correct"] == -1


def test_segment_analysis_rejects_mismatched_rows() -> None:
    with pytest.raises(ValueError, match="identical row IDs"):
        analyze_segments(
            [row("a", 1, True)],
            [row("b", 1, True)],
            segments={"all": (1,)},
        )
