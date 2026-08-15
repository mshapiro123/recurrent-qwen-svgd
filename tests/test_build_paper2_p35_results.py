import pytest

from scripts.build_paper2_p35_results import exact_paired_pvalue, paired_binary


def test_exact_paired_pvalue_handles_ties_and_direction() -> None:
    assert exact_paired_pvalue(0, 0) == 1.0
    assert exact_paired_pvalue(0, 10) == pytest.approx(2 / 1024)
    assert exact_paired_pvalue(10, 0) == pytest.approx(2 / 1024)


def test_paired_binary_reports_right_minus_left() -> None:
    result = paired_binary(
        [True, True, False, False],
        [True, False, True, True],
    )
    assert result["left_only"] == 1
    assert result["right_only"] == 2
    assert result["net_right_minus_left"] == 1
