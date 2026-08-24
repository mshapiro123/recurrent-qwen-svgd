from analysis.analyze_paper2_bicameral_w1_generation import exact_mcnemar_p


def test_exact_mcnemar_handles_no_discordance() -> None:
    assert exact_mcnemar_p(0, 0) == 1.0


def test_exact_mcnemar_is_symmetric() -> None:
    assert exact_mcnemar_p(9, 2) == exact_mcnemar_p(2, 9)


def test_exact_mcnemar_detects_one_sided_change() -> None:
    assert exact_mcnemar_p(10, 0) == 2 / (2**10)
