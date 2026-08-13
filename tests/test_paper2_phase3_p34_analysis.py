from analysis.build_paper2_phase3_p34_campaign_handoff import (
    exact_sign_test,
    minimum_share_margin,
)


def test_exact_sign_test_is_symmetric_and_neutral_without_discordance() -> None:
    assert exact_sign_test(0, 0) == 1.0
    assert exact_sign_test(53, 44) == exact_sign_test(44, 53)


def test_share_margin_applies_lower_floors_and_preserve_ceiling() -> None:
    event = {
        "read": {
            "bounds": {"aim": 0.15, "kl": 0.35, "preserve": 0.25},
            "shares": {"aim": 0.20, "kl": 0.34, "preserve": 0.10},
        }
    }
    assert abs(minimum_share_margin(event) + 0.01) < 1e-12
