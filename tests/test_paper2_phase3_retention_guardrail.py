from __future__ import annotations

from eval.eval_paper2_phase3_retention_guardrail import calibrate


def test_retention_guardrail_is_init_relative_and_estimator_specific() -> None:
    rows = []
    for seed in (0, 1):
        for index in range(1024):
            rows.append(
                {
                    "seed": seed,
                    "record_id": f"{seed}-{index}",
                    "retained": index % 100 != 0,
                }
            )
    prior = {
        "binding_noise_model_candidate": {
            "paired_discordant_probability": 0.006,
            "adjacent_checkpoint_autocorrelation": 0.7,
        }
    }
    result = calibrate(
        step0_rows=rows,
        prior_empirical_summary=prior,
        panel_sha256="abc",
        campaigns=20_000,
        looks=20,
        seed=7,
        tier_s_fwer=0.001,
    )
    assert result["threshold_reference"] == "step0_init_relative"
    assert result["task_level_capability_scoring"] is False
    assert result["superseded_task_scale_thresholds"][
        "delta_cat_minus_8p5_points"
    ] == "void_for_p33"
    for tier in ("tier_s", "tier_w"):
        assert set(result[tier]["requested_power"]) == {
            "drop_0.005",
            "drop_0.010",
            "drop_0.020",
        }
    assert all(result["assertions"].values())
