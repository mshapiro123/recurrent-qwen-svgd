from __future__ import annotations

import pytest

from eval.analyze_phase_g_oracle_localization import analyze


def arm(default: float, nondefault: float, loop_one: float, loop_four: float) -> dict:
    return {
        "gate_status": "blocked",
        "passed": False,
        "transition_control": {
            "default": {"control_rate": default},
            "nondefault": {"control_rate": nondefault},
            "overall": {"control_rate": 0.25},
            "by_loop_index": {
                "1": {"control_rate": loop_one, "legality_rate": 0.9},
                "4": {"control_rate": loop_four, "legality_rate": 0.2},
            },
        },
    }


def trace(route: str) -> list[dict]:
    return [
        {
            "route": route,
            "step": step,
            "loss": float(11 - step),
            "gradient_norm": float(step),
            "oracle_reentry_residual_rms_ratio": float(step) / 100,
        }
        for step in range(1, 11)
    ]


def test_analyze_reports_gate_and_posthoc_localization() -> None:
    result = analyze(
        {
            "arms": {
                "additive": arm(0.7, 0.1, 0.6, 0.05),
                "film": arm(0.6, 0.2, 0.5, 0.1),
            },
            "measured_reading": "BOTH_FAIL",
            "interpretation": "reentry_conditioning_closed_on_frozen_substrate",
            "automatic_successor_authorized": False,
        },
        {"additive": trace("additive"), "film": trace("film")},
        window=2,
    )

    assert result["registered_reading"] == "BOTH_FAIL"
    assert result["automatic_successor_authorized"] is False
    additive = result["routes"]["additive"]
    assert additive["heldout"]["default_minus_nondefault_control_rate"] == pytest.approx(
        0.6
    )
    assert additive["heldout"]["loop_1_minus_loop_4_control_rate"] == pytest.approx(
        0.55
    )
    assert additive["training_liveness"]["first_window_mean_loss"] == pytest.approx(
        9.5
    )
    assert additive["training_liveness"]["last_window_mean_loss"] == pytest.approx(
        1.5
    )


def test_analyze_rejects_noncontiguous_trace() -> None:
    bad_trace = trace("additive")
    bad_trace[-1]["step"] = 12
    with pytest.raises(ValueError, match="contiguous"):
        analyze(
            {
                "arms": {
                    "additive": arm(0.7, 0.1, 0.6, 0.05),
                    "film": arm(0.6, 0.2, 0.5, 0.1),
                },
                "measured_reading": "BOTH_FAIL",
                "interpretation": "closed",
                "automatic_successor_authorized": False,
            },
            {"additive": bad_trace, "film": trace("film")},
            window=2,
        )
