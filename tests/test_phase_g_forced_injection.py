from __future__ import annotations

from training.phase_g_forced_injection_spec import (
    LOCKED_INJECTION_FACTORS,
    score_forced_injection_probe,
    summarize_factor_rows,
)


def factor_metrics(switching: int, validity: float, fidelity: float = 0.25) -> dict:
    return {
        "switching_groups": switching,
        "K1_validity": validity,
        "selected_target_fidelity": fidelity,
    }


def arm(label: str, points: list[tuple[int, float]]) -> dict:
    return {
        "label": label,
        "factors": {
            f"{factor:g}": factor_metrics(switching, validity)
            for factor, (switching, validity) in zip(
                LOCKED_INJECTION_FACTORS,
                points,
            )
        },
    }


def test_channel_exists_if_either_arm_switches_while_valid() -> None:
    quiet = [(4, 0.75)] * 5
    responsive = [(4, 0.75), (8, 0.74), (16, 0.70), (20, 0.65), (24, 0.55)]

    result = score_forced_injection_probe(
        [arm("kl_0p001", quiet), arm("kl_0p0001_confirmation", responsive)]
    )

    assert result["measured_verdict"] == "CHANNEL-EXISTS"
    assert result["authorization"] == "successor_spec_authorized"


def test_no_channel_if_switching_stays_below_eight() -> None:
    quiet = [(4, 0.75), (5, 0.74), (6, 0.70), (7, 0.65), (7, 0.55)]

    result = score_forced_injection_probe(
        [arm("kl_0p001", quiet), arm("kl_0p0001_confirmation", quiet)]
    )

    assert result["measured_verdict"] == "NO-CHANNEL"
    assert result["authorization"] == "closed"


def test_no_channel_if_switching_appears_only_after_validity_collapse() -> None:
    collapsed = [(4, 0.75), (8, 0.60), (16, 0.49), (20, 0.30), (24, 0.10)]

    result = score_forced_injection_probe(
        [arm("kl_0p001", collapsed), arm("kl_0p0001_confirmation", collapsed)]
    )

    assert result["measured_verdict"] == "NO-CHANNEL"


def test_intermediate_outcome_is_ambiguous_and_closed_by_default() -> None:
    intermediate = [(4, 0.75), (8, 0.70), (12, 0.65), (15, 0.60), (15, 0.55)]

    result = score_forced_injection_probe(
        [arm("kl_0p001", intermediate), arm("kl_0p0001_confirmation", intermediate)]
    )

    assert result["measured_verdict"] == "AMBIGUOUS"
    assert result["authorization"] == "closed_by_default"


def test_factor_summary_counts_switching_fidelity_and_validity() -> None:
    rows = []
    for group in range(32):
        variants = 4 if group < 10 else 3
        for variant in range(variants):
            prediction = "A" if variant == 0 or group >= 16 else "B"
            rows.append(
                {
                    "base_problem_id": f"group_{group}",
                    "target": "A" if variant % 2 == 0 else "B",
                    "prediction": prediction,
                    "factor_1_prediction": "A",
                    "reachable_symbols": ["A", "B"],
                    "phase_g_metrics": {"phase_g_injection_scale": 0.1},
                }
            )
    assert len(rows) == 106

    summary = summarize_factor_rows(rows)

    assert summary["groups"] == 32
    assert summary["switching_groups"] == 16
    assert summary["K1_validity"] == 1.0
    assert summary["changed_from_factor_1"] > 0
