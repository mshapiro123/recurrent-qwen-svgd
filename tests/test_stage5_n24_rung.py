from __future__ import annotations

from colab import stage5_n24_rung as rung


def cell(correct: int, total: int = 128) -> dict[str, float | int]:
    return {"correct": correct, "total": total, "accuracy": correct / total}


def test_n24_locked_gate_constants() -> None:
    gates = rung.locked_gate_summary()

    assert gates["n_symbols"] == 24
    assert gates["support_depth"] == 12
    assert gates["max_eval_depth"] == 22
    assert gates["strong_depths"] == [16, 17]
    assert gates["strong_scaling_min_correct"] == 91
    assert gates["chance_rejection_min_correct"] == 10
    assert gates["nonregression_floors"]["8"] == 0.93
    assert gates["nonregression_floors"]["12"] == 0.85


def test_n24_score_detects_strong_four_point_law() -> None:
    active_summary = {
        "active_matrix": {
            **{str(depth): {str(depth): cell(120)} for depth in range(1, 13)},
            **{str(depth): {str(depth): cell(20)} for depth in range(13, 23)},
            "16": {"16": cell(91)},
            "17": {"17": cell(91)},
        }
    }

    score = rung.score_n24_rung(active_summary)

    assert score["nonregression_pass"] is True
    assert score["strong_scaling_pass"] is True
    assert score["scaling_pass"] is True
    assert score["verdict"] == "strong_four_point_law"


def test_n24_score_marks_ceiling_when_depth15_below_bar() -> None:
    active_summary = {
        "active_matrix": {
            **{str(depth): {str(depth): cell(120)} for depth in range(1, 13)},
            "13": {"13": cell(52)},
            "14": {"14": cell(40)},
            "15": {"15": cell(80)},
            "16": {"16": cell(60)},
            "17": {"17": cell(12)},
        }
    }

    score = rung.score_n24_rung(active_summary)

    assert score["asymptote_broken"] is True
    assert score["strong_scaling_pass"] is False
    assert score["verdict"] == "law_broken_or_ceiling_at_14"


def test_tier1_canary_policy_hard_stops_on_accuracy_or_ppl_damage() -> None:
    assert rung.tier1_canary_verdict(accuracy_delta=-0.031, ppl_relative_delta=0.0)["status"] == "red_hard_stop"
    assert rung.tier1_canary_verdict(accuracy_delta=0.0, ppl_relative_delta=0.051)["status"] == "red_hard_stop"
    assert rung.tier1_canary_verdict(accuracy_delta=-0.02, ppl_relative_delta=0.0)["status"] == "yellow_review"
    assert rung.tier1_canary_verdict(accuracy_delta=0.0, ppl_relative_delta=0.0)["status"] == "green_continue"
