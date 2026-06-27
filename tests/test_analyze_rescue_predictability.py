from __future__ import annotations

from eval.analyze_rescue_predictability import (
    auc_for_binary,
    binary_gate_sweep,
    discrimination_for_feature,
    label_examples,
)


def toy_examples():
    return [
        {
            "id": "rescued",
            "base_predicted_margin": 0.1,
            "loop_hits": {1: False, 2: True, 3: False},
            "loop_predicted_margins": {1: 0.2, 2: 0.8, 3: 0.5},
            "loop_answer_margins": {1: -0.1, 2: 0.3, 3: -0.2},
            "loop_scores": {1: {"A": 0.0, "B": 0.2}, 2: {"A": 0.8, "B": 0.0}, 3: {"A": 0.0, "B": 0.5}},
            "loop_diagnostics": {1: {"mean_expected_loops": 1.8, "mean_halt_entropy": 0.6}},
        },
        {
            "id": "harmed",
            "base_predicted_margin": 3.0,
            "loop_hits": {1: True, 2: False, 3: False},
            "loop_predicted_margins": {1: 3.0, 2: 0.4, 3: 0.2},
            "loop_answer_margins": {1: 2.0, 2: -0.4, 3: -0.2},
            "loop_scores": {1: {"A": 3.0, "B": 0.0}, 2: {"A": 0.0, "B": 0.4}, 3: {"A": 0.0, "B": 0.2}},
            "loop_diagnostics": {1: {"mean_expected_loops": 1.1, "mean_halt_entropy": 0.2}},
        },
        {
            "id": "stable_correct",
            "base_predicted_margin": 2.5,
            "loop_hits": {1: True, 2: True, 3: True},
            "loop_predicted_margins": {1: 2.0, 2: 2.1, 3: 2.2},
            "loop_answer_margins": {1: 2.0, 2: 2.1, 3: 2.2},
            "loop_scores": {1: {"A": 2.0, "B": 0.0}, 2: {"A": 2.1, "B": 0.0}, 3: {"A": 2.2, "B": 0.0}},
            "loop_diagnostics": {1: {"mean_expected_loops": 1.2, "mean_halt_entropy": 0.2}},
        },
        {
            "id": "stable_wrong",
            "base_predicted_margin": 1.0,
            "loop_hits": {1: False, 2: False, 3: False},
            "loop_predicted_margins": {1: 1.0, 2: 1.1, 3: 0.9},
            "loop_answer_margins": {1: -1.0, 2: -1.1, 3: -0.9},
            "loop_scores": {1: {"A": 0.0, "B": 1.0}, 2: {"A": 0.0, "B": 1.1}, 3: {"A": 0.0, "B": 0.9}},
            "loop_diagnostics": {1: {"mean_expected_loops": 1.4, "mean_halt_entropy": 0.3}},
        },
    ]


def test_auc_for_binary_handles_perfect_low_predicts_positive() -> None:
    auc = auc_for_binary([0.0, 1.0, 2.0, 3.0], [True, True, False, False])
    assert auc == 0.0


def test_label_examples_marks_rescue_and_harm() -> None:
    labelled = label_examples(toy_examples(), [1, 2, 3])

    assert [row["category"] for row in labelled] == [
        "rescuable",
        "harmable",
        "stable_correct",
        "stable_wrong",
    ]


def test_discrimination_reports_oriented_auc() -> None:
    labelled = label_examples(toy_examples(), [1, 2, 3])
    row = discrimination_for_feature(labelled, "base_predicted_margin", positive_label="rescuable")

    assert row["auc"] == 0.0
    assert row["oriented_auc"] == 1.0
    assert row["direction"] == "low_predicts_positive"


def test_binary_gate_can_capture_rescue_without_harm() -> None:
    labelled = label_examples(toy_examples(), [1, 2, 3])
    gates = binary_gate_sweep(labelled, [1, 2, 3], ["base_predicted_margin"])

    best = gates[0]
    assert best["delta_vs_loop1"] >= 1
    assert best["rescue_captured"] >= 1
    assert best["harm_triggered"] == 0
