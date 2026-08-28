from __future__ import annotations

import pytest

from eval.eval_paper2_recirculation_phase_a_desk_rider import (
    analyze_arm,
    classify_token_piece,
    first_divergence,
    first_token_transition_summary,
    length_conditioning,
    onset_summary,
    quantile,
)


def _row(
    item_id: str,
    *,
    tokens: list[int],
    margins: list[float],
    correct: bool,
    prediction: str,
    battery: str = "gsm8k",
) -> dict:
    return {
        "item_id": item_id,
        "battery": battery,
        "reader": "reader_v1",
        "augmented_correct": correct,
        "prediction": prediction,
        "generated_text": "generated-" + "-".join(str(token) for token in tokens),
        "generated_token_ids": tokens,
        "generated_tokens": len(tokens),
        "answer_token_margins": margins,
    }


def test_first_divergence_includes_prefix_exhaustion() -> None:
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    assert first_divergence([1, 2], [1, 2]) is None


def test_quantile_uses_linear_order_statistic_interpolation() -> None:
    assert quantile([0.0, 1.0, 2.0, 3.0], 0.25) == pytest.approx(0.75)
    assert quantile([2.0], 0.9) == 2.0
    assert quantile([], 0.5) is None


def test_token_classification_is_exclusive_and_handles_sequence_end() -> None:
    assert classify_token_piece("  ", token_id=1) == "whitespace"
    assert classify_token_piece("12", token_id=2) == "numeric"
    assert classify_token_piece("+", token_id=3) == "operator"
    assert classify_token_piece("12+", token_id=4) == "numeric_operator"
    assert classify_token_piece("<eos>", token_id=5, special_token_ids={5}) == "special"
    assert classify_token_piece(None, token_id=None) == "sequence_end"


def test_arm_analysis_localizes_regression_and_fix_onsets() -> None:
    baseline = [
        _row("reg", tokens=[1, 2, 3, 4], margins=[1.0, 0.1, 0.5, 0.6], correct=True, prediction="4"),
        _row("fix", tokens=[5, 6], margins=[0.8, 0.4], correct=False, prediction="6"),
        _row("stable", tokens=[8, 8, 8], margins=[0.9, 0.8, 0.7], correct=True, prediction="8"),
    ]
    arm = [
        _row("reg", tokens=[1, 9], margins=[1.2, 0.2], correct=False, prediction="9"),
        _row("fix", tokens=[5, 7], margins=[0.7, 0.3], correct=True, prediction="7"),
        _row("stable", tokens=[8, 8, 8], margins=[1.0, 0.9, 0.8], correct=True, prediction="8"),
    ]
    pieces = {1: "a", 2: "2", 3: "b", 4: "4", 5: "x", 6: "word", 7: "+", 8: "z", 9: "+"}
    rows = analyze_arm(
        baseline,
        arm,
        arm_name="test_arm",
        decode_token=lambda token_id: pieces[token_id],
        special_token_ids=set(),
        pooled_baseline_gsm8k_margins=[0.1, 0.4, 0.5, 0.8, 1.0],
        low_margin_threshold=0.4,
    )

    regression = rows[0]
    assert regression["transition"] == "regression"
    assert regression["first_divergence_index_zero_based"] == 1
    assert regression["first_divergence_position_one_based"] == 2
    assert regression["baseline_prefix_fraction_retained"] == 0.25
    assert regression["baseline_onset_top1_margin"] == 0.1
    assert regression["baseline_onset_low_margin_q25"] is True
    assert regression["either_onset_numeric_or_operator"] is True

    fix = rows[1]
    assert fix["transition"] == "fix"
    assert fix["baseline_onset_token_class"] == "lexical"
    assert fix["arm_onset_token_class"] == "operator"

    stable = rows[2]
    assert stable["transition"] == "preserved_correct"
    assert stable["first_divergence_index_zero_based"] is None
    assert stable["baseline_prefix_fraction_retained"] == 1.0

    first_tokens = first_token_transition_summary(rows)
    assert first_tokens["first_token_changed_rows"] == 0
    assert first_tokens["top_transition_pairs"][0]["rows"] == 1
    assert first_tokens["status"] == "post_primary_receipt_descriptive_extension"


def test_identical_tokens_cannot_change_reader_outcome() -> None:
    baseline = [_row("row", tokens=[1], margins=[0.5], correct=True, prediction="yes")]
    arm = [_row("row", tokens=[1], margins=[0.6], correct=False, prediction="no")]
    with pytest.raises(RuntimeError, match="identical token sequence"):
        analyze_arm(
            baseline,
            arm,
            arm_name="bad_arm",
            decode_token=lambda token_id: str(token_id),
            special_token_ids=set(),
            pooled_baseline_gsm8k_margins=[0.5],
            low_margin_threshold=0.5,
        )


def test_onset_and_length_summaries_use_registered_views() -> None:
    rows = [
        {
            "item_id": "a",
            "battery": "gsm8k",
            "transition": "regression",
            "baseline_correct": True,
            "baseline_length": 20,
            "arm_length": 4,
            "arm_to_baseline_length_ratio": 0.2,
            "first_divergence_index_zero_based": 1,
            "first_divergence_position_one_based": 2,
            "baseline_prefix_fraction_retained": 0.05,
            "baseline_onset_top1_margin": 0.1,
            "arm_onset_top1_margin": 0.2,
            "baseline_onset_low_margin_q25": True,
            "either_onset_numeric_or_operator": True,
            "baseline_onset_token_class": "numeric",
            "arm_onset_token_class": "operator",
            "baseline_onset_token_id": 1,
        },
        {
            "item_id": "b",
            "battery": "gsm8k",
            "transition": "preserved_correct",
            "baseline_correct": True,
            "baseline_length": 200,
            "arm_length": 180,
            "arm_to_baseline_length_ratio": 0.9,
            "first_divergence_index_zero_based": 119,
            "first_divergence_position_one_based": 120,
            "baseline_prefix_fraction_retained": 0.595,
            "baseline_onset_top1_margin": 0.8,
            "arm_onset_top1_margin": 0.7,
            "baseline_onset_low_margin_q25": False,
            "either_onset_numeric_or_operator": False,
            "baseline_onset_token_class": "lexical",
            "arm_onset_token_class": "lexical",
            "baseline_onset_token_id": 2,
        },
    ]
    onset = onset_summary(
        rows,
        transition="regression",
        absolute_early_positions=8,
        normalized_early_fraction=0.1,
        normalized_late_fraction=0.5,
    )
    assert onset["absolute_early_rows"] == 1
    assert onset["normalized_early_rows"] == 1
    assert onset["normalized_late_rows"] == 0

    length = length_conditioning(
        rows,
        transition="regression",
        eligibility="baseline_correct",
        bins=[[1, 32], [33, 128], [129, 256]],
    )
    assert length["eligible_rows"] == 2
    assert length["fixed_bins"][0]["event_rate"] == 1.0
    assert length["fixed_bins"][2]["event_rate"] == 0.0
    assert length["nonempty_bin_rates_monotone_non_decreasing"] is False
