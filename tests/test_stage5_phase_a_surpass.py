from __future__ import annotations

from colab.stage5_phase_a_surpass import (
    PHASE_A_ROWS_PER_DEPTH,
    fisher_one_sided_greater,
    phase_a_preregistration,
    surpass_gate,
)


def test_fisher_one_sided_greater_prefers_large_a_margin() -> None:
    neutral = fisher_one_sided_greater(64, 64, PHASE_A_ROWS_PER_DEPTH)
    strong = fisher_one_sided_greater(100, 70, PHASE_A_ROWS_PER_DEPTH)

    assert neutral > 0.45
    assert strong < 0.01


def test_surpass_gate_requires_three_consecutive_significant_depths() -> None:
    a_counts = {"8": 80, "9": 100, "10": 101, "11": 99, "12": 82}
    b_counts = {"8": 79, "9": 70, "10": 72, "11": 70, "12": 80}

    gate = surpass_gate(a_counts, b_counts)

    assert gate["pass"] is True
    assert gate["passing_consecutive_depths"] == [9, 10, 11]
    assert gate["per_depth"]["8"]["a_beats_b"] is True
    assert gate["per_depth"]["8"]["p_one_sided_fisher"] > 0.05


def test_phase_a_preregistration_names_same_reader_metric_and_compute_policy() -> None:
    payload = phase_a_preregistration()

    assert payload["kind"] == "stage5_phase_a_surpass_preregistration"
    assert "same-reader final-symbol metric" in payload["arms"]["A_looped"]
    assert "full-model" in payload["arms"]["B_dense_direct"]
    assert "FP32 parameters/moments" in payload["compute_ledger"]["optimizer_protocol"]
    assert "do not claim raw FLOP advantage" in payload["compute_ledger"]["flop_claim_policy"]
