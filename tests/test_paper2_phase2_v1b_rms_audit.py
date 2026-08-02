from __future__ import annotations

import pytest

from analysis.audit_paper2_phase2_v1b_rms import (
    deduplicate_position_records,
    recommend_rms_cap,
    sequence_position_bucket,
)


def test_position_records_deduplicate_across_radii() -> None:
    rows = [
        {
            "cohort": "oracle_help",
            "row_index": 3,
            "position": 7,
            "c_value": c_value,
            "state_rms": 2.0,
            "gradient_l2": 4.0,
            "margin_before": 1.0,
            "stratum": "code",
        }
        for c_value in (0.01, 0.02, 0.05)
    ]
    assert len(deduplicate_position_records(rows)) == 1


def test_position_deduplication_rejects_invariant_drift() -> None:
    rows = [
        {
            "cohort": "oracle_help",
            "row_index": 3,
            "position": 7,
            "c_value": 0.01,
            "state_rms": 2.0,
            "gradient_l2": 4.0,
            "margin_before": 1.0,
            "stratum": "code",
        },
        {
            "cohort": "oracle_help",
            "row_index": 3,
            "position": 7,
            "c_value": 0.05,
            "state_rms": 3.0,
            "gradient_l2": 4.0,
            "margin_before": 1.0,
            "stratum": "code",
        },
    ]
    with pytest.raises(RuntimeError, match="invariant drift"):
        deduplicate_position_records(rows)


def test_rms_cap_prefers_p99_for_attention_sink_tail() -> None:
    recommendation = recommend_rms_cap(
        median=2.0,
        p99=8.0,
        high_rms_positions=[0, 1, 2, 20],
        tail_hurt_rate=0.0,
        body_hurt_rate=0.0,
    )
    assert recommendation["form"] == "p99_state_rms_cap"
    assert recommendation["value"] == 8.0


def test_sequence_position_buckets_are_stable() -> None:
    assert sequence_position_bucket(0, 100) == "position_0"
    assert sequence_position_bucket(3, 100) == "positions_1_3"
    assert sequence_position_bucket(12, 100) == "early_quartile"
    assert sequence_position_bucket(60, 100) == "middle_half"
    assert sequence_position_bucket(90, 100) == "late_quartile"
