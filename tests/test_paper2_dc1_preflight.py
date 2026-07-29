from __future__ import annotations

import pytest

from training.paper2_dc1 import (
    DEV_C_TOKENS,
    EVAL_C_TOKENS,
    PREFLIGHT_POSITION_BUDGET,
    assert_dc1_document_disjoint,
    dc1_preflight_spec,
    scale_interpolation_schedule,
)


def test_dc1_partition_and_probe_sizes_are_locked() -> None:
    assert DEV_C_TOKENS == 500_000
    assert EVAL_C_TOKENS == 200_000
    assert PREFLIGHT_POSITION_BUDGET == 50_000


def test_dc1_scale_schedule_has_registered_landmarks_and_raw_endpoint() -> None:
    schedule = scale_interpolation_schedule(embedding_rms=0.015, raw_rms=10.0)
    assert [row["label"] for row in schedule] == [
        "matched",
        "3x",
        "10x",
        "30x",
        "100x",
        "300x",
        "raw",
    ]
    assert schedule[0]["target_rms"] == pytest.approx(0.015)
    assert schedule[-1]["target_rms"] == pytest.approx(10.0)
    assert all(left["target_rms"] < right["target_rms"] for left, right in zip(schedule, schedule[1:]))


def test_dc1_disjointness_fails_closed() -> None:
    rows = [{"document_id": "new:a"}, {"document_id": "old:b"}]
    with pytest.raises(RuntimeError, match="overlaps"):
        assert_dc1_document_disjoint(rows, prior_document_ids={"old:b"}, partition="dev_c")


def test_dc1_preflight_is_descriptive_and_does_not_authorize_training() -> None:
    spec = dc1_preflight_spec()
    assert spec["training_authorized"] is False
    assert spec["evaluation_c_touched"] is False
    assert spec["horizontal_append_cap"] == 3
    assert spec["vertical_loops"] == 1
    assert spec["preflight"]["findings_are_gates"] is False
    assert spec["preconditions_before_training"] == [
        "dc1_p_packet_banked",
        "rg4_green",
        "rg11_green_with_declared_precision_policy",
        "dc1_preregistration_locked_to_drive_with_sha256",
    ]

