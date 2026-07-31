from __future__ import annotations

import pytest
import torch

from eval.eval_paper2_dc0_depth_by_append import parameter_fingerprint
from eval.eval_paper2_dc1_preflight import assert_frozen_instance_unchanged

from training.paper2_dc1 import (
    DEV_C_TOKENS,
    EVAL_C_SEED,
    EVAL_C_TOKENS,
    PREFLIGHT_POSITION_BUDGET,
    assert_dc1_document_disjoint,
    dc1_preflight_spec,
    eval_c_freeze_receipt,
    scale_interpolation_schedule,
)


def test_dc1_partition_and_probe_sizes_are_locked() -> None:
    assert DEV_C_TOKENS == 500_000
    assert EVAL_C_TOKENS == 200_000
    assert EVAL_C_SEED == 20260730
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


def test_dc1_parameter_integrity_is_scoped_to_each_loaded_instance() -> None:
    first = torch.nn.Linear(3, 2, bias=False)
    second = torch.nn.Linear(3, 2, bias=False)
    first_before = parameter_fingerprint(first)
    second_before = parameter_fingerprint(second)

    assert first_before != second_before
    assert_frozen_instance_unchanged(first, before=first_before, instance="first")
    assert_frozen_instance_unchanged(second, before=second_before, instance="second")

    with torch.no_grad():
        second.weight[0, 0].add_(1.0)
    with pytest.raises(RuntimeError, match="mutated frozen second parameters"):
        assert_frozen_instance_unchanged(second, before=second_before, instance="second")


def test_eval_c_public_freeze_receipt_is_hash_only_and_unspent() -> None:
    receipt = eval_c_freeze_receipt(
        source_revisions={"general": "frozen", "code": "frozen"},
        data_jsonl_sha256="a" * 64,
        private_manifest_sha256="b" * 64,
        teacher_cache_sha256="c" * 64,
        disjointness={"document_disjoint": True, "overlap_count": 0},
        teacher_model="teacher",
        teacher_revision="revision",
    )

    assert receipt["status"] == "complete_unread_unscored"
    assert receipt["scores_exposed"] is False
    assert receipt["read_once_scoring_spent"] is False
    assert receipt["training_started"] is False
    assert receipt["optimizer_steps"] == 0
    serialized = str(receipt)
    for prohibited in ("accepted_positions", "rejected_positions", "accuracy", "agreement"):
        assert prohibited not in serialized
