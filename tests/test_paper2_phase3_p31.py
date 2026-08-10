from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.paper2_phase3_p31 import (
    ALL_BATTERIES,
    PairedNullDesign,
    build_split_ledger,
    claim_partition_lease,
    eval_half,
    paired_false_stop,
    paired_upper_confidence_bound,
    simulate_false_stop_probability,
    update_partition_lease,
)


def _row(battery: str, index: int, native_split: str) -> dict[str, object]:
    return {
        "battery": battery,
        "item_id": f"{battery}-{native_split}-{index}",
        "document_id": f"{battery}-doc-{native_split}-{index}",
        "native_split": native_split,
        "prompt": f"prompt {index}",
        "answer": str(index),
        "tests": [f"assert answer == {index}"] if battery == "mbpp" else None,
        "reader": f"{battery}-reader-v1",
        "programmatic_verifier_available": native_split == "train",
    }


def test_split_ledger_is_score_blind_document_disjoint_and_role_locked() -> None:
    rows = []
    for battery in ALL_BATTERIES:
        if battery not in {"arc_easy", "mmlu", "tier1"}:
            rows.extend(_row(battery, index, "train") for index in range(2))
        rows.extend(_row(battery, index, "evaluation") for index in range(8))
    revisions = {battery: f"{battery}-revision" for battery in ALL_BATTERIES}
    readers = {battery: f"{battery}-reader-v1" for battery in ALL_BATTERIES}

    ledger = build_split_ledger(rows, dataset_revisions=revisions, reader_versions=readers)

    assert ledger["status"] == "score_blind_split_frozen"
    assert ledger["scores_computed"] is False
    assert ledger["confirm_scoring_spent"] is False
    assert not any(ledger["document_overlap"].values())
    assert ledger["battery_roles"]["gsm8k"] == "target_primary"
    assert ledger["battery_roles"]["arc_easy"] == "floor_retention_only"
    assert sum(ledger["macro_weights"].values()) == pytest.approx(1.0)
    assert eval_half("stable-document") == eval_half("stable-document")


def test_split_ledger_rejects_cross_partition_document_reuse() -> None:
    rows = [
        _row(battery, 0, "train")
        for battery in ALL_BATTERIES
        if battery not in {"arc_easy", "mmlu", "tier1"}
    ]
    evaluation = _row("gsm8k", 1, "evaluation")
    evaluation["document_id"] = rows[0]["document_id"]
    rows.append(evaluation)
    revisions = {battery: f"{battery}-revision" for battery in ALL_BATTERIES}
    readers = {battery: f"{battery}-reader-v1" for battery in ALL_BATTERIES}
    with pytest.raises(ValueError, match="document partition overlap"):
        build_split_ledger(rows, dataset_revisions=revisions, reader_versions=readers)


def test_floor_only_battery_cannot_feed_training_objective() -> None:
    rows = [_row("arc_easy", 0, "train")]
    revisions = {battery: f"{battery}-revision" for battery in ALL_BATTERIES}
    readers = {battery: f"{battery}-reader-v1" for battery in ALL_BATTERIES}
    with pytest.raises(ValueError, match="floor-only battery"):
        build_split_ledger(rows, dataset_revisions=revisions, reader_versions=readers)


def test_confirm_lease_is_atomic_and_spent_when_scoring_starts(tmp_path: Path) -> None:
    path = tmp_path / "confirm_lease.json"
    lease = claim_partition_lease(
        path,
        partition="confirm",
        run_id="registered-pass",
        preregistration_sha256="a" * 64,
    )
    with pytest.raises(RuntimeError, match="already exists"):
        claim_partition_lease(
            path,
            partition="confirm",
            run_id="second-pass",
            preregistration_sha256="a" * 64,
        )
    updated = update_partition_lease(path, lease, scoring_started=True)
    assert updated["confirm_scoring_spent"] is True


def test_paired_stop_requires_two_consecutive_upper_bounds_below_margin() -> None:
    safe = np.array([-0.03] * 100 + [0.03] * 100)
    harmful = np.array([-0.10] * 180 + [0.0] * 20)
    assert paired_upper_confidence_bound(harmful, alpha=0.01) < -0.03
    one = paired_false_stop([safe, harmful, safe], alpha=0.01)
    two = paired_false_stop([safe, harmful, harmful], alpha=0.01)
    assert one["stopped"] is False
    assert two["stopped"] is True
    assert two["stop_look"] == 3


def test_false_stop_simulation_is_seeded_and_reports_conservative_target() -> None:
    design = PairedNullDesign(rows=64, looks=4, adjacent_correlation=0.5)
    first = simulate_false_stop_probability(
        design, alpha=0.001, campaigns=200, seed=20260809, batch_campaigns=50
    )
    second = simulate_false_stop_probability(
        design, alpha=0.001, campaigns=200, seed=20260809, batch_campaigns=50
    )
    assert first == second
    assert first["consecutive_looks_required"] == 2
    assert first["target_probability"] == 1e-4
