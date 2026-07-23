from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from training.oracle_train_readout_spec import (
    MATCHED_GROUPS,
    MATCHED_ROWS,
    MATCHED_TRANSITIONS,
    classify_nondefault_control,
    preregistration_payload,
    row_id_sha256,
    select_matched_training_rows,
)
from training.oracle_interface_probe_spec import summarize_oracle_arm


ROOT = Path(__file__).resolve().parents[1]
TRAIN = (
    ROOT
    / "outputs/stage5/stage5_phase_g_multitarget_control_20260718/data/train.jsonl"
)


def read_train() -> list[dict]:
    return [
        json.loads(line)
        for line in TRAIN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_matched_train_subset_exactly_mirrors_heldout_denominators() -> None:
    rows = select_matched_training_rows(read_train())

    assert len(rows) == MATCHED_ROWS
    assert len({row["base_problem_id"] for row in rows}) == MATCHED_GROUPS
    assert sum(row["depth"] for row in rows) == MATCHED_TRANSITIONS
    assert Counter(row["depth"] for row in rows) == {1: 16, 2: 22, 3: 27, 4: 41}
    assert Counter(
        next(row["depth"] for row in rows if row["base_problem_id"] == group)
        for group in {row["base_problem_id"] for row in rows}
    ) == {1: 8, 2: 8, 3: 8, 4: 8}


def test_matched_train_subset_is_seeded_and_deterministic() -> None:
    rows = read_train()
    first = select_matched_training_rows(rows)
    second = select_matched_training_rows(rows)
    alternate = select_matched_training_rows(rows, seed=20260723)

    assert row_id_sha256(first) == row_id_sha256(second)
    assert row_id_sha256(first) != row_id_sha256(alternate)


def test_readout_bands_and_prohibitions_are_locked() -> None:
    preregistration = preregistration_payload()

    assert classify_nondefault_control(0.85) == "fit_seen_command_mapping"
    assert classify_nondefault_control(0.25) == "did_not_fit_command_mapping"
    assert classify_nondefault_control(0.50) == "partial_fit"
    assert preregistration["registered_heldout_verdict_mutable"] is False
    assert "training" in preregistration["prohibitions"]


def test_oracle_summarizer_accepts_explicit_posthoc_denominators() -> None:
    transitions = [
        {
            "id": "a",
            "base_problem_id": "g1",
            "depth": 1,
            "loop_index": 1,
            "command_is_default": True,
            "controlled": True,
            "legal": True,
            "target_margin": 1.0,
        },
        {
            "id": "b",
            "base_problem_id": "g2",
            "depth": 2,
            "loop_index": 1,
            "command_is_default": False,
            "controlled": False,
            "legal": True,
            "target_margin": -1.0,
        },
        {
            "id": "b",
            "base_problem_id": "g2",
            "depth": 2,
            "loop_index": 2,
            "command_is_default": False,
            "controlled": True,
            "legal": True,
            "target_margin": 1.0,
        },
    ]
    terminals = [
        {"id": "a", "base_problem_id": "g1", "valid": True},
        {"id": "b", "base_problem_id": "g2", "valid": True},
    ]

    summary = summarize_oracle_arm(
        transitions,
        terminals,
        route="additive",
        identity_exact=True,
        frozen_lineage_unchanged=True,
        expected_rows=2,
        expected_groups=2,
        expected_transitions=3,
    )

    assert summary["transition_control"]["nondefault"]["total"] == 2
