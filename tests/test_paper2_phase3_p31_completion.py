from __future__ import annotations

from pathlib import Path

import pytest

from training.paper2_phase3_p31 import ALL_BATTERIES
from training.paper2_phase3_p31_completion import (
    build_sentinel_panel,
    reference_score_table,
    seal_confirm_membership,
    verified_stratum_counts,
)


def _ledger() -> dict[str, object]:
    return {
        "scores_computed": False,
        "confirm_scoring_spent": False,
        "split_seed": 20260809,
        "complete_ledger_sha256": "a" * 64,
        "counts": {battery: {"confirm": 2} for battery in ALL_BATTERIES},
        "partition_hashes": {
            battery: {"confirm": f"{index:064x}"}
            for index, battery in enumerate(ALL_BATTERIES, start=1)
        },
    }


def test_confirm_seals_are_idempotent_but_immutable(tmp_path: Path) -> None:
    first = seal_confirm_membership(
        _ledger(),
        output_dir=tmp_path,
        source_rows_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
    )
    second = seal_confirm_membership(
        _ledger(),
        output_dir=tmp_path,
        source_rows_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
    )
    assert first["seal_set_sha256"] == second["seal_set_sha256"]
    changed = _ledger()
    changed["counts"]["gsm8k"]["confirm"] = 3
    with pytest.raises(RuntimeError, match="seal changed"):
        seal_confirm_membership(
            changed,
            output_dir=tmp_path,
            source_rows_sha256="b" * 64,
            source_manifest_sha256="c" * 64,
        )


def test_reference_table_excludes_floor_from_headline_and_uses_delta_only() -> None:
    rows = []
    for battery in ALL_BATTERIES:
        role = "floor_retention_only" if battery in {"arc_easy", "mmlu", "tier1"} else "target"
        for index in range(6):
            rows.append(
                {
                    "battery": battery,
                    "battery_role": role,
                    "partition": "dev",
                    "document_id": f"{battery}-{index}",
                    "item_id": f"{battery}-{index}",
                    "base_correct": index % 2 == 0,
                    "teacher_14b_correct": index % 2 == 0,
                }
            )
    table = reference_score_table(rows, bootstrap_replicates=100)
    assert set(table["headline_batteries"]) == {"gsm8k", "mbpp", "arc_challenge"}
    assert table["batteries"]["gsm8k"]["reporting_rule"] == "delta_only_denominator_ci_touches_zero"


def test_verified_counts_and_sentinel_are_exact_and_confirm_free() -> None:
    rows = []
    scores = []
    batteries = ["arc_easy", "arc_challenge", "mmlu", "gsm8k", "mbpp"]
    for index in range(500):
        battery = batteries[index % len(batteries)]
        partition = "dev" if index % 4 == 0 else "verified_train"
        row = {
            "battery": battery,
            "battery_role": "target_primary",
            "partition": partition,
            "document_id": f"doc-{index}",
            "item_id": f"item-{index}",
            "content_sha256": f"{index:064x}",
        }
        rows.append(row)
        scores.append(
            {
                **row,
                "base_correct": index % 3 == 0,
                "teacher_14b_correct": index % 3 != 1,
            }
        )
    verified = verified_stratum_counts(
        [row for row in scores if row["partition"] == "verified_train"]
    )
    assert verified["counts_all"]["total"] == 375
    panel, receipt = build_sentinel_panel(rows, scored_rows=scores, size=60)
    assert len(panel) == 60
    assert receipt["confirm_rows"] == 0
    assert set(receipt["cohort_counts"]) == {
        "consensus_no_op",
        "stable_missing_knowledge",
        "procedural_reasoning",
        "mixed",
        "paired_counterfactuals",
        "paraphrase_ood",
    }
