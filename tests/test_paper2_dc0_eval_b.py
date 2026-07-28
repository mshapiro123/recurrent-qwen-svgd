from __future__ import annotations

import pytest

from training.paper2_dc0 import (
    assert_eval_b_document_disjoint,
    eval_b_document_manifest,
    layer_application_costs,
)
from eval.eval_paper2_dc0_depth_by_append import (
    anchor_registered_k0,
    cluster_bootstrap_log_ratio,
    group_batches,
    transition_counts,
)

import torch


def _row(document_id: str, row_id: str) -> dict[str, object]:
    return {
        "document_id": document_id,
        "row_id": row_id,
        "input_ids": [1, 2, 3],
        "token_count": 3,
        "stratum": "general",
    }


def test_eval_b_manifest_hashes_sorted_document_ids() -> None:
    first = eval_b_document_manifest([_row("b", "2"), _row("a", "1")])
    second = eval_b_document_manifest([_row("a", "1"), _row("b", "2")])
    assert first["document_id_list_sha256"] == second["document_id_list_sha256"]
    assert first["documents"] == 2


def test_eval_b_fails_closed_on_prior_document_overlap() -> None:
    with pytest.raises(RuntimeError, match="overlaps prior D0 documents"):
        assert_eval_b_document_disjoint(
            [_row("new", "1"), _row("reused", "2")],
            prior_document_ids={"reused"},
        )


def test_dc0_layer_application_costs_match_registered_surgery() -> None:
    costs = layer_application_costs()
    assert costs["loop_one_total_per_position"] == 24
    assert costs["inplace_extra_per_loop"] == 12
    assert costs["append_extra_per_slot"] == 24
    assert costs["append_to_inplace_first_marginal_ratio"] == 2.0


def test_dc0_transition_counts_identical_help_hurt_definition() -> None:
    teacher = torch.tensor([1, 1, 1, 1])
    before = torch.tensor([0, 1, 1, 0])
    after = torch.tensor([1, 0, 1, 0])
    result = transition_counts(before, after, teacher)
    assert result["helps"] == 1
    assert result["hurts"] == 1
    assert result["neutral"] == 2


def test_dc0_group_batches_never_mix_sequence_lengths() -> None:
    rows = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5]},
        {"input_ids": [6, 7, 8]},
    ]
    batches = group_batches(rows, batch_size=2)
    assert all(len({len(rows[index]["input_ids"]) for index in batch}) == 1 for batch in batches)


def test_dc0_append_grid_anchors_registered_k0_and_receipts_cached_disagreement() -> None:
    registered_k0 = torch.tensor([1, 2, 3, 4])
    cached_grid = torch.tensor(
        [
            [1, 5, 6, 7],
            [9, 8, 7, 6],
            [3, 2, 1, 0],
            [8, 4, 3, 2],
        ]
    )

    anchored, cached_k0, diagnostics = anchor_registered_k0(
        cached_grid,
        registered_k0,
    )

    assert torch.equal(anchored[:, 0], registered_k0)
    assert torch.equal(anchored[:, 1:], cached_grid[:, 1:])
    assert torch.equal(cached_k0, cached_grid[:, 0])
    assert diagnostics == {
        "positions": 4,
        "prediction_disagreements": 2,
        "prediction_disagreement_rate": 0.5,
        "primary_k0_source": "registered_full_sequence_depth_1",
        "append_k_positive_source": "incremental_cache_append",
    }
    assert torch.equal(cached_grid[:, 0], torch.tensor([1, 9, 3, 8]))


def test_cluster_bootstrap_reports_row_cluster_interval() -> None:
    result = cluster_bootstrap_log_ratio([(1, 3), (2, 5), (1, 4)], draws=100)
    assert result["cluster_unit"] == "source_row"
    assert result["ratio_ci95"][0] > 1.0
