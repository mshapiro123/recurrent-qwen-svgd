from __future__ import annotations

import json

import torch

from eval.cache_paper2_phase3_agreement_oracle import (
    _selected_student_receipts,
    actual_head_equivalence,
    analytic_oracle_directions,
    read_strict_records,
)


def test_selected_hidden_receipts_skip_unneeded_sparse_shards() -> None:
    samples = [
        {"anchor_index": anchor}
        for anchor in (10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33)
    ]
    summary = {
        "model_caches": {
            "student_0p5b": {
                "shards": [
                    {"path": "/cache/rows_000000_000004.pt"},
                    {"path": "/cache/rows_000004_000008.pt"},
                    {"path": "/cache/rows_000008_000012.pt"},
                ]
            }
        }
    }
    selected = _selected_student_receipts(
        summary=summary, samples=samples, needed={20, 31}
    )
    assert [receipt["path"] for receipt in selected] == [
        "/cache/rows_000004_000008.pt",
        "/cache/rows_000008_000012.pt",
    ]
from eval.eval_paper2_phase3_linear_forecast import (
    fit_ridge,
    fit_ridge_family,
    predict_ridge,
)


def test_analytic_oracle_matches_actual_linear_head_autograd() -> None:
    generator = torch.Generator().manual_seed(20260810)
    weight = torch.randn(31, 16, generator=generator)
    hidden = torch.randn(9, 16, generator=generator)
    source = torch.arange(9) + 1
    target = torch.arange(9) + 11
    analytic = analytic_oracle_directions(
        lm_head_weight=weight,
        source_tokens=source,
        target_tokens=target,
    )
    expected = weight.index_select(0, target) - weight.index_select(0, source)
    expected = expected / expected.norm(dim=-1, keepdim=True)
    assert torch.allclose(analytic, expected)
    receipt = actual_head_equivalence(
        hidden=hidden,
        source_tokens=source,
        target_tokens=target,
        lm_head_weight=weight,
    )
    assert receipt["passed"]
    assert receipt["optimized_vs_autograd_max_abs_difference"] < 1e-6


def test_strict_reader_keeps_only_concurrent_disagreements(tmp_path) -> None:
    path = tmp_path / "coverage.jsonl"
    rows = [
        {
            "record_id": "keep",
            "flip_candidate_14b": True,
            "cross_scale_consistent": True,
        },
        {
            "record_id": "uncovered",
            "flip_candidate_14b": True,
            "cross_scale_consistent": False,
        },
        {
            "record_id": "agreement",
            "flip_candidate_14b": False,
            "cross_scale_consistent": True,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert [row["record_id"] for row in read_strict_records(path)] == ["keep"]


def test_shared_eigensystem_ridge_matches_independent_solves() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(80, 12, generator=generator)
    y = torch.randn(80, 5, generator=generator)
    ridges = [0.01, 0.1, 1.0]
    family = fit_ridge_family(x, y, ridges=ridges)
    probe = torch.randn(11, 12, generator=generator)
    for ridge, shared in zip(ridges, family):
        independent = fit_ridge(x, y, ridge=ridge)
        assert torch.allclose(
            predict_ridge(shared, probe),
            predict_ridge(independent, probe),
            atol=2e-4,
            rtol=2e-4,
        )
