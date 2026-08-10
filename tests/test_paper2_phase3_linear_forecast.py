from __future__ import annotations

import torch

from eval.eval_paper2_phase3_linear_forecast import document_split, run_forecast


def test_document_split_is_disjoint_and_stable() -> None:
    documents = [f"doc-{index // 4}" for index in range(400)]
    first = document_split(documents, seed=20260810)
    second = document_split(documents, seed=20260810)
    assert torch.equal(first.train, second.train)
    assert torch.equal(first.calibration, second.calibration)
    assert torch.equal(first.holdout, second.holdout)
    sets = [
        {documents[index] for index in values.tolist()}
        for values in (first.train, first.calibration, first.holdout)
    ]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]


def test_linear_forecast_recovers_document_disjoint_direction_map() -> None:
    generator = torch.Generator().manual_seed(20260810)
    rows = 600
    features = torch.randn(rows, 12, generator=generator)
    weight = torch.randn(12, 8, generator=generator)
    directions = features @ weight + 0.01 * torch.randn(rows, 8, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    documents = [f"doc-{index // 3}" for index in range(rows)]
    result = run_forecast(
        features=features,
        directions=directions,
        documents=documents,
        ridge_candidates=[0.001, 0.1, 10.0],
        seed=20260810,
        bootstrap_replicates=200,
    )
    assert result["split"]["document_disjoint"]
    assert result["holdout_cosine"]["mean"] > 0.99
    assert result["holdout_cosine"]["ci95_low"] > 0.99
    assert result["p33_training_authorized"] is False
    assert "neither an upper nor a lower bound" in result["interpretation"]
