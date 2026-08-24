from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from analysis.analyze_paper2_bicameral_stage0 import (
    blockwise_wht,
    common_mode_receipt,
    cross_fitted_residual_correlation,
    reachable_fraction,
    validate_lock,
)


def test_blockwise_wht_is_orthonormal_and_round_trips() -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn((5, 256), generator=generator)
    transformed = blockwise_wht(values, block_size=128)
    restored = blockwise_wht(transformed, block_size=128)
    assert torch.allclose(restored, values, atol=2e-6, rtol=2e-6)
    assert transformed.square().sum() == pytest.approx(values.square().sum(), rel=2e-6)


def test_reachable_fraction_detects_bandwise_rescaling() -> None:
    generator = torch.Generator().manual_seed(23)
    states = torch.randn((9, 256), generator=generator)
    state_wht = blockwise_wht(states, block_size=128).reshape(9, 2, 128)
    coefficients = torch.linspace(-0.5, 0.8, 128)
    correction_wht = state_wht * coefficients.view(1, 1, -1)
    corrections = blockwise_wht(correction_wht.reshape(9, 256), block_size=128)
    receipt = reachable_fraction(corrections, states, block_size=128)
    assert receipt["rho_reach"] == pytest.approx(1.0, abs=2e-6)
    assert len(receipt["coefficients"]) == 128


def test_common_mode_projection_removes_shared_direction() -> None:
    generator = torch.Generator().manual_seed(29)
    noise = torch.randn((64, 32), generator=generator) * 0.02
    values = noise
    values[:, 0] += 1.0
    receipt = common_mode_receipt(values)
    assert receipt["common_mode_fraction"] > 0.95
    assert abs(receipt["rho_res"]) < 0.03


def test_crossfit_residual_estimator_reports_registered_shape() -> None:
    generator = torch.Generator().manual_seed(31)
    rows, dimensions = 40, 32
    labels = torch.tensor([0] * 20 + [1] * 20)
    cluster_signal = torch.zeros((2, dimensions))
    cluster_signal[0, 0] = 2.0
    cluster_signal[1, 1] = 2.0
    corrections = cluster_signal[labels] + 0.3 * torch.randn(
        (rows, dimensions), generator=generator
    )
    receipt = cross_fitted_residual_correlation(
        corrections,
        labels,
        splits=3,
        seed_base=20260823,
        rank_start=1,
        rank_max=2,
        persistence_fraction=0.8,
    )
    assert receipt["splits"] == 3
    assert set(receipt["terminal_clusters"]) == {"0", "1"}
    assert len(receipt["by_rank"]["1"]["0"]["folds"]) == 6
    assert receipt["same_sample_projection_prohibited"] is True


def test_checked_in_lock_is_stage0_only() -> None:
    lock = json.loads(
        Path("training/paper2_bicameral_stage0_lock.json").read_text(encoding="utf-8")
    )
    validate_lock(lock)
    assert lock["sealed_partitions_authorized"] is False
    assert lock["architecture_reference"]["available_at_lock_materialization"] is True
    assert lock["analysis"]["crossfit_splits"] == 20
