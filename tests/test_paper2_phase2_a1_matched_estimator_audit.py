from __future__ import annotations

import torch

from eval.eval_paper2_phase2_a1_matched_estimator_audit import (
    calibration_measurement_batches,
    fixed_dev_batches,
    share_contract,
    summarize_norm_rows,
)


def test_calibration_batches_reproduce_original_one_based_measurement_slice() -> None:
    train = torch.arange(200)
    observed = calibration_measurement_batches(
        train,
        seed=3,
        batch_size=8,
        sampled_batches=100,
        first_batch=50,
        last_batch=100,
    )
    generator = torch.Generator().manual_seed(3 + 34001)
    expected = [
        train.index_select(0, torch.randint(train.numel(), (8,), generator=generator))
        for _ in range(100)
    ][49:100]
    assert len(observed) == 51
    assert all(torch.equal(left, right) for left, right in zip(observed, expected))


def test_fixed_dev_batches_are_common_and_deterministic() -> None:
    rows = torch.arange(101)
    first = fixed_dev_batches(rows, batch_size=7, count=51)
    second = fixed_dev_batches(rows, batch_size=7, count=51)
    assert len(first) == 51
    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_amended_share_contract_is_inequality_not_equal_target() -> None:
    passing = {"flow": 0.70, "functional_probe_kl": 0.10, "counterfactual_preserve_kl": 0.20}
    assert share_contract(passing)["joint"]
    assert not share_contract({**passing, "flow": 0.49})["joint"]
    assert not share_contract({**passing, "functional_probe_kl": 0.251})["joint"]


def test_summary_uses_aggregate_norms_and_bootstraps_batches() -> None:
    rows = [
        {"flow": 6.0, "functional_probe_kl": 2.0, "counterfactual_preserve_kl": 2.0}
        for _ in range(51)
    ]
    summary = summarize_norm_rows(
        rows,
        weights={"flow": 1.0, "functional_probe_kl": 1.0, "counterfactual_preserve_kl": 1.0},
        bootstrap_seed=17,
        bootstrap_draws=100,
    )
    assert summary["aggregate_shares"] == {
        "flow": 0.6,
        "functional_probe_kl": 0.2,
        "counterfactual_preserve_kl": 0.2,
    }
    assert summary["aggregate_contract"]["joint"] is True
    assert summary["batch_contract_fraction"]["joint"] == 1.0
    assert summary["bootstrap"]["draws"] == 100
