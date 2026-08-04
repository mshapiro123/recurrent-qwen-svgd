from __future__ import annotations

import pytest
import torch

from eval.eval_paper2_phase2_layer_mode_bound import (
    FIT_SEEDS,
    SPREAD_GATE,
    _fit_concat_seed,
    build_concat_design,
    layer_mode_decision,
    resource_plan,
)


def test_resource_plan_forbids_dense_covariance_for_registered_width() -> None:
    plan = resource_plan(rows=166_708, width=15_360, rank=256)
    assert plan["method"] == "randomized_low_rank_svd_streamed_design"
    assert plan["dense_covariance_materialized"] is False
    assert plan["fit_seeds"] == list(FIT_SEEDS)
    assert plan["dense_covariance_bytes_fp64"] > 1_800_000_000


def test_concat_design_rms_normalizes_each_layer_before_stacking() -> None:
    states = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]],
        ]
    )
    design = build_concat_design(states, chunk_size=1)
    reshaped = design.view(2, 3, 2)
    rms = reshaped.square().mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_layer_mode_decision_requires_spread_gate_before_swap() -> None:
    noisy = layer_mode_decision(
        agreement_ci=(0.01, 0.02),
        stratum_deltas={"code": 0.01, "text": 0.01},
        concat_agreements=[0.10, 0.104, 0.101],
    )
    assert noisy["primary"] == "learned_mixture_rrr"
    assert noisy["swap_branch_expired"]
    assert noisy["reason"] == "concat_fit_spread_exceeds_0p25pp_noise_gate"

    passing = layer_mode_decision(
        agreement_ci=(0.005, 0.02),
        stratum_deltas={"code": 0.01, "text": 0.006},
        concat_agreements=[0.10, 0.101, 0.102],
    )
    assert passing["primary"] == "concat_rrr"
    assert passing["authorize_tucker_layer_rank_2"]
    assert passing["max_within_concat_agreement_range"] < SPREAD_GATE


def test_layer_mode_decision_keeps_locked_arm_when_ci_misses_half_point() -> None:
    decision = layer_mode_decision(
        agreement_ci=(0.0049, 0.02),
        stratum_deltas={"code": 0.02, "text": 0.01},
        concat_agreements=[0.10, 0.101, 0.102],
    )
    assert decision["primary"] == "learned_mixture_rrr"
    assert not decision["authorize_tucker_layer_rank_2"]


def test_spread_gate_is_exactly_quarter_point() -> None:
    assert SPREAD_GATE == pytest.approx(0.0025)


def test_concat_fit_writes_resume_safe_calibration_and_holdout_metrics(tmp_path) -> None:
    torch.manual_seed(41)
    calibration_rows = 24
    holdout_rows = 8
    width = 6
    hidden = 12
    topk = 3
    vocab = 20
    calibration_design = torch.randn(calibration_rows, width)
    holdout_design = torch.randn(holdout_rows, width)
    calibration_targets = torch.randn(calibration_rows, 8 * 128)
    calibration_hidden = torch.randn(calibration_rows, hidden).to(torch.bfloat16)
    calibration_horizons = torch.randint(1, 5, (calibration_rows,))
    calibration_ids = torch.randint(0, vocab, (calibration_rows, topk))
    calibration_log_probs = torch.randn(calibration_rows, topk).to(torch.bfloat16)
    holdout_targets = {
        "horizons": torch.randint(1, 5, (holdout_rows,)),
        "topk_ids": torch.randint(0, vocab, (holdout_rows, topk)),
        "topk_log_probs": torch.randn(holdout_rows, topk).to(torch.bfloat16),
    }
    cache = tmp_path / "concat.pt"
    kwargs = {
        "seed": FIT_SEEDS[0],
        "calibration_design": calibration_design,
        "holdout_design": holdout_design,
        "calibration_targets": calibration_targets,
        "calibration_hidden": calibration_hidden,
        "calibration_horizons": calibration_horizons,
        "calibration_topk_ids": calibration_ids,
        "calibration_topk_log_probs": calibration_log_probs,
        "holdout_targets": holdout_targets,
        "lm_head": torch.randn(vocab, hidden).to(torch.bfloat16),
        "cache_path": cache,
    }

    first = _fit_concat_seed(**kwargs)
    second = _fit_concat_seed(**kwargs)

    assert cache.is_file()
    assert first["protocol_version"] == "layer_mode_bound_randomized_rrr_r2"
    assert first["calibration_metrics"]["teacher_top1"].shape == (calibration_rows,)
    assert first["metrics"]["teacher_top1"].shape == (holdout_rows,)
    assert torch.equal(first["metrics"]["future_kl"], second["metrics"]["future_kl"])
