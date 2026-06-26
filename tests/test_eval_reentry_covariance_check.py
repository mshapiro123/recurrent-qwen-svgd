from __future__ import annotations

from pathlib import Path

import torch

from eval.eval_reentry_covariance_check import covariance_match_check, recommendation, write_markdown


def rotation(theta: float) -> torch.Tensor:
    c = torch.cos(torch.tensor(theta, dtype=torch.double))
    s = torch.sin(torch.tensor(theta, dtype=torch.double))
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def test_covariance_check_identifies_same_spectrum_rotation() -> None:
    entry = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.double))
    r = rotation(0.73)
    exit_cov = r.T @ entry @ r

    result = covariance_match_check(exit_cov, entry)

    assert result["before"] > 0.1
    assert result["after_orthogonal"] < 1e-6
    assert result["after_linear"] < 1e-6
    assert result["spectrum_rel_l2"] < 1e-6


def test_covariance_check_separates_rotation_from_axis_scaling() -> None:
    entry = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.double))
    r = rotation(0.41)
    exit_cov = r.T @ torch.diag(torch.tensor([2.5, 2.5], dtype=torch.double)) @ r

    result = covariance_match_check(exit_cov, entry)

    assert result["before"] > 0.1
    assert result["after_orthogonal"] > 0.1
    assert result["after_linear"] < 1e-6
    assert result["spectrum_rel_l2"] > 0.1


def test_recommendation_prefers_orthogonal_when_spectra_match() -> None:
    result = {
        "before": 0.6,
        "after_orthogonal": 0.05,
        "after_linear": 0.0,
        "spectrum_rel_l2": 0.01,
        "linear_map_condition": 1.2,
    }

    rec = recommendation(result, current_rank=8, subspace_rank=8, max_condition=16.0)

    assert rec["action"] == "orthogonal_directional_adapter"
    assert rec["rank_sufficient_for_subspace"] is True


def test_recommendation_flags_general_linear_and_rank_shortfall() -> None:
    result = {
        "before": 0.6,
        "after_orthogonal": 0.55,
        "after_linear": 0.02,
        "spectrum_rel_l2": 0.8,
        "linear_map_condition": 4.0,
    }

    rec = recommendation(result, current_rank=4, subspace_rank=8, max_condition=16.0)

    assert rec["action"] == "general_linear_directional_adapter"
    assert rec["rank_sufficient_for_subspace"] is False
    assert "current_rank_4_below_subspace_rank_8" in rec["reasons"]


def test_markdown_writer_names_prebuild_gate(tmp_path: Path) -> None:
    summary = {
        "run_id": "run",
        "rank_audit": {"current_adapter_rank": 8, "required_min_rank": 8},
        "recommendation": {"action": "general_linear_directional_adapter", "reasons": ["x"]},
        "projected_covariance_check": {
            "dimension": 8,
            "before": 0.5,
            "after_orthogonal": 0.4,
            "after_linear": 0.01,
            "spectrum_rel_l2": 0.3,
            "linear_map_condition": 3.0,
            "delta_effective_rank": 4.0,
        },
        "full_hidden_covariance_check": {
            "dimension": 896,
            "before": 0.5,
            "after_orthogonal": 0.4,
            "after_linear": 0.01,
            "linear_map_condition": 3.0,
        },
    }

    out = tmp_path / "summary.md"
    write_markdown(summary, out)

    text = out.read_text(encoding="utf-8")
    assert "general_linear_directional_adapter" in text
    assert "pre-build gate" in text
