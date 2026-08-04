from __future__ import annotations

import torch

from eval.eval_paper2_phase2_matched_alpha_audit import (
    demanded_permitted_audit,
    effective_gradient_attribution,
    historical_row_comparison,
    quantile_summary,
    trust_history_audit,
)


def test_quantile_summary_ignores_nonfinite_values() -> None:
    result = quantile_summary(torch.tensor([1.0, 2.0, float("nan"), float("inf")]))
    assert result["count"] == 2
    assert result["mean"] == 1.5
    assert result["p50"] == 1.5
    assert result["max"] == 2.0


def test_effective_gradient_attribution_reports_clipping_and_signed_shares() -> None:
    result = effective_gradient_attribution(
        {
            "a": torch.tensor([3.0, 0.0]),
            "b": torch.tensor([0.0, 4.0]),
        },
        weights={"a": 1.0, "b": 1.0},
        ceiling=2.5,
    )
    assert result["preclip_total_norm"] == 5.0
    assert result["clip_scale"] == 0.5
    assert result["clipped"] is True
    assert abs(result["losses"]["a"]["postclip_norm_share"] - 3.0 / 7.0) < 1e-7
    assert abs(result["losses"]["b"]["postclip_norm_share"] - 4.0 / 7.0) < 1e-7
    assert abs(
        sum(value["signed_update_alignment_share"] for value in result["losses"].values())
        - 1.0
    ) < 1e-7


def test_demanded_permitted_audit_reproduces_registered_conflict() -> None:
    rows = {
        "flow_start": torch.zeros(2, 1, 2),
        "target_scratch": torch.full((2, 1, 2), 2.0),
        "flow_state": torch.zeros(2, 1, 2),
        "endpoint_ratio": torch.tensor([[0.2], [0.4]]),
    }
    result = demanded_permitted_audit(rows)
    assert result["beta"] == 0.5
    assert result["trust_ceiling"] == 0.5
    assert result["fraction_demand_exceeds_permission"] == 0.0
    assert result["demanded_over_permitted"]["p50"] == 1.0
    assert result["demanded_over_permitted_by_beta"]["1.0"][
        "fraction_demand_exceeds_permission"
    ] == 1.0
    assert result["huber_linear_regime_fraction"] == 0.0


def test_trust_history_audit_does_not_invent_missing_training_magnitudes() -> None:
    arm = {
        "history": [
            {"step": 0, "losses": {"flow": 2.0, "trust": 4.0}},
            {"step": 100, "losses": {"flow": 1.0, "trust": 2.0}},
        ]
    }
    checkpoint = {"trust_history": [True] * 51 + [False] * 49}
    result = trust_history_audit(arm, checkpoint)
    assert result["rolling_stop_rule_reproduced"] is True
    assert result["per_training_step_trust_magnitude_recoverable"] is False
    assert len(result["scheduled_evaluation_proxy"]) == 2


def test_historical_row_comparison_reports_missing_artifact_without_fabrication(
    tmp_path,
) -> None:
    arm = {
        "history": [{"step": 100, "quality_noninferior": True}],
        "checkpoint": {"path": str(tmp_path / "resume.pt")},
        "step": 193,
    }
    result = historical_row_comparison(
        arm=arm,
        exact_rows={
            "acceptance_delta": torch.zeros(2),
            "quality_correct": torch.ones(2),
        },
    )
    assert result["available"] is False
    assert result["reason"] == "scheduled row artifact missing"


def test_audit_sources_are_read_only_and_preserve_guardrail_taxonomy() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    evaluator = (root / "eval/eval_paper2_phase2_matched_alpha_audit.py").read_text(
        encoding="utf-8"
    )
    runner = (root / "colab/run_stage5_paper2_phase2_matched_alpha_audit.py").read_text(
        encoding="utf-8"
    )
    combined = evaluator + runner
    assert "torch.optim" not in combined
    assert "optimizer.step" not in combined
    assert '"model_parameter_updates": 0' in evaluator
    assert '"endpoint_qualification_not_catastrophe"' in evaluator
    assert '"pending_empirical_catastrophe_threshold"' in evaluator
    assert '"retention_0p997"' in evaluator
    assert '"shapers"' in evaluator
    assert "Per-training-step trust-rent magnitudes were not stored" in evaluator
    assert "frozen E1 evaluation partition" in combined
    assert "stage5_paper2_phase2_matched_alpha_audit_20260804" in runner
