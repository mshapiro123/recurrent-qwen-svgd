from __future__ import annotations

import torch

from training.paper2_phase3_p32 import GateLabel
from training.paper2_phase3_p33_prep import (
    fixed_random_projection,
    forced_audit_write,
    intervene_state,
    observatory_metrics,
    observatory_event_rows,
    paired_astate_execution,
    prepare_training_rows,
    state_sketches,
)


def _record(index: int, *, positive: bool) -> dict[str, object]:
    return {
        "record_id": f"row-{index}",
        "flip_candidate_14b": positive,
        "cross_scale_consistent": positive,
        "teachability": 0.8 if positive else 0.1,
        "prediction_position": 1,
        "horizon": index % 4 + 1,
        "student_top1_probability": 0.9 - index * 1e-5,
        "teacher_14b_top1_probability": 0.8 - index * 1e-5,
        "teacher_js_divergence": 0.02,
    }


def test_p33_staging_holds_out_audit_and_uses_ranked_three_to_one_negatives() -> None:
    records = [_record(index, positive=True) for index in range(100)]
    records += [_record(100 + index, positive=False) for index in range(340)]
    staged, audit, negative_audit, receipt = prepare_training_rows(
        records, audit_rows=20, negative_audit_rows=30
    )
    assert len(audit) == 20
    assert receipt["train_positive_count"] == 80
    assert receipt["train_negative_count"] == 240
    assert receipt["negative_to_positive_ratio"] == 3.0
    assert len(negative_audit) == 30
    assert receipt["audit_cohorts_disjoint"] is True
    assert receipt["negative_audit_excluded_from_training"] is True
    audit_ids = {row["record_id"] for row in audit}
    assert all(
        row["gate_label"] == int(GateLabel.IGNORED)
        for row in staged
        if row["record_id"] in audit_ids
    )
    negative_audit_ids = {row["record_id"] for row in negative_audit}
    assert audit_ids.isdisjoint(negative_audit_ids)
    assert all(
        row["gate_label"] == int(GateLabel.IGNORED)
        and row["audit_role"] == "negative"
        for row in staged
        if row["record_id"] in negative_audit_ids
    )


def test_fixed_projection_and_observatory_contracts() -> None:
    projection = fixed_random_projection(input_dim=16, output_dim=4)
    assert torch.allclose(projection @ projection.T, torch.eye(4), atol=1e-5)
    states = torch.randn(3, 4, 2, 8)
    writes = torch.randn(3, 3, 5, 12)
    metrics = observatory_metrics(states=states, writes=writes, loss_gradient=writes)
    assert metrics["bridge_write_ratio_r_b"].shape == (3, 3)
    assert metrics["effective_rank"].shape == (4,)
    events = observatory_event_rows(
        record_ids=["a", "b", "c"], metrics=metrics
    )
    assert len(events) == 9
    sketches = state_sketches(states, random_projection=projection)
    assert sketches["fixed_random_projection"].shape == (3, 4, 4)
    state = torch.randn(4, 2, 8)
    random, bypass = intervene_state(state, mode="norm_matched_random", seed=7)
    assert bypass is False
    assert torch.allclose(
        random.flatten(1).norm(dim=1), state.flatten(1).norm(dim=1), rtol=1e-5
    )
    cross, bypass = intervene_state(state, mode="cross_example", seed=9)
    assert bypass is False
    assert torch.allclose(
        cross.flatten(1).norm(dim=1), state.flatten(1).norm(dim=1), rtol=1e-5
    )


def test_astate_reports_unclipped_numerator_and_denominator() -> None:
    state = torch.ones(2, 1, 3)

    def forward(value: torch.Tensor, bypass: bool) -> torch.Tensor:
        return torch.zeros(2) if bypass else value.flatten(1).sum(dim=1)

    result = paired_astate_execution(
        forward,
        cached_state=state,
        baseline_without_recurrence=torch.ones(2),
        mode="bypass",
        seed=0,
    )
    assert result["ratio_clipped"] is False
    assert torch.equal(result["numerator"], torch.tensor([3.0, 3.0]))
    assert torch.equal(result["denominator"], torch.tensor([2.0, 2.0]))


def test_forced_audit_radius_is_not_a_training_gate_factor() -> None:
    delta = torch.randn(3, 4, 8)
    hidden = torch.randn_like(delta)
    write = forced_audit_write(delta, hidden, radius=0.15, rms_cap=0.55)
    expected = (
        hidden.float().square().mean(-1).sqrt().clamp_max(0.55) * 0.15
    )
    observed = write.float().square().mean(-1).sqrt()
    assert torch.allclose(observed, expected, atol=1e-5, rtol=1e-5)
