from __future__ import annotations

import torch

from eval.eval_paper2_phase2_oracle_overlap import summarize_oracle_overlap


def test_oracle_and_hurt_overlap_are_position_exact() -> None:
    teacher = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6])]
    baseline = [torch.tensor([0, 2, 3, 9]), torch.tensor([5, 0])]
    trained = [torch.tensor([1, 8, 0, 4]), torch.tensor([5, 6])]
    untrained = [torch.tensor([1, 8, 3, 9]), torch.tensor([5, 6])]
    inplace = [torch.tensor([0, 7, 0, 4]), torch.tensor([8, 6])]
    strata = ["general", "code"]

    summary = summarize_oracle_overlap(
        teacher_rows=teacher,
        baseline_rows=baseline,
        trained_rows=trained,
        untrained_rows=untrained,
        inplace_rows=inplace,
        strata=strata,
    )

    pooled = summary["pooled"]
    assert pooled["scored_positions"] == 6
    assert pooled["trained_append"]["helps"] == 3
    assert pooled["trained_append"]["hurts"] == 2
    assert pooled["trained_append"]["oracle_gain"] == 3
    assert pooled["trained_append"]["oracle_correct"] == 6
    assert pooled["untrained_append"]["helps"] == 2
    assert pooled["untrained_append"]["hurts"] == 1

    overlap = pooled["trained_append_vs_inplace_hurts"]
    assert overlap["trained_hurts"] == 2
    assert overlap["inplace_hurts"] == 3
    assert overlap["intersection"] == 2
    assert overlap["union"] == 3
    assert overlap["jaccard"] == 2 / 3
    assert overlap["trained_contained_in_inplace"] == 1.0
    assert overlap["inplace_contained_in_trained"] == 2 / 3


def test_public_summary_contains_aggregates_only() -> None:
    summary = summarize_oracle_overlap(
        teacher_rows=[torch.tensor([1])],
        baseline_rows=[torch.tensor([0])],
        trained_rows=[torch.tensor([1])],
        untrained_rows=[torch.tensor([0])],
        inplace_rows=[torch.tensor([0])],
        strata=["general"],
    )
    serialized = repr(summary)
    assert "row_id" not in serialized
    assert "teacher_token" not in serialized
    assert "prediction" not in serialized
