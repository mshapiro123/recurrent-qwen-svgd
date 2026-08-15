from __future__ import annotations

import torch

from eval.eval_paper2_phase3_p34_oracle_refresh import direction_refresh_read


def test_direction_refresh_distinguishes_registered_and_persistent_sources() -> None:
    torch.manual_seed(3)
    weight = torch.randn(11, 8)
    rows = [
        {
            "population": "positive",
            "base_top1": 1,
            "cached_student_top1": 1,
            "deployed_top1": 2,
            "teacher_top1": 3,
        },
        {
            "population": "positive",
            "base_top1": 4,
            "cached_student_top1": 4,
            "deployed_top1": 5,
            "teacher_top1": 5,
        },
    ]
    read = direction_refresh_read(rows, weight)
    assert read["registered_estimator_direction_stale"] is False
    assert read["persistent_serving_reanchor_needed"] is True
    assert read["deployed_target_already_reached_rows"] == 1
