from __future__ import annotations

import torch

from eval.eval_paper2_phase3_retention_step0 import position_buckets


def test_position_buckets_match_phase2_training_contract() -> None:
    positions = torch.tensor([0, 1, 3, 4, 31, 32, 127, 128, 400])
    assert position_buckets(positions).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4]
