from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from training.paper2_stage2a_objective import stage2a_answer_region_objective


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260817)
    student = torch.randn((2, 4, 140), generator=generator, dtype=torch.float64)
    teacher = torch.randn((2, 4, 128), generator=generator, dtype=torch.float64)
    lattice = torch.arange(128).expand(2, 4, 128)
    targets = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    mask = torch.tensor(
        [[False, True, False, False], [False, True, True, True]], dtype=torch.bool
    )
    return {
        "student_logits": student,
        "teacher_topk_token_ids": lattice,
        "teacher_topk_logits": teacher,
        "teacher_token_ids": targets,
        "answer_region_mask": mask,
    }


def test_stage2a_objective_matches_registered_per_example_reduction() -> None:
    inputs = _inputs()
    read = stage2a_answer_region_objective(**inputs)
    student = inputs["student_logits"]
    teacher = inputs["teacher_topk_logits"]
    lattice = inputs["teacher_topk_token_ids"]
    targets = inputs["teacher_token_ids"]
    mask = inputs["answer_region_mask"]
    gathered = student.gather(-1, lattice)
    t_log = F.log_softmax(teacher, dim=-1)
    s_log = F.log_softmax(gathered, dim=-1)
    kl_positions = (t_log.exp() * (t_log - s_log)).sum(-1)
    ce_positions = F.cross_entropy(
        student.reshape(-1, 140), targets.reshape(-1), reduction="none"
    ).reshape(2, 4)
    expected_ce = torch.stack((ce_positions[0, 1], ce_positions[1, 1:].mean())).mean()
    expected_kl = torch.stack((kl_positions[0, 1], kl_positions[1, 1:].mean())).mean()
    torch.testing.assert_close(read.cross_entropy, expected_ce.float())
    torch.testing.assert_close(read.forward_kl, expected_kl.float())
    torch.testing.assert_close(read.loss, 0.5 * (expected_ce + expected_kl).float())
    assert read.answer_positions_per_example.tolist() == [1, 3]


def test_stage2a_objective_ignores_prompt_and_formatting_positions() -> None:
    inputs = _inputs()
    baseline = stage2a_answer_region_objective(**inputs)
    modified = {key: value.clone() for key, value in inputs.items()}
    modified["student_logits"][:, 0] = 10_000.0
    modified["teacher_topk_logits"][:, 0] = -10_000.0
    modified["student_logits"][0, 2:] = -20_000.0
    read = stage2a_answer_region_objective(**modified)
    torch.testing.assert_close(read.loss, baseline.loss)


def test_stage2a_objective_rejects_position_zero_and_wrong_lattice() -> None:
    inputs = _inputs()
    inputs["answer_region_mask"][0, 0] = True
    with pytest.raises(ValueError, match="position zero"):
        stage2a_answer_region_objective(**inputs)
    inputs = _inputs()
    inputs["teacher_topk_token_ids"] = inputs["teacher_topk_token_ids"][..., :127]
    inputs["teacher_topk_logits"] = inputs["teacher_topk_logits"][..., :127]
    with pytest.raises(ValueError, match="exactly 128"):
        stage2a_answer_region_objective(**inputs)
