from __future__ import annotations

import torch

from eval.eval_paper2_dc1_stage_a import build_scoring_rows
from training.train_paper2_dc1_stage_a import (
    assert_only_allowlist_gradients,
    build_stage_a_example,
    freeze_except_allowlist,
)


class TinyStageA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.horizontal_bridge = torch.nn.Module()
        self.horizontal_bridge.delta = torch.nn.Linear(3, 3, bias=False)
        self.frozen = torch.nn.Linear(3, 2, bias=False)


def test_stage_a_example_reads_teacher_from_appended_slot_logits() -> None:
    row = {"input_ids": [10, 11, 12, 13]}
    teacher = torch.tensor([20, 21, 22])
    input_ids, attention, labels = build_stage_a_example(
        row=row,
        teacher_ids=teacher,
        local_position=1,
        latent_token_id=99,
        terminal_token_id=2,
        device="cpu",
    )

    assert input_ids.tolist() == [[10, 11, 99, 2]]
    assert attention.tolist() == [[1, 1, 1, 1]]
    assert labels.tolist() == [[-100, -100, -100, 21]]
    assert input_ids[0, -1].item() != labels[0, -1].item()


def test_stage_a_trainable_set_and_gradient_audit_are_exact() -> None:
    model = TinyStageA()
    allowlist = {"horizontal_bridge.delta.weight"}
    receipt = freeze_except_allowlist(model, allowlist)
    assert receipt["parameter_names"] == ["horizontal_bridge.delta.weight"]
    assert model.horizontal_bridge.delta.weight.requires_grad
    assert not model.frozen.weight.requires_grad

    model.horizontal_bridge.delta.weight.grad = torch.ones_like(
        model.horizontal_bridge.delta.weight
    )
    assert_only_allowlist_gradients(model, allowlist)
    model.frozen.weight.grad = torch.ones_like(model.frozen.weight)
    try:
        assert_only_allowlist_gradients(model, allowlist)
    except RuntimeError as error:
        assert "frozen parameter" in str(error)
    else:
        raise AssertionError("nonzero frozen gradient was not rejected")


def test_stage_a_immutable_rows_keep_row_clusters_and_all_arms() -> None:
    rows = [{"row_id": "row-a", "stratum": "code", "input_ids": [1, 2, 3]}]
    teacher_rows = {0: {"teacher_greedy_token_id": torch.tensor([2, 8])}}
    inplace = [torch.tensor([[2, 8, 7], [4, 8, 8]])]
    trained = [torch.tensor([2, 8])]
    untrained = [torch.tensor([5, 8])]
    cache = build_scoring_rows(
        rows=rows,
        teacher_rows=teacher_rows,
        inplace_rows=inplace,
        trained_rows=trained,
        untrained_rows=untrained,
    )

    assert cache[0]["row_id"] == "row-a"
    assert cache[0]["scored_positions"] == 2
    assert set(cache[0]["arms"]) == {
        "trained_append_k1",
        "untrained_append_k1",
        "inplace_depth2_descriptive",
        "inplace_depth3_descriptive",
    }
    assert cache[0]["arms"]["trained_append_k1"]["helps"] == 1
    assert cache[0]["arms"]["trained_append_k1"]["hurts"] == 0
