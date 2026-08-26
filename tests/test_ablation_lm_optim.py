from __future__ import annotations

import torch
from torch import nn

from models.ablation_lm.optim import (
    FULL_MATRIX_MUON_GEOMETRY,
    MODE_WISE_MUON_SUPPORTED,
    OptimizerTarget,
    ParameterRole,
    mark_muon_eligible,
    partition_optimizer_parameters,
    tag_optimizer_role,
)


class _Engram(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.randn(4, width))
        tag_optimizer_role(self, "table", ParameterRole.ENGRAM)


class _RoleFixture(nn.Module):
    def __init__(self, *, include_novel: bool = True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.hidden = mark_muon_eligible(
            nn.Linear(8, 8),
            post_muon_multiplier=0.25,
        )
        self.norm = nn.LayerNorm(8)
        self.gate = nn.Parameter(torch.zeros(()))
        tag_optimizer_role(self, "gate", ParameterRole.GATE)
        self.engram = _Engram(8)
        if include_novel:
            # Matrix rank does not silently promote an experimental module.
            self.novel_projection = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.lm_head.weight = self.embedding.weight


def _unique_trainables(model: nn.Module) -> set[int]:
    return {id(parameter) for parameter in model.parameters() if parameter.requires_grad}


def _partition_ids(model: nn.Module) -> list[int]:
    partition = partition_optimizer_parameters(model)
    return [id(parameter) for group in partition.groups for parameter in group.parameters]


def test_partition_covers_every_trainable_exactly_once_by_semantic_role() -> None:
    model = _RoleFixture()
    partition = partition_optimizer_parameters(model)
    partition_ids = _partition_ids(model)

    assert set(partition_ids) == _unique_trainables(model)
    assert len(partition_ids) == len(set(partition_ids))

    hidden = partition.assignment_for("hidden.weight")
    assert hidden.target is OptimizerTarget.MUON_ELIGIBLE
    assert hidden.role is ParameterRole.DENSE_HIDDEN_WEIGHT
    assert hidden.post_muon_multiplier == 0.25
    assert len(partition.muon_groups) == 1
    assert partition.muon_groups[0].post_muon_multiplier == 0.25
    assert partition.muon_groups[0].update_geometry == FULL_MATRIX_MUON_GEOMETRY
    assert MODE_WISE_MUON_SUPPORTED is False

    for name in (
        "hidden.bias",
        "norm.weight",
        "norm.bias",
        "gate",
        "engram.table",
        "novel_projection.weight",
    ):
        assert partition.assignment_for(name).target is OptimizerTarget.AUXILIARY_ADAMW


def test_tied_embedding_and_head_weight_is_listed_once_and_stays_adamw() -> None:
    model = _RoleFixture()
    partition = partition_optimizer_parameters(model)
    tied = partition.assignment_for("embedding.weight")

    assert tied.parameter is model.embedding.weight is model.lm_head.weight
    assert tied.aliases == ("embedding.weight", "lm_head.weight")
    assert tied.role is ParameterRole.EMBEDDING
    assert tied.target is OptimizerTarget.AUXILIARY_ADAMW
    assert sum(parameter is tied.parameter for parameter in partition.auxiliary_adamw_parameters) == 1
    assert all(parameter is not tied.parameter for parameter in partition.muon_parameters)


def test_roles_are_stable_when_optional_novel_module_is_added() -> None:
    without_novel = partition_optimizer_parameters(_RoleFixture(include_novel=False))
    with_novel = partition_optimizer_parameters(_RoleFixture(include_novel=True))

    stable_names = (
        "embedding.weight",
        "hidden.weight",
        "hidden.bias",
        "norm.weight",
        "norm.bias",
        "gate",
        "engram.table",
    )
    for name in stable_names:
        left = without_novel.assignment_for(name)
        right = with_novel.assignment_for(name)
        assert (left.role, left.target, left.post_muon_multiplier) == (
            right.role,
            right.target,
            right.post_muon_multiplier,
        )

    novel = with_novel.assignment_for("novel_projection.weight")
    assert novel.role is ParameterRole.NOVEL_MODULE
    assert novel.target is OptimizerTarget.AUXILIARY_ADAMW
