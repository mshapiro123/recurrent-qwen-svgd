from __future__ import annotations

import copy
import pytest
import torch
from torch import nn
from training.muon import split_muon_and_adamw_params

from models.ablation_lm.optim import (
    FULL_MATRIX_MUON_GEOMETRY,
    LEGACY_RANK_ONLY_MUON_SPLITTER_SUPPORTED,
    MODE_WISE_MUON_SUPPORTED,
    OptimizerTarget,
    ParameterRole,
    REQUIRE_CLOSED_MUON_ALLOWLIST_ATTR,
    mark_coupled_mode_adamw,
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


def _tiny_ablation_lm() -> nn.Module:
    from models.ablation_lm import AblationLM, AblationLMConfig

    return AblationLM(
        AblationLMConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            n_prelude_layers=1,
            n_core_blocks=1,
            n_coda_layers=1,
            max_sequence_length=16,
            recurrent_steps=1,
            max_recurrent_steps=2,
            scratch_width=8,
        )
    )


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
    assert LEGACY_RANK_ONLY_MUON_SPLITTER_SUPPORTED is False

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


def test_dense_mu_and_factored_delta_pair_cannot_split_optimizer_families() -> None:
    class _ModePair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mu = mark_coupled_mode_adamw(nn.Linear(8, 8, bias=False))
            self.delta_u = mark_coupled_mode_adamw(nn.Linear(8, 2, bias=False))
            self.delta_v = mark_coupled_mode_adamw(nn.Linear(2, 8, bias=False))

    partition = partition_optimizer_parameters(_ModePair())

    for name in ("mu.weight", "delta_u.weight", "delta_v.weight"):
        assignment = partition.assignment_for(name)
        assert assignment.role is ParameterRole.COUPLED_MODE
        assert assignment.target is OptimizerTarget.AUXILIARY_ADAMW


def test_closed_ablation_allowlist_rejects_a_new_muon_hypothesis() -> None:
    model = _RoleFixture()
    setattr(model, REQUIRE_CLOSED_MUON_ALLOWLIST_ATTR, True)

    with pytest.raises(RuntimeError, match="closed Muon allowlist differs"):
        partition_optimizer_parameters(model)


def test_closed_ablation_allowlist_is_enforced_through_a_wrapper() -> None:
    class _Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _tiny_ablation_lm()
            self.extra = mark_muon_eligible(nn.Linear(32, 32, bias=False))

    with pytest.raises(RuntimeError, match="closed Muon allowlist differs"):
        partition_optimizer_parameters(_Wrapper())


def test_legacy_rank_only_splitter_fails_closed_for_ablation_lm() -> None:
    model = _tiny_ablation_lm()

    with pytest.raises(RuntimeError, match="rank-only Muon splitting is prohibited"):
        split_muon_and_adamw_params(model.named_parameters())
    cloned = copy.deepcopy(model)
    with pytest.raises(RuntimeError, match="rank-only Muon splitting is prohibited"):
        split_muon_and_adamw_params(cloned.named_parameters())
    cloned_core_rows = tuple(
        (name, parameter)
        for name, parameter in cloned.named_parameters()
        if name.startswith("core_blocks.")
    )
    assert cloned_core_rows
    assert all(
        bool(getattr(parameter, "_ablation_lm_rank_only_muon_prohibited", False))
        for _name, parameter in cloned_core_rows
    )
    with pytest.raises(RuntimeError, match="rank-only Muon splitting is prohibited"):
        split_muon_and_adamw_params(cloned_core_rows)
    with pytest.raises(RuntimeError, match="rank-only Muon splitting is prohibited"):
        split_muon_and_adamw_params(cloned.core_blocks.named_parameters())
    cloned_core_only = copy.deepcopy(model.core_blocks)
    with pytest.raises(RuntimeError, match="rank-only Muon splitting is prohibited"):
        split_muon_and_adamw_params(cloned_core_only.named_parameters())


def test_legacy_splitter_still_accepts_the_existing_recurrent_qwen_wrapper() -> None:
    from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM
    from tests.test_recurrent_wrapper_tiny import TinyCausalLM

    wrapper = RecurrentQwenForCausalLM(
        TinyCausalLM(),
        layer_split=LayerSplit(1, 3),
    )
    muon, adamw = split_muon_and_adamw_params(wrapper.named_parameters())

    assert muon
    assert adamw


def test_meta_assign_load_restores_tied_identity_and_optimizer_safety() -> None:
    model = _tiny_ablation_lm()
    assert all(isinstance(value, torch.Tensor) for value in model.state_dict().values())
    checkpoint = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    unique_before = len(tuple(model.parameters()))

    model.to("meta")
    assert model.token_embedding.weight is model.lm_head.weight
    assert len(tuple(model.parameters())) == unique_before
    model.load_state_dict(checkpoint, assign=True)

    assert model.token_embedding.weight is model.lm_head.weight
    assert len(tuple(model.parameters())) == unique_before
    assert all(
        bool(getattr(parameter, "_ablation_lm_rank_only_muon_prohibited", False))
        for parameter in model.parameters()
    )
    with pytest.raises(RuntimeError, match="rank-only Muon splitting is prohibited"):
        split_muon_and_adamw_params(model.core_blocks.named_parameters())

    inconsistent = dict(checkpoint)
    inconsistent["lm_head.weight"] = inconsistent["lm_head.weight"] + 1
    with pytest.raises(RuntimeError, match="disagree"):
        model.load_state_dict(inconsistent, assign=True)

    tied_only = {
        "token_embedding.weight": checkpoint["token_embedding.weight"],
        "lm_head.weight": checkpoint["lm_head.weight"],
    }
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(tied_only, strict=True, assign=True)
    assert model.token_embedding.weight is model.lm_head.weight
    assert all(
        bool(getattr(parameter, "_ablation_lm_rank_only_muon_prohibited", False))
        for parameter in model.parameters()
    )

    vocabulary_before = model.token_embedding.weight.detach().clone()
    with pytest.raises(RuntimeError, match="strict=False, assign=False"):
        model.load_state_dict(
            {"lm_head.weight": torch.zeros_like(vocabulary_before)},
            strict=False,
            assign=True,
        )
    torch.testing.assert_close(model.token_embedding.weight, vocabulary_before)
    assert model.token_embedding.weight is model.lm_head.weight


def test_safetensors_shared_model_roundtrip_preserves_rng_state_and_tying(
    tmp_path,
) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    source = _tiny_ablation_lm()
    source.core_blocks[0].attention.dropout_rng.next_generator(
        torch.device("cpu"),
        coordinate=0,
    )
    checkpoint_path = tmp_path / "weft1-tiny.safetensors"

    safetensors.save_model(source, checkpoint_path)
    restored = _tiny_ablation_lm()
    missing, unexpected = safetensors.load_model(
        restored,
        checkpoint_path,
        strict=True,
    )

    assert missing == set()
    assert unexpected == []
    assert restored.token_embedding.weight is restored.lm_head.weight
    assert (
        restored.core_blocks[0].attention.dropout_rng.draw_indices
        == source.core_blocks[0].attention.dropout_rng.draw_indices
    )
    for source_parameter, restored_parameter in zip(
        source.parameters(),
        restored.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(source_parameter, restored_parameter, rtol=0, atol=0)
