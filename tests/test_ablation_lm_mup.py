from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from models.ablation_lm.config import AblationLMConfig
from models.ablation_lm.model import AblationLM
from models.ablation_lm.mup import (
    MUP_BASE_VOCAB_SIZE,
    MUP_ETA_BASE,
    MUP_WEIGHT_DECAY,
    MuPClassificationError,
    MuPParameterClass,
    audit_mup_parameters,
    build_provisional_mup_adamw,
    classify_mup_parameters,
    initialize_mup_parameters,
    output_multiplier,
    scale_tied_readout,
)


def _minimal_model(width: int = 128) -> AblationLM:
    return AblationLM(
        AblationLMConfig(
            vocab_size=MUP_BASE_VOCAB_SIZE,
            d_model=width,
            n_heads=width // 64,
            n_kv_heads=width // 128,
            d_ff=11 * width // 4,
            n_prelude_layers=1,
            n_core_blocks=1,
            n_coda_layers=0,
            max_sequence_length=8,
            initialization_seed=20_260_902,
            run_seed=20_260_902,
        )
    )


def test_closed_map_is_complete_duplicate_free_and_preserves_tied_alias() -> None:
    model = _minimal_model()
    parameterization = classify_mup_parameters(model, width=128)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert len(parameterization.assignments) == len(expected)
    tied = parameterization.assignment_for("token_embedding.weight")
    assert tied.parameter_class is MuPParameterClass.INPUT
    assert tied.aliases == ("lm_head.weight", "token_embedding.weight")
    assert tied.shape == (32_768, 128)
    assert parameterization.assignment_for("lm_head.weight") is tied
    assert parameterization.output_multiplier == 4.0


def test_unknown_tensor_is_retained_by_audit_and_rejected_by_promotion() -> None:
    class Unknown(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mystery = nn.Parameter(torch.ones(8, 8))

    model = Unknown()
    audit = audit_mup_parameters(model, width=128)
    assert audit.assignments == ()
    assert len(audit.issues) == 1
    assert audit.issues[0].canonical_name == "mystery"
    assert audit.issues[0].shape == (8, 8)
    with pytest.raises(MuPClassificationError, match=r"mystery shape=\(8, 8\)"):
        classify_mup_parameters(model, width=128)


def test_pf3_initialization_is_bit_replayable_and_uses_bound_scales() -> None:
    left = _minimal_model()
    right = _minimal_model()
    left_vector_before = left.final_norm.weight.detach().clone()
    left_map = initialize_mup_parameters(left, width=128, base_seed=20_260_902)
    right_map = initialize_mup_parameters(right, width=128, base_seed=20_260_902)
    assert left_map == right_map
    left_parameters = dict(left.named_parameters())
    right_parameters = dict(right.named_parameters())
    assert set(left_parameters) == set(right_parameters)
    assert all(
        torch.equal(left_parameters[name], right_parameters[name])
        for name in left_parameters
    )
    assert torch.equal(left.final_norm.weight, left_vector_before)
    q_weight = left.prelude_blocks[0].attention.q_proj.weight.detach().float()
    expected_std = 1.0 / math.sqrt(128)
    assert float(q_weight.std(unbiased=False)) == pytest.approx(expected_std, rel=0.03)
    embedding_std = float(left.token_embedding.weight.detach().float().std(unbiased=False))
    assert embedding_std == pytest.approx(0.02, rel=0.01)


def test_provisional_adamw_has_constant_effective_hidden_decay() -> None:
    model = _minimal_model()
    initialize_mup_parameters(model, width=128, base_seed=20_260_902)
    optimizer, parameterization = build_provisional_mup_adamw(model, width=128)
    groups = {group["mup_class"]: group for group in optimizer.param_groups}
    assert set(groups) == {"hidden", "input", "vector"}
    hidden = groups["hidden"]
    assert hidden["lr"] == MUP_ETA_BASE / 0.25
    assert hidden["weight_decay"] == MUP_WEIGHT_DECAY * 0.25
    assert hidden["lr"] * hidden["weight_decay"] == pytest.approx(
        MUP_ETA_BASE * MUP_WEIGHT_DECAY
    )
    assert groups["input"]["lr"] == groups["vector"]["lr"] == MUP_ETA_BASE
    assert groups["input"]["weight_decay"] == groups["vector"]["weight_decay"] == 0.0
    optimizer_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(optimizer_ids) == len(set(optimizer_ids)) == len(parameterization.assignments)


def test_readout_multiplier_is_external_and_has_no_second_parameter() -> None:
    logits = torch.tensor([[1.0, -2.0]])
    scaled = scale_tied_readout(logits, width=128)
    assert output_multiplier(128) == 4.0
    assert torch.equal(scaled, logits * 4.0)
    model = _minimal_model()
    assert model.lm_head.weight is model.token_embedding.weight


def test_ltm_shape_that_does_not_scale_both_axes_fails_closed() -> None:
    class Holder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.long_term_memory = nn.Module()
            self.long_term_memory.query = nn.Linear(128, 12, bias=False)

    with pytest.raises(MuPClassificationError, match="does not prove both fan-in and fan-out"):
        classify_mup_parameters(Holder(), width=128)
