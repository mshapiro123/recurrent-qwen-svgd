from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from models.ablation_lm.bicameral_core import BicameralTransformerBlock
from models.ablation_lm.bicameral_recurrent import (
    BicameralRecurrenceReceipt,
    BicameralRecurrentCore,
)


REGISTERED_K = (1, 2, 4, 8)


def _block(path: str) -> BicameralTransformerBlock:
    return BicameralTransformerBlock(
        16,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        max_sequence_length=16,
        rank=4,
        initialization_seed=2_026_090_2,
        module_path=path,
    ).eval()


def test_recurrent_seam_reuses_caller_cache_and_applies_c_over_k() -> None:
    first = _block("model.bicameral_core.seam.0")
    second = _block("model.bicameral_core.seam.1")
    core = BicameralRecurrentCore((first, second)).eval()
    anchor = torch.randn(1, 5, 16)
    h_a = anchor.clone()
    h_b = anchor.clone()
    positions = torch.arange(5).view(1, -1)
    caches = (
        first.project_kv(anchor, position_ids=positions),
        second.project_kv(anchor, position_ids=positions),
    )

    with (
        patch.object(first, "forward", wraps=first.forward) as first_visit,
        patch.object(second, "forward", wraps=second.forward) as second_visit,
        patch.object(first, "project_kv", wraps=first.project_kv) as first_project,
        patch.object(second, "project_kv", wraps=second.project_kv) as second_project,
    ):
        output = core(
            h_a,
            h_b,
            projected_kv=caches,
            recurrent_steps=4,
            recurrence_c=0.8,
            position_ids=positions,
            force_math_attention=True,
        )

    assert first_project.call_count == second_project.call_count == 0
    assert first_visit.call_count == second_visit.call_count == 4
    assert all(call.kwargs["residual_scale"] == 0.2 for call in first_visit.call_args_list)
    assert output.h_a.shape == output.h_b.shape == anchor.shape
    assert output.receipt.executed_block_passes == 8
    assert output.receipt.cache_policy == "caller-owned (C-S5-2 unbound)"
    assert output.receipt.terminal_recombination == "not executed (C-S5-1 unbound)"


@pytest.mark.parametrize("steps", REGISTERED_K)
def test_recurrent_seam_executes_every_registered_k_exactly(steps: int) -> None:
    block = _block(f"model.bicameral_core.seam.k{steps}")
    core = BicameralRecurrentCore((block,)).eval()
    anchor = torch.randn(1, 3, 16)
    cache = block.project_kv(anchor)
    with patch.object(block, "forward", wraps=block.forward) as visit:
        output = core(
            anchor,
            anchor.clone(),
            projected_kv=(cache,),
            recurrent_steps=steps,
            recurrence_c=0.8,
            force_math_attention=True,
        )
    assert visit.call_count == steps
    assert output.receipt.recurrent_steps == steps
    assert output.receipt.executed_block_passes == steps
    assert output.receipt.residual_scale == 0.8 / steps


def test_recurrent_seam_is_dense_equivalent_when_disagreement_is_zero() -> None:
    block = _block("model.bicameral_core.seam.symmetric")
    with torch.no_grad():
        for projection in block.swap_linears:
            projection.dU.zero_()
            projection.dV.zero_()
    core = BicameralRecurrentCore((block,)).eval()
    anchor = torch.randn(2, 4, 16)
    cache = block.project_kv(anchor)

    output = core(
        anchor,
        anchor.clone(),
        projected_kv=(cache,),
        recurrent_steps=3,
        recurrence_c=0.9,
        force_math_attention=True,
    )

    assert torch.equal(output.h_a, output.h_b)


def test_recurrent_seam_keeps_all_paired_projection_modes_live() -> None:
    block = _block("model.bicameral_core.seam.liveness").train()
    core = BicameralRecurrentCore((block,))
    anchor = torch.randn(2, 4, 16, requires_grad=True)
    h_a = torch.randn(2, 4, 16, requires_grad=True)
    h_b = torch.randn(2, 4, 16, requires_grad=True)
    cache = block.project_kv(anchor)

    output = core(
        h_a,
        h_b,
        projected_kv=(cache,),
        recurrent_steps=4,
        recurrence_c=0.8,
    )
    (output.h_a.square().mean() + 0.37 * output.h_b.square().mean()).backward()

    for name, projection in zip(
        ("q", "k", "v", "o", "gate", "up", "down"),
        block.swap_linears,
        strict=True,
    ):
        for parameter_name in ("mu", "dU", "dV"):
            gradient = getattr(projection, parameter_name).grad
            assert gradient is not None, f"{name}.{parameter_name} is disconnected"
            assert bool(torch.isfinite(gradient).all())
            assert bool(gradient.ne(0).any()), f"{name}.{parameter_name} is frozen"


def test_recurrent_seam_fails_closed_on_cache_owner_and_open_decision_claims() -> None:
    first = _block("model.bicameral_core.seam.owner.0")
    second = _block("model.bicameral_core.seam.owner.1")
    core = BicameralRecurrentCore((first,)).eval()
    anchor = torch.randn(1, 3, 16)
    wrong_cache = second.project_kv(anchor)

    with pytest.raises(ValueError, match="different bicameral block"):
        core(
            anchor,
            anchor,
            projected_kv=(wrong_cache,),
            recurrent_steps=1,
            recurrence_c=1.0,
        )

    valid = BicameralRecurrenceReceipt(
        recurrent_steps=2,
        unique_core_blocks=1,
        executed_block_passes=2,
        recurrence_c=1.0,
        residual_scale=0.5,
        cache_policy="caller-owned (C-S5-2 unbound)",
        terminal_recombination="not executed (C-S5-1 unbound)",
    )
    with pytest.raises(ValueError, match="selected cache policy"):
        replace(valid, cache_policy="paired eigenmode")
    with pytest.raises(ValueError, match="terminal recombination"):
        replace(valid, terminal_recombination="unit-circle")


@pytest.mark.parametrize(
    ("updates", "error", "message"),
    [
        ({"recurrent_steps": 0}, ValueError, "positive integer"),
        ({"recurrent_steps": True}, ValueError, "positive integer"),
        ({"recurrence_c": 0.0}, ValueError, "strictly positive"),
        ({"recurrence_c": float("nan")}, ValueError, "strictly positive"),
        ({"recurrence_c": True}, TypeError, "real scalar"),
    ],
)
def test_recurrent_seam_validates_registered_schedule_literals(
    updates: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    block = _block("model.bicameral_core.seam.validation")
    core = BicameralRecurrentCore((block,)).eval()
    anchor = torch.randn(1, 3, 16)
    kwargs: dict[str, object] = {
        "projected_kv": (block.project_kv(anchor),),
        "recurrent_steps": 1,
        "recurrence_c": 1.0,
    }
    kwargs.update(updates)
    with pytest.raises(error, match=message):
        core(anchor, anchor, **kwargs)  # type: ignore[arg-type]


def test_recurrent_seam_rejects_missing_cache_and_invalid_states() -> None:
    block = _block("model.bicameral_core.seam.invalid")
    core = BicameralRecurrentCore((block,)).eval()
    anchor = torch.randn(1, 3, 16)
    cache = block.project_kv(anchor)
    with pytest.raises(ValueError, match="one entry"):
        core(
            anchor,
            anchor,
            projected_kv=(),
            recurrent_steps=1,
            recurrence_c=1.0,
        )
    with pytest.raises(ValueError, match="identical shapes"):
        core(
            anchor,
            anchor[:, :2],
            projected_kv=(cache,),
            recurrent_steps=1,
            recurrence_c=1.0,
        )
