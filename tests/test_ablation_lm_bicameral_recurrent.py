from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from models.ablation_lm.bicameral_core import BicameralTransformerBlock
from models.ablation_lm.bicameral_recurrent import (
    AfterBlockResult,
    BicameralRecurrenceReceipt,
    BicameralRecurrentCore,
    execute_bicameral_recurrence,
)


REGISTERED_K = (1, 2, 4, 8)


class _AdditiveBlock(torch.nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        events = kwargs.get("execution_events")
        prefix = kwargs.get("execution_prefix")
        if isinstance(events, list) and isinstance(prefix, str):
            events.extend((f"{prefix}.attention", f"{prefix}.feed_forward"))
        return h_a + self.increment, h_b - self.increment


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


def test_static_policy_reuses_caller_cache_and_applies_c_over_k() -> None:
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
            kv_policy="static",
            position_ids=positions,
            force_math_attention=True,
        )

    assert first_project.call_count == second_project.call_count == 0
    assert first_visit.call_count == second_visit.call_count == 4
    assert all(call.kwargs["residual_scale"] == 0.2 for call in first_visit.call_args_list)
    assert output.h_a.shape == output.h_b.shape == anchor.shape
    assert output.receipt.executed_block_passes == 8
    assert output.receipt.kv_policy == "static"
    assert output.receipt.kv_cache_multiplier_at_serving == 1
    assert output.receipt.terminal_recombination == "not executed (separated-state seam)"


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
            kv_policy="static",
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
        kv_policy="static",
        force_math_attention=True,
    )

    assert torch.equal(output.h_a, output.h_b)


def test_recurrent_seam_keeps_all_paired_projection_modes_live() -> None:
    block = _block("model.bicameral_core.seam.liveness").train()
    core = BicameralRecurrentCore((block,))
    anchor = torch.randn(2, 4, 16, requires_grad=True)
    h_a = torch.randn(2, 4, 16, requires_grad=True)
    h_b = torch.randn(2, 4, 16, requires_grad=True)
    output = core(
        h_a,
        h_b,
        recurrent_steps=4,
        recurrence_c=0.8,
        kv_policy="live",
    )
    (output.h_a.square().mean() + 0.37 * output.h_b.square().mean()).backward()

    for name, projection in zip(
        ("q", "o", "gate", "up", "down"),
        block.swap_linears,
        strict=True,
    ):
        for parameter_name in ("mu", "dU", "dV"):
            gradient = getattr(projection, parameter_name).grad
            assert gradient is not None, f"{name}.{parameter_name} is disconnected"
            assert bool(torch.isfinite(gradient).all())
            assert bool(gradient.ne(0).any()), f"{name}.{parameter_name} is frozen"
    for name, projection in zip(("k", "v"), block.shared_kv_linears, strict=True):
        gradient = projection.weight.grad
        assert gradient is not None, f"{name}.weight is disconnected"
        assert bool(torch.isfinite(gradient).all())
        assert bool(gradient.ne(0).any()), f"{name}.weight is frozen"


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
            kv_policy="static",
        )

    valid = BicameralRecurrenceReceipt(
        recurrent_steps=2,
        unique_core_blocks=1,
        executed_block_passes=2,
        recurrence_c=1.0,
        residual_scale=0.5,
        kv_policy="static",
        kv_cache_multiplier_at_serving=1,
        after_block_modules=(),
        visit_schedule=(
            "setup.block[0].use_projected_kv.static",
            "visit[0].block[0].attention",
            "visit[0].block[0].feed_forward",
            "visit[1].block[0].attention",
            "visit[1].block[0].feed_forward",
        ),
        terminal_recombination="not executed (separated-state seam)",
    )
    with pytest.raises(ValueError, match="kv_policy"):
        replace(valid, kv_policy="paired eigenmode")
    with pytest.raises(ValueError, match="multiplier"):
        replace(valid, kv_cache_multiplier_at_serving=2)
    with pytest.raises(ValueError, match="terminal recombination"):
        replace(valid, terminal_recombination="unit-circle")
    with pytest.raises(ValueError, match="exact execution trace"):
        replace(
            valid,
            visit_schedule=(
                "setup.block[0].use_projected_kv.static",
                "visit[0].block[0].feed_forward",
                "visit[0].block[0].attention",
                "visit[1].block[0].attention",
                "visit[1].block[0].feed_forward",
            ),
        )


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
        "kv_policy": "static",
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
            kv_policy="static",
        )
    with pytest.raises(ValueError, match="identical shapes"):
        core(
            anchor,
            anchor[:, :2],
            projected_kv=(cache,),
            recurrent_steps=1,
            recurrence_c=1.0,
            kv_policy="static",
        )


def test_k1_live_equals_static_with_multiple_blocks_bit_exactly() -> None:
    first = _block("model.bicameral_core.k1_identity.0")
    second = _block("model.bicameral_core.k1_identity.1")
    core = BicameralRecurrentCore((first, second)).eval()
    anchor = torch.randn(2, 5, 16)
    positions = torch.arange(5).view(1, -1).expand(2, -1)
    static_caches = (
        first.project_kv(anchor, position_ids=positions),
        second.project_kv(anchor, position_ids=positions),
    )

    static = core(
        anchor,
        anchor.clone(),
        recurrent_steps=1,
        recurrence_c=0.8,
        projected_kv=static_caches,
        kv_policy="static",
        position_ids=positions,
        force_math_attention=True,
    )
    live = core(
        anchor,
        anchor.clone(),
        recurrent_steps=1,
        recurrence_c=0.8,
        position_ids=positions,
        force_math_attention=True,
    )

    assert live.receipt.kv_policy == "live"
    assert torch.equal(live.h_a, static.h_a)
    assert torch.equal(live.h_b, static.h_b)


def test_live_projects_every_block_from_one_visit_entry_snapshot() -> None:
    first = _block("model.bicameral_core.live_snapshot.0")
    second = _block("model.bicameral_core.live_snapshot.1")
    core = BicameralRecurrentCore((first, second)).eval()
    anchor = torch.randn(1, 4, 16)

    with (
        patch.object(first, "project_kv", wraps=first.project_kv) as first_project,
        patch.object(second, "project_kv", wraps=second.project_kv) as second_project,
    ):
        output = core(
            anchor,
            anchor.clone(),
            recurrent_steps=2,
            recurrence_c=0.8,
            force_math_attention=True,
        )

    assert first_project.call_count == second_project.call_count == 2
    for visit in range(2):
        first_args = first_project.call_args_list[visit].args
        second_args = second_project.call_args_list[visit].args
        assert first_args[0] is second_args[0]
        assert first_args[1] is second_args[1]
    assert first_project.call_args_list[0].args[0] is not first_project.call_args_list[1].args[0]
    assert output.receipt.kv_cache_multiplier_at_serving == 4


def test_midpoint_refreshes_once_from_the_visit_entry_pair() -> None:
    first = _block("model.bicameral_core.midpoint.0")
    second = _block("model.bicameral_core.midpoint.1")
    core = BicameralRecurrentCore((first, second)).eval()
    anchor = torch.randn(1, 4, 16)
    initial = (first.project_kv(anchor), second.project_kv(anchor))

    with (
        patch.object(first, "project_kv", wraps=first.project_kv) as first_project,
        patch.object(second, "project_kv", wraps=second.project_kv) as second_project,
    ):
        output = core(
            anchor,
            anchor.clone(),
            recurrent_steps=4,
            recurrence_c=0.8,
            projected_kv=initial,
            kv_policy="midpoint",
            force_math_attention=True,
        )

    assert first_project.call_count == second_project.call_count == 1
    assert first_project.call_args.args[0] is second_project.call_args.args[0]
    assert first_project.call_args.args[1] is second_project.call_args.args[1]
    assert output.receipt.kv_policy == "midpoint"
    assert output.receipt.kv_cache_multiplier_at_serving == 2


@pytest.mark.parametrize(
    ("policy", "steps", "expected_multiplier"),
    (("live", 3, 6), ("static", 3, 1), ("midpoint", 3, 2)),
)
def test_receipt_binds_serving_cache_multiplier(
    policy: str,
    steps: int,
    expected_multiplier: int,
) -> None:
    block = _block(f"model.bicameral_core.receipt.{policy}")
    core = BicameralRecurrentCore((block,)).eval()
    anchor = torch.randn(1, 3, 16)
    caches = None if policy == "live" else (block.project_kv(anchor),)
    output = core(
        anchor,
        anchor.clone(),
        recurrent_steps=steps,
        recurrence_c=0.6,
        projected_kv=caches,
        kv_policy=policy,
        force_math_attention=True,
    )
    assert output.receipt.kv_policy == policy
    assert output.receipt.kv_cache_multiplier_at_serving == expected_multiplier


def test_kv_policy_and_cache_ownership_fail_closed() -> None:
    block = _block("model.bicameral_core.policy_validation")
    core = BicameralRecurrentCore((block,)).eval()
    anchor = torch.randn(1, 3, 16)
    cache = block.project_kv(anchor)
    with pytest.raises(ValueError, match="kv_policy"):
        core(
            anchor,
            anchor,
            recurrent_steps=1,
            recurrence_c=1.0,
            kv_policy="first",
        )
    with pytest.raises(ValueError, match="forbids caller caches"):
        core(
            anchor,
            anchor,
            recurrent_steps=1,
            recurrence_c=1.0,
            projected_kv=(cache,),
            kv_policy="live",
        )
    with pytest.raises(TypeError, match="must be a sequence"):
        core(
            anchor,
            anchor,
            recurrent_steps=1,
            recurrence_c=1.0,
            projected_kv=object(),  # type: ignore[arg-type]
            kv_policy="static",
        )


@pytest.mark.parametrize("policy", ("live", "static", "midpoint"))
def test_every_kv_policy_remains_exactly_causal_across_packing(policy: str) -> None:
    block = _block(f"model.bicameral_core.causality.{policy}")
    core = BicameralRecurrentCore((block,)).eval()
    anchor = torch.randn(1, 6, 16, requires_grad=True)
    h_a = torch.randn(1, 6, 16, requires_grad=True)
    h_b = torch.randn(1, 6, 16, requires_grad=True)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    document_ids = torch.tensor([[0, 0, 1, 1, 1, -1]])
    position_ids = torch.tensor([[0, 1, 0, 1, 2, 0]])
    caches = (
        None
        if policy == "live"
        else (block.project_kv(anchor, position_ids=position_ids),)
    )
    output = core(
        h_a,
        h_b,
        recurrent_steps=2,
        recurrence_c=0.8,
        projected_kv=caches,
        kv_policy=policy,
        attention_mask=attention_mask,
        document_ids=document_ids,
        position_ids=position_ids,
        force_math_attention=True,
    )
    gradients = torch.autograd.grad(
        output.h_a[0, 3].sum() + 0.31 * output.h_b[0, 3].sum(),
        (anchor, h_a, h_b),
        allow_unused=True,
    )
    anchor_gradient, query_a_gradient, query_b_gradient = gradients
    if policy == "live":
        assert anchor_gradient is None
    else:
        assert anchor_gradient is not None
        assert torch.count_nonzero(anchor_gradient[0, :2]) == 0
        assert torch.count_nonzero(anchor_gradient[0, 4:]) == 0
    for query_gradient in (query_a_gradient, query_b_gradient):
        assert query_gradient is not None
        assert torch.count_nonzero(query_gradient[0, :2]) == 0
        assert torch.count_nonzero(query_gradient[0, 4:]) == 0


def test_after_block_hook_runs_immediately_after_every_block_in_schedule_order() -> None:
    core = BicameralRecurrentCore((_AdditiveBlock(1.0), _AdditiveBlock(2.0)))
    hidden = torch.zeros(1, 1, 1)
    calls: list[tuple[int, int, float, float]] = []

    def lane_update(
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        *,
        visit: int,
        block_index: int,
        residual_scale: float,
    ) -> AfterBlockResult:
        calls.append((visit, block_index, residual_scale, h_a.item()))
        return AfterBlockResult(
            h_a=h_a + 10.0,
            h_b=h_b - 10.0,
            executed_modules=("PositionAlignedScratch.step_bicameral",),
        )

    output = core(
        hidden,
        hidden.clone(),
        recurrent_steps=2,
        recurrence_c=0.8,
        projected_kv=(object(), object()),
        kv_policy="static",
        after_block=lane_update,
        expected_after_block_modules=("PositionAlignedScratch.step_bicameral",),
    )

    assert calls == [
        (0, 0, 0.4, 1.0),
        (0, 1, 0.4, 13.0),
        (1, 0, 0.4, 24.0),
        (1, 1, 0.4, 36.0),
    ]
    torch.testing.assert_close(output.h_a, torch.full_like(hidden, 46.0))
    torch.testing.assert_close(output.h_b, torch.full_like(hidden, -46.0))


def test_explicit_no_hook_is_bit_identical_to_omitted_hook() -> None:
    block = _block("model.bicameral_core.no_hook_identity")
    core = BicameralRecurrentCore((block,)).eval()
    hidden = torch.randn(1, 4, 16)
    caches = (block.project_kv(hidden),)
    common = {
        "recurrent_steps": 2,
        "recurrence_c": 0.8,
        "projected_kv": caches,
        "kv_policy": "static",
        "force_math_attention": True,
    }

    omitted = core(hidden, hidden.clone(), **common)
    explicit = core(hidden, hidden.clone(), after_block=None, **common)

    assert torch.equal(explicit.h_a, omitted.h_a)
    assert torch.equal(explicit.h_b, omitted.h_b)
    assert explicit.receipt == omitted.receipt


def test_functional_executor_accepts_canonical_modulelist_without_new_owner() -> None:
    block = _block("model.bicameral_core.functional_executor")
    canonical_blocks = torch.nn.ModuleList((block,))
    hidden = torch.randn(1, 4, 16)
    caches = (block.project_kv(hidden),)

    output = execute_bicameral_recurrence(
        canonical_blocks,
        hidden,
        hidden.clone(),
        recurrent_steps=2,
        recurrence_c=0.8,
        projected_kv=caches,
        kv_policy="static",
        force_math_attention=True,
    )

    assert output.h_a.shape == output.h_b.shape == hidden.shape
    assert output.receipt.unique_core_blocks == 1
    assert list(canonical_blocks.children()) == [block]


@pytest.mark.parametrize(
    ("hook", "error", "message"),
    [
        (object(), TypeError, "callable"),
        (lambda *_args, **_kwargs: None, TypeError, "AfterBlockResult"),
        (
            lambda h_a, h_b, **_kwargs: AfterBlockResult(
                h_a=h_a,
                h_b=h_b,
                executed_modules=("unexpected",),
            ),
            RuntimeError,
            "disagree with the bound receipt",
        ),
        (
            lambda h_a, h_b, **_kwargs: AfterBlockResult(
                h_a=h_a[..., :1],
                h_b=h_b[..., :1],
                executed_modules=(),
            ),
            ValueError,
            "state shape",
        ),
    ],
)
def test_after_block_hook_contract_fails_closed(
    hook: object,
    error: type[Exception],
    message: str,
) -> None:
    core = BicameralRecurrentCore((_AdditiveBlock(1.0),))
    hidden = torch.zeros(1, 1, 2)
    with pytest.raises(error, match=message):
        core(
            hidden,
            hidden.clone(),
            recurrent_steps=1,
            recurrence_c=0.8,
            projected_kv=(object(),),
            kv_policy="static",
            after_block=hook,  # type: ignore[arg-type]
            expected_after_block_modules=(),
        )
