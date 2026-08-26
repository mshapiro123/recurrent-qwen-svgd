from __future__ import annotations

import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from models.ablation_lm import AblationLM, AblationLMConfig, parameter_accounting
from models.ablation_lm.memory import ReadOnlyLatentMemory
from models.ablation_lm.optim import (
    OptimizerTarget,
    ParameterRole,
    partition_optimizer_parameters,
)


def _tiny_config(**updates: object) -> AblationLMConfig:
    config = AblationLMConfig(
        vocab_size=64,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        n_prelude_layers=1,
        n_core_blocks=2,
        n_coda_layers=1,
        recurrent_steps=1,
        max_recurrent_steps=4,
        max_sequence_length=16,
        scratch_width=8,
        engram_hashes_per_order=2,
        engram_table_size=31,
        engram_row_dim=2,
        long_term_memory_slots=16,
        long_term_memory_width=8,
    )
    return replace(config, **updates)


def _model(config: AblationLMConfig) -> AblationLM:
    memory = None
    if config.use_long_term_memory:
        generator = torch.Generator().manual_seed(1_337)
        memory = ReadOnlyLatentMemory(
            config.d_model,
            keys=torch.randn(
                config.long_term_memory_slots,
                config.long_term_memory_width,
                generator=generator,
            ),
            values=torch.randn(
                config.long_term_memory_slots,
                config.long_term_memory_width,
                generator=generator,
            ),
            provenance_ids=torch.arange(config.long_term_memory_slots),
            layer_scale=config.long_term_memory_layer_scale,
            norm_eps=config.norm_eps,
        )
    return AblationLM(config, long_term_memory=memory)


def _manual_dense_forward(model: AblationLM, tokens: torch.Tensor) -> torch.Tensor:
    hidden = model.token_embedding(tokens)
    positions = torch.arange(tokens.shape[1]).view(1, -1).expand(tokens.shape[0], -1)
    for block in model.prelude_blocks:
        hidden = block(hidden, position_ids=positions)
    for block in model.core_blocks:
        hidden = block(hidden, position_ids=positions)
    for block in model.coda_blocks:
        hidden = block(hidden, position_ids=positions)
    return model.lm_head(model.final_norm(hidden))


def _manual_static_kv_forward(model: AblationLM, tokens: torch.Tensor) -> torch.Tensor:
    hidden = model.token_embedding(tokens)
    positions = torch.arange(tokens.shape[1]).view(1, -1).expand(tokens.shape[0], -1)
    for block in model.prelude_blocks:
        hidden = block(hidden, position_ids=positions)
    anchor = hidden
    assert model.loop_embedding is not None
    hidden = hidden + model.loop_embedding.weight[0]
    for block in model.core_blocks:
        hidden = block(
            hidden,
            projected_kv=block.project_kv(anchor, position_ids=positions),
            position_ids=positions,
        )
    for block in model.coda_blocks:
        hidden = block(hidden, position_ids=positions)
    return model.lm_head(model.final_norm(hidden))


def _manual_static_visit_logits(
    model: AblationLM,
    tokens: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    document_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    effective_documents = model._contiguous_document_segments(
        tokens,
        attention_mask,
        document_ids,
    )
    assert effective_documents is not None
    positions = model._document_position_ids(effective_documents)
    embedded = model.token_embedding(tokens)
    hidden = embedded
    if model.front_hadamard is not None:
        hidden = model.front_hadamard(hidden)
    for block_index, block in enumerate(model.prelude_blocks):
        hidden = block(
            hidden,
            attention_mask=attention_mask,
            position_ids=positions,
            document_ids=effective_documents,
        )
        if block_index == 0 and model.engram is not None:
            hidden, _engram_audit = model.engram(
                hidden,
                tokens,
                document_ids=effective_documents,
                enabled=True,
            )
    prelude = hidden
    lanes = model.scratch.initialize(prelude) if model.scratch is not None else None
    core_kv_cache = model._project_core_kv(prelude, position_ids=positions)
    alpha = model.config.recurrence_scale(model.config.recurrent_steps)
    visit_logits: list[torch.Tensor] = []
    for step_index in range(model.config.recurrent_steps):
        if (
            model.config.static_kv_midpoint_refresh
            and step_index == model.config.recurrent_steps // 2
        ):
            core_kv_cache = model._project_core_kv(hidden, position_ids=positions)
        hidden, lanes = model._run_recurrent_visit(
            hidden,
            prelude=prelude,
            lanes=lanes,
            core_kv_cache=core_kv_cache,
            step_index=step_index,
            alpha=alpha,
            attention_mask=attention_mask,
            position_ids=positions,
            document_ids=effective_documents,
        )
        readout_hidden = hidden
        if model.long_term_memory is not None:
            readout_hidden, _memory_audit = model.long_term_memory(
                readout_hidden,
                record_ids=None,
            )
        for block in model.coda_blocks:
            readout_hidden = block(
                readout_hidden,
                attention_mask=attention_mask,
                position_ids=positions,
                document_ids=effective_documents,
            )
        visit_logits.append(model.lm_head(model.final_norm(readout_hidden)))
    return embedded, visit_logits


def test_t1_disabled_graph_is_exactly_the_dense_transformer() -> None:
    torch.manual_seed(3)
    model = _model(_tiny_config()).eval()
    tokens = torch.randint(0, 64, (2, 7))

    actual = model(tokens).logits
    reference = _manual_dense_forward(model, tokens)

    assert model.lm_head.weight is model.token_embedding.weight
    assert model.loop_embedding is None
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    with pytest.raises(ValueError, match="structural recurrence"):
        model(tokens, recurrent_steps=2)


def test_s0_causality_gradient_is_exactly_zero_for_future_positions() -> None:
    torch.manual_seed(4)
    model = _model(_tiny_config()).eval()
    tokens = torch.randint(0, 64, (1, 8))
    captured: list[torch.Tensor] = []

    def retain_embedding_gradient(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        output.retain_grad()
        captured.append(output)

    handle = model.token_embedding.register_forward_hook(retain_embedding_gradient)
    try:
        logits = model(tokens).logits
        logits[0, 3, 7].backward()
    finally:
        handle.remove()

    assert len(captured) == 1 and captured[0].grad is not None
    gradient = captured[0].grad
    assert torch.count_nonzero(gradient[0, :4]) > 0
    assert torch.count_nonzero(gradient[0, 4:]) == 0


def test_t15_model_static_kv_matches_fresh_anchor_projection_at_k1() -> None:
    torch.manual_seed(6)
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=1,
        use_static_kv_core=True,
    )
    model = _model(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 7))

    actual = model(tokens).logits
    reference = _manual_static_kv_forward(model, tokens)

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_static_kv_removes_the_recurrent_visit_multiplier_from_projection() -> None:
    torch.manual_seed(7)
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=3,
        use_static_kv_core=True,
    )
    model = _model(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 7))
    projection_calls = 0

    def count_projection(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal projection_calls
        projection_calls += 1

    handles = [
        block.attention.k_proj.register_forward_hook(count_projection)
        for block in model.core_blocks
    ]
    try:
        output = model(tokens, return_diagnostics=True)
    finally:
        for handle in handles:
            handle.remove()

    assert projection_calls == config.n_core_blocks
    expected_elements = (
        tokens.shape[0]
        * config.n_core_blocks
        * tokens.shape[1]
        * 2
        * config.n_kv_heads
        * config.head_dim
    )
    assert output.diagnostics["main_graph_core_kv_projection_events"] == 1
    assert output.diagnostics["main_graph_core_kv_linear_projection_calls"] == (
        2 * config.n_core_blocks
    )
    assert output.diagnostics["static_kv_elements_per_generation"] == expected_elements
    assert output.diagnostics["static_kv_bytes_per_generation"] == 4 * expected_elements
    position_elements = tokens.numel()
    assert output.diagnostics[
        "static_kv_position_metadata_elements_per_generation"
    ] == position_elements
    assert output.diagnostics[
        "static_kv_position_metadata_bytes_per_generation"
    ] == position_elements * torch.tensor([], dtype=torch.long).element_size()
    assert output.diagnostics["static_kv_total_bytes_per_generation"] == (
        4 * expected_elements
        + position_elements * torch.tensor([], dtype=torch.long).element_size()
    )
    assert output.diagnostics["static_kv_cumulative_projected_elements"] == (
        expected_elements
    )


def test_static_recurrent_visit_fails_closed_without_caller_owned_cache() -> None:
    config = _tiny_config(
        use_recurrence=True,
        use_static_kv_core=True,
    )
    model = _model(config).eval()
    prelude = torch.randn(1, 5, config.d_model)

    with pytest.raises(ValueError, match="projected once by the caller"):
        model._run_recurrent_visit(
            prelude,
            prelude=prelude,
            lanes=None,
            step_index=0,
            alpha=1.0,
            attention_mask=None,
            position_ids=None,
            document_ids=None,
        )


def test_static_recurrent_visit_rejects_trusted_query_position_mismatch() -> None:
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=2,
        use_static_kv_core=True,
    )
    model = _model(config).eval()
    hidden = torch.randn(1, 4, config.d_model)
    positions = torch.arange(4).view(1, -1)
    cache = model._project_core_kv(hidden, position_ids=positions)

    with pytest.raises(ValueError, match="position IDs differ"):
        model._run_recurrent_visit(
            hidden,
            prelude=hidden,
            lanes=None,
            core_kv_cache=cache,
            step_index=0,
            alpha=config.recurrence_scale(),
            attention_mask=None,
            position_ids=positions + 1,
            document_ids=None,
        )


def test_static_kv_position_receipt_is_immutable_against_caller_mutation() -> None:
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=2,
        use_static_kv_core=True,
    )
    model = _model(config).eval()
    hidden = torch.randn(1, 4, config.d_model)
    positions = torch.arange(4).view(1, -1)
    original_positions = positions.clone()
    cache = model._project_core_kv(hidden, position_ids=positions)

    positions.add_(1)
    torch.testing.assert_close(cache[0].position_ids, original_positions)
    with pytest.raises(ValueError, match="position IDs differ"):
        model._run_recurrent_visit(
            hidden,
            prelude=hidden,
            lanes=None,
            core_kv_cache=cache,
            step_index=0,
            alpha=config.recurrence_scale(),
            attention_mask=None,
            position_ids=positions,
            document_ids=None,
        )


def test_t2_static_kv_parameters_receive_live_gradients_at_first_backward() -> None:
    torch.manual_seed(71)
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=3,
        use_static_kv_core=True,
    )
    model = _model(config).train()
    tokens = torch.randint(0, config.vocab_size, (2, 7))

    output = model(tokens, labels=tokens)
    assert output.loss is not None
    output.loss.backward()

    for block in model.core_blocks:
        for projection in (block.attention.k_proj, block.attention.v_proj):
            gradient = projection.weight.grad
            assert gradient is not None
            assert bool(torch.isfinite(gradient).all())
            assert torch.count_nonzero(gradient) > 0


def test_fork_b_prime_refreshes_static_kv_once_at_the_midpoint() -> None:
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=4,
        use_static_kv_core=True,
        static_kv_midpoint_refresh=True,
    )
    model = _model(config).eval()
    reference = copy.deepcopy(model)
    tokens = torch.randint(0, config.vocab_size, (1, 6))
    attention_mask = torch.ones_like(tokens)
    document_ids = torch.zeros_like(tokens)
    k_projection_calls = 0
    v_projection_calls = 0

    def count_k(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal k_projection_calls
        k_projection_calls += 1

    def count_v(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal v_projection_calls
        v_projection_calls += 1

    handles = []
    for block in model.core_blocks:
        handles.append(block.attention.k_proj.register_forward_hook(count_k))
        handles.append(block.attention.v_proj.register_forward_hook(count_v))
    try:
        with patch.object(
            model,
            "_project_core_kv",
            wraps=model._project_core_kv,
        ) as project:
            output = model(
                tokens,
                attention_mask=attention_mask,
                document_ids=document_ids,
                return_diagnostics=True,
            )
            projection_sources = [
                call.args[0].detach().clone() for call in project.call_args_list
            ]
    finally:
        for handle in handles:
            handle.remove()

    embedded, reference_visits = _manual_static_visit_logits(
        reference,
        tokens,
        attention_mask=attention_mask,
        document_ids=document_ids,
    )
    del embedded
    effective_documents = reference._contiguous_document_segments(
        tokens,
        attention_mask,
        document_ids,
    )
    assert effective_documents is not None
    positions = reference._document_position_ids(effective_documents)
    expected_hidden = reference.token_embedding(tokens)
    for block in reference.prelude_blocks:
        expected_hidden = block(
            expected_hidden,
            attention_mask=attention_mask,
            position_ids=positions,
            document_ids=effective_documents,
        )
    expected_prelude = expected_hidden
    expected_cache = reference._project_core_kv(
        expected_prelude,
        position_ids=positions,
    )
    alpha = config.recurrence_scale(config.recurrent_steps)
    for step_index in range(config.recurrent_steps // 2):
        expected_hidden, _ = reference._run_recurrent_visit(
            expected_hidden,
            prelude=expected_prelude,
            lanes=None,
            core_kv_cache=expected_cache,
            step_index=step_index,
            alpha=alpha,
            attention_mask=attention_mask,
            position_ids=positions,
            document_ids=effective_documents,
        )

    assert output.diagnostics["main_graph_core_kv_projection_events"] == 2
    assert output.diagnostics["static_kv_peak_elements_upper_bound"] == (
        2 * output.diagnostics["static_kv_elements_per_generation"]
    )
    assert output.diagnostics["static_kv_midpoint_refresh_requested"] is True
    assert output.diagnostics["static_kv_midpoint_refresh_executed"] is True
    assert output.diagnostics["static_kv_midpoint_refresh_visit"] == 2
    assert "local_jacobian_cache_semantics_by_visit" not in output.diagnostics
    assert k_projection_calls == 2 * config.n_core_blocks
    assert v_projection_calls == 2 * config.n_core_blocks
    assert len(projection_sources) == 2
    torch.testing.assert_close(projection_sources[0], expected_prelude, rtol=0, atol=0)
    torch.testing.assert_close(projection_sources[1], expected_hidden, rtol=0, atol=0)
    torch.testing.assert_close(output.logits, reference_visits[-1], rtol=0, atol=0)


def test_fork_b_prime_reports_requested_but_unexecuted_at_k1() -> None:
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=1,
        use_static_kv_core=True,
        static_kv_midpoint_refresh=True,
    )
    output = _model(config).eval()(
        torch.randint(0, config.vocab_size, (1, 6)),
        return_diagnostics=True,
    )

    assert output.diagnostics["static_kv_midpoint_refresh_requested"] is True
    assert output.diagnostics["static_kv_midpoint_refresh_executed"] is False
    assert output.diagnostics["static_kv_midpoint_refresh_visit"] is None
    assert output.diagnostics["main_graph_core_kv_projection_events"] == 1


@pytest.mark.parametrize(
    ("steps", "midpoint_refresh"),
    ((1, False), (2, False), (3, False), (4, True)),
)
def test_t14b_static_kv_is_exactly_causal_at_every_visit_horizon(
    steps: int,
    midpoint_refresh: bool,
) -> None:
    torch.manual_seed(8 + steps)
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=steps,
        use_static_kv_core=True,
        static_kv_midpoint_refresh=midpoint_refresh,
    )
    model = _model(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 8))
    captured: list[torch.Tensor] = []

    def retain_embedding_gradient(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        output.retain_grad()
        captured.append(output)

    handle = model.token_embedding.register_forward_hook(retain_embedding_gradient)
    try:
        logits = model(tokens).logits
        logits[0, 3, 7].backward()
    finally:
        handle.remove()

    assert len(captured) == 1 and captured[0].grad is not None
    gradient = captured[0].grad
    assert torch.count_nonzero(gradient[0, :4]) > 0
    assert torch.count_nonzero(gradient[0, 4:]) == 0


@pytest.mark.parametrize("midpoint_refresh", (False, True))
def test_t14b_static_kv_checks_every_k_with_packing_and_padding(
    midpoint_refresh: bool,
) -> None:
    torch.manual_seed(121)
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=4,
        use_static_kv_core=True,
        static_kv_midpoint_refresh=midpoint_refresh,
        use_front_hadamard_experts=True,
        use_reentry_bridge=True,
        use_scratch=True,
        use_lane_carrier=True,
        use_engram=True,
        use_long_term_memory=True,
    )
    model = _model(config).eval()
    assert model.front_hadamard is not None
    assert model.reentry_bridge is not None
    assert model.scratch is not None and model.scratch.carrier is not None
    assert model.engram is not None
    assert model.long_term_memory is not None
    tokens = torch.tensor([[1, 2, 3, 10, 11, 12, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    document_ids = torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1]])
    embedded, visit_logits = _manual_static_visit_logits(
        model,
        tokens,
        attention_mask=attention_mask,
        document_ids=document_ids,
    )

    assert len(visit_logits) == config.recurrent_steps
    for logits in visit_logits:
        first_document_gradient = torch.autograd.grad(
            logits[0, 1, 7],
            embedded,
            retain_graph=True,
        )[0]
        assert torch.count_nonzero(first_document_gradient[0, :2]) > 0
        assert torch.count_nonzero(first_document_gradient[0, 2:]) == 0

        second_document_gradient = torch.autograd.grad(
            logits[0, 4, 7],
            embedded,
            retain_graph=True,
        )[0]
        assert torch.count_nonzero(second_document_gradient[0, :3]) == 0
        assert torch.count_nonzero(second_document_gradient[0, 3:5]) > 0
        assert torch.count_nonzero(second_document_gradient[0, 5:]) == 0


def test_full_active_graph_is_causal_under_future_token_perturbation() -> None:
    torch.manual_seed(5)
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=2,
        use_static_kv_core=True,
        use_front_hadamard_experts=True,
        use_reentry_bridge=True,
        use_scratch=True,
        use_lane_carrier=True,
        use_engram=True,
        use_long_term_memory=True,
    )
    model = _model(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    changed = tokens.clone()
    changed[:, 5:] = torch.randint(0, config.vocab_size, (2, 3))

    original_logits = model(tokens).logits
    changed_logits = model(changed).logits

    torch.testing.assert_close(original_logits[:, :5], changed_logits[:, :5], rtol=0, atol=0)


def test_packed_document_boundaries_isolate_attention_engram_and_scratch() -> None:
    torch.manual_seed(9)
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=2,
        use_scratch=True,
        use_engram=True,
    )
    model = _model(config).eval()
    tokens = torch.tensor([[1, 2, 3, 10, 11, 12, 13]])
    documents = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    changed = tokens.clone()
    changed[:, :3] = torch.tensor([[31, 32, 33]])

    original = model(tokens, document_ids=documents).logits
    perturbed = model(changed, document_ids=documents).logits

    torch.testing.assert_close(original[:, 3:], perturbed[:, 3:], rtol=0, atol=0)
    assert model._document_position_ids(documents).tolist() == [[0, 1, 2, 0, 1, 2, 3]]


def test_reused_document_labels_cannot_reconnect_noncontiguous_segments() -> None:
    torch.manual_seed(10)
    config = _tiny_config(use_engram=True)
    model = _model(config).eval()
    tokens = torch.tensor([[1, 2, 10, 11, 20, 21]])
    documents = torch.tensor([[0, 0, 1, 1, 0, 0]])
    changed = tokens.clone()
    changed[:, :2] = torch.tensor([[31, 32]])

    original = model(tokens, document_ids=documents).logits
    perturbed = model(changed, document_ids=documents).logits

    torch.testing.assert_close(original[:, 4:], perturbed[:, 4:], rtol=0, atol=0)


def test_padding_tokens_do_not_enter_attention_or_engram_suffixes() -> None:
    torch.manual_seed(11)
    config = _tiny_config(use_engram=True)
    model = _model(config).eval()
    tokens = torch.tensor([[50, 51, 1, 2, 3, 4]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
    changed = tokens.clone()
    changed[:, :2] = torch.tensor([[60, 61]])

    original = model(tokens, attention_mask=mask).logits
    perturbed = model(changed, attention_mask=mask).logits

    torch.testing.assert_close(original[:, 2:], perturbed[:, 2:], rtol=0, atol=0)


def test_shifted_loss_masks_both_source_and_target_padding_positions() -> None:
    model = _model(_tiny_config())
    logits = torch.randn(1, 5, 64, requires_grad=True)
    labels = torch.tensor([[9, 8, 1, 2, 3]])
    mask = torch.tensor([[0, 0, 1, 1, 1]])

    loss = model._language_model_loss(logits, labels, mask, None)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[:, 1].abs().sum().item() == 0.0
    assert logits.grad[:, 2].abs().sum().item() > 0.0


def test_active_modules_and_inner_branches_are_gradient_live() -> None:
    torch.manual_seed(13)
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=2,
        use_front_hadamard_experts=True,
        use_reentry_bridge=True,
        use_scratch=True,
        use_lane_carrier=True,
        use_engram=True,
        use_long_term_memory=True,
    )
    model = _model(config)
    tokens = torch.randint(0, config.vocab_size, (2, 7))
    record_ids = torch.tensor([[0] * 7, [1] * 7])
    output = model(
        tokens,
        labels=tokens,
        memory_record_ids=record_ids,
        return_diagnostics=True,
        capture_loop_gradients=True,
    )
    assert output.loss is not None
    output.loss.backward()

    selected = (
        model.front_hadamard.router.weight,
        model.front_hadamard.expert_gains,
        model.reentry_bridge.projection.weight,
        model.scratch.initializer.weight,
        model.scratch.update_out.weight,
        model.scratch.readout.weight,
        model.scratch.carrier.raw_rho,
        model.engram.query_proj.weight,
        next(iter(model.engram.tables.values())).weight,
        model.long_term_memory.query.weight,
        model.long_term_memory.output.weight,
    )
    for parameter in selected:
        assert parameter.grad is not None
        assert parameter.grad.abs().sum().item() > 0
    for row in range(config.recurrent_steps):
        assert model.scratch.step_embedding.weight.grad[row].abs().sum().item() > 0
    assert output.diagnostics["alpha_t"] == 0.5
    assert output.diagnostics["recurrence_enabled"] is True
    assert output.diagnostics["executed_core_visits"] == 2
    assert output.diagnostics["executed_core_block_passes"] == 4
    assert output.diagnostics["loop_rms"].shape == (2,)
    assert output.diagnostics["loop_update_rms"].shape == (2,)
    assert output.diagnostics["trajectory_state_count"] == 3
    assert "trajectory_jets" in output.diagnostics
    assert output.diagnostics["trajectory_jets"]["velocity_rms"].shape[0] == 2
    assert output.diagnostics["trajectory_jets"]["plane_probes"].shape[-1] == (
        config.jet_plane_probe_count
    )
    gradient_metrics = output.diagnostics["loop_gradient_probe"].metrics()
    assert gradient_metrics.gradient_rms.shape == (2,)
    assert gradient_metrics.adjacent_cosines.shape == (1,)
    assert output.diagnostics["lane_carrier_minimum_retention"].item() >= 0.9
    assert set(output.diagnostics["hadamard_router"]) == {
        "m",
        "s",
        "load",
        "load_std",
        "routing_entropy",
    }


def test_semantic_optimizer_partition_and_parameter_accounting_cover_full_model() -> None:
    config = _tiny_config(use_engram=True, use_long_term_memory=True)
    model = _model(config)
    partition = partition_optimizer_parameters(model)
    accounting = parameter_accounting(model)
    grouped = [parameter for group in partition.groups for parameter in group.parameters]

    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert (
        partition.assignment_for("token_embedding.weight").target
        is OptimizerTarget.AUXILIARY_ADAMW
    )
    assert (
        partition.assignment_for("core_blocks.0.attention.q_proj.weight").target
        is OptimizerTarget.MUON_ELIGIBLE
    )
    assert partition.assignment_for("final_norm.weight").role is ParameterRole.NORMALIZATION
    assert partition.assignment_for("engram.query_proj.weight").role is ParameterRole.ENGRAM
    assert accounting.vocabulary == config.vocab_size * config.d_model
    assert accounting.engram_tables == (
        len(config.engram_orders)
        * config.engram_hashes_per_order
        * config.engram_table_size
        * config.engram_row_dim
    )
    assert accounting.engram_interface > 0
    assert accounting.total == sum(parameter.numel() for parameter in model.parameters())
    assert accounting.total == (
        accounting.vocabulary
        + accounting.engram_tables
        + accounting.engram_interface
        + accounting.long_term_memory_trainable
        + accounting.non_vocabulary_dense
    )
    assert accounting.engram_frozen_table_elements == 0
    assert accounting.long_term_memory_store_elements == (
        2 * config.long_term_memory_slots * config.long_term_memory_width
    )


def test_all_coupled_scratch_mode_matrices_stay_together_on_adamw() -> None:
    model = _model(_tiny_config(use_scratch=True, use_lane_carrier=True))
    partition = partition_optimizer_parameters(model)
    names = (
        "scratch.initializer.weight",
        "scratch.context_projection.weight",
        "scratch.update_in.weight",
        "scratch.update_out.weight",
        "scratch.readout.weight",
    )

    for name in names:
        assignment = partition.assignment_for(name)
        assert assignment.role is ParameterRole.COUPLED_MODE
        assert assignment.target is OptimizerTarget.AUXILIARY_ADAMW


def test_parameter_accounting_is_invariant_to_a_transparent_model_wrapper() -> None:
    class _Wrapper(nn.Module):
        def __init__(self, model: AblationLM) -> None:
            super().__init__()
            self.model = model

    model = _model(_tiny_config(use_engram=True, use_long_term_memory=True))

    assert parameter_accounting(_Wrapper(model)) == parameter_accounting(model)


def test_scratch_and_birkhoff_carrier_are_independent_structural_arms() -> None:
    scratch_only = _model(_tiny_config(use_scratch=True))
    scratch_with_carrier = _model(
        _tiny_config(use_scratch=True, use_lane_carrier=True)
    )

    assert scratch_only.scratch.carrier is None
    assert scratch_with_carrier.scratch.carrier is not None


def test_accounting_handles_a_deliberately_frozen_engram_table() -> None:
    config = _tiny_config(use_engram=True)
    model = _model(config)
    first_table = next(iter(model.engram.tables.values()))
    first_table.weight.requires_grad_(False)

    accounting = parameter_accounting(model)

    assert accounting.engram_frozen_table_elements == first_table.weight.numel()
    assert accounting.engram_tables == (
        (len(config.engram_orders) * config.engram_hashes_per_order - 1)
        * config.engram_table_size
        * config.engram_row_dim
    )


def test_seeded_dense_initialization_is_depth_scaled_and_ablation_invariant() -> None:
    baseline_config = _tiny_config()
    baseline = _model(baseline_config)
    with_engram = _model(replace(baseline_config, use_engram=True))

    torch.testing.assert_close(
        baseline.token_embedding.weight,
        with_engram.token_embedding.weight,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        baseline.core_blocks[0].attention.q_proj.weight,
        with_engram.core_blocks[0].attention.q_proj.weight,
        rtol=0,
        atol=0,
    )
    ordinary = torch.cat(
        [block.attention.q_proj.weight.flatten() for block in baseline.core_blocks]
    )
    residual = torch.cat(
        [block.attention.output_proj.weight.flatten() for block in baseline.core_blocks]
    )
    physical_blocks = (
        baseline_config.n_prelude_layers
        + baseline_config.n_core_blocks
        + baseline_config.n_coda_layers
    )
    assert ordinary.std().item() == pytest.approx(0.02, rel=0.2)
    assert residual.std().item() == pytest.approx(
        0.02 / (2 * physical_blocks) ** 0.5,
        rel=0.2,
    )


def test_executed_zero_scale_engram_matches_structural_off_graph_exactly() -> None:
    config = _tiny_config()
    structural_off = _model(config).eval()
    attached = _model(replace(config, use_engram=True)).eval()
    with torch.no_grad():
        attached.engram.raw_residual_scale.zero_()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])

    off_logits = structural_off(tokens).logits
    attached_logits = attached(tokens).logits

    torch.testing.assert_close(off_logits, attached_logits, rtol=0, atol=0)


def test_long_term_store_is_frozen_and_excludes_matching_provenance() -> None:
    config = _tiny_config(
        use_long_term_memory=True,
        long_term_memory_slots=2,
        long_term_memory_width=8,
    )
    model = _model(config).eval()
    tokens = torch.tensor([[1, 2, 3]])
    record_ids = torch.zeros_like(tokens)

    before = model(tokens, memory_record_ids=record_ids).logits
    with torch.no_grad():
        model.long_term_memory.memory_keys[0].fill_(1_000.0)
        model.long_term_memory.memory_values[0].fill_(1_000.0)
    after = model(tokens, memory_record_ids=record_ids).logits

    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert not model.long_term_memory.memory_keys.requires_grad
    assert not model.long_term_memory.memory_values.requires_grad


def test_training_with_long_term_memory_requires_leave_one_out_ids() -> None:
    config = _tiny_config(use_long_term_memory=True)
    model = _model(config).train()
    tokens = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError, match="leave-one-out"):
        model(tokens, labels=tokens)


def test_cache_is_a_hard_gate_until_identity_and_accounting_pass() -> None:
    model = _model(_tiny_config())
    with pytest.raises(NotImplementedError, match="identity"):
        model(torch.ones(1, 3, dtype=torch.long), use_cache=True)


@pytest.mark.parametrize(
    "mask",
    (
        torch.tensor([[0.0, 1.0, 1.0]]),
        torch.tensor([[0, -1, 1]]),
        torch.tensor([[0, 2, 1]]),
    ),
)
def test_attention_mask_rejects_nonbinary_or_floating_conventions(mask: torch.Tensor) -> None:
    model = _model(_tiny_config())

    with pytest.raises((TypeError, ValueError), match="attention_mask"):
        model(torch.ones(1, 3, dtype=torch.long), attention_mask=mask)


def test_forward_rejects_silent_label_and_recurrence_coercions() -> None:
    model = _model(_tiny_config(use_recurrence=True))
    tokens = torch.ones(1, 3, dtype=torch.long)

    with pytest.raises(TypeError, match="recurrent_steps"):
        model(tokens, recurrent_steps=2.5)
    with pytest.raises(TypeError, match="labels"):
        model(tokens, labels=tokens.float())
    with pytest.raises(ValueError, match="valid vocabulary"):
        model(tokens, labels=torch.tensor([[1, 2, 64]]))


def test_one_active_checkpoint_keeps_recurrent_k_inference_controllable() -> None:
    torch.manual_seed(122)
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=1,
        recurrence_coefficient=0.75,
        use_reentry_bridge=True,
    )
    model = _model(config).eval()
    assert model.reentry_bridge is not None
    tokens = torch.randint(0, config.vocab_size, (2, 7))

    expected_bridge_visits = {1: 0, 2: 1, 4: 3}
    outputs = {}
    with patch.object(
        model.reentry_bridge,
        "forward",
        wraps=model.reentry_bridge.forward,
    ) as bridge_forward:
        previous_calls = 0
        for steps in (1, 2, 4):
            output = model(tokens, recurrent_steps=steps, return_diagnostics=True)
            outputs[steps] = output.logits
            assert model.config.recurrent_steps == 1
            assert output.diagnostics["alpha_t"] == pytest.approx(0.75 / steps)
            receipt = output.diagnostics["composition_receipt"]
            assert receipt["requested_visits"] == steps
            assert receipt["executed_visits"] == steps
            assert output.diagnostics["reentry_bridge_requested"] is True
            assert (
                output.diagnostics["reentry_bridge_executed_visits"]
                == expected_bridge_visits[steps]
            )
            assert bridge_forward.call_count - previous_calls == expected_bridge_visits[steps]
            previous_calls = bridge_forward.call_count

    with patch.object(model, "reentry_bridge", None):
        bypassed = model(tokens, recurrent_steps=1, return_diagnostics=True)
        assert bypassed.diagnostics["reentry_bridge_executed_visits"] == 0
    torch.testing.assert_close(outputs[1], bypassed.logits, rtol=0, atol=0)
    assert model.reentry_bridge is not None
    assert model.config.recurrent_steps == 1

    with pytest.raises(ValueError, match="configured recurrence cap"):
        model(tokens, recurrent_steps=0)
    with pytest.raises(ValueError, match="configured recurrence cap"):
        model(tokens, recurrent_steps=config.max_recurrent_steps + 1)
    with pytest.raises(TypeError, match="recurrent_steps"):
        model(tokens, recurrent_steps=2.0)


def test_z_loss_uses_the_same_valid_next_token_mask_as_cross_entropy() -> None:
    model = _model(_tiny_config(z_loss_coefficient=0.1))
    logits = torch.randn(1, 5, 64, requires_grad=True)
    labels = torch.tensor([[9, 8, 1, 2, 3]])
    mask = torch.tensor([[0, 0, 1, 1, 1]])

    total, cross_entropy, z_loss = model._language_model_loss_components(
        logits,
        labels,
        mask,
        None,
    )
    valid_log_partition = torch.logsumexp(logits[:, 2:4].float(), dim=-1)

    torch.testing.assert_close(z_loss, 0.1 * valid_log_partition.square().mean())
    torch.testing.assert_close(total, cross_entropy + z_loss)


def test_zero_z_loss_coefficient_is_structural_cross_entropy_identity(monkeypatch) -> None:
    model = _model(_tiny_config(z_loss_coefficient=0.0))
    logits = torch.randn(1, 4, 64)
    labels = torch.tensor([[1, 2, 3, 4]])

    def forbidden_logsumexp(*_args, **_kwargs):
        raise AssertionError("zero z-loss must not execute logsumexp")

    monkeypatch.setattr(torch, "logsumexp", forbidden_logsumexp)
    total, cross_entropy, z_loss = model._language_model_loss_components(
        logits,
        labels,
        None,
        None,
    )

    torch.testing.assert_close(total, cross_entropy, rtol=0, atol=0)
    assert z_loss.item() == 0.0


def test_model_level_recurrent_jacobian_probe_is_finite_per_visit() -> None:
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=2,
        use_scratch=True,
        use_lane_carrier=True,
    )
    model = _model(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[1, 1, 0, 0]])

    output = model(
        tokens,
        attention_mask=mask,
        return_diagnostics=True,
        jacobian_probe_iterations=2,
    )
    changed_padding = model(
        torch.tensor([[1, 2, 55, 56]]),
        attention_mask=mask,
        return_diagnostics=True,
        jacobian_probe_iterations=2,
    )

    estimates = output.diagnostics["loop_jacobian_spectral_norm"]
    assert estimates.shape == (2,)
    assert bool(torch.isfinite(estimates).all())
    assert bool(estimates.gt(0).all())
    assert output.diagnostics["local_jacobian_cache_semantics_by_visit"] == (
        (0, "dynamic_kv_total_derivative"),
        (1, "dynamic_kv_total_derivative"),
    )
    horizon = output.diagnostics["horizon_jacobian_spectral_norm"]
    assert horizon.ndim == 0
    assert torch.isfinite(horizon)
    assert horizon.item() > 0
    assert horizon.item() < 10
    assert output.diagnostics["joint_state_lane_metric_scale"] == pytest.approx(
        (config.d_model / (2 * config.scratch_width)) ** 0.5
    )
    torch.testing.assert_close(
        estimates,
        changed_padding.diagnostics["loop_jacobian_spectral_norm"],
    )
    torch.testing.assert_close(
        horizon,
        changed_padding.diagnostics["horizon_jacobian_spectral_norm"],
    )
    torch.testing.assert_close(
        output.diagnostics["scratch_mu_rms"],
        changed_padding.diagnostics["scratch_mu_rms"],
    )
    torch.testing.assert_close(
        output.diagnostics["scratch_delta_rms"],
        changed_padding.diagnostics["scratch_delta_rms"],
    )


def test_static_midpoint_jacobian_receipts_distinguish_partial_and_total_visits() -> None:
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=2,
        use_static_kv_core=True,
        static_kv_midpoint_refresh=True,
    )
    output = _model(config).eval()(
        torch.tensor([[1, 2, 3, 4]]),
        return_diagnostics=True,
        jacobian_probe_iterations=1,
    )

    assert output.diagnostics["local_jacobian_cache_semantics_by_visit"] == (
        (0, "fixed_cache_partial_derivative"),
        (1, "refresh_cache_total_derivative"),
    )


def test_hadamard_router_audit_excludes_padding_tokens() -> None:
    model = _model(_tiny_config(use_front_hadamard_experts=True)).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[1, 1, 0, 0]])

    output = model(tokens, attention_mask=mask, return_diagnostics=True)
    changed_padding = model(
        torch.tensor([[1, 2, 55, 56]]),
        attention_mask=mask,
        return_diagnostics=True,
    )
    assert model.front_hadamard is not None
    embeddings = model.token_embedding(tokens)
    logits = model.front_hadamard.router(model.front_hadamard.norm(embeddings)).float()
    valid_logits = logits[mask.bool()]

    torch.testing.assert_close(output.diagnostics["hadamard_router"]["m"], valid_logits.mean())
    torch.testing.assert_close(
        output.diagnostics["hadamard_router"]["s"],
        valid_logits.std(unbiased=False),
    )
    for name, value in output.diagnostics["hadamard_router"].items():
        torch.testing.assert_close(value, changed_padding.diagnostics["hadamard_router"][name])
    torch.testing.assert_close(
        output.diagnostics["loop_rms"],
        changed_padding.diagnostics["loop_rms"],
    )
    torch.testing.assert_close(
        output.diagnostics["loop_update_rms"],
        changed_padding.diagnostics["loop_update_rms"],
    )


def test_full_active_graph_is_finite_in_bfloat16_with_fp32_loss() -> None:
    config = _tiny_config(
        use_recurrence=True,
        recurrent_steps=2,
        use_static_kv_core=True,
        use_front_hadamard_experts=True,
        use_reentry_bridge=True,
        use_scratch=True,
        use_lane_carrier=True,
        use_engram=True,
        use_long_term_memory=True,
    )
    model = _model(config).to(torch.bfloat16)
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    record_ids = torch.tensor([[0] * 4, [1] * 4])

    output = model(tokens, labels=tokens, memory_record_ids=record_ids)

    assert output.logits.dtype is torch.bfloat16
    assert bool(torch.isfinite(output.logits).all())
    assert output.loss is not None and output.loss.dtype is torch.float32
    assert bool(torch.isfinite(output.loss))
