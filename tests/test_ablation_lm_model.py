from __future__ import annotations

from dataclasses import replace

import pytest
import torch

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


def test_full_active_graph_is_causal_under_future_token_perturbation() -> None:
    torch.manual_seed(5)
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
    assert partition.assignment_for("token_embedding.weight").target is OptimizerTarget.AUXILIARY_ADAMW
    assert partition.assignment_for("core_blocks.0.attention.q_proj.weight").target is OptimizerTarget.MUON_ELIGIBLE
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


def test_full_active_graph_is_finite_in_bfloat16_with_fp32_loss() -> None:
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
    model = _model(config).to(torch.bfloat16)
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    record_ids = torch.tensor([[0] * 4, [1] * 4])

    output = model(tokens, labels=tokens, memory_record_ids=record_ids)

    assert output.logits.dtype is torch.bfloat16
    assert bool(torch.isfinite(output.logits).all())
    assert output.loss is not None and output.loss.dtype is torch.float32
    assert bool(torch.isfinite(output.loss))
