from __future__ import annotations

import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from models.coconut_composite import (
    CoconutRecurrentQwen,
    DepthByAppendOutput,
    HorizontalIdentityBridge,
    assert_parameter_group_coverage,
    configure_composite_trainable_set,
)
from models.lora import LoRALinear, apply_lora_to_recurrent_block
from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM
from eval.eval_coconut_composite_integrity import optimizer_parameter_names


LATENT_ID = 29


def test_depth_by_append_output_rejects_failed_eviction_accounting() -> None:
    with pytest.raises(ValueError, match="evicted-slot count"):
        DepthByAppendOutput(
            predictions=torch.zeros((3, 4), dtype=torch.long),
            requested_append_steps=3,
            executed_decision_positions=3,
            total_grid_applications=12,
            feedback_grid_applications=9,
            evicted_slots=8,
            eviction_assertions=3,
            diagnostics={},
        ).assert_accounting()


def test_depth_by_append_output_accepts_exact_m7_counters() -> None:
    output = DepthByAppendOutput(
        predictions=torch.zeros((3, 4), dtype=torch.long),
        requested_append_steps=3,
        executed_decision_positions=3,
        total_grid_applications=12,
        feedback_grid_applications=9,
        evicted_slots=9,
        eviction_assertions=3,
        diagnostics={},
    )
    assert output.assert_accounting()["status"] == "exact"


def test_depth_by_append_output_counts_transient_readout_queries() -> None:
    output = DepthByAppendOutput(
        predictions=torch.zeros((2, 2), dtype=torch.long),
        requested_append_steps=1,
        executed_decision_positions=2,
        total_grid_applications=6,
        feedback_grid_applications=2,
        readout_grid_applications=2,
        evicted_slots=4,
        eviction_assertions=2,
        diagnostics={"read_at_t_query": True},
    )
    assert output.assert_accounting()["evicted_slots"] == 4


def test_m7_append_then_evict_is_invisible_to_later_real_tokens() -> None:
    model = tiny_composite().eval()
    inputs = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)
    baseline = model.depth_by_append(
        input_ids=inputs,
        append_steps=0,
        feedback_mode="raw",
        capture_real_logits=True,
    )
    appended = model.depth_by_append(
        input_ids=inputs,
        append_steps=2,
        feedback_mode="raw",
        capture_real_logits=True,
    )
    assert baseline.real_logits is not None and appended.real_logits is not None
    torch.testing.assert_close(appended.real_logits, baseline.real_logits, atol=1e-6, rtol=1e-6)
    assert appended.diagnostics["real_position_ids"] == [0, 1, 2, 3]
    assert appended.diagnostics["cache_lengths_after_eviction"] == [1, 2, 3, 4]


def test_m7_read_at_t_query_has_causal_transient_accounting() -> None:
    model = tiny_composite().eval()
    result = model.depth_by_append(
        input_ids=torch.tensor([[2, 3, 4]], dtype=torch.long),
        append_steps=1,
        feedback_mode="raw",
        read_at_t_query=True,
    )
    assert result.predictions.shape == (1, 2, 2)
    assert result.feedback_grid_applications == 2
    assert result.readout_grid_applications == 2
    assert result.evicted_slots == 4


def tiny_composite(*, dtype: torch.dtype = torch.float32) -> CoconutRecurrentQwen:
    torch.manual_seed(20260725)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
    )
    base = Qwen2ForCausalLM(config).to(dtype=dtype)
    recurrent = RecurrentQwenForCausalLM(base, layer_split=LayerSplit(1, 3))
    return CoconutRecurrentQwen(recurrent, latent_token_id=LATENT_ID)


def batch(horizontal_steps: int = 2) -> dict[str, torch.Tensor]:
    ids = [1, 2, 3] + [LATENT_ID] * horizontal_steps + [4, 5]
    input_ids = torch.tensor([ids], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def q_proj_weight(model: CoconutRecurrentQwen) -> torch.nn.Parameter:
    module = model.recurrent.qwen.layers[1].self_attn.q_proj
    return module.base.weight if isinstance(module, LoRALinear) else module.weight


@pytest.mark.parametrize("loops", [1, 2])
def test_rg1_h0_is_exact_recurrent_identity(loops: int) -> None:
    model = tiny_composite().eval()
    inputs = batch(horizontal_steps=0)
    with torch.no_grad():
        direct = model.recurrent(
            **inputs,
            max_loops=loops,
            use_cache=False,
            return_dict=True,
        )
        composite = model(**inputs, horizontal_steps=0, max_loops=loops)

    assert torch.equal(composite.logits, direct.logits)
    assert torch.equal(composite.loss, direct.loss)
    assert composite.total_grid_applications == loops


def test_rg1_zero_init_adapter_budget_preserves_h0_identity() -> None:
    model = tiny_composite().eval()
    assert apply_lora_to_recurrent_block(model.recurrent, rank=2, alpha=4) > 0
    configure_composite_trainable_set(
        model,
        budget="adapter_r16",
        horizontal_bridge_trainable=False,
    )
    inputs = batch(horizontal_steps=0)
    with torch.no_grad():
        direct = model.recurrent(
            **inputs,
            max_loops=2,
            use_cache=False,
            return_dict=True,
        )
        composite = model(**inputs, horizontal_steps=0, max_loops=2)

    assert torch.equal(composite.logits, direct.logits)


def test_rg2_zero_delta_bridge_is_exact_identity() -> None:
    bridge = HorizontalIdentityBridge(16)
    states = torch.randn(2, 16)

    assert torch.equal(bridge(states), states)
    assert torch.count_nonzero(bridge.delta.weight) == 0


def test_rg3_horizontal_and_prompt_paths_are_gradient_live_without_input_leakage() -> None:
    model = tiny_composite().eval()
    output = model(**batch(), horizontal_steps=2, max_loops=1)
    fed_grads = torch.autograd.grad(
        output.loss,
        (*output.horizontal_fed_states, output.input_embeddings),
        retain_graph=True,
    )

    assert all(torch.isfinite(gradient).all() for gradient in fed_grads)
    assert all(float(gradient.norm()) > 0.0 for gradient in fed_grads[:-1])
    embedding_grad = fed_grads[-1]
    assert float(embedding_grad[:, :3].norm()) > 0.0
    assert torch.count_nonzero(embedding_grad[:, 3:5]) == 0


def test_rg4_finite_difference_matches_horizontal_autograd_direction() -> None:
    model = tiny_composite().eval()
    inputs = batch(horizontal_steps=1)
    direction = torch.randn(1, 16)
    direction = direction / direction.norm()
    epsilon = torch.tensor(0.0, requires_grad=True)
    output = model(
        **inputs,
        horizontal_steps=1,
        max_loops=1,
        horizontal_state_additions={1: epsilon * direction},
    )
    analytic = torch.autograd.grad(output.loss, epsilon)[0]

    step = 1e-3
    with torch.no_grad():
        plus = model(
            **inputs,
            horizontal_steps=1,
            max_loops=1,
            horizontal_state_additions={1: step * direction},
        ).loss
        minus = model(
            **inputs,
            horizontal_steps=1,
            max_loops=1,
            horizontal_state_additions={1: -step * direction},
        ).loss
    finite_difference = (plus - minus) / (2.0 * step)

    assert torch.allclose(analytic, finite_difference, atol=2e-4, rtol=0.03)


def test_rg5_sliced_cache_matches_recompute_logits_loss_and_gradients() -> None:
    reference = tiny_composite().eval()
    cached = copy.deepcopy(reference).eval()
    inputs = batch(horizontal_steps=2)

    recompute = reference(**inputs, horizontal_steps=2, max_loops=1, execution_mode="recompute")
    recompute_gradient = torch.autograd.grad(recompute.loss, q_proj_weight(reference))[0]
    cache = cached(**inputs, horizontal_steps=2, max_loops=1, execution_mode="sliced_cache")
    cache_gradient = torch.autograd.grad(cache.loss, q_proj_weight(cached))[0]

    assert torch.allclose(recompute.logits, cache.logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(recompute.loss, cache.loss, atol=1e-7, rtol=1e-6)
    assert torch.allclose(recompute_gradient, cache_gradient, atol=1e-7, rtol=1e-5)
    assert cache.cache_prefix_lengths == (3, 4)
    assert cache.executed_execution_mode == "sliced_cache"


def test_rg5_cache_rejects_vertical_recurrence_and_checkpointing() -> None:
    model = tiny_composite().train()
    with pytest.raises(ValueError, match="only for one vertical loop"):
        model(**batch(1), horizontal_steps=1, max_loops=2, execution_mode="sliced_cache")

    model.recurrent.qwen.gradient_checkpointing = True
    with pytest.raises(ValueError, match="checkpointing"):
        model(**batch(1), horizontal_steps=1, max_loops=1, execution_mode="sliced_cache")


def test_cached_fallback_mask_without_attention_mask_covers_past_keys() -> None:
    model = tiny_composite()
    hidden = torch.zeros(1, 2, 16)
    mask = model.recurrent._fallback_causal_mask(
        None,
        hidden,
        cache_position=torch.tensor([3, 4]),
    )

    assert mask.shape == (1, 1, 2, 5)
    assert torch.count_nonzero(mask) == 1
    assert mask[0, 0, 0, 4] == torch.finfo(hidden.dtype).min


def test_rg6_adapter_budget_freezes_base_but_keeps_lora_and_feedback_live() -> None:
    model = tiny_composite().train()
    replaced = apply_lora_to_recurrent_block(
        model.recurrent,
        rank=2,
        alpha=4,
        adapter_dtype=torch.float32,
    )
    assert replaced > 0
    trainable = configure_composite_trainable_set(
        model,
        budget="adapter_r16",
        horizontal_bridge_trainable=False,
    )
    output = model(**batch(1), horizontal_steps=1, max_loops=1)
    output.loss.backward()

    base_parameters = [
        parameter
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in module.base.parameters()
    ]
    lora_parameters = [
        parameter
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in module.lora_parameters()
    ]
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in base_parameters)
    assert any(parameter.grad is not None and float(parameter.grad.norm()) > 0.0 for parameter in lora_parameters)
    assert all(state.grad is not None and float(state.grad.norm()) > 0.0 for state in output.horizontal_fed_states)
    assert trainable


def test_rg7_two_dimensional_grid_is_live_and_accounted_exactly() -> None:
    model = tiny_composite().eval()
    forward_calls = 0
    backward_calls = 0

    def forward_hook(_module, _inputs, _output) -> None:
        nonlocal forward_calls
        forward_calls += 1

    layer = model.recurrent.qwen.layers[1]
    forward_handle = layer.register_forward_hook(forward_hook)
    output = model(**batch(2), horizontal_steps=2, max_loops=2)
    grid_states = tuple(state for column in output.recurrent_application_states for state in column)

    def count_backward(gradient: torch.Tensor) -> torch.Tensor:
        nonlocal backward_calls
        backward_calls += 1
        return gradient

    tensor_handles = [state.register_hook(count_backward) for state in grid_states]
    output.loss.backward(retain_graph=True)
    forward_handle.remove()
    for handle in tensor_handles:
        handle.remove()
    gradients = torch.autograd.grad(output.loss, grid_states, allow_unused=True)

    assert len(output.recurrent_application_states) == 3
    assert all(len(column) == 2 for column in output.recurrent_application_states)
    assert output.feedback_grid_applications == 4
    assert output.total_grid_applications == 6
    assert forward_calls == output.total_grid_applications
    assert backward_calls == output.total_grid_applications
    assert all(gradient is not None and float(gradient.norm()) > 0.0 for gradient in gradients)


def test_rg8_parameter_group_coverage_hashes_exact_sets() -> None:
    names = {"one.weight", "two.bias"}
    receipt = assert_parameter_group_coverage(names, reversed(sorted(names)), names)

    assert receipt["passed"] is True
    assert receipt["optimizer_name_sha256"] == receipt["ema_name_sha256"]
    assert receipt["parameter_names"] == sorted(names)
    with pytest.raises(AssertionError, match="coverage differs"):
        assert_parameter_group_coverage(names, {"one.weight"}, names)


def test_rg8_optimizer_parameter_names_reads_instantiated_groups() -> None:
    model = tiny_composite()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.horizontal_bridge.delta.weight.requires_grad_(True)
    optimizer = torch.optim.AdamW([model.horizontal_bridge.delta.weight], lr=1e-4)

    assert optimizer_parameter_names(model, optimizer) == {
        "horizontal_bridge.delta.weight"
    }


def test_rg8_replaced_input_slot_has_no_gradient_even_with_tied_output_row() -> None:
    model = tiny_composite().eval()
    embedding = model.recurrent.qwen.embed_tokens
    model.recurrent.base_model.lm_head.weight = embedding.weight
    output = model(**batch(1), horizontal_steps=1, max_loops=1)
    activation_gradient, parameter_gradient = torch.autograd.grad(
        output.loss,
        (output.input_embeddings, embedding.weight),
    )

    assert torch.count_nonzero(activation_gradient[:, 3]) == 0
    # The shared row remains an LM-head class, so this is not input leakage.
    assert float(parameter_gradient[LATENT_ID].norm()) > 0.0


def test_rg9_anomaly_detection_accepts_one_full_backward() -> None:
    model = tiny_composite().eval()
    with torch.autograd.detect_anomaly():
        output = model(**batch(1), horizontal_steps=1, max_loops=1)
        output.loss.backward()
    assert torch.isfinite(output.loss)


def test_rg10_checkpointed_and_plain_recompute_match() -> None:
    plain = tiny_composite().train()
    checkpointed = copy.deepcopy(plain).train()
    checkpointed.recurrent.qwen.gradient_checkpointing = True
    inputs = batch(1)

    plain_output = plain(**inputs, horizontal_steps=1, max_loops=2)
    plain_gradient = torch.autograd.grad(plain_output.loss, q_proj_weight(plain))[0]
    checkpointed_output = checkpointed(**inputs, horizontal_steps=1, max_loops=2)
    checkpointed_gradient = torch.autograd.grad(
        checkpointed_output.loss,
        q_proj_weight(checkpointed),
    )[0]

    assert torch.allclose(plain_output.logits, checkpointed_output.logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(plain_gradient, checkpointed_gradient, atol=1e-7, rtol=1e-5)


def test_rg11_bfloat16_and_float32_gradient_directions_agree() -> None:
    fp32 = tiny_composite(dtype=torch.float32).eval()
    bf16 = tiny_composite(dtype=torch.float32).eval()
    bf16.load_state_dict(fp32.state_dict())
    bf16.to(dtype=torch.bfloat16)
    inputs = batch(1)

    fp32_output = fp32(**inputs, horizontal_steps=1, max_loops=1)
    fp32_gradient = torch.autograd.grad(fp32_output.loss, q_proj_weight(fp32))[0].float()
    bf16_output = bf16(**inputs, horizontal_steps=1, max_loops=1)
    bf16_gradient = torch.autograd.grad(bf16_output.loss, q_proj_weight(bf16))[0].float()
    cosine = torch.nn.functional.cosine_similarity(
        fp32_gradient.flatten(),
        bf16_gradient.flatten(),
        dim=0,
    )

    assert torch.isfinite(bf16_output.loss)
    assert all(torch.isfinite(state).all() for state in bf16_output.horizontal_fed_states)
    assert float(cosine) >= 0.99


def test_rg12_pilot_is_not_implicitly_authorized() -> None:
    model = tiny_composite()

    assert model.horizontal_bridge.delta.weight.requires_grad
    assert not hasattr(model, "rg12_training_loss_floor")
