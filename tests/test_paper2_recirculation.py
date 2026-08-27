from __future__ import annotations

import torch
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from eval.eval_paper2_recirculation_phase0 import projection_receipt
from models.recirculation import (
    PaperNativeRecirculationEvaluator,
    RecirculationConfig,
    graph_receipt,
)


def _qwen() -> Qwen2ForCausalLM:
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(config).eval()


def _gemma() -> Gemma3ForCausalLM:
    torch.manual_seed(11)
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=3,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    return Gemma3ForCausalLM(config).eval()


def _tokens() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    return input_ids, torch.ones_like(input_ids)


def test_graph_receipt_freezes_first_pass_readout_and_kv_ownership() -> None:
    receipt = graph_receipt(sequence_length=3, num_layers=4, destination_layer=2)
    rows = receipt["rows"]
    assert receipt["readout"] == "first_iteration_only"
    assert receipt["tap_convention"] == "post_block_hidden_states_index_equals_paper_layer"
    assert any(
        row["position"] == 0
        and row["layer"] == 3
        and row["architecture_copy"] == 0
        and row["status"] == "provisional_then_discarded"
        for row in rows
    )
    assert any(
        row["position"] == 0
        and row["layer"] == 3
        and row["architecture_copy"] == 1
        and row["status"] == "committed"
        for row in rows
    )
    final_rows = [row for row in rows if row["position"] == 2]
    assert all(row["architecture_copy"] == 0 for row in final_rows)
    assert all(row["kv_owner"] == "scored_stack" for row in final_rows)


def test_qwen_alpha_zero_is_bit_exact_for_logits_and_committed_cache() -> None:
    input_ids, attention_mask = _tokens()
    evaluator = PaperNativeRecirculationEvaluator(
        _qwen(), RecirculationConfig(source_layer=4, destination_layer=2, alpha=0.0)
    )
    receipt = evaluator.identity_receipt(
        input_ids=input_ids, attention_mask=attention_mask
    )
    assert receipt["bit_exact"] is True
    assert receipt["scored_logits_maximum_absolute_difference"] == 0.0
    assert receipt["committed_cache"]["maximum_absolute_difference"] == 0.0


def test_gemma_sliding_cache_alpha_zero_is_bit_exact() -> None:
    input_ids, attention_mask = _tokens()
    evaluator = PaperNativeRecirculationEvaluator(
        _gemma(), RecirculationConfig(source_layer=4, destination_layer=2, alpha=0.0)
    )
    assert evaluator.identity_receipt(
        input_ids=input_ids, attention_mask=attention_mask
    )["bit_exact"]


def test_nonzero_recirculation_changes_only_later_scored_positions() -> None:
    input_ids, attention_mask = _tokens()
    model = _qwen()
    intact = PaperNativeRecirculationEvaluator(
        model, RecirculationConfig(source_layer=4, destination_layer=2, alpha=0.0)
    )
    recirculated = PaperNativeRecirculationEvaluator(
        model, RecirculationConfig(source_layer=4, destination_layer=2, alpha=0.2)
    )
    intact_logits, _ = intact.forward_sequence(
        input_ids=input_ids, attention_mask=attention_mask
    )
    recirculated_logits, _ = recirculated.forward_sequence(
        input_ids=input_ids, attention_mask=attention_mask
    )
    differences = (intact_logits - recirculated_logits).abs().amax(dim=-1)
    assert differences[0, 0].item() == 0.0
    assert bool((differences[0, 1:] > 0).all())


def test_prefill_and_incremental_advance_use_future_cache_only() -> None:
    input_ids, attention_mask = _tokens()
    evaluator = PaperNativeRecirculationEvaluator(
        _qwen(), RecirculationConfig(source_layer=4, destination_layer=2, alpha=0.2)
    )
    state, output = evaluator.prefill_cached(
        input_ids=input_ids[:, :3], attention_mask=attention_mask[:, :3]
    )
    assert output.augmented_logits.shape == (1, 64)
    state, advanced = evaluator.advance_cached(
        state=state, selected_tokens=torch.tensor([4])
    )
    assert state.processed_positions == 4
    assert state.attention_mask.shape == (1, 4)
    assert advanced.augmented_logits.shape == (1, 64)


def test_cost_projection_prices_the_complete_registered_phase_a() -> None:
    receipt = projection_receipt(
        phase0_elapsed=100.0,
        qwen_pilot={"recirculated": {"elapsed_seconds": 50.0}},
        battery_elapsed=25.0,
    )
    assert receipt["coarse_pairs"] == 32
    assert receipt["coarse_cells"] == 96
    assert receipt["refinement_perplexity_cells"] == 13
    assert receipt["battery_cells"] == 2
    assert receipt["projected_total_seconds"] > 100.0
    assert receipt["ceiling_a100_hours"] == 8.0
