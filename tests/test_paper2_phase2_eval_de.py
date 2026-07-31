from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from eval.prepare_paper2_phase2_eval_de import (
    boundary_layer_indices,
    cache_own_base_features,
    public_feature_receipt,
)
from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM


def test_boundary_layer_indices_cover_prelude_block_and_coda() -> None:
    assert boundary_layer_indices(prelude_end=6, recurrent_end=18, layers=24) == {
        "post_prelude": 6,
        "post_recurrent": 18,
        "post_coda": 24,
    }


def test_feature_receipt_exposes_hashes_not_hidden_tensors() -> None:
    payload = {
        "kind": "paper2_phase2_own_base_feature_shard",
        "rows": [
            {
                "row_index": 0,
                "features": {
                    "post_prelude": torch.zeros(2, 8, dtype=torch.bfloat16),
                    "post_recurrent": torch.ones(2, 8, dtype=torch.bfloat16),
                },
            }
        ],
    }
    receipt = public_feature_receipt(
        partition="eval_d",
        shard_receipts=[{"path": "private.pt", "sha256": "a" * 64, "rows": 1}],
        layer_indices={"post_prelude": 6, "post_recurrent": 18, "post_coda": 24},
        positions=2,
    )
    assert receipt["positions"] == 2
    assert receipt["storage_dtype"] == "bfloat16"
    assert "features" not in repr(receipt)
    assert "tensor(" not in repr(receipt)
    assert "private.pt" not in repr(receipt)
    assert "rows" in payload


def test_own_base_cache_extracts_three_boundary_states(tmp_path) -> None:
    config = Qwen2Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    base = Qwen2ForCausalLM(config).eval()
    wrapper = RecurrentQwenForCausalLM(
        base, layer_split=LayerSplit(prelude_end=1, recurrent_end=3)
    ).eval()
    receipt = cache_own_base_features(
        wrapper=wrapper,
        rows=[{"input_ids": [1, 2, 3]}],
        partition="eval_d",
        destination=tmp_path,
        checkpoint_sha256="a" * 64,
        device="cpu",
    )
    shard = torch.load(next(tmp_path.glob("*.pt")), weights_only=False)
    features = shard["rows"][0]["features"]
    assert set(features) == {"post_prelude", "post_recurrent", "post_coda"}
    assert all(tensor.shape == (2, 16) for tensor in features.values())
    assert all(tensor.dtype == torch.bfloat16 for tensor in features.values())
    assert receipt["positions"] == 2
