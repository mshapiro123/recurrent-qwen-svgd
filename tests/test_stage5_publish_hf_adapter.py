from __future__ import annotations

import torch

from colab.run_stage5_publish_hf_adapter import (
    build_export_metadata,
    checkpoint_value_from_payload,
    render_model_card,
    should_upload,
)


def test_checkpoint_value_from_payload_prefers_followup_recovered_checkpoint() -> None:
    payload = {
        "metadata": {"recovered_checkpoint": "outputs/stage5/run/recovered.pt"},
        "autopilot_compact": {"final_checkpoint": "outputs/stage5/other/final.pt"},
    }

    assert checkpoint_value_from_payload(payload) == "outputs/stage5/run/recovered.pt"


def test_checkpoint_value_from_payload_reads_autopilot_final_checkpoint() -> None:
    payload = {
        "compact": {"final_checkpoint": "outputs/stage5/parent/final.pt"},
        "child_run_ids": {"curriculum": "parent_curriculum"},
    }

    assert checkpoint_value_from_payload(payload) == "outputs/stage5/parent/final.pt"


def test_build_export_metadata_captures_checkpoint_hash_and_config(tmp_path) -> None:
    checkpoint = tmp_path / "phase1_step_10.pt"
    torch.save(
        {
            "phase": "phase1",
            "step": 10,
            "config": {
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "layer_split": "6,18",
                "max_loops": 4,
                "adapter_dtype": "float32",
                "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
            },
            "trainable_state_dict": {"bridge.weight": torch.eye(2)},
        },
        checkpoint,
    )
    checkpoint_metadata = {
        "phase": "phase1",
        "step": 10,
        "config": {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "layer_split": "6,18",
            "max_loops": 4,
            "adapter_dtype": "float32",
            "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
        },
        "trainable_key_count": 1,
    }

    metadata = build_export_metadata(
        checkpoint=checkpoint,
        checkpoint_metadata=checkpoint_metadata,
        source_summary=None,
        source_payload={"deltas": {"recovered_vs_base": {"selected_exact_delta": 1}}},
    )

    assert metadata["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert metadata["checkpoint_sha256"]
    assert metadata["checkpoint_bytes"] == checkpoint.stat().st_size
    assert metadata["architecture"]["layer_split"] == "6,18"
    assert metadata["eval_snapshot"]["deltas"]["recovered_vs_base"]["selected_exact_delta"] == 1


def test_render_model_card_includes_loading_sketch_and_benchmark_delta() -> None:
    metadata = {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "architecture": {
            "wrapper": "RecurrentQwenForCausalLM",
            "layer_split": "6,18",
            "max_loops": 4,
            "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
            "adapter_dtype": "float32",
        },
        "source": {"commit": "abc123", "repo": "https://github.com/mshapiro123/recurrent-qwen-svgd.git"},
        "eval_snapshot": {
            "deltas": {"recovered_vs_base": {"best_of_k_exact_delta": 2}},
            "compact": {"particle_passed": False},
        },
    }

    card = render_model_card(metadata)

    assert "Recurrent-Depth Qwen Adapter" in card
    assert "Qwen/Qwen2.5-0.5B-Instruct" in card
    assert "best_of_k_exact_delta" in card
    assert "load_trainable_checkpoint" in card


def test_should_upload_requires_repo_and_token_unless_disabled() -> None:
    assert should_upload("mshapiro123/recurrent-qwen-test", "token", "auto") is True
    assert should_upload("mshapiro123/recurrent-qwen-test", "", "auto") is False
    assert should_upload("", "token", "auto") is False
    assert should_upload("mshapiro123/recurrent-qwen-test", "token", "0") is False
