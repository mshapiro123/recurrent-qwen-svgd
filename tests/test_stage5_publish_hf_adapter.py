from __future__ import annotations

import torch

from colab.run_stage5_publish_hf_adapter import (
    build_export_metadata,
    checkpoint_value_from_payload,
    compact_recipe_control_evidence,
    compact_selector_conversion_evidence,
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
    assert metadata["architecture_evidence"] == {}


def test_build_export_metadata_captures_recipe_control_evidence(tmp_path) -> None:
    checkpoint = tmp_path / "phase1_step_10.pt"
    torch.save(
        {
            "phase": "phase1",
            "step": 10,
            "config": {"model_name": "Qwen/Qwen2.5-0.5B-Instruct"},
            "trainable_state_dict": {"bridge.weight": torch.eye(2)},
        },
        checkpoint,
    )
    recipe_path = tmp_path / "recipe" / "summary.json"
    recipe_path.parent.mkdir()
    recipe_payload = {
        "gate": "stage5_same_recipe_architecture",
        "status": "needs_selector_conversion",
        "passed": False,
        "reason": "candidate coverage improved",
        "next_step": "rescore",
        "dense_summary": "outputs/stage5/dense/summary.json",
        "recurrent_summary": "outputs/stage5/recurrent/summary.json",
        "hard_bucket": "hard",
        "decision_evidence": {
            "aggregate": {"delta_exact": 0, "wins": 0, "losses": 0, "ties": 20},
            "hard": {"delta_exact": 0, "wins": 0, "losses": 0, "ties": 6},
            "aggregate_best_of_k": {"delta_exact": 3, "wins": 3, "losses": 0, "ties": 17},
            "hard_best_of_k": {"delta_exact": 2, "wins": 2, "losses": 0, "ties": 4},
        },
    }

    metadata = build_export_metadata(
        checkpoint=checkpoint,
        checkpoint_metadata={
            "phase": "phase1",
            "step": 10,
            "config": {"model_name": "Qwen/Qwen2.5-0.5B-Instruct"},
            "trainable_key_count": 1,
        },
        source_summary=None,
        source_payload=None,
        recipe_control_summary=recipe_path,
        recipe_control_payload=recipe_payload,
    )

    evidence = metadata["architecture_evidence"]
    assert evidence["status"] == "needs_selector_conversion"
    assert evidence["aggregate_best_of_k"]["delta_exact"] == 3
    assert evidence["hard_best_of_k"]["delta_exact"] == 2


def test_build_export_metadata_captures_selector_conversion_evidence(tmp_path) -> None:
    checkpoint = tmp_path / "phase1_step_10.pt"
    torch.save(
        {
            "phase": "phase1",
            "step": 10,
            "config": {"model_name": "Qwen/Qwen2.5-0.5B-Instruct"},
            "trainable_state_dict": {"bridge.weight": torch.eye(2)},
        },
        checkpoint,
    )
    selector_path = tmp_path / "selector_conversion" / "summary.json"
    selector_path.parent.mkdir()
    selector_payload = {
        "gate": "stage5_same_recipe_selector_conversion",
        "kind": "recipe_selector_conversion",
        "status": "passed",
        "passed": True,
        "reason": "selector converts candidate coverage",
        "next_step": "release gate",
        "recipe_control_summary": "outputs/stage5/recipe/summary.json",
        "selector_rescore_summary": "outputs/stage5/selector/summary.json",
        "dense_summary": "outputs/stage5/dense/summary.json",
        "hard_bucket": "hard",
        "passing_selectors": [{"label": "recovered", "selection_strategy": "reliability_vote"}],
        "best_selector": {"label": "recovered", "selection_strategy": "reliability_vote"},
        "selector_evidence": [
            {
                "label": "recovered",
                "selection_strategy": "reliability_vote",
                "aggregate": {"delta_exact": 3, "wins": 3, "losses": 0, "ties": 17},
                "hard": {"delta_exact": 2, "wins": 2, "losses": 0, "ties": 4},
                "aggregate_best_of_k": {"delta_exact": 4, "wins": 4, "losses": 0, "ties": 16},
                "hard_best_of_k": {"delta_exact": 2, "wins": 2, "losses": 0, "ties": 4},
            }
        ],
    }

    metadata = build_export_metadata(
        checkpoint=checkpoint,
        checkpoint_metadata={
            "phase": "phase1",
            "step": 10,
            "config": {"model_name": "Qwen/Qwen2.5-0.5B-Instruct"},
            "trainable_key_count": 1,
        },
        source_summary=None,
        source_payload=None,
        selector_conversion_summary=selector_path,
        selector_conversion_payload=selector_payload,
    )

    evidence = metadata["selector_conversion_evidence"]
    assert evidence["status"] == "passed"
    assert evidence["best_selector"]["selection_strategy"] == "reliability_vote"
    assert evidence["best_aggregate_selected"]["delta_exact"] == 3
    assert evidence["best_hard_selected"]["delta_exact"] == 2


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
        "architecture_evidence": {
            "summary_path": "outputs/stage5/recipe/summary.json",
            "status": "passed",
            "passed": True,
            "reason": "hard-tail selected lift",
            "next_step": "replicate",
            "aggregate_selected": {"delta_exact": 1, "wins": 1, "losses": 0, "ties": 10},
            "hard_selected": {"delta_exact": 1, "wins": 1, "losses": 0, "ties": 4},
            "aggregate_best_of_k": {"delta_exact": 2, "wins": 2, "losses": 0, "ties": 9},
            "hard_best_of_k": {"delta_exact": 1, "wins": 1, "losses": 0, "ties": 4},
        },
        "selector_conversion_evidence": {
            "summary_path": "outputs/stage5/selector_conversion/summary.json",
            "status": "passed",
            "passed": True,
            "reason": "selector converted candidate coverage",
            "next_step": "release gate",
            "best_selector": {"label": "recovered", "selection_strategy": "reliability_vote"},
            "best_aggregate_selected": {"delta_exact": 3, "wins": 3, "losses": 0, "ties": 10},
            "best_hard_selected": {"delta_exact": 2, "wins": 2, "losses": 0, "ties": 4},
            "best_aggregate_best_of_k": {"delta_exact": 4, "wins": 4, "losses": 0, "ties": 9},
            "best_hard_best_of_k": {"delta_exact": 2, "wins": 2, "losses": 0, "ties": 4},
        },
    }

    card = render_model_card(metadata)

    assert "Recurrent-Depth Qwen Adapter" in card
    assert "Qwen/Qwen2.5-0.5B-Instruct" in card
    assert "best_of_k_exact_delta" in card
    assert "load_trainable_checkpoint" in card
    assert "Same-Recipe Architecture Evidence" in card
    assert "hard-tail selected lift" in card
    assert "Aggregate best-of-K recurrent-vs-dense: delta 2" in card
    assert "Same-Recipe Selector Conversion Evidence" in card
    assert "selector converted candidate coverage" in card
    assert "Best aggregate selected recurrent-vs-dense: delta 3" in card


def test_compact_recipe_control_evidence_returns_empty_without_payload(tmp_path) -> None:
    assert compact_recipe_control_evidence(tmp_path / "missing.json", None) == {}


def test_compact_selector_conversion_evidence_returns_empty_without_payload(tmp_path) -> None:
    assert compact_selector_conversion_evidence(tmp_path / "missing.json", None) == {}


def test_should_upload_requires_repo_and_token_unless_disabled() -> None:
    assert should_upload("mshapiro123/recurrent-qwen-test", "token", "auto") is True
    assert should_upload("mshapiro123/recurrent-qwen-test", "", "auto") is False
    assert should_upload("", "token", "auto") is False
    assert should_upload("mshapiro123/recurrent-qwen-test", "token", "0") is False
