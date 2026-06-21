from __future__ import annotations

from pathlib import Path

from colab.run_stage5_reasoning_dataset_audit import (
    audit_command,
    recommendation_for,
    sort_recommendations,
    summary_markdown,
)


def test_audit_command_includes_dataset_name_and_adapter() -> None:
    spec = {
        "dataset_id": "Glint-Research/Fable-5-traces",
        "name": "pi_agent",
        "split": "train",
        "adapter": "fable_pi_agent",
    }

    cmd = audit_command("fable5_pi_agent", spec, Path("outputs/audit.json"))

    assert "--dataset_id" in cmd
    assert "Glint-Research/Fable-5-traces" in cmd
    assert "--name" in cmd
    assert "pi_agent" in cmd
    assert "--adapter" in cmd
    assert "fable_pi_agent" in cmd


def test_audit_command_includes_explicit_hf_file() -> None:
    spec = {
        "dataset_id": "Glint-Research/Fable-5-traces",
        "hf_file": "fable5_cot_merged.jsonl",
        "split": "train",
        "adapter": "fable_flat",
    }

    cmd = audit_command("fable5_flat", spec, Path("outputs/audit.json"))

    assert "--hf_file" in cmd
    assert "fable5_cot_merged.jsonl" in cmd


def test_recommendation_promotes_compatible_opus_trace_source() -> None:
    spec = {
        "dataset_id": "lordx64/reasoning-distill-opus-4-7-max-sft",
        "priority": "immediate",
        "license": "apache-2.0",
    }
    report = {
        "converted_rows": 900,
        "conversion_rate": 0.9,
        "adapter_success_counts": {"qwen_text": 900},
        "training_role": {"priority": "immediate_candidate", "primary_role": "reasoning_trace_sft"},
        "token_stats": {"total_tokens": {"p90": 1024}, "cot_tokens": {"p90": 512}},
    }

    item = recommendation_for("opus47_sft", spec, report)

    assert item["status"] == "promote_to_small_train_mix"
    assert "filtered subset" in item["recommendation"]


def test_recommendation_holds_fable_even_when_convertible() -> None:
    spec = {
        "dataset_id": "Glint-Research/Fable-5-traces",
        "priority": "later",
        "license": "agpl-3.0",
    }
    report = {
        "converted_rows": 700,
        "conversion_rate": 0.7,
        "adapter_success_counts": {"fable_flat": 700},
        "training_role": {"priority": "later", "primary_role": "agent_tool_trace_or_coding_diversity"},
        "token_stats": {"total_tokens": {"p90": 1800}, "cot_tokens": {"p90": 900}},
    }

    item = recommendation_for("fable5_flat", spec, report)

    assert item["status"] == "hold_for_agent_tool_filter"
    assert "do not mix into ARC/GPQA" in item["recommendation"]


def test_sort_recommendations_keeps_immediate_before_later() -> None:
    items = [
        {"key": "fable", "role_priority": "later", "status": "hold_for_agent_tool_filter"},
        {"key": "opus", "role_priority": "immediate_candidate", "status": "promote_to_small_train_mix"},
    ]

    assert sort_recommendations(items)[0]["key"] == "opus"


def test_summary_markdown_renders_table() -> None:
    payload = {
        "run_id": "run",
        "status": "ok",
        "registry": "config/reasoning_dataset_registry.yaml",
        "limit": 10,
        "next_step": "Train carefully.",
        "errors": {},
        "recommendations": [
            {
                "key": "opus47_sft",
                "status": "promote_to_small_train_mix",
                "converted_rows": 9,
                "conversion_rate": 0.9,
                "total_tokens_p90": 100,
                "primary_role": "reasoning_trace_sft",
                "recommendation": "Use it.",
            }
        ],
    }

    markdown = summary_markdown(payload)

    assert "# Stage 5 Reasoning Dataset Audit - run" in markdown
    assert "| `opus47_sft` |" in markdown
    assert "90.0%" in markdown
