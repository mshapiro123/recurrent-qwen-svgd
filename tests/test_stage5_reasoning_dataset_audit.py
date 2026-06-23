from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

import colab.run_stage5_reasoning_dataset_audit as module
from colab.run_stage5_reasoning_dataset_audit import (
    REGISTRY_PATH,
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
        "curriculum_signal": {
            "recommendation": "direct_recovery_candidate",
            "direct_candidate_rows": 800,
            "deep_narrow_candidate_rows": 50,
            "estimated_target_mix": {"direct": 800, "deep_narrow": 50, "wide": 0},
            "fit_rates_total_tokens": {"512": 0.9, "1024": 1.0, "2048": 1.0},
        },
        "token_stats": {"total_tokens": {"p90": 1024}, "cot_tokens": {"p90": 512}},
    }

    item = recommendation_for("opus47_sft", spec, report)

    assert item["status"] == "promote_to_direct_recovery_mix"
    assert "depth-1/direct" in item["recommendation"]
    assert item["direct_candidate_rows"] == 800
    assert item["estimated_target_mix"] == {"direct": 800, "deep_narrow": 50, "wide": 0}


def test_recommendation_promotes_long_trace_source_for_deep_narrow() -> None:
    spec = {
        "dataset_id": "Jackrong/Claude-opus-4.7-TraceInversion-5000x",
        "priority": "immediate_audit_candidate",
        "license": "inspect_before_use",
    }
    report = {
        "converted_rows": 400,
        "conversion_rate": 0.8,
        "adapter_success_counts": {"trace_inversion": 400},
        "training_role": {"priority": "immediate_candidate", "primary_role": "reasoning_trace_sft"},
        "curriculum_signal": {
            "recommendation": "deep_narrow_candidate",
            "direct_candidate_rows": 50,
            "deep_narrow_candidate_rows": 300,
            "estimated_target_mix": {"direct": 50, "deep_narrow": 300, "wide": 0},
            "fit_rates_total_tokens": {"512": 0.2, "1024": 0.8, "2048": 1.0},
        },
        "token_stats": {"total_tokens": {"p90": 1500}, "cot_tokens": {"p90": 900}},
    }

    item = recommendation_for("jackrong_opus47_trace_inversion", spec, report)

    assert item["status"] == "promote_to_deep_narrow_mix"
    assert "learned recurrence/depth" in item["recommendation"]
    assert item["deep_narrow_candidate_rows"] == 300


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
        "curriculum_signal": {
            "recommendation": "hold_for_wide_or_agentic_filter",
            "estimated_target_mix": {"direct": 0, "deep_narrow": 0, "wide": 700},
        },
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
                "status": "promote_to_direct_recovery_mix",
                "converted_rows": 9,
                "conversion_rate": 0.9,
                "estimated_target_mix": {"direct": 8, "deep_narrow": 1, "wide": 0},
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
    assert "8/1/0" in markdown


def test_registry_tracks_core_and_extended_trace_sources() -> None:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    datasets = registry["datasets"]

    for key in [
        "opus47_sft",
        "opus47_raw",
        "jackrong_opus47_trace_inversion",
        "jackrong_opus46_trace_inversion",
        "fable5_pi_agent",
        "fable5_flat",
        "fable5_agentic_sft",
        "fable5_complete_2m",
        "jackrong_glm51_reasoning_1m",
        "jackrong_kimi25_reasoning_1m",
        "gryphe_opus46_reasoning_24k",
        "angrygiraffe_opus46_47_reasoning_87k",
        "withinus_claude_mythos_25k",
        "avtrkrb_combined_reasoning_1m",
    ]:
        assert key in datasets

    assert datasets["opus47_sft"]["priority"] == "immediate"
    assert datasets["fable5_flat"]["priority"] == "audit"
    assert datasets["fable5_complete_2m"]["streaming"] is True


def test_commit_results_adds_only_safe_audit_artifacts(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "audit"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "summary.md").write_text("# ok\n", encoding="utf-8")
    (run_dir / "adapter.pt").write_bytes(b"checkpoint")

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *, check: bool = True, log_name: str | None = None):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "PUSH_RESULTS", True)
    monkeypatch.setattr(module, "run", fake_run)

    module.commit_results()

    add_commands = [cmd for cmd in commands if cmd[:3] == ["git", "add", "-f"]]
    assert add_commands
    added = " ".join(add_commands[0])
    assert "summary.json" in added
    assert "summary.md" in added
    assert "adapter.pt" not in added
