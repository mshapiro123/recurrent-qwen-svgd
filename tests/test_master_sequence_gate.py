from __future__ import annotations

import json
from pathlib import Path

import pytest

from colab.master_sequence_gate import (
    checkpoint_value_from_payload,
    phase1_checkpoint_for_breadth,
    phase1_depth_gate_passed,
    require_phase1_depth_gate_for_breadth,
)


def write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_pointer(root, rel: str) -> None:
    pointer = root / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(rel + "\n", encoding="utf-8")


def test_phase2_gate_blocks_reentry_norm_pointer(tmp_path, monkeypatch) -> None:
    rel = "outputs/stage5/reentry_norm/summary.json"
    write_json(tmp_path / rel, {"kind": "stage5_reentry_norm_eval_only", "status": "entry_rms_safe_for_smoke"})
    write_pointer(tmp_path, rel)
    monkeypatch.delenv("STAGE5_ALLOW_PRE_PHASE1_BREADTH", raising=False)

    with pytest.raises(SystemExit, match="blocked by the master sequence"):
        require_phase1_depth_gate_for_breadth(root=tmp_path, action_name="candidate_conversion_diagnostic")


def test_phase2_gate_allows_direct_same_recipe_assessment() -> None:
    payload = {
        "kind": "stage5_mcq_recipe_control_assessment",
        "gate": "stage5_same_recipe_mcq_architecture",
        "status": "hard_tail_lift_vs_dense",
        "passed": True,
    }

    assert phase1_depth_gate_passed(payload, root=Path(".")) is True


def test_checkpoint_value_from_payload_reads_stage5_checkpoint_shapes() -> None:
    assert (
        checkpoint_value_from_payload({"metadata": {"recovered_checkpoint": "outputs/stage5/run/recovered.pt"}})
        == "outputs/stage5/run/recovered.pt"
    )
    assert (
        checkpoint_value_from_payload(
            {
                "stages": [
                    {"selected_checkpoint": {"checkpoint": "old.pt"}},
                    {"selected_checkpoint": "outputs/stage5/final/phase1_step_150.pt"},
                ]
            }
        )
        == "outputs/stage5/final/phase1_step_150.pt"
    )


def test_phase1_checkpoint_resolver_walks_dense_assessment_to_stage4_checkpoint(tmp_path) -> None:
    stage4_rel = "outputs/stage5/stage4/summary.json"
    benchmark_rel = "outputs/stage5/benchmark/summary.json"
    benchmark_assessment_rel = "outputs/stage5/benchmark_assessment/summary.json"
    dense_rel = "outputs/stage5/dense/summary.json"
    checkpoint = "outputs/stage5/stage4/phase1/phase1_step_75.pt"
    write_json(
        tmp_path / stage4_rel,
        {
            "kind": "stage5_reentry_recovery_training",
            "checkpoint": checkpoint,
            "validation_checks": {"status": "validation_sane", "issues": []},
            "post_reentry_health_checks": {"status": "reentry_health_sane", "issues": []},
        },
    )
    write_json(
        tmp_path / benchmark_rel,
        {
            "kind": "stage5_benchmark_suite",
            "source_summary": stage4_rel,
            "checkpoint": checkpoint,
        },
    )
    write_json(
        tmp_path / benchmark_assessment_rel,
        {
            "kind": "stage5_broader_benchmark_suite_assessment",
            "gate": "stage5_broader_benchmark_suite",
            "status": "passed",
            "passed": True,
            "source_summary": benchmark_rel,
        },
    )
    dense_payload = {
        "kind": "stage5_dense_mcq_trace_sft_control",
        "source_summary": benchmark_assessment_rel,
        "recurrent_benchmark_summary": benchmark_rel,
        "recipe_control_assessment": {
            "ran": True,
            "status": "hard_tail_lift_vs_dense",
            "passed": True,
            "recurrent_summary": benchmark_rel,
        },
    }
    write_json(tmp_path / dense_rel, dense_payload)

    resolved = phase1_checkpoint_for_breadth(dense_payload, root=tmp_path, summary_path=tmp_path / dense_rel)

    assert resolved is not None
    assert resolved["checkpoint"] == checkpoint
    assert resolved["checkpoint_source_summary"].replace("\\", "/").endswith(benchmark_rel)


def test_phase2_gate_returns_resolved_checkpoint_for_dense_control_pointer(tmp_path, monkeypatch) -> None:
    stage4_rel = "outputs/stage5/stage4/summary.json"
    benchmark_rel = "outputs/stage5/benchmark/summary.json"
    benchmark_assessment_rel = "outputs/stage5/benchmark_assessment/summary.json"
    dense_rel = "outputs/stage5/dense/summary.json"
    checkpoint = "outputs/stage5/stage4/phase1/phase1_step_75.pt"
    write_json(tmp_path / stage4_rel, {"kind": "stage5_reentry_recovery_training", "checkpoint": checkpoint})
    write_json(tmp_path / benchmark_rel, {"kind": "stage5_benchmark_suite", "source_summary": stage4_rel, "checkpoint": checkpoint})
    write_json(
        tmp_path / benchmark_assessment_rel,
        {
            "kind": "stage5_broader_benchmark_suite_assessment",
            "gate": "stage5_broader_benchmark_suite",
            "status": "passed",
            "passed": True,
            "source_summary": benchmark_rel,
        },
    )
    write_json(
        tmp_path / dense_rel,
        {
            "kind": "stage5_dense_mcq_trace_sft_control",
            "source_summary": benchmark_assessment_rel,
            "recurrent_benchmark_summary": benchmark_rel,
            "recipe_control_assessment": {
                "ran": True,
                "status": "hard_tail_lift_vs_dense",
                "passed": True,
                "recurrent_summary": benchmark_rel,
            },
        },
    )
    write_pointer(tmp_path, dense_rel)
    monkeypatch.delenv("STAGE5_ALLOW_PRE_PHASE1_BREADTH", raising=False)

    result = require_phase1_depth_gate_for_breadth(root=tmp_path, action_name="candidate_conversion_diagnostic")

    assert result["allowed"] is True
    assert result["checkpoint"] == checkpoint
    assert result["checkpoint_source_summary"].replace("\\", "/").endswith(benchmark_rel)


def test_phase2_gate_blocks_passed_assessment_without_checkpoint_lineage(tmp_path, monkeypatch) -> None:
    rel = "outputs/stage5/dense/summary.json"
    write_json(
        tmp_path / rel,
        {
            "kind": "stage5_mcq_recipe_control_assessment",
            "gate": "stage5_same_recipe_mcq_architecture",
            "status": "hard_tail_lift_vs_dense",
            "passed": True,
        },
    )
    write_pointer(tmp_path, rel)
    monkeypatch.delenv("STAGE5_ALLOW_PRE_PHASE1_BREADTH", raising=False)

    with pytest.raises(SystemExit, match="could not resolve the recurrent checkpoint"):
        require_phase1_depth_gate_for_breadth(root=tmp_path, action_name="candidate_conversion_diagnostic")


def test_phase2_gate_requires_dense_control_upstream_base_gate(tmp_path) -> None:
    benchmark_rel = "outputs/stage5/benchmark_assessment/summary.json"
    dense_payload = {
        "kind": "stage5_dense_mcq_trace_sft_control",
        "source_summary": benchmark_rel,
        "recipe_control_assessment": {
            "ran": True,
            "status": "hard_tail_lift_vs_dense",
            "passed": True,
        },
    }
    write_json(
        tmp_path / benchmark_rel,
        {
            "kind": "stage5_benchmark_assessment",
            "gate": "stage5_broader_benchmark_suite",
            "status": "needs_recurrent_recovery",
            "passed": False,
        },
    )

    assert phase1_depth_gate_passed(dense_payload, root=tmp_path) is False

    write_json(
        tmp_path / benchmark_rel,
        {
            "kind": "stage5_benchmark_assessment",
            "gate": "stage5_broader_benchmark_suite",
            "status": "passed",
            "passed": True,
        },
    )

    assert phase1_depth_gate_passed(dense_payload, root=tmp_path) is True


def test_phase2_gate_override_is_explicit(tmp_path, monkeypatch) -> None:
    rel = "outputs/stage5/reentry_norm/summary.json"
    write_json(tmp_path / rel, {"kind": "stage5_reentry_norm_eval_only"})
    write_pointer(tmp_path, rel)
    monkeypatch.setenv("STAGE5_ALLOW_PRE_PHASE1_BREADTH", "1")

    result = require_phase1_depth_gate_for_breadth(root=tmp_path, action_name="effective_pathways_diagnostic")

    assert result == {"allowed": True, "override": "STAGE5_ALLOW_PRE_PHASE1_BREADTH"}
