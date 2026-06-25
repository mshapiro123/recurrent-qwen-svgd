import json
from pathlib import Path

from colab.review_stage5_phase1_gate import build_review


def write_summary(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_pointer(path: Path, target: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(target), encoding="utf-8")
    return path


def benchmark_payload(**overrides) -> dict:
    payload = {
        "kind": "stage5_broader_benchmark_suite_assessment",
        "gate": "stage5_broader_benchmark_suite",
        "status": "passed",
        "passed": True,
        "benchmarks": [
            {
                "benchmark": "arc_challenge",
                "correct_delta_recurrent_vs_base": 2,
            }
        ],
    }
    payload.update(overrides)
    return payload


def dense_payload(**overrides) -> dict:
    payload = {
        "kind": "stage5_mcq_recipe_control_assessment",
        "gate": "stage5_same_recipe_mcq_architecture",
        "status": "hard_tail_lift_vs_dense",
        "passed": True,
    }
    payload.update(overrides)
    return payload


def test_phase1_gate_ignores_stale_benchmark_when_current_pointer_is_before_stage4(tmp_path: Path) -> None:
    write_summary(
        tmp_path / "outputs" / "stage5" / "old_bench" / "summary.json",
        benchmark_payload(status="needs_recurrent_recovery", passed=False),
    )
    current = write_summary(
        tmp_path / "outputs" / "stage5" / "stage2_norm" / "summary.json",
        {"kind": "stage5_reentry_norm_eval_only"},
    )
    pointer = write_pointer(tmp_path / "config" / "stage5_current_source_summary.txt", current)

    review = build_review(tmp_path / "outputs" / "stage5", pointer=pointer)

    assert review["action"] == "wait_for_reentry_recovery_training"
    assert review["benchmark_assessment"] is None
    assert review["launch_env"] == {}


def test_phase1_gate_waits_when_no_benchmark_assessment(tmp_path: Path) -> None:
    review = build_review(tmp_path / "outputs" / "stage5", pointer=tmp_path / "config" / "stage5_current_source_summary.txt")

    assert review["action"] == "wait_for_debiased_benchmark_suite"
    assert review["next_target"] == ""
    assert review["launch_env"] == {}


def test_phase1_gate_stops_when_recurrent_benchmark_does_not_pass(tmp_path: Path) -> None:
    benchmark = write_summary(
        tmp_path / "outputs" / "stage5" / "bench" / "summary.json",
        benchmark_payload(status="failed", passed=False),
    )
    pointer = write_pointer(tmp_path / "config" / "stage5_current_source_summary.txt", benchmark)

    review = build_review(tmp_path / "outputs" / "stage5", pointer=pointer)

    assert review["action"] == "stop_recurrent_not_base_competitive"
    assert review["next_target"] == ""
    assert review["launch_env"] == {}
    assert review["current_pointer"]["preferred"] is True


def test_phase1_gate_routes_passed_benchmark_to_dense_same_recipe_control(tmp_path: Path) -> None:
    benchmark = write_summary(
        tmp_path / "outputs" / "stage5" / "bench" / "summary.json",
        benchmark_payload(),
    )
    pointer = write_pointer(tmp_path / "config" / "stage5_current_source_summary.txt", benchmark)

    review = build_review(tmp_path / "outputs" / "stage5", pointer=pointer)

    assert review["action"] == "run_dense_same_curriculum_control"
    assert review["next_target"] == "dense_mcq_trace_sft_control"
    assert review["arc_challenge_delta_recurrent_vs_base"] == 2
    assert review["launch_env"]["STAGE5_CURRENT_A100_TARGET"] == "dense_mcq_trace_sft_control"
    assert review["launch_env"]["STAGE5_CURRENT_A100_SOURCE_SUMMARY"].endswith("outputs/stage5/bench/summary.json")


def test_phase1_gate_declares_architecture_signal_after_dense_hard_tail_lift(tmp_path: Path) -> None:
    benchmark = write_summary(
        tmp_path / "outputs" / "stage5" / "bench" / "summary.json",
        benchmark_payload(),
    )
    dense = write_summary(
        tmp_path / "outputs" / "stage5" / "dense" / "summary.json",
        dense_payload(),
    )
    pointer = write_pointer(tmp_path / "config" / "stage5_current_source_summary.txt", dense)

    review = build_review(tmp_path / "outputs" / "stage5", pointer=pointer)

    assert review["benchmark_assessment"].endswith("outputs/stage5/bench/summary.json")
    assert review["dense_control_assessment"].endswith("outputs/stage5/dense/summary.json")
    assert review["action"] == "phase1_architecture_signal"
    assert review["next_target"] == "phase2_breadth_diagnostic_after_review"
    assert review["launch_env"] == {}
    assert review["current_pointer"]["preferred"] is True


def test_phase1_gate_stops_when_dense_control_shows_no_architecture_lift(tmp_path: Path) -> None:
    write_summary(
        tmp_path / "outputs" / "stage5" / "bench" / "summary.json",
        benchmark_payload(),
    )
    write_summary(
        tmp_path / "outputs" / "stage5" / "dense" / "summary.json",
        dense_payload(status="no_architecture_lift_vs_dense", passed=False),
    )

    review = build_review(tmp_path / "outputs" / "stage5", pointer=tmp_path / "config" / "stage5_current_source_summary.txt")

    assert review["action"] == "no_architecture_lift_vs_dense"
    assert review["next_target"] == ""
    assert review["launch_env"] == {}
