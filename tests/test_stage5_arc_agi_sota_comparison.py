from __future__ import annotations

import json

from colab.build_stage5_arc_agi_sota_comparison import (
    build_sota_comparison,
    candidate_evidence,
    latest_candidate_summary,
    main,
)


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _criterion(payload: dict, name: str) -> dict:
    return next(row for row in payload["criteria"] if row["name"] == name)


def _candidate(path, *, selected_exact: int = 12, examples: int = 100, params_b: float | None = 0.5):
    metadata = {
        "arc_version": "1",
        "eval_split": "evaluation",
        "eval_task_limit": examples,
    }
    if params_b is not None:
        metadata["params_b"] = params_b
    _write(
        path,
        {
            "run_id": "candidate",
            "metadata": metadata,
            "phase1_arc_agi_tuned": {
                "selected_exact": selected_exact,
                "best_of_k_exact": selected_exact,
                "examples_with_targets": examples,
            },
        },
    )
    return path


def _recovered_candidate(path, *, selected_exact: int = 12, examples: int = 100, params_b: float | None = 0.5):
    metadata = {
        "arc_version": "1",
        "arc_split": "evaluation",
    }
    if params_b is not None:
        metadata["params_b"] = params_b
    _write(
        path,
        {
            "run_id": "recovered_benchmark",
            "metadata": metadata,
            "base": {"summary": {"selected_exact": 10, "best_of_k_exact": 10, "examples_with_targets": examples}},
            "phase1_start": {
                "summary": {"selected_exact": 5, "best_of_k_exact": 5, "examples_with_targets": examples}
            },
            "recovered": {
                "summary": {
                    "selected_exact": selected_exact,
                    "best_of_k_exact": selected_exact,
                    "examples_with_targets": examples,
                }
            },
        },
    )
    return path


def _candidate_without_metadata(path, *, selected_exact: int = 12, examples: int = 100):
    _write(
        path,
        {
            "run_id": "candidate",
            "phase1_arc_agi_tuned": {
                "selected_exact": selected_exact,
                "best_of_k_exact": selected_exact,
                "examples_with_targets": examples,
            },
        },
    )
    return path


def _registry(path, *, accuracy: float = 0.1, arc_version: str = "1", arc_split: str = "evaluation"):
    _write(
        path,
        {
            "benchmark": "ARC-AGI public evaluation",
            "arc_version": arc_version,
            "arc_split": arc_split,
            "metric": "selected_accuracy",
            "same_size_band": {"min_params_b": 0.3, "max_params_b": 1.0},
            "baselines": [
                {
                    "name": "same-size-baseline",
                    "params_b": 0.5,
                    "arc_version": arc_version,
                    "arc_split": arc_split,
                    "metric": "selected_accuracy",
                    "accuracy": accuracy,
                    "evidence_type": "official_leaderboard",
                    "source": "https://arcprize.org/leaderboard",
                    "accessed_date": "2026-06-20",
                }
            ],
        },
    )
    return path


def test_sota_comparison_passes_when_candidate_beats_same_size_baseline(tmp_path) -> None:
    payload = build_sota_comparison(
        candidate_summary=_candidate(tmp_path / "candidate" / "summary.json", selected_exact=12),
        baseline_registry=_registry(tmp_path / "baselines.json", accuracy=0.1),
        candidate_label="phase1_arc_agi_tuned",
        metric="selected_accuracy",
        min_examples=100,
        min_margin=0.0,
    )

    assert payload["status"] == "passed"
    assert payload["passed"] is True
    assert payload["candidate"]["params_b"] == 0.5
    assert payload["candidate_in_same_size_band"] is True
    assert payload["delta_accuracy_vs_best_baseline"] == 0.01999999999999999


def test_sota_comparison_auto_label_prefers_recovered_benchmark_summary(tmp_path) -> None:
    candidate = _recovered_candidate(tmp_path / "candidate" / "summary.json", selected_exact=12)

    payload = build_sota_comparison(
        candidate_summary=candidate,
        baseline_registry=_registry(tmp_path / "baselines.json", accuracy=0.1),
        candidate_label="auto",
        metric="selected_accuracy",
        min_examples=100,
        min_margin=0.0,
    )

    assert payload["status"] == "passed"
    assert payload["candidate"]["requested_label"] == "auto"
    assert payload["candidate"]["label"] == "recovered"
    assert payload["candidate"]["accuracy"] == 0.12


def test_latest_candidate_summary_prefers_main_summary_over_newer_child_eval(tmp_path, monkeypatch) -> None:
    import colab.build_stage5_arc_agi_sota_comparison as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    main_summary = tmp_path / "outputs" / "stage5" / "main" / "summary.json"
    child_summary = tmp_path / "outputs" / "stage5" / "child" / "summary.json"
    _recovered_candidate(main_summary, selected_exact=12, examples=100)
    _write(
        child_summary,
        {
            "run_id": "child_eval",
            "summary": {
                "selected_exact": 99,
                "best_of_k_exact": 99,
                "examples_with_targets": 100,
            },
        },
    )

    assert latest_candidate_summary() == main_summary


def test_latest_candidate_summary_prefers_model_size_metadata(tmp_path, monkeypatch) -> None:
    import colab.build_stage5_arc_agi_sota_comparison as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    missing_params = tmp_path / "outputs" / "stage5" / "missing_params" / "summary.json"
    with_params = tmp_path / "outputs" / "stage5" / "with_params" / "summary.json"
    _recovered_candidate(missing_params, selected_exact=30, examples=100, params_b=None)
    _recovered_candidate(with_params, selected_exact=12, examples=100, params_b=0.5)

    assert latest_candidate_summary() == with_params


def test_candidate_evidence_reports_requested_missing_label(tmp_path) -> None:
    candidate = _recovered_candidate(tmp_path / "candidate" / "summary.json", selected_exact=12)

    evidence = candidate_evidence(candidate, label="phase1_arc_agi_tuned", metric="selected_accuracy")

    assert evidence["present"] is False
    assert "phase1_arc_agi_tuned" in evidence["reason"]


def test_sota_comparison_fails_when_candidate_trails_baseline(tmp_path) -> None:
    payload = build_sota_comparison(
        candidate_summary=_candidate(tmp_path / "candidate" / "summary.json", selected_exact=8),
        baseline_registry=_registry(tmp_path / "baselines.json", accuracy=0.1),
        candidate_label="phase1_arc_agi_tuned",
        metric="selected_accuracy",
        min_examples=100,
        min_margin=0.0,
    )

    assert payload["status"] == "failed"
    assert payload["passed"] is False


def test_sota_comparison_requires_baseline_registry(tmp_path) -> None:
    payload = build_sota_comparison(
        candidate_summary=_candidate(tmp_path / "candidate" / "summary.json", selected_exact=12),
        baseline_registry=tmp_path / "missing_baselines.json",
        candidate_label="phase1_arc_agi_tuned",
        metric="selected_accuracy",
        min_examples=100,
        min_margin=0.0,
    )

    assert payload["status"] == "needs_baseline_registry"
    assert _criterion(payload, "baseline_registry_present")["passed"] is False


def test_sota_comparison_requires_matching_registry_arc_metadata(tmp_path) -> None:
    payload = build_sota_comparison(
        candidate_summary=_candidate(tmp_path / "candidate" / "summary.json", selected_exact=12),
        baseline_registry=_registry(tmp_path / "baselines.json", accuracy=0.1, arc_version="2"),
        candidate_label="phase1_arc_agi_tuned",
        metric="selected_accuracy",
        min_examples=100,
        min_margin=0.0,
    )

    assert payload["status"] == "needs_matching_baseline_registry"
    assert payload["candidate_registry_arc_match"] is False
    assert _criterion(payload, "baseline_registry_matches_candidate_arc")["passed"] is False


def test_sota_comparison_requires_candidate_size_metadata(tmp_path) -> None:
    payload = build_sota_comparison(
        candidate_summary=_candidate(tmp_path / "candidate" / "summary.json", selected_exact=12, params_b=None),
        baseline_registry=_registry(tmp_path / "baselines.json", accuracy=0.1),
        candidate_label="phase1_arc_agi_tuned",
        metric="selected_accuracy",
        min_examples=100,
        min_margin=0.0,
    )

    assert payload["status"] == "needs_candidate_size_metadata"
    assert _criterion(payload, "candidate_params_present")["passed"] is False


def test_sota_comparison_requires_candidate_inside_same_size_band(tmp_path) -> None:
    payload = build_sota_comparison(
        candidate_summary=_candidate(tmp_path / "candidate" / "summary.json", selected_exact=12, params_b=1.5),
        baseline_registry=_registry(tmp_path / "baselines.json", accuracy=0.1),
        candidate_label="phase1_arc_agi_tuned",
        metric="selected_accuracy",
        min_examples=100,
        min_margin=0.0,
    )

    assert payload["status"] == "needs_candidate_size_match"
    assert payload["candidate_in_same_size_band"] is False
    assert _criterion(payload, "candidate_inside_same_size_band")["passed"] is False


def test_sota_comparison_requires_arc_agi_candidate_metadata(tmp_path) -> None:
    payload = build_sota_comparison(
        candidate_summary=_candidate_without_metadata(tmp_path / "candidate" / "summary.json", selected_exact=12),
        baseline_registry=_registry(tmp_path / "baselines.json", accuracy=0.1),
        candidate_label="phase1_arc_agi_tuned",
        metric="selected_accuracy",
        min_examples=100,
        min_margin=0.0,
    )

    assert payload["status"] == "needs_arc_agi_candidate_metadata"
    assert _criterion(payload, "candidate_arc_agi_metadata_present")["passed"] is False


def test_sota_comparison_rejects_invalid_baseline_registry(tmp_path) -> None:
    registry = tmp_path / "baselines.json"
    _write(
        registry,
        {
            "benchmark": "ARC-AGI public evaluation",
            "arc_version": "1",
            "arc_split": "evaluation",
            "metric": "selected_accuracy",
            "same_size_band": {"min_params_b": 0.3, "max_params_b": 1.0},
            "baselines": [
                {
                    "name": "placeholder-baseline",
                    "params_b": 0.5,
                    "arc_version": "1",
                    "arc_split": "evaluation",
                    "metric": "selected_accuracy",
                    "accuracy": 0.1,
                    "evidence_type": "official_leaderboard",
                    "source": "REPLACE_WITH_AUTHORITATIVE_SOURCE",
                }
            ],
        },
    )

    payload = build_sota_comparison(
        candidate_summary=_candidate(tmp_path / "candidate" / "summary.json", selected_exact=12),
        baseline_registry=registry,
        candidate_label="phase1_arc_agi_tuned",
        metric="selected_accuracy",
        min_examples=100,
        min_margin=0.0,
    )

    assert payload["status"] == "needs_baseline_registry"
    assert payload["baseline_registry"]["validation"]["passed"] is False


def test_sota_comparison_rejects_mixed_arc_baseline_row(tmp_path) -> None:
    registry = _registry(tmp_path / "baselines.json", accuracy=0.1)
    payload_data = json.loads(registry.read_text(encoding="utf-8"))
    payload_data["baselines"][0]["arc_split"] = "training"
    registry.write_text(json.dumps(payload_data), encoding="utf-8")

    payload = build_sota_comparison(
        candidate_summary=_candidate(tmp_path / "candidate" / "summary.json", selected_exact=12),
        baseline_registry=registry,
        candidate_label="phase1_arc_agi_tuned",
        metric="selected_accuracy",
        min_examples=100,
        min_margin=0.0,
    )

    assert payload["status"] == "needs_baseline_registry"
    assert payload["baseline_registry"]["validation"]["passed"] is False
    assert any(
        row["path"] == "$.baselines[0].arc_split"
        for row in payload["baseline_registry"]["validation"]["issues"]
    )


def test_sota_comparison_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    candidate = _candidate(tmp_path / "candidate" / "summary.json", selected_exact=12)
    registry = _registry(tmp_path / "baselines.json", accuracy=0.1)
    output_json = tmp_path / "comparison.json"
    output_md = tmp_path / "comparison.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_stage5_arc_agi_sota_comparison.py",
            "--candidate_summary_json",
            str(candidate),
            "--baseline_registry_json",
            str(registry),
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ],
    )

    assert main() == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "passed"
    assert "Same-Size Comparison" in output_md.read_text(encoding="utf-8")
