from __future__ import annotations

import json

from colab.build_stage5_arc_agi_sota_comparison import build_sota_comparison, candidate_evidence, main


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(path, *, selected_exact: int = 12, examples: int = 100):
    _write(
        path,
        {
            "run_id": "candidate",
            "metadata": {
                "arc_version": "1",
                "eval_split": "evaluation",
                "eval_task_limit": examples,
            },
            "phase1_arc_agi_tuned": {
                "selected_exact": selected_exact,
                "best_of_k_exact": selected_exact,
                "examples_with_targets": examples,
            },
        },
    )
    return path


def _recovered_candidate(path, *, selected_exact: int = 12, examples: int = 100):
    _write(
        path,
        {
            "run_id": "recovered_benchmark",
            "metadata": {
                "arc_version": "1",
                "arc_split": "evaluation",
            },
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
                    "metric": "selected_accuracy",
                    "accuracy": accuracy,
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
    assert payload["criteria"][3]["passed"] is False


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
    assert payload["criteria"][4]["name"] == "baseline_registry_matches_candidate_arc"
    assert payload["criteria"][4]["passed"] is False


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
    assert payload["criteria"][2]["name"] == "candidate_arc_agi_metadata_present"
    assert payload["criteria"][2]["passed"] is False


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
                    "metric": "selected_accuracy",
                    "accuracy": 0.1,
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
