from __future__ import annotations

import json
from pathlib import Path

from colab.assess_stage5_gate1 import assess_gate1, latest_summary, main


def _paired(delta: int, wins: int, losses: int, ties: int, paired_examples: int = 30) -> dict[str, object]:
    return {
        "delta_exact": delta,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "paired_examples": paired_examples,
        "bootstrap_delta_accuracy_ci95": {"low": -0.05, "high": 0.15},
    }


def _payload(aggregate: dict[str, object], hard: dict[str, object] | None) -> dict[str, object]:
    comparison: dict[str, object] = {
        "metrics": {"selected_exact": aggregate},
    }
    if hard is not None:
        comparison["difficulty_metrics"] = {"selected_exact": {"hard": hard}}
    return {
        "run_id": "selector",
        "source_run_dir": "outputs/stage5/source",
        "strategies": ["reliability_vote"],
        "rows": [],
        "best_by_label": {},
        "paired_comparisons": {
            "recovered__selector_reliability_vote_vs_source": comparison,
        },
    }


def test_gate1_assessment_passes_hard_tail_lift_without_aggregate_harm() -> None:
    payload = _payload(
        aggregate=_paired(0, wins=1, losses=1, ties=28),
        hard=_paired(2, wins=2, losses=0, ties=8, paired_examples=10),
    )

    assessment = assess_gate1(payload, source_summary="summary.json")

    assert assessment["status"] == "passed"
    assert assessment["passed"] is True
    assert assessment["passing_comparisons"] == ["recovered__selector_reliability_vote_vs_source"]


def test_gate1_assessment_flags_hard_tail_tradeoff() -> None:
    payload = _payload(
        aggregate=_paired(-1, wins=1, losses=2, ties=27),
        hard=_paired(2, wins=2, losses=0, ties=8, paired_examples=10),
    )

    assessment = assess_gate1(payload, source_summary="summary.json")

    assert assessment["status"] == "needs_review"
    assert assessment["passed"] is False
    assert assessment["tradeoff_comparisons"] == ["recovered__selector_reliability_vote_vs_source"]


def test_gate1_assessment_needs_more_evidence_for_aggregate_only_lift() -> None:
    payload = _payload(
        aggregate=_paired(2, wins=2, losses=0, ties=28),
        hard=None,
    )

    assessment = assess_gate1(payload, source_summary="summary.json")

    assert assessment["status"] == "needs_more_evidence"
    assert assessment["aggregate_only_comparisons"] == ["recovered__selector_reliability_vote_vs_source"]


def test_latest_summary_finds_paired_comparison_summary(tmp_path) -> None:
    old = tmp_path / "old" / "summary.json"
    old.parent.mkdir()
    old.write_text(json.dumps({"rows": []}), encoding="utf-8")
    latest = tmp_path / "latest" / "summary.json"
    latest.parent.mkdir()
    latest.write_text(json.dumps(_payload(_paired(1, 1, 0, 29), _paired(1, 1, 0, 9, 10))), encoding="utf-8")

    assert latest_summary(tmp_path) == latest


def test_gate1_assessment_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source_summary.json"
    output_json = tmp_path / "gate1.json"
    output_md = tmp_path / "gate1.md"
    source.write_text(
        json.dumps(_payload(_paired(0, 1, 1, 28), _paired(2, 2, 0, 8, 10))),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "assess_stage5_gate1.py",
            "--summary_json",
            str(source),
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ],
    )

    assert main() == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "passed"
    assert "Stage 5 Gate 1 Assessment" in output_md.read_text(encoding="utf-8")
