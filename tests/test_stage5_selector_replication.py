from __future__ import annotations

import json

from colab.assess_stage5_selector_replication import assess_selector_replication, main


def _gate1(*, comparison: str, status: str = "passed") -> dict:
    passed = status == "passed"
    return {
        "gate": "stage5_gate1_selector_tta",
        "status": status,
        "passed": passed,
        "reason": "selector evidence",
        "next_step": "replicate",
        "source_summary": "outputs/stage5/source/summary.json",
        "source_kind": "selector_rescore",
        "passing_comparisons": [comparison] if passed else [],
        "evidence": [
            {
                "comparison": comparison,
                "passed": passed,
                "aggregate_evidence": "delta 1",
                "hard_evidence": "delta 1",
            }
        ],
    }


def test_selector_replication_passes_when_same_comparison_passes_twice(tmp_path) -> None:
    comparison = "recovered__selector_reliability_vote_vs_source"
    payload = assess_selector_replication(
        discovery_path=tmp_path / "discovery.json",
        confirmation_path=tmp_path / "confirmation.json",
        discovery_payload=_gate1(comparison=comparison),
        confirmation_payload=_gate1(comparison=comparison),
    )

    assert payload["status"] == "passed"
    assert payload["passed"] is True
    assert payload["replicated_comparisons"] == [comparison]


def test_selector_replication_needs_confirmation_when_second_gate_missing(tmp_path) -> None:
    payload = assess_selector_replication(
        discovery_path=tmp_path / "discovery.json",
        confirmation_path=None,
        discovery_payload=_gate1(comparison="a"),
        confirmation_payload=None,
    )

    assert payload["status"] == "needs_confirmation"
    assert payload["passed"] is False


def test_selector_replication_fails_when_passing_comparisons_do_not_overlap(tmp_path) -> None:
    payload = assess_selector_replication(
        discovery_path=tmp_path / "discovery.json",
        confirmation_path=tmp_path / "confirmation.json",
        discovery_payload=_gate1(comparison="recovered__selector_reliability_vote_vs_source"),
        confirmation_payload=_gate1(comparison="recovered__selector_self_consistency_vs_source"),
    )

    assert payload["status"] == "failed"
    assert payload["passed"] is False
    assert payload["replicated_comparisons"] == []


def test_selector_replication_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    discovery = tmp_path / "discovery.json"
    confirmation = tmp_path / "confirmation.json"
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"
    comparison = "recovered__selector_reliability_vote_vs_source"
    discovery.write_text(json.dumps(_gate1(comparison=comparison)), encoding="utf-8")
    confirmation.write_text(json.dumps(_gate1(comparison=comparison)), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "assess_stage5_selector_replication.py",
            "--discovery_gate1_json",
            str(discovery),
            "--confirmation_gate1_json",
            str(confirmation),
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ],
    )

    assert main() == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "passed"
    assert "Selector Replication" in output_md.read_text(encoding="utf-8")
