from __future__ import annotations

import json

from colab.assess_stage5_gate2 import assess_gate2, latest_summary, main


def _variant(
    *,
    selected_delta: float,
    best_delta: float,
    seeds: int = 5,
    non_negative: int = 5,
    passed: bool = True,
) -> dict[str, object]:
    return {
        "passed": passed,
        "evaluated_seed_count": seeds,
        "non_negative_seed_count": non_negative,
        "mean_delta_vs_tuned": {
            "selected_delta": selected_delta,
            "best_of_k_delta": best_delta,
            "first_delta": 0,
            "valid_rate_delta": 0.0,
        },
    }


def _payload(variant: dict[str, object], *, recovery_passed: bool = True, particle_passed: bool = True) -> dict[str, object]:
    return {
        "run_id": "particle",
        "recovery_decision": {"passed": recovery_passed, "evidence": {}},
        "particle_decision": {
            "passed": particle_passed,
            "evidence": {
                "best_replicated_variant": "svgd",
                "variants": {"svgd": variant},
            },
        },
    }


def test_gate2_passes_replicated_selected_lift() -> None:
    assessment = assess_gate2(_payload(_variant(selected_delta=1, best_delta=2)), source_summary="summary.json")

    assert assessment["status"] == "passed"
    assert assessment["passed"] is True
    assert assessment["best_variant"]["variant"] == "svgd"


def test_gate2_needs_selector_conversion_for_coverage_only_lift() -> None:
    assessment = assess_gate2(_payload(_variant(selected_delta=0, best_delta=2)), source_summary="summary.json")

    assert assessment["status"] == "needs_selector_conversion"
    assert assessment["passed"] is False


def test_gate2_needs_more_evidence_for_too_few_seeds() -> None:
    assessment = assess_gate2(
        _payload(_variant(selected_delta=1, best_delta=2, seeds=1, non_negative=1)),
        source_summary="summary.json",
        min_seed_count=3,
    )

    assert assessment["status"] == "needs_more_evidence"
    assert assessment["passed"] is False


def test_gate2_fails_when_recovery_failed() -> None:
    assessment = assess_gate2(
        _payload(_variant(selected_delta=1, best_delta=2), recovery_passed=False),
        source_summary="summary.json",
    )

    assert assessment["status"] == "failed"
    assert "Deterministic recurrent recovery" in assessment["reason"]


def test_gate2_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    source = tmp_path / "particle_summary.json"
    output_json = tmp_path / "gate2.json"
    output_md = tmp_path / "gate2.md"
    source.write_text(json.dumps(_payload(_variant(selected_delta=1, best_delta=2))), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "assess_stage5_gate2.py",
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
    assert "Stage 5 Gate 2 Assessment" in output_md.read_text(encoding="utf-8")


def test_gate2_latest_summary_finds_recovery_particle_summary(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "particle" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(_payload(_variant(selected_delta=1, best_delta=2))), encoding="utf-8")

    assert latest_summary(scan_root) == source
