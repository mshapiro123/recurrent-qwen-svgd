from __future__ import annotations

import json

from colab.validate_arc_agi_baseline_registry import main, validate_baseline_registry, validate_registry_payload


def _registry(*, params_b: float = 0.5, source: str = "https://arcprize.org/leaderboard") -> dict:
    return {
        "benchmark": "ARC-AGI public evaluation",
        "metric": "selected_accuracy",
        "same_size_band": {"min_params_b": 0.3, "max_params_b": 1.0},
        "baselines": [
            {
                "name": "sourced-small-baseline",
                "params_b": params_b,
                "metric": "selected_accuracy",
                "accuracy": 0.1,
                "source": source,
                "accessed_date": "2026-06-20",
            }
        ],
    }


def test_baseline_registry_validation_passes_sourced_same_size_registry() -> None:
    payload = validate_registry_payload(_registry())

    assert payload["status"] == "passed"
    assert payload["passed"] is True
    assert payload["valid_baseline_count"] == 1
    assert payload["best_baseline"]["accuracy"] == 0.1


def test_baseline_registry_validation_rejects_placeholder_source() -> None:
    payload = validate_registry_payload(_registry(source="REPLACE_WITH_AUTHORITATIVE_SOURCE"))

    assert payload["status"] == "needs_baseline_registry"
    assert payload["passed"] is False
    assert any(row["path"] == "$.baselines[0].source" for row in payload["issues"])


def test_baseline_registry_validation_rejects_out_of_band_params() -> None:
    payload = validate_registry_payload(_registry(params_b=2.0))

    assert payload["status"] == "needs_baseline_registry"
    assert payload["passed"] is False
    assert any(row["path"] == "$.baselines[0].params_b" for row in payload["issues"])


def test_baseline_registry_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "baselines.json"
    registry.write_text(json.dumps(_registry()), encoding="utf-8")
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_arc_agi_baseline_registry.py",
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
    assert "Baseline Registry" in output_md.read_text(encoding="utf-8")


def test_missing_baseline_registry_returns_gate_payload(tmp_path) -> None:
    payload = validate_baseline_registry(tmp_path / "missing.json")

    assert payload["status"] == "needs_baseline_registry"
    assert payload["passed"] is False
    assert payload["criteria"][0]["passed"] is False
