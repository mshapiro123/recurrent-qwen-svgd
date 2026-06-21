from __future__ import annotations

import json

from colab.assess_stage5_release_gate import (
    assess_release_gate,
    latest_matching,
    is_hf_export,
    is_recipe_control,
    is_recovered_benchmark,
    main,
)


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _summary(selected: int, best: int, examples: int = 100) -> dict[str, object]:
    return {
        "summary": {
            "selected_exact": selected,
            "best_of_k_exact": best,
            "examples_with_targets": examples,
        }
    }


def _benchmark(path, *, selected_delta: int = 1, best_delta: int = 1, examples: int = 100):
    payload = {
        "run_id": "bench",
        "base": _summary(10, 12, examples),
        "phase1_start": _summary(8, 9, examples),
        "recovered": _summary(10 + selected_delta, 12 + best_delta, examples),
        "deltas": {
            "recovered_vs_base": {
                "selected_exact_delta": selected_delta,
                "best_of_k_exact_delta": best_delta,
            }
        },
    }
    _write(path, payload)
    return path


def _recipe(path, *, status: str = "passed", passed: bool = True):
    payload = {
        "run_id": "recipe",
        "gate": "stage5_same_recipe_architecture",
        "status": status,
        "passed": passed,
        "reason": "architecture evidence",
        "next_step": "replicate",
        "decision_evidence": {
            "aggregate": {"delta_exact": 1},
            "hard": {"delta_exact": 1},
            "aggregate_best_of_k": {"delta_exact": 2},
            "hard_best_of_k": {"delta_exact": 1},
        },
    }
    _write(path, payload)
    return path


def _selector_conversion(path, *, passed: bool = True):
    payload = {
        "run_id": "selector_conversion",
        "gate": "stage5_same_recipe_selector_conversion",
        "kind": "recipe_selector_conversion",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "reason": "selector converted candidate coverage",
        "next_step": "reassess architecture",
        "passing_selectors": [{"label": "recovered", "selection_strategy": "reliability_vote"}] if passed else [],
        "best_selector": {"label": "recovered", "selection_strategy": "reliability_vote"},
    }
    _write(path, payload)
    return path


def _hf_export(path):
    payload = {
        "run_id": "export",
        "export_dir": "outputs/hf_exports/export",
        "checkpoint": "outputs/stage5/run/phase1.pt",
        "hf_repo_id": "mshapiro123/recurrent-qwen-test",
        "uploaded": False,
        "metadata": {
            "checkpoint_sha256": "abc123",
            "architecture_evidence": {"status": "passed"},
        },
    }
    _write(path, payload)
    return path


def test_release_gate_ready_when_all_evidence_present(tmp_path) -> None:
    benchmark = _benchmark(tmp_path / "outputs" / "stage5" / "bench" / "summary.json")
    recipe = _recipe(tmp_path / "outputs" / "stage5" / "recipe" / "summary.json")
    export = _hf_export(tmp_path / "outputs" / "hf_exports" / "export" / "summary.json")

    payload = assess_release_gate(
        benchmark_summary=benchmark,
        recipe_control_summary=recipe,
        hf_export_summary=export,
        min_arc_examples=100,
    )

    assert payload["status"] == "ready_for_broader_benchmarks"
    assert payload["passed"] is True
    assert [row["passed"] for row in payload["criteria"]] == [True, True, True]


def test_release_gate_routes_selector_conversion_before_release(tmp_path) -> None:
    benchmark = _benchmark(tmp_path / "outputs" / "stage5" / "bench" / "summary.json")
    recipe = _recipe(
        tmp_path / "outputs" / "stage5" / "recipe" / "summary.json",
        status="needs_selector_conversion",
        passed=False,
    )
    export = _hf_export(tmp_path / "outputs" / "hf_exports" / "export" / "summary.json")

    payload = assess_release_gate(
        benchmark_summary=benchmark,
        recipe_control_summary=recipe,
        hf_export_summary=export,
        min_arc_examples=100,
    )

    assert payload["status"] == "needs_selector_conversion"
    assert payload["passed"] is False
    assert "selector" in payload["next_step"].lower()


def test_release_gate_accepts_passed_selector_conversion_as_architecture_evidence(tmp_path) -> None:
    benchmark = _benchmark(tmp_path / "outputs" / "stage5" / "bench" / "summary.json")
    recipe = _recipe(
        tmp_path / "outputs" / "stage5" / "recipe" / "summary.json",
        status="needs_selector_conversion",
        passed=False,
    )
    selector_conversion = _selector_conversion(
        tmp_path / "outputs" / "stage5" / "selector_conversion" / "summary.json",
        passed=True,
    )
    export = _hf_export(tmp_path / "outputs" / "hf_exports" / "export" / "summary.json")

    payload = assess_release_gate(
        benchmark_summary=benchmark,
        recipe_control_summary=recipe,
        selector_conversion_summary=selector_conversion,
        hf_export_summary=export,
        min_arc_examples=100,
    )

    assert payload["status"] == "ready_for_broader_benchmarks"
    assert payload["passed"] is True
    assert payload["selector_conversion_summary"] is not None
    assert payload["criteria"][1]["passed"] is True


def test_release_gate_requires_benchmark_size_before_export_status(tmp_path) -> None:
    benchmark = _benchmark(tmp_path / "outputs" / "stage5" / "bench" / "summary.json", examples=20)
    recipe = _recipe(tmp_path / "outputs" / "stage5" / "recipe" / "summary.json")

    payload = assess_release_gate(
        benchmark_summary=benchmark,
        recipe_control_summary=recipe,
        hf_export_summary=None,
        min_arc_examples=100,
    )

    assert payload["status"] == "needs_benchmark_confirmation"
    assert payload["criteria"][0]["passed"] is False


def test_latest_matching_finds_expected_artifact_types(tmp_path) -> None:
    benchmark = _benchmark(tmp_path / "outputs" / "stage5" / "bench" / "summary.json")
    recipe = _recipe(tmp_path / "outputs" / "stage5" / "recipe" / "summary.json")
    export = _hf_export(tmp_path / "outputs" / "hf_exports" / "export" / "summary.json")
    stage5_files = [benchmark, recipe]
    hf_files = [export]

    assert latest_matching(stage5_files, is_recovered_benchmark) == benchmark
    assert latest_matching(stage5_files, is_recipe_control) == recipe
    assert latest_matching(hf_files, is_hf_export) == export


def test_release_gate_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    benchmark = _benchmark(tmp_path / "outputs" / "stage5" / "bench" / "summary.json")
    recipe = _recipe(tmp_path / "outputs" / "stage5" / "recipe" / "summary.json")
    export = _hf_export(tmp_path / "outputs" / "hf_exports" / "export" / "summary.json")
    output_json = tmp_path / "release_gate.json"
    output_md = tmp_path / "release_gate.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "assess_stage5_release_gate.py",
            "--benchmark_summary",
            str(benchmark),
            "--recipe_control_summary",
            str(recipe),
            "--hf_export_summary",
            str(export),
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ],
    )

    assert main() == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "ready_for_broader_benchmarks"
    assert "Stage 5 Release / Benchmark Gate" in output_md.read_text(encoding="utf-8")
