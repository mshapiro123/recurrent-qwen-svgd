from __future__ import annotations

import json
from pathlib import Path

from colab.summarize_stage5_progress import scan_progress, write_report


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _summary(selected: int, best: int, examples: int = 20) -> dict[str, object]:
    return {
        "selected_exact": selected,
        "best_of_k_exact": best,
        "first_exact": selected,
        "examples_with_targets": examples,
        "selected_accuracy": selected / examples,
        "best_of_k_accuracy": best / examples,
        "valid_candidate_rate": 1.0,
    }


def test_progress_ledger_reads_recovered_benchmark_and_gaps(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "bench" / "summary.json"
    _write(
        source,
        {
            "run_id": "bench",
            "base": {"summary": _summary(8, 9)},
            "phase1_start": {"summary": _summary(2, 3)},
            "recovered": {"summary": _summary(6, 7)},
            "deltas": {},
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["parsed_records"] == 3
    assert payload["best_by_arm"]["base"]["selected_exact"] == 8
    assert payload["best_by_arm"]["recovered"]["best_of_k_exact"] == 7
    assert payload["recovered_vs_base_gaps"] == [
        {
            "run_id": "bench",
            "examples": 20,
            "selected_delta_recovered_vs_base": -2,
            "best_of_k_delta_recovered_vs_base": -2,
            "path": str(source),
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reads_selector_rescore_rows(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "selector" / "summary.json"
    _write(
        source,
        {
            "run_id": "selector",
            "source_run_dir": "outputs/stage5/source",
            "strategies": ["self_consistency"],
            "rows": [
                {
                    "label": "recovered",
                    "selection_strategy": "self_consistency",
                    "examples": 50,
                    "selected_exact": 12,
                    "best_of_k_exact": 14,
                    "valid_candidate_rate": 0.9,
                }
            ],
            "best_by_label": {},
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["parsed_records"] == 1
    record = payload["records"][0]
    assert record["kind"] == "selector_rescore"
    assert record["arm"] == "recovered"
    assert record["label"] == "self_consistency"
    assert record["selected_exact"] == 12
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reads_recovery_particle_gate(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "particle" / "summary.json"
    _write(
        source,
        {
            "run_id": "particle",
            "recovered_checkpoint": {
                "checkpoint": "outputs/stage5/recovered.pt",
                "summary": _summary(3, 4, examples=10),
            },
            "recovery_decision": {"passed": True, "evidence": {}},
            "particle_decision": {
                "passed": True,
                "evidence": {
                    "variants": {
                        "svgd": {
                            "passed": True,
                            "mean_delta_vs_tuned": {
                                "selected_delta": 1,
                                "best_of_k_delta": 2,
                            },
                        }
                    }
                },
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["parsed_records"] == 2
    recovered = next(record for record in payload["records"] if record["arm"] == "recovered")
    particle = next(record for record in payload["records"] if record["arm"] == "particle")
    assert recovered["selected_exact"] == 3
    assert particle["label"] == "svgd"
    assert particle["selected_delta_vs_recovered"] == 1.0
    assert particle["best_of_k_delta_vs_recovered"] == 2.0
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reads_dense_sft_control(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "dense" / "summary.json"
    _write(
        source,
        {
            "run_id": "dense",
            "kind": "dense_sft_control",
            "base": {"summary": _summary(5, 6, examples=10)},
            "dense_tuned": {"summary": _summary(7, 8, examples=10)},
            "phase1_start": {"summary": _summary(4, 5, examples=10)},
            "deltas": {},
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["parsed_records"] == 3
    dense = next(record for record in payload["records"] if record["arm"] == "dense_tuned")
    assert dense["kind"] == "dense_sft_control"
    assert dense["selected_exact"] == 7
    assert payload["best_by_arm"]["dense_tuned"]["best_of_k_exact"] == 8
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reads_recurrent_sft_summary(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "recurrent" / "summary.json"
    _write(
        source,
        {
            "run_id": "recurrent",
            "base": _summary(5, 6, examples=10),
            "phase1_start": _summary(4, 5, examples=10),
            "phase1_arc_agi_tuned": _summary(7, 8, examples=10),
            "tuned_checkpoint": "outputs/stage5/recurrent/phase1.pt",
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["parsed_records"] == 3
    tuned = next(record for record in payload["records"] if record["arm"] == "recurrent_tuned")
    assert tuned["kind"] == "recurrent_sft"
    assert tuned["selected_exact"] == 7
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_gate1_assessments(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "gate1" / "summary.json"
    _write(
        source,
        {
            "run_id": "gate1",
            "gate": "stage5_gate1_selector_tta",
            "status": "passed",
            "passed": True,
            "source_summary": "outputs/stage5/selector/summary.json",
            "source_kind": "selector_rescore",
            "reason": "hard-tail lift",
            "next_step": "replicate",
            "passing_comparisons": ["selector_vs_source"],
            "tradeoff_comparisons": [],
            "num_comparisons": 1,
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["gate1_assessments"] == [
        {
            "path": str(source),
            "run_id": "gate1",
            "status": "passed",
            "passed": True,
            "source_summary": "outputs/stage5/selector/summary.json",
            "source_kind": "selector_rescore",
            "reason": "hard-tail lift",
            "next_step": "replicate",
            "passing_comparisons": ["selector_vs_source"],
            "tradeoff_comparisons": [],
            "num_comparisons": 1,
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_gate2_assessments(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "gate2" / "summary.json"
    _write(
        source,
        {
            "run_id": "gate2",
            "gate": "stage5_gate2_particle_mechanism",
            "status": "passed",
            "passed": True,
            "source_summary": "outputs/stage5/particle/summary.json",
            "source_kind": "recovery_particle_gate",
            "reason": "replicated selected lift",
            "next_step": "replicate",
            "best_variant": {
                "variant": "svgd",
                "selected_delta": 1,
                "best_of_k_delta": 2,
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["gate2_assessments"] == [
        {
            "path": str(source),
            "run_id": "gate2",
            "status": "passed",
            "passed": True,
            "source_summary": "outputs/stage5/particle/summary.json",
            "source_kind": "recovery_particle_gate",
            "reason": "replicated selected lift",
            "next_step": "replicate",
            "best_variant": "svgd",
            "selected_delta": 1.0,
            "best_of_k_delta": 2.0,
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_reports_recipe_control_assessments(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    source = scan_root / "recipe" / "summary.json"
    _write(
        source,
        {
            "run_id": "recipe",
            "gate": "stage5_same_recipe_architecture",
            "status": "passed",
            "passed": True,
            "dense_summary": "outputs/stage5/dense/summary.json",
            "recurrent_summary": "outputs/stage5/recurrent/summary.json",
            "reason": "hard-tail lift",
            "next_step": "replicate",
            "decision_evidence": {
                "aggregate": {"delta_exact": 1},
                "hard": {"delta_exact": 2},
            },
        },
    )

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["recipe_control_assessments"] == [
        {
            "path": str(source),
            "run_id": "recipe",
            "status": "passed",
            "passed": True,
            "dense_summary": "outputs/stage5/dense/summary.json",
            "recurrent_summary": "outputs/stage5/recurrent/summary.json",
            "reason": "hard-tail lift",
            "next_step": "replicate",
            "aggregate_selected_delta": 1,
            "hard_selected_delta": 2,
        }
    ]
    assert payload["recommended_next_plan_source"] == str(source)


def test_progress_ledger_skips_empty_and_malformed_eval_summaries(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    _write(scan_root / "empty" / "base_summary.json", {"summary": {}})
    _write(
        scan_root / "zero_evidence" / "base_summary.json",
        {
            "summary": {
                "first_exact": 0,
                "selected_exact": 0,
                "best_of_k_exact": 0,
                "examples_with_targets": 0,
                "valid_candidate_rate": 0.0,
            }
        },
    )
    malformed = scan_root / "broken" / "summary.json"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{not json", encoding="utf-8")

    payload = scan_progress(scan_root, run_id="ledger")

    assert payload["scanned_files"] == 3
    assert payload["parsed_records"] == 0
    assert payload["skipped_files"] == [str(malformed)]
    assert payload["recommended_next_plan_source"] is None


def test_progress_ledger_writes_summary_markdown(tmp_path) -> None:
    scan_root = tmp_path / "outputs" / "stage5"
    _write(
        scan_root / "bench" / "summary.json",
        {
            "run_id": "bench",
            "base": {"summary": _summary(8, 9)},
            "phase1_start": {"summary": _summary(2, 3)},
            "recovered": {"summary": _summary(6, 7)},
            "deltas": {},
        },
    )
    payload = scan_progress(scan_root, run_id="ledger")
    output_dir = tmp_path / "ledger"

    write_report(payload, output_dir)

    report = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Stage 5 ARC-AGI Progress Ledger" in report
    assert "| `base` | 8 | 9 | 20 | `base` |" in report
    assert "selected delta `-2`" in report
    assert (output_dir / "summary.json").exists()
