from __future__ import annotations

import json

from colab.run_stage5_benchmark_suite import (
    EvalJob,
    BenchmarkSpec,
    build_summary,
    checkpoint_candidates_from_payload,
    compare_arm_summaries,
    parse_csv,
    resolve_checkpoint,
    summarize_rows,
)


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_parse_csv_trims_empty_items() -> None:
    assert parse_csv(" arc_challenge, gpqa_lite ,,") == ["arc_challenge", "gpqa_lite"]


def test_summarize_rows_groups_by_aggregate() -> None:
    rows = [
        {"aggregate": "mean", "hit": True},
        {"aggregate": "mean", "hit": False},
        {"aggregate": "vote", "hit": True},
    ]

    summary = summarize_rows(rows)

    assert summary["mean"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert summary["vote"] == {"correct": 1, "total": 1, "accuracy": 1.0}


def test_compare_arm_summaries_reports_recurrent_delta() -> None:
    comparison = compare_arm_summaries(
        {"mean": {"correct": 3, "total": 5, "accuracy": 0.6}},
        {"mean": {"correct": 4, "total": 5, "accuracy": 0.8}},
    )

    assert comparison["mean"]["correct_delta_recurrent_vs_base"] == 1
    assert comparison["mean"]["accuracy_delta_recurrent_vs_base"] == 0.20000000000000007


def test_checkpoint_candidates_include_hf_export_checkpoint(tmp_path) -> None:
    source = tmp_path / "outputs" / "hf_exports" / "export" / "summary.json"
    payload = {
        "checkpoint": "missing/original.pt",
        "export_dir": str(source.parent),
    }

    candidates = checkpoint_candidates_from_payload(source, payload)

    assert source.parent / "recurrent_adapter_checkpoint.pt" in candidates


def test_resolve_checkpoint_prefers_existing_export_adapter(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    source = tmp_path / "outputs" / "hf_exports" / "export" / "summary.json"
    checkpoint = source.parent / "recurrent_adapter_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    payload = {
        "checkpoint": str(tmp_path / "missing.pt"),
        "export_dir": str(source.parent),
    }
    monkeypatch.setattr(module, "EXPLICIT_CHECKPOINT", "")

    assert resolve_checkpoint(source, payload) == checkpoint


def test_build_summary_compares_base_and_recurrent_rows(tmp_path) -> None:
    base_jsonl = tmp_path / "arc_base_label.jsonl"
    recurrent_jsonl = tmp_path / "arc_recurrent_label.jsonl"
    _write_jsonl(
        base_jsonl,
        [
            {"aggregate": "mean", "hit": True},
            {"aggregate": "mean", "hit": False},
        ],
    )
    _write_jsonl(
        recurrent_jsonl,
        [
            {"aggregate": "mean", "hit": True},
            {"aggregate": "mean", "hit": True},
        ],
    )

    payload = build_summary(
        source_summary=None,
        checkpoint=tmp_path / "checkpoint.pt",
        specs=[BenchmarkSpec("arc_challenge", tmp_path / "arc.jsonl", [])],
        jobs=[
            EvalJob("arc_challenge", "base", "label", base_jsonl, []),
            EvalJob("arc_challenge", "recurrent", "label", recurrent_jsonl, []),
        ],
        failures=[],
        elapsed_seconds=12.5,
    )

    delta = payload["comparisons"]["arc_challenge"]["label"]["mean"]
    assert payload["status"] == "completed"
    assert delta["correct_delta_recurrent_vs_base"] == 1
    assert delta["base"]["correct"] == 1
    assert delta["recurrent"]["correct"] == 2
