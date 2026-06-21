from __future__ import annotations

import json
from pathlib import Path

from training.arc_agi_training_signal import (
    summarize_rows,
    summarize_training_signal,
    task_family,
    training_signal_markdown,
    warnings_for_summary,
    write_training_signal_report,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row(
    task_id: str,
    *,
    completion: str = "12\n34",
    trace_mode: str = "none",
    trace_source: str | None = None,
    source_dataset: str = "arc-agi",
    category: str = "arc_original_test_pair",
    selected: bool = False,
    selector_generated: bool = False,
    selected_exceeds_best_of_k: bool = False,
) -> dict[str, object]:
    return {
        "prompt": "solve",
        "completion": completion,
        "cot": completion,
        "cot_tokens": 3,
        "source_dataset": source_dataset,
        "category": category,
        "task_id": task_id,
        "test_index": 0,
        "trace_mode": trace_mode,
        "trace_source": trace_source,
        "selected": selected,
        "selector_generated": selector_generated,
        "selected_exceeds_best_of_k": selected_exceeds_best_of_k,
    }


def test_task_family_extracts_synthetic_modes() -> None:
    assert task_family("synthetic_move_recolor_000123") == "move_recolor"
    assert task_family("synthetic_frame_object_000001:loo0") == "frame_object"
    assert task_family("0d3d703e") == "arc"


def test_summarize_rows_profiles_training_signal() -> None:
    rows = [
        _row("0d3d703e"),
        _row(
            "synthetic_move_recolor_000001",
            completion="<think>\nprogram:\n  return grid\n</think>\n12",
            trace_mode="symbolic_program",
            trace_source="move_non_background",
        ),
        _row(
            "synthetic_move_recolor_000002",
            completion="<think>\ntrace\n</think>\n34",
            trace_mode="symbolic",
            trace_source="move_non_background",
        ),
        _row(
            "synthetic_constant_output_000003",
            source_dataset="arc-agi-candidate-distill",
            category="symbolic_constant_output",
            selected=True,
            selector_generated=True,
            selected_exceeds_best_of_k=True,
        ),
        _row(
            "synthetic_crop_recolor_000004",
            completion="<think>\nprogram state trace:\nstep 1: grid = crop_non_background(test_input, background=0)\n1\n</think>\n2",
            trace_mode="symbolic_state_trace",
            trace_source="crop_non_background+color_map",
        ),
    ]

    summary = summarize_rows(rows)

    assert summary["rows"] == 5
    assert summary["public_arc_rows"] == 1
    assert summary["synthetic_rows"] == 4
    assert summary["candidate_distill_rows"] == 1
    assert summary["candidate_distill_selector_generated_rows"] == 1
    assert summary["candidate_distill_selected_rows"] == 1
    assert summary["candidate_distill_selected_exceeds_best_of_k_rows"] == 1
    assert summary["trace_rows"] == 3
    assert summary["program_trace_rows"] == 2
    assert summary["task_family_counts"]["move_recolor"] == 2
    assert summary["task_family_counts"]["crop_recolor"] == 1
    assert summary["source_dataset_counts"]["arc-agi-candidate-distill"] == 1
    assert summary["trace_source_counts"]["move_non_background"] == 2
    assert summary["completion_chars"]["max"] >= len("<think>")


def test_warnings_flag_missing_expected_signal() -> None:
    summary = summarize_rows([_row("0d3d703e")])

    warnings = warnings_for_summary(
        summary,
        {
            "trace_mode": "symbolic_state_trace",
            "trace_filter": "covered",
            "synthetic_tasks": 10,
            "candidate_distill_jsonls": ["candidates.jsonl"],
        },
    )

    assert "Trace mode `symbolic_state_trace` was requested but no traced rows were found." in warnings
    assert "Trace filter `covered` was requested but some rows have no trace." in warnings
    assert "Synthetic tasks were requested but no synthetic rows were found." in warnings
    assert "Candidate distillation sources were configured but no candidate-distill rows were found." in warnings


def test_summarize_training_signal_includes_val_and_metadata(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_jsonl(
        train,
        [
            _row(
                "synthetic_crop_recolor_000001",
                completion="<think>\nprogram:\n  return grid\n</think>\n12",
                trace_mode="symbolic_program",
                trace_source="crop_non_background+color_map",
            )
        ],
    )
    _write_jsonl(val, [_row("0d3d703e")])

    payload = summarize_training_signal(
        train,
        val_jsonl=val,
        metadata={
            "trace_mode": "symbolic_program",
            "trace_filter": "covered",
            "synthetic_tasks": 1,
            "synthetic_modes": "crop_recolor",
            "grid_format": "compact",
        },
    )

    assert payload["train"]["rows"] == 1
    assert payload["train"]["program_trace_rows"] == 1
    assert payload["val"]["rows"] == 1
    assert payload["metadata_projection"]["synthetic_modes"] == "crop_recolor"
    assert payload["warnings"] == []


def test_training_signal_report_writes_json_and_markdown(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    _write_jsonl(train, [_row("0d3d703e")])
    payload = summarize_training_signal(train)

    write_training_signal_report(payload, tmp_path / "signal.json", tmp_path / "signal.md")

    loaded = json.loads((tmp_path / "signal.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "signal.md").read_text(encoding="utf-8")
    assert loaded["train"]["rows"] == 1
    assert "ARC-AGI Training Signal Audit" in markdown
    assert "| `arc` | 1 |" in markdown
    assert training_signal_markdown(payload) == markdown
