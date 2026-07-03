import json

from eval.analyze_synthetic_reader_alignment import analyze


def write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_reader_alignment_suspends_metric_when_readers_differ(tmp_path) -> None:
    active_rows = tmp_path / "active_rows.jsonl"
    final_rows = tmp_path / "final_rows.jsonl"
    final_data = tmp_path / "final_data.jsonl"
    chain_data = tmp_path / "chain_data.jsonl"
    active_summary = tmp_path / "active_summary.json"
    final_summary = tmp_path / "final_summary.json"

    write_jsonl(
        active_rows,
        [
            {
                "id": "x",
                "depth": 2,
                "forced_loop_count": 2,
                "prediction": "P",
                "target": "P",
                "hit": True,
            }
        ],
    )
    write_jsonl(
        final_rows,
        [
            {
                "id": "x",
                "depth": 2,
                "forced_loop_count": 2,
                "prediction": "B",
                "answer": "D",
                "hit": False,
            }
        ],
    )
    write_jsonl(
        final_data,
        [
            {
                "id": "x",
                "choices": {"A": "F", "B": "E", "C": "D", "D": "P"},
                "answer": "D",
                "target": "P",
            }
        ],
    )
    write_jsonl(
        chain_data,
        [
            {
                "id": "x",
                "choices": {"A": "G", "B": "I", "C": "P", "D": "D"},
                "answer": "C",
                "target": "P",
            }
        ],
    )
    active_summary.write_text(
        json.dumps(
            {
                "data_jsonl": "chain.jsonl",
                "prediction_space": "full_symbols",
                "prompt_style": "question_only",
            }
        ),
        encoding="utf-8",
    )
    final_summary.write_text(
        json.dumps(
            {
                "data_jsonl": "final.jsonl",
                "score_target": "option_text",
            }
        ),
        encoding="utf-8",
    )

    payload = analyze(
        active_rows_path=active_rows,
        final_rows_path=final_rows,
        final_data_jsonl=final_data,
        chain_data_jsonl=chain_data,
        active_summary_path=active_summary,
        final_summary_path=final_summary,
    )

    assert payload["final_answer_metric_suspended"] is True
    assert payload["by_depth"]["2"]["table"]["active_right_final_wrong"] == 1
    mismatch = payload["by_depth"]["2"]["active_right_final_wrong"]
    assert mismatch["active_prediction_in_final_choices"] == 1
    assert mismatch["active_target_equals_final_answer_symbol"] == 1
    assert mismatch["active_prediction_equals_final_prediction_symbol"] == 0
