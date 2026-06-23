import json
from pathlib import Path

from colab.run_stage5_model_viability_probe import parse_int_csv, summarize_pair


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def row(row_id: str, hit: bool, aggregate: str = "mean", expected_loops=None) -> dict:
    payload = {
        "id": row_id,
        "hit": hit,
        "aggregate": aggregate,
    }
    if expected_loops is not None:
        payload["loop_diagnostics"] = {"mean_expected_loops": expected_loops}
    return payload


def test_parse_int_csv_requires_at_least_one_loop():
    assert parse_int_csv("1,2, 3") == [1, 2, 3]

    try:
        parse_int_csv(" ")
    except ValueError as exc:
        assert "Expected at least one loop" in str(exc)
    else:
        raise AssertionError("parse_int_csv should reject an empty loop list")


def test_summarize_pair_reports_paired_delta_and_loop_telemetry(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    recurrent = tmp_path / "recurrent.jsonl"
    write_jsonl(
        base,
        [
            row("a", True),
            row("b", False),
            row("c", True),
        ],
    )
    write_jsonl(
        recurrent,
        [
            row("a", True, expected_loops=1.0),
            row("b", True, expected_loops=2.0),
            row("c", False, expected_loops=3.0),
        ],
    )

    summary = summarize_pair(base, recurrent)["mean"]

    assert summary["paired_examples"] == 3
    assert summary["base_correct"] == 2
    assert summary["recurrent_correct"] == 2
    assert summary["delta"] == 0
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["ties"] == 1
    assert summary["mean_expected_loops"] == 2.0
