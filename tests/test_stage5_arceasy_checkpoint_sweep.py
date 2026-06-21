from __future__ import annotations

import json

from colab.run_stage5_arceasy_checkpoint_sweep import build_summary, parse_steps


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_parse_steps_ignores_empty_items() -> None:
    assert parse_steps("50, 100,,200") == [50, 100, 200]


def test_build_summary_reports_arm_deltas_and_paired_counts(tmp_path) -> None:
    base = tmp_path / "base.jsonl"
    arm = tmp_path / "arm.jsonl"
    _write_jsonl(
        base,
        [
            {"id": "a", "aggregate": "mean", "hit": True},
            {"id": "b", "aggregate": "mean", "hit": False},
            {"id": "c", "aggregate": "mean", "hit": True},
        ],
    )
    _write_jsonl(
        arm,
        [
            {"id": "a", "aggregate": "mean", "hit": False},
            {"id": "b", "aggregate": "mean", "hit": True},
            {"id": "c", "aggregate": "mean", "hit": True},
        ],
    )

    payload = build_summary(base, [("candidate", tmp_path / "checkpoint.pt", arm)])

    assert payload["base"]["summary"]["correct"] == 2
    assert payload["arms"][0]["summary"]["correct"] == 2
    assert payload["arms"][0]["delta_vs_base"] == 0
    assert payload["arms"][0]["paired_vs_base"]["wins"] == 1
    assert payload["arms"][0]["paired_vs_base"]["losses"] == 1
