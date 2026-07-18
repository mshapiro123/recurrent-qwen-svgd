from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.train_phase_g_alpha import truncate_jsonl_after_step


def test_truncate_phase_g_trace_drops_future_rows_and_torn_final_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "loss": 3.0}),
                json.dumps({"step": 2, "loss": 2.0}),
                json.dumps({"step": 3, "loss": 1.0}),
                '{"step": 4, "loss":',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    truncate_jsonl_after_step(path, 2)

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["step"] for row in rows] == [1, 2]


def test_truncate_phase_g_trace_rejects_corrupt_nonfinal_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        '{"step": 1}\n{"step":\n{"step": 3}\n',
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        truncate_jsonl_after_step(path, 2)
