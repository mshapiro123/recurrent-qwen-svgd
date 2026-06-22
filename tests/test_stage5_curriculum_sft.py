from __future__ import annotations

import json
from pathlib import Path

import pytest

import colab.run_stage5_curriculum_sft as runner


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def positive_row(index: int) -> dict:
    return {
        "prompt": f"<|im_start|>user\nProblem {index}<|im_end|>\n<|im_start|>assistant\n",
        "completion": f"Reasoning {index}\nANSWER: {index}",
        "trace_role": "positive_depth",
        "curriculum_id": f"p{index}",
        "curriculum_mode": "deep_narrow",
        "routing_type": "deep_narrow",
        "target_loop_count": 3,
    }


def test_split_train_val_is_deterministic_and_held_out() -> None:
    rows = [positive_row(index) for index in range(10)]

    train_a, val_a = runner.split_train_val(rows, val_fraction=0.2, val_min_rows=1, seed=7)
    train_b, val_b = runner.split_train_val(rows, val_fraction=0.2, val_min_rows=1, seed=7)

    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 8
    assert len(val_a) == 2
    assert {row["curriculum_id"] for row in train_a}.isdisjoint({row["curriculum_id"] for row in val_a})


def test_split_train_val_requires_two_rows() -> None:
    with pytest.raises(ValueError, match="At least two"):
        runner.split_train_val([positive_row(1)])


def test_validate_drive_backup_blocks_missing_drive(tmp_path) -> None:
    missing = tmp_path / "missing-drive-root"

    with pytest.raises(RuntimeError, match="requires a mounted Drive backup"):
        runner.validate_drive_backup(drive_root=missing, allow_no_backup=False)

    assert runner.validate_drive_backup(drive_root=missing, allow_no_backup=True)["available"] is False


def test_restore_work_dir_from_drive_backup(monkeypatch, tmp_path) -> None:
    local = tmp_path / "workspace" / "data" / "curriculum" / "run_001"
    backup_root = tmp_path / "drive" / "curriculum_runs"
    backup = backup_root / "run_001"
    backup.mkdir(parents=True)
    (backup / "summary.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (backup / "positive_sft.jsonl").write_text(json.dumps(positive_row(1)) + "\n", encoding="utf-8")

    monkeypatch.setattr(runner, "CURRICULUM_INPUT_BACKUP_DIR", str(backup_root))
    monkeypatch.setattr(runner, "SUMMARY_JSON", "")
    monkeypatch.setattr(runner, "mount_drive_if_possible", lambda: None)

    result = runner.restore_work_dir_if_needed(local)

    assert result["restored"] is True
    assert (local / "summary.json").exists()
    assert (local / "positive_sft.jsonl").exists()


def test_restore_work_dir_raises_when_local_and_backup_missing(monkeypatch, tmp_path) -> None:
    local = tmp_path / "workspace" / "data" / "curriculum" / "run_001"
    backup_root = tmp_path / "drive" / "curriculum_runs"

    monkeypatch.setattr(runner, "CURRICULUM_INPUT_BACKUP_DIR", str(backup_root))
    monkeypatch.setattr(runner, "SUMMARY_JSON", "")
    monkeypatch.setattr(runner, "mount_drive_if_possible", lambda: None)

    with pytest.raises(FileNotFoundError, match="Missing curriculum work dir"):
        runner.restore_work_dir_if_needed(local)


def test_prepare_train_val_uses_gate_positive_sft(monkeypatch, tmp_path) -> None:
    positive_sft = tmp_path / "positive_sft.jsonl"
    write_jsonl(positive_sft, [positive_row(index) for index in range(5)])

    monkeypatch.setattr(runner, "DATA_DIR", tmp_path / "prepared")
    monkeypatch.setattr(runner, "MIN_POSITIVE_ROWS", 4)
    monkeypatch.setattr(runner, "VAL_FRACTION", 0.4)
    monkeypatch.setattr(runner, "VAL_MIN_ROWS", 1)
    monkeypatch.setattr(runner, "SPLIT_SEED", 3)

    train_jsonl, val_jsonl, summary = runner.prepare_train_val(
        {"artifacts": {"positive_sft": str(positive_sft)}}
    )

    assert train_jsonl.exists()
    assert val_jsonl.exists()
    assert summary["rows"] == 5
    assert summary["train_rows"] == 3
    assert summary["val_rows"] == 2
    train_ids = {row["curriculum_id"] for row in runner.read_jsonl(train_jsonl)}
    val_ids = {row["curriculum_id"] for row in runner.read_jsonl(val_jsonl)}
    assert train_ids.isdisjoint(val_ids)


def test_prepare_train_val_blocks_tiny_shard(monkeypatch, tmp_path) -> None:
    positive_sft = tmp_path / "positive_sft.jsonl"
    write_jsonl(positive_sft, [positive_row(1), positive_row(2)])
    monkeypatch.setattr(runner, "MIN_POSITIVE_ROWS", 3)

    with pytest.raises(RuntimeError, match="positive_sft has 2 rows"):
        runner.prepare_train_val({"artifacts": {"positive_sft": str(positive_sft)}})
