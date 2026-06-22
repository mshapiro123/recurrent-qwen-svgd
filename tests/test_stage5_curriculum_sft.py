from __future__ import annotations

import json
import subprocess
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


def test_default_curriculum_sft_target_is_programmatic_direct_deep_shard() -> None:
    assert runner.WORK_DIR.as_posix() == "data/curriculum/programmatic_direct_deep_001"
    assert runner.MIN_POSITIVE_ROWS == 2000
    assert runner.MIN_MODE_ROWS == "direct=1000,deep_narrow=1000"


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


def test_validate_drive_backup_creates_default_root_when_drive_mounted(monkeypatch, tmp_path) -> None:
    mydrive = tmp_path / "MyDrive"
    mydrive.mkdir()
    backup_root = mydrive / "recurrent-qwen-svgd-artifacts"

    monkeypatch.setattr(runner, "mydrive_root", lambda: mydrive)
    monkeypatch.setattr(runner, "drive_backup_root", lambda: backup_root)
    monkeypatch.setattr(runner, "mount_drive_if_possible", lambda: None)

    payload = runner.validate_drive_backup(allow_no_backup=False)

    assert payload["available"] is True
    assert backup_root.exists()


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


def test_grouped_eval_metrics_extracts_curriculum_mode_metrics() -> None:
    metrics = {
        "loss": 2.0,
        "group/curriculum_mode/direct/examples": 12.0,
        "group/curriculum_mode/direct/mean_expected_loops": 1.2,
        "group/curriculum_mode/deep_narrow/examples": 8.0,
        "group/curriculum_mode/deep_narrow/mean_expected_loops": 3.1,
    }

    grouped = runner.grouped_eval_metrics(metrics, group_field="curriculum_mode")

    assert grouped == {
        "deep_narrow": {"examples": 8.0, "mean_expected_loops": 3.1},
        "direct": {"examples": 12.0, "mean_expected_loops": 1.2},
    }


def test_eval_jsonl_requests_single_pass_curriculum_mode_groups(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, *, check=True, log_name=None):
        calls.append([str(item) for item in cmd])
        return subprocess.CompletedProcess(cmd, 0, "examples=2\ngroup/curriculum_mode/direct/examples=1\n", None)

    monkeypatch.setattr(runner, "run", fake_run)
    metrics = runner.eval_jsonl("val", tmp_path / "val.jsonl", tmp_path / "phase1.pt")

    assert metrics["group/curriculum_mode/direct/examples"] == 1.0
    assert "--group_by_field" in calls[0]
    assert calls[0][calls[0].index("--group_by_field") + 1] == "curriculum_mode"


def test_curriculum_sft_updates_current_source_summary(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "curriculum_sft"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(runner, "RUN_ID", "curriculum_sft")

    runner.write_summary(
        {
            "sft_gate": {"status": "go"},
            "input_restore": {"restored": False},
            "dataset": {"rows": 16, "train_rows": 15, "val_rows": 1},
            "drive_preflight": {"available": True},
            "resume_from": None,
            "phase1_checkpoint": "outputs/stage5/curriculum_sft/phase1/phase1_step_150.pt",
            "config": {"max_steps": 150, "max_loops": 4},
            "phase1_val": {"loss": 2.5},
            "phase1_val_by_mode": {"direct": {"loss": 2.0}, "deep_narrow": {"loss": 3.0}},
        }
    )

    assert (run_dir / "summary.json").exists()
    assert "Validation By Curriculum Mode" in (run_dir / "summary.md").read_text(encoding="utf-8")
    assert (tmp_path / "config" / "stage5_current_source_summary.txt").read_text(
        encoding="utf-8"
    ) == "outputs/stage5/curriculum_sft/summary.json\n"


def test_curriculum_sft_commit_stages_current_source_pointer(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "curriculum_sft"
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    run_dir.mkdir(parents=True)
    pointer.parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    pointer.write_text("outputs/stage5/curriculum_sft/summary.json\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, check=True, log_name=None):
        commands.append([str(item) for item in cmd])
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(runner, "run", fake_run)

    runner.git_commit_results()

    add_commands = [cmd for cmd in commands if cmd[:2] == ["git", "add"]]
    assert add_commands
    staged = {item for cmd in add_commands for item in cmd[3:]}
    assert "outputs/stage5/curriculum_sft/summary.json" in staged
    assert "config/stage5_current_source_summary.txt" in staged
