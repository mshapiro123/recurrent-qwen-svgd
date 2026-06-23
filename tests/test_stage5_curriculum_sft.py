from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import colab.run_stage5_curriculum_sft as runner


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def positive_row(index: int, *, mode: str = "deep_narrow") -> dict:
    return {
        "prompt": f"<|im_start|>user\nProblem {index}<|im_end|>\n<|im_start|>assistant\n",
        "completion": f"Reasoning {index}\nANSWER: {index}",
        "trace_role": "positive_depth",
        "curriculum_id": f"p{index}",
        "curriculum_mode": mode,
        "routing_type": mode,
        "target_loop_count": 1 if mode == "direct" else 3,
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


def test_split_train_val_stratifies_direct_and_deep_modes() -> None:
    rows = [positive_row(index, mode="direct") for index in range(8)]
    rows.extend(positive_row(index + 100, mode="deep_narrow") for index in range(8))

    train, val = runner.split_train_val(rows, val_fraction=0.125, val_min_rows=2, seed=11)

    train_modes = {row["curriculum_mode"] for row in train}
    val_modes = {row["curriculum_mode"] for row in val}
    assert len(val) == 2
    assert train_modes == {"direct", "deep_narrow"}
    assert val_modes == {"direct", "deep_narrow"}


def test_split_train_val_balances_imbalanced_modes_proportionally() -> None:
    rows = [positive_row(index, mode="direct") for index in range(33)]
    rows.extend(positive_row(index + 100, mode="deep_narrow") for index in range(40))

    train, val = runner.split_train_val(rows, val_fraction=0.1, val_min_rows=1, seed=17)

    train_counts = {
        mode: sum(1 for row in train if row["curriculum_mode"] == mode)
        for mode in {"direct", "deep_narrow"}
    }
    val_counts = {
        mode: sum(1 for row in val if row["curriculum_mode"] == mode)
        for mode in {"direct", "deep_narrow"}
    }
    assert len(val) == 7
    assert val_counts == {"direct": 3, "deep_narrow": 4}
    assert train_counts == {"direct": 30, "deep_narrow": 36}


def test_insert_depth_hint_keeps_chat_turn_order() -> None:
    prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"

    updated = runner.insert_depth_hint(prompt, "Depth hint: answer directly.")

    assert updated.startswith("<|im_start|>user\nDepth hint: answer directly.\n\nWhat is 2+2?")
    assert updated.endswith("<|im_start|>assistant\n")


def test_depth_hint_for_row_follows_curriculum_mode() -> None:
    assert "shallow" in runner.depth_hint_for_row(positive_row(1, mode="direct"))
    assert "multi-step" in runner.depth_hint_for_row(positive_row(2, mode="deep_narrow"))


def test_split_train_val_keeps_singleton_mode_in_train() -> None:
    rows = [positive_row(1, mode="wide")]
    rows.extend(positive_row(index + 10, mode="deep_narrow") for index in range(5))

    train, val = runner.split_train_val(rows, val_fraction=0.33, val_min_rows=2, seed=3)

    assert any(row["curriculum_mode"] == "wide" for row in train)
    assert all(row["curriculum_mode"] != "wide" for row in val)


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


def test_backup_to_drive_skips_mount_when_no_backup_allowed(monkeypatch, tmp_path) -> None:
    calls = {"mount": 0}
    missing_root = tmp_path / "missing-drive-root"

    monkeypatch.setattr(runner, "ALLOW_NO_DRIVE_BACKUP", True)
    monkeypatch.setattr(runner, "drive_backup_root", lambda: missing_root)
    monkeypatch.setattr(runner, "mount_drive_if_possible", lambda: calls.__setitem__("mount", calls["mount"] + 1))

    payload = runner.backup_to_drive(tmp_path / "train.jsonl", tmp_path / "val.jsonl")

    assert payload == {
        "backed_up": False,
        "drive_root": str(missing_root),
        "reason": "allow_no_drive_backup",
    }
    assert calls["mount"] == 0


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


def test_run_sft_gate_can_allow_answer_line_verified_trace_shards(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(runner, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(runner, "SUMMARY_JSON", str(tmp_path / "work" / "summary.json"))
    monkeypatch.setattr(runner, "MIN_POSITIVE_ROWS", 16)
    monkeypatch.setattr(runner, "MAX_LOOPS", 4)
    monkeypatch.setattr(runner, "MIN_MODE_ROWS", "")
    monkeypatch.setattr(runner, "ALLOW_ANSWER_LINE_VERIFICATION", True)
    monkeypatch.setattr(runner, "write_gate_outputs", lambda payload, *, output_json, output_md: None)

    def fake_build_gate_payload(args):
        captured["allow_answer_line_verification"] = args.allow_answer_line_verification
        return {"go": True, "artifacts": {"positive_sft": str(tmp_path / "positive_sft.jsonl")}}

    monkeypatch.setattr(runner, "build_gate_payload", fake_build_gate_payload)

    assert runner.run_sft_gate()["go"] is True
    assert captured["allow_answer_line_verification"] is True


def test_run_sft_gate_can_allow_cross_model_only_answer_shards(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(runner, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(runner, "SUMMARY_JSON", str(tmp_path / "work" / "summary.json"))
    monkeypatch.setattr(runner, "MIN_POSITIVE_ROWS", 16)
    monkeypatch.setattr(runner, "MAX_LOOPS", 4)
    monkeypatch.setattr(runner, "MIN_MODE_ROWS", "")
    monkeypatch.setattr(runner, "ALLOW_CROSS_MODEL_ONLY_ANSWERS", True)
    monkeypatch.setattr(runner, "write_gate_outputs", lambda payload, *, output_json, output_md: None)

    def fake_build_gate_payload(args):
        captured["require_programmatic_answer_check"] = args.require_programmatic_answer_check
        return {"go": True, "artifacts": {"positive_sft": str(tmp_path / "positive_sft.jsonl")}}

    monkeypatch.setattr(runner, "build_gate_payload", fake_build_gate_payload)

    assert runner.run_sft_gate()["go"] is True
    assert captured["require_programmatic_answer_check"] is False


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


def test_validation_checks_report_sane_depth_gradient() -> None:
    checks = runner.validation_checks(
        {"loss": 2.0, "mean_expected_loops": 2.4},
        {
            "direct": {"mean_expected_loops": 1.1},
            "deep_narrow": {"mean_expected_loops": 2.8},
        },
    )

    assert checks["status"] == "validation_sane"
    assert checks["issues"] == []
    assert checks["depth_gradient"]["observed"] is True


def test_validation_checks_flag_nonfinite_and_loop_collapse() -> None:
    checks = runner.validation_checks(
        {"loss": float("nan"), "mean_expected_loops": 1.0},
        {
            "direct": {"mean_expected_loops": 1.2},
            "deep_narrow": {"mean_expected_loops": 1.3},
        },
    )

    assert checks["status"] == "validation_needs_review"
    assert checks["nonfinite_metrics"] == ["loss"]
    assert "mean_expected_loops_collapsed" in checks["issues"]
    assert "depth_gradient_not_observed" in checks["issues"]
    assert checks["depth_gradient"]["observed"] is False


def test_validation_checks_require_depth_gradient() -> None:
    checks = runner.validation_checks(
        {"loss": 2.0, "mean_expected_loops": 2.4},
        {
            "direct": {"mean_expected_loops": 1.4},
            "deep_narrow": {"mean_expected_loops": 1.5},
        },
    )

    assert checks["status"] == "validation_needs_review"
    assert "depth_gradient_not_observed" in checks["issues"]
    assert checks["depth_gradient"]["observed"] is False


def test_validation_checks_flag_missing_depth_gradient_metrics() -> None:
    checks = runner.validation_checks(
        {"loss": 2.0, "mean_expected_loops": 2.4},
        {"direct": {"mean_expected_loops": 1.4}},
    )

    assert checks["status"] == "validation_needs_review"
    assert "missing_depth_gradient_metrics" in checks["issues"]
    assert checks["depth_gradient"]["available"] is False


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
            "validation_checks": {"status": "validation_sane", "issues": []},
        }
    )

    assert (run_dir / "summary.json").exists()
    assert "Validation By Curriculum Mode" in (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "Validation Checks" in (run_dir / "summary.md").read_text(encoding="utf-8")
    assert (tmp_path / "config" / "stage5_current_source_summary.txt").read_text(
        encoding="utf-8"
    ) == "outputs/stage5/curriculum_sft/summary.json\n"


def test_curriculum_sft_commit_stages_current_source_pointer(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "curriculum_sft"
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    run_dir.mkdir(parents=True)
    phase1_dir = run_dir / "phase1"
    phase1_dir.mkdir()
    (phase1_dir / "phase1_step_50.pt").write_bytes(b"old checkpoint")
    (phase1_dir / "phase1_step_150.pt").write_bytes(b"latest checkpoint")
    pointer.parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    pointer.write_text("outputs/stage5/curriculum_sft/summary.json\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, check=True, log_name=None):
        commands.append([str(item) for item in cmd])
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(runner, "COMMIT_CHECKPOINTS", True)
    monkeypatch.setattr(runner, "run", fake_run)

    runner.git_commit_results()

    add_commands = [cmd for cmd in commands if cmd[:2] == ["git", "add"]]
    assert add_commands
    staged = {item for cmd in add_commands for item in cmd[3:]}
    assert "outputs/stage5/curriculum_sft/summary.json" in staged
    assert "config/stage5_current_source_summary.txt" in staged
    assert "outputs/stage5/curriculum_sft/phase1/phase1_step_150.pt" in staged
    assert "outputs/stage5/curriculum_sft/phase1/phase1_step_50.pt" not in staged
