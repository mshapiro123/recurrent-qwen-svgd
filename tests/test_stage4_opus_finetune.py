from __future__ import annotations

import subprocess

import pytest

from colab.run_stage4_opus_finetune import (
    stage_current_source_pointer,
    update_current_source_summary,
    validate_dataset_source,
    validate_drive_backup,
)


def test_validate_dataset_source_allows_approved_opus_sft() -> None:
    payload = validate_dataset_source(
        dataset_id="lordx64/reasoning-distill-opus-4-7-max-sft",
        allow_unapproved=False,
    )

    assert payload["approved"] is True
    assert payload["allow_unapproved"] is False


def test_validate_dataset_source_blocks_unapproved_trace_source_by_default() -> None:
    with pytest.raises(ValueError, match="restricted to approved recovery datasets"):
        validate_dataset_source(
            dataset_id="Jackrong/Claude-opus-4.7-TraceInversion-5000x",
            allow_unapproved=False,
        )


def test_validate_dataset_source_allows_explicit_nondefault_override() -> None:
    payload = validate_dataset_source(
        dataset_id="Jackrong/Claude-opus-4.7-TraceInversion-5000x",
        allow_unapproved=True,
    )

    assert payload["approved"] is False
    assert payload["allow_unapproved"] is True


def test_validate_drive_backup_requires_existing_directory_by_default(tmp_path) -> None:
    missing = tmp_path / "missing-drive"

    with pytest.raises(RuntimeError, match="requires a mounted Drive backup directory"):
        validate_drive_backup(drive_root=missing, allow_no_backup=False)


def test_validate_drive_backup_allows_existing_directory(tmp_path) -> None:
    drive = tmp_path / "drive"
    drive.mkdir()

    payload = validate_drive_backup(drive_root=drive, allow_no_backup=False)

    assert payload["available"] is True
    assert payload["allow_no_backup"] is False


def test_validate_drive_backup_allows_explicit_nondefault_override(tmp_path) -> None:
    missing = tmp_path / "missing-drive"

    payload = validate_drive_backup(drive_root=missing, allow_no_backup=True)

    assert payload["available"] is False
    assert payload["allow_no_backup"] is True


def test_stage4_updates_current_source_summary(monkeypatch, tmp_path) -> None:
    import colab.run_stage4_opus_finetune as module

    summary = tmp_path / "outputs" / "stage4" / "opus" / "summary.json"
    summary.parent.mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)

    written = update_current_source_summary(summary)

    assert written == tmp_path / "config" / "stage5_current_source_summary.txt"
    assert written.read_text(encoding="utf-8") == "outputs/stage4/opus/summary.json\n"


def test_stage4_stages_current_source_pointer(monkeypatch, tmp_path) -> None:
    import colab.run_stage4_opus_finetune as module

    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("outputs/stage4/opus/summary.json\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, check=True, log_name=None):
        commands.append([str(item) for item in cmd])
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "run", fake_run)

    stage_current_source_pointer()

    assert commands == [["git", "add", "-f", "config/stage5_current_source_summary.txt"]]
