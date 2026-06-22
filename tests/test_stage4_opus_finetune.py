from __future__ import annotations

import pytest

from colab.run_stage4_opus_finetune import validate_dataset_source, validate_drive_backup


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
