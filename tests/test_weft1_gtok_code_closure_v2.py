from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import training.weft1_gtok_code_closure_v2 as closure


def _git(monkeypatch: pytest.MonkeyPatch, *, dirty: bool = False) -> None:
    def run(command, **_kwargs):
        if command[1:3] == ["status", "--porcelain=v1"]:
            return SimpleNamespace(stdout=" M governed.py\n" if dirty else "")
        if command[1:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="1" * 40 + "\n")
        raise AssertionError(command)

    monkeypatch.setattr(closure.subprocess, "run", run)


def test_code_closure_rejects_dirty_checkout_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(monkeypatch, dirty=True)
    with pytest.raises(closure.GTokCodeClosureV2Error, match="clean"):
        closure.capture_gtok_code_closure_v2(tmp_path)


def test_code_closure_binds_and_revalidates_behavior_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    governed = tmp_path / "governed.py"
    governed.write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    _git(monkeypatch)
    monkeypatch.setattr(closure, "_behavior_paths", lambda _root: ("governed.py",))
    receipt = closure.capture_gtok_code_closure_v2(tmp_path)
    closure.validate_gtok_code_closure_v2(receipt, repository_root=tmp_path)
    governed.write_text("VALUE = 2\n", encoding="utf-8", newline="\n")
    with pytest.raises(closure.GTokCodeClosureV2Error, match="changed"):
        closure.validate_gtok_code_closure_v2(receipt, repository_root=tmp_path)


def test_every_behavior_bearing_path_is_forced_to_lf() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = closure._behavior_paths(root)
    completed = subprocess.run(
        ["git", "check-attr", "eol", "--", *paths],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = tuple(line for line in completed.stdout.splitlines() if line)
    assert len(rows) == len(paths)
    assert all(row.endswith(": eol: lf") for row in rows)
