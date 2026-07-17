"""Path normalization helpers for resumable Stage 5 runners."""

from __future__ import annotations

from pathlib import Path


def resolve_repo_path(root: Path, raw: str | Path) -> Path:
    """Resolve a path reported as either repo-relative or absolute."""

    path = Path(raw)
    return path if path.is_absolute() else root / path


def repo_relative_text(root: Path, raw: str | Path) -> str:
    """Return stable repo-relative text without assuming the input path form."""

    path = Path(raw)
    if not path.is_absolute():
        return path.as_posix()
    return path.relative_to(root).as_posix()
