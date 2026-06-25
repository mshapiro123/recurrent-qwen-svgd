"""Helpers for publishing lightweight Stage 5 artifacts to GitHub."""

from __future__ import annotations

from pathlib import Path


DEFAULT_PUBLISH_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}

BLOCKED_CHECKPOINT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".pt",
    ".pth",
    ".safetensors",
}


def update_current_source_summary(root: str | Path, summary_path: str | Path) -> Path:
    """Point the generic Stage 5 launcher at a newly published summary.

    The Stage 5 Colab flow intentionally uses one notebook plus a current
    source-summary pointer. Launchers that publish a new readout should advance
    that pointer in the same commit, otherwise restarted notebooks can route
    backward to an already-completed phase.
    """

    root = Path(root)
    summary = Path(summary_path)
    if not summary.is_absolute():
        summary = root / summary
    try:
        rel_summary = summary.relative_to(root)
    except ValueError:
        rel_summary = summary
    pointer = root / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(rel_summary.as_posix() + "\n", encoding="utf-8")
    return pointer


def publishable_artifact_paths(
    root: str | Path,
    *,
    allowed_suffixes: set[str] | None = None,
    blocked_suffixes: set[str] | None = None,
) -> list[Path]:
    """Return files that should be force-added despite broad output ignores.

    Stage 5 output directories are ignored by default, so Colab launchers often
    need ``git add -f``. This helper keeps that force-add scoped to evidence
    artifacts and prevents model checkpoints from entering Git history.
    """

    root = Path(root)
    allowed = {suffix.lower() for suffix in (allowed_suffixes or DEFAULT_PUBLISH_SUFFIXES)}
    blocked = {suffix.lower() for suffix in (blocked_suffixes or BLOCKED_CHECKPOINT_SUFFIXES)}
    if not root.exists():
        return []
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in blocked:
            continue
        if allowed and suffix not in allowed:
            continue
        paths.append(path)
    return sorted(paths)
