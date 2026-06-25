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
