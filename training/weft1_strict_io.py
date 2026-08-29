"""Shared strict JSON and lexical-path boundary helpers for WEFT-1.

Governed ledgers use UTF-8, LF-only JSON with exactly one final LF.  Loading
them through ordinary ``json.loads`` would silently accept duplicate keys and
non-finite numbers, so every ledger entrypoint routes through this module.

Source-cache paths are checked lexically for symlinks/reparse points before any
``resolve()``, metadata read, or file open.  The later resolved containment
check remains necessary; these checks address distinct attack surfaces.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any


class StrictJsonError(ValueError):
    """A governed JSON artifact is not strict canonical JSON."""


class StrictPathError(ValueError):
    """A governed filesystem path contains a symlink/reparse boundary."""


def _object_no_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise StrictJsonError(f"JSON object repeats key: {key}")
        value[key] = item
    return value


def _read_strict_json_bytes(path: Path) -> bytes:
    if not isinstance(path, Path):
        raise TypeError("strict JSON path must be a pathlib.Path")
    assert_no_symlink_ancestors(path)
    try:
        return path.read_bytes()
    except OSError as error:
        raise StrictJsonError(f"cannot read strict JSON: {path}") from error


def _parse_strict_json_object(raw: bytes) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_object_no_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                StrictJsonError(f"JSON uses non-finite constant: {constant}")
            ),
        )
    except StrictJsonError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise StrictJsonError("artifact is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise StrictJsonError("strict JSON root must be an object")
    return value


def load_strict_json_object(path: Path) -> dict[str, Any]:
    """Load strict UTF-8 JSON while rejecting duplicates and non-finite values."""

    raw = _read_strict_json_bytes(path)
    if not raw or raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw:
        raise StrictJsonError("strict JSON must be nonempty UTF-8 without BOM/NUL")
    return _parse_strict_json_object(raw)


def load_canonical_json_snapshot(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read once, then return the immutable bytes and parsed canonical object.

    The two checked-in ledgers intentionally use different wrapping choices
    for short arrays, so canonical transport is defined independently of a
    reformatter: UTF-8 without BOM, LF-only, no NUL/tab/trailing line space,
    an object at the root, and exactly one final LF.
    """

    raw = _read_strict_json_bytes(path)
    if not raw or raw.startswith(b"\xef\xbb\xbf"):
        raise StrictJsonError("canonical JSON must be nonempty UTF-8 without BOM")
    if b"\x00" in raw or b"\r" in raw or b"\t" in raw:
        raise StrictJsonError("canonical JSON must use LF-only text without NUL or tabs")
    if not raw.startswith(b"{") or not raw.endswith(b"}\n"):
        raise StrictJsonError("canonical JSON must be one object with one final LF")
    if raw.endswith(b"\n\n") or any(
        line.endswith((b" ", b"\t")) for line in raw[:-1].split(b"\n")
    ):
        raise StrictJsonError("canonical JSON has excess or trailing whitespace")
    return raw, _parse_strict_json_object(raw)


def load_canonical_json_object(path: Path) -> dict[str, Any]:
    """Load one canonical governed JSON object without semantic ambiguity."""

    _, value = load_canonical_json_snapshot(path)
    return value


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise StrictPathError(f"cannot inspect lexical path component: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def assert_no_symlink_ancestors(path: Path) -> Path:
    """Reject every existing symlink/reparse component without resolving it."""

    if not isinstance(path, Path):
        raise TypeError("strict filesystem path must be a pathlib.Path")
    # ``abspath`` normalizes relative syntax but does not dereference symlinks.
    lexical = Path(os.path.abspath(os.fspath(path)))
    chain = tuple(reversed((lexical, *lexical.parents)))
    for component in chain:
        if _is_link_or_reparse(component):
            raise StrictPathError(
                f"governed path contains a symlink/reparse ancestor: {component}"
            )
    return lexical


__all__ = [
    "StrictJsonError",
    "StrictPathError",
    "assert_no_symlink_ancestors",
    "load_canonical_json_object",
    "load_canonical_json_snapshot",
    "load_strict_json_object",
]
