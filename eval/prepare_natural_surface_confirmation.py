"""Freeze a natural-surface confirmation split disjoint from the selection canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def hash_values(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_slice(
    rows: list[dict[str, Any]],
    *,
    offset_per_depth: int,
    rows_per_depth: int,
    max_depth: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for depth in range(1, int(max_depth) + 1):
        depth_rows = [row for row in rows if int(row["depth"]) == depth]
        chunk = depth_rows[int(offset_per_depth) : int(offset_per_depth) + int(rows_per_depth)]
        if len(chunk) != int(rows_per_depth):
            raise ValueError(
                f"Insufficient rows at depth {depth}: requested offset={offset_per_depth}, "
                f"count={rows_per_depth}, available={len(depth_rows)}"
            )
        selected.extend(chunk)
    return selected


def freeze_confirmation_split(
    relay_rows: list[dict[str, Any]],
    pointer_rows: list[dict[str, Any]],
    *,
    selection_rows_per_family_depth: int = 16,
    confirmation_rows_per_family_depth: int = 16,
    max_depth: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection_relay = balanced_slice(
        relay_rows,
        offset_per_depth=0,
        rows_per_depth=selection_rows_per_family_depth,
        max_depth=max_depth,
    )
    selection_pointer = balanced_slice(
        pointer_rows,
        offset_per_depth=0,
        rows_per_depth=selection_rows_per_family_depth,
        max_depth=max_depth,
    )
    confirmation_relay = balanced_slice(
        relay_rows,
        offset_per_depth=selection_rows_per_family_depth,
        rows_per_depth=confirmation_rows_per_family_depth,
        max_depth=max_depth,
    )
    confirmation_pointer = balanced_slice(
        pointer_rows,
        offset_per_depth=selection_rows_per_family_depth,
        rows_per_depth=confirmation_rows_per_family_depth,
        max_depth=max_depth,
    )
    selection = selection_relay + selection_pointer
    confirmation = confirmation_relay + confirmation_pointer
    selection_pair_ids = {str(row.get("paired_instance_id") or row["id"]) for row in selection}
    confirmation_pair_ids = {str(row.get("paired_instance_id") or row["id"]) for row in confirmation}
    overlap = sorted(selection_pair_ids & confirmation_pair_ids)
    if overlap:
        raise ValueError(f"Selection and confirmation rows overlap: {overlap[:5]}")
    counts = Counter((str(row.get("verbal_surface_family")), int(row["depth"])) for row in confirmation)
    expected = {
        (family, depth): int(confirmation_rows_per_family_depth)
        for family in ("relay", "pointer")
        for depth in range(1, int(max_depth) + 1)
    }
    if counts != expected:
        raise ValueError(f"Confirmation balance mismatch: {dict(counts)} != {expected}")
    manifest = {
        "kind": "natural_surface_confirmation_split",
        "selection": {
            "offset_per_family_depth": 0,
            "rows_per_family_depth": int(selection_rows_per_family_depth),
            "row_ids_sha256": hash_values([str(row["id"]) for row in selection]),
            "paired_ids_sha256": hash_values(sorted(selection_pair_ids)),
            "rows": len(selection),
        },
        "confirmation": {
            "offset_per_family_depth": int(selection_rows_per_family_depth),
            "rows_per_family_depth": int(confirmation_rows_per_family_depth),
            "row_ids_sha256": hash_values([str(row["id"]) for row in confirmation]),
            "paired_ids_sha256": hash_values(sorted(confirmation_pair_ids)),
            "rows": len(confirmation),
            "by_family_depth": {f"{family}_d{depth}": count for (family, depth), count in sorted(counts.items())},
        },
        "max_depth": int(max_depth),
        "paired_overlap": overlap,
        "status": "frozen_disjoint_confirmation",
    }
    return confirmation, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-jsonl", required=True)
    parser.add_argument("--pointer-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--selection-rows-per-family-depth", type=int, default=16)
    parser.add_argument("--confirmation-rows-per-family-depth", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()
    rows, manifest = freeze_confirmation_split(
        read_jsonl(args.relay_jsonl),
        read_jsonl(args.pointer_jsonl),
        selection_rows_per_family_depth=args.selection_rows_per_family_depth,
        confirmation_rows_per_family_depth=args.confirmation_rows_per_family_depth,
        max_depth=args.max_depth,
    )
    manifest["source"] = {
        "relay_jsonl": str(args.relay_jsonl),
        "relay_sha256": sha256_file(args.relay_jsonl),
        "pointer_jsonl": str(args.pointer_jsonl),
        "pointer_sha256": sha256_file(args.pointer_jsonl),
    }
    write_jsonl(args.output_jsonl, rows)
    manifest_path = Path(args.output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
