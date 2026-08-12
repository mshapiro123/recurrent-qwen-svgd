"""Freeze the ratified P3.4 step-zero loss-share calibration batches."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from training.paper2_phase3_p32 import GateLabel


P34_SHARE_CALIBRATION_SEED = 20260812
P34_SHARE_ROWS_PER_STRATUM = 128
P34_SHARE_POSITIVES_PER_STRATUM = 32
P34_SHARE_NEGATIVES_PER_STRATUM = 96
P34_SHARE_STRATA = ("code", "general")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    material = "\n".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) for row in rows
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _rank(seed: int, stratum: str, label: int, record_id: str) -> int:
    payload = f"{seed}:{stratum}:{label}:{record_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def select_share_calibration_rows(
    rows: Iterable[Mapping[str, Any]], *, seed: int = P34_SHARE_CALIBRATION_SEED
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quotas = {
        (stratum, int(GateLabel.POSITIVE)): P34_SHARE_POSITIVES_PER_STRATUM
        for stratum in P34_SHARE_STRATA
    }
    quotas.update(
        {
            (stratum, int(GateLabel.NEGATIVE)): P34_SHARE_NEGATIVES_PER_STRATUM
            for stratum in P34_SHARE_STRATA
        }
    )
    heaps: dict[tuple[str, int], list[tuple[int, str, dict[str, Any]]]] = {
        key: [] for key in quotas
    }
    eligible_counts: Counter[tuple[str, int]] = Counter()
    for source in rows:
        if not bool(source.get("training_eligible")) or bool(source.get("audit_holdout")):
            continue
        stratum = str(source.get("stratum"))
        label = int(source.get("gate_label", int(GateLabel.IGNORED)))
        key = (stratum, label)
        if key not in quotas:
            continue
        row = dict(source)
        record_id = str(row["record_id"])
        rank = _rank(seed, stratum, label, record_id)
        eligible_counts[key] += 1
        item = (-rank, record_id, row)
        heap = heaps[key]
        if len(heap) < quotas[key]:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    selected: list[dict[str, Any]] = []
    group_receipts: dict[str, Any] = {}
    for key in sorted(quotas):
        stratum, label = key
        if len(heaps[key]) != quotas[key]:
            raise RuntimeError(f"P3.4 share calibration quota unavailable: {key}")
        group = [item[2] for item in heaps[key]]
        group.sort(key=lambda row: (_rank(seed, stratum, label, str(row["record_id"])), str(row["record_id"])))
        selected.extend(group)
        label_name = "positive" if label == int(GateLabel.POSITIVE) else "negative"
        group_receipts[f"{stratum}_{label_name}"] = {
            "eligible_rows": eligible_counts[key],
            "selected_rows": len(group),
            "selection_sha256": canonical_rows_sha256(group),
        }
    selected.sort(key=lambda row: (str(row["stratum"]), int(row["gate_label"]), str(row["record_id"])))
    if len(selected) != len(P34_SHARE_STRATA) * P34_SHARE_ROWS_PER_STRATUM:
        raise RuntimeError("P3.4 share calibration row count changed")
    if len({str(row["record_id"]) for row in selected}) != len(selected):
        raise RuntimeError("P3.4 share calibration contains duplicate records")
    receipt = {
        "kind": "paper2_phase3_p34_share_calibration_selection_v1",
        "seed": seed,
        "strata": list(P34_SHARE_STRATA),
        "rows_per_stratum": P34_SHARE_ROWS_PER_STRATUM,
        "positive_rows_per_stratum": P34_SHARE_POSITIVES_PER_STRATUM,
        "negative_rows_per_stratum": P34_SHARE_NEGATIVES_PER_STRATUM,
        "selection_rule": (
            "smallest SHA256(seed:stratum:gate_label:record_id) ranks among "
            "training-eligible non-audit rows"
        ),
        "groups": group_receipts,
        "rows": len(selected),
        "selection_sha256": canonical_rows_sha256(selected),
        "source_counts": dict(Counter(str(row["source"]) for row in selected)),
    }
    return selected, receipt


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged_labels", type=Path, required=True)
    parser.add_argument("--output_rows", type=Path, required=True)
    parser.add_argument("--output_receipt", type=Path, required=True)
    parser.add_argument("--expected_source_sha256", required=True)
    args = parser.parse_args()
    observed = sha256_file(args.staged_labels)
    if observed != args.expected_source_sha256:
        raise RuntimeError("P3.4 staged-label source hash changed")
    selected, receipt = select_share_calibration_rows(read_jsonl(args.staged_labels))
    write_jsonl(args.output_rows, selected)
    receipt.update(
        {
            "source_path": str(args.staged_labels),
            "source_sha256": observed,
            "output_rows_path": str(args.output_rows),
            "output_rows_file_sha256": sha256_file(args.output_rows),
        }
    )
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
