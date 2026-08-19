"""Build the score-blind, deterministic Stage 2B-D DEV-2 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from training.paper2_stage2b_depth import paired_sign_test_power


DEV2_ROWS = 2_048
DEV2_SEED = 20_260_818


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    available = sum(counts.values())
    exact = {key: total * value / available for key, value in counts.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    for key, _ in sorted(
        exact.items(), key=lambda item: (item[1] - allocation[item[0]], item[0]), reverse=True
    )[: total - sum(allocation.values())]:
        allocation[key] += 1
    return allocation


def merge_reference_scores(
    reference_rows: Iterable[dict[str, Any]],
    score_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference = list(reference_rows)
    scores = list(score_rows)
    by_id = {str(row["item_id"]): row for row in scores}
    if len(by_id) != len(scores):
        raise RuntimeError("P3.1 merged score table contains duplicate item IDs")
    expected = {
        str(row["item_id"])
        for row in reference
        if row.get("partition") != "confirm"
    }
    if set(by_id) != expected:
        raise RuntimeError(
            "P3.1 merged score coverage mismatch "
            f"missing={len(expected - set(by_id))} extra={len(set(by_id) - expected)}"
        )
    invariant_fields = (
        "battery",
        "battery_role",
        "document_id",
        "content_sha256",
        "partition",
    )
    merged = []
    for source in reference:
        row = dict(source)
        if row.get("partition") != "confirm":
            score = by_id[str(row["item_id"])]
            changed = [field for field in invariant_fields if score[field] != row[field]]
            if changed:
                raise RuntimeError(
                    f"P3.1 merged score identity changed for {row['item_id']}: {changed}"
                )
            row["base_correct"] = bool(score["base_correct"])
            row["teacher_14b_correct"] = bool(score["teacher_14b_correct"])
        merged.append(row)
    return merged


def build_dev2(
    reference_rows: Iterable[dict[str, Any]],
    dev1_rows: Iterable[dict[str, Any]],
    *,
    seed: int = DEV2_SEED,
    size: int = DEV2_ROWS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = list(reference_rows)
    required = {
        "item_id",
        "document_id",
        "battery",
        "battery_role",
        "partition",
        "content_sha256",
    }
    missing_fields = sorted(
        {
            field
            for row in reference
            for field in required - set(row)
        }
    )
    if missing_fields:
        raise RuntimeError(
            "DEV-2 construction requires the P3.1 partition table; "
            f"missing fields: {missing_fields}"
        )
    missing_scores = sorted(
        str(row["item_id"])
        for row in reference
        if row.get("partition") != "confirm"
        and not {"base_correct", "teacher_14b_correct"}.issubset(row)
    )
    if missing_scores:
        raise RuntimeError(
            "DEV-2 construction requires joined P3.1 DEV/verified scores; "
            f"missing rows: {len(missing_scores)}"
        )
    dev1_ids = {str(row["item_id"]) for row in dev1_rows}
    if len(dev1_ids) != 1_024:
        raise RuntimeError(f"expected 1,024 DEV-1 IDs, observed {len(dev1_ids)}")
    by_id = {str(row["item_id"]): row for row in reference}
    if len(by_id) != len(reference):
        raise RuntimeError("reference table contains duplicate item IDs")
    missing_dev1 = dev1_ids - set(by_id)
    if missing_dev1:
        raise RuntimeError(f"DEV-1 contains {len(missing_dev1)} IDs absent from reference")
    dev1_confirm = [
        item_id for item_id in dev1_ids if by_id[item_id].get("partition") == "confirm"
    ]
    if dev1_confirm:
        raise RuntimeError(
            f"DEV-1 overlaps sealed CONFIRM on {len(dev1_confirm)} rows"
        )
    confirm_rows = [row for row in reference if row.get("partition") == "confirm"]
    reserve = [
        row
        for item_id, row in by_id.items()
        if item_id not in dev1_ids and row.get("partition") != "confirm"
    ]
    if len(reserve) != 9_207:
        raise RuntimeError(f"expected 9,207 rows outside DEV-1, observed {len(reserve)}")
    if len(confirm_rows) != 1_502:
        raise RuntimeError(f"expected 1,502 sealed CONFIRM rows, observed {len(confirm_rows)}")

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reserve:
        key = f"{row['battery']}::{row['partition']}"
        strata[key].append(row)
    allocation = _largest_remainder({key: len(rows) for key, rows in strata.items()}, size)
    selected = []
    excluded = []
    for index, key in enumerate(sorted(strata)):
        rows = sorted(strata[key], key=lambda row: str(row["item_id"]))
        random.Random(seed + index).shuffle(rows)
        cut = allocation[key]
        selected.extend(rows[:cut])
        excluded.extend(rows[cut:])
    selected.sort(key=lambda row: str(row["item_id"]))
    if len(selected) != size or len({row["item_id"] for row in selected}) != size:
        raise RuntimeError("DEV-2 selection is not a unique 2,048-row set")

    manifest_rows = [
        {
            "item_id": row["item_id"],
            "document_id": row["document_id"],
            "battery": row["battery"],
            "battery_role": row["battery_role"],
            "source_partition": row["partition"],
            "content_sha256": row["content_sha256"],
            "base_correct_reference_only": bool(row["base_correct"]),
            "teacher_14b_correct_reference_only": bool(row["teacher_14b_correct"]),
        }
        for row in selected
    ]
    receipt = {
        "kind": "paper2_stage2b_dev2_manifest_receipt_v1",
        "status": "score_blind_manifest_complete",
        "seed": seed,
        "rows": size,
        "candidate_rows": len(reserve),
        "dev1_rows_excluded": len(dev1_ids),
        "confirm_contact": False,
        "eval_e_contact": False,
        "model_loaded": False,
        "model_scores_computed": False,
        "optimizer_constructed": False,
        "strata": dict(sorted(Counter(f"{r['battery']}::{r['source_partition']}" for r in manifest_rows).items())),
        "battery_counts": dict(sorted(Counter(r["battery"] for r in manifest_rows).items())),
        "battery_role_counts": dict(sorted(Counter(r["battery_role"] for r in manifest_rows).items())),
        "power_at_plus_30": [
            paired_sign_test_power(
                rows=size, net_improvement=30, discordance_rate=rate
            )
            for rate in (0.10, 0.20, 0.30)
        ],
        "excluded_rows": len(excluded),
    }
    return manifest_rows, receipt


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-rows", required=True)
    parser.add_argument("--reference-scores", required=True)
    parser.add_argument("--dev1-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    reference = merge_reference_scores(
        read_jsonl(args.reference_rows), read_jsonl(args.reference_scores)
    )
    rows, receipt = build_dev2(reference, read_jsonl(args.dev1_rows))
    receipt["manifest_sha256"] = write_jsonl(output / "dev2_manifest.jsonl", rows)
    receipt_path = output / "dev2_manifest_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
