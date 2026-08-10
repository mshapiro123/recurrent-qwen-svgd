"""Seal P3.1, merge frozen-model scores, and emit currency receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from training.paper2_phase3_p31_completion import (
    build_sentinel_panel,
    reference_score_table,
    seal_confirm_membership,
    sha256_file,
    verified_stratum_counts,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(path)


def merge_scores(
    rows: list[dict[str, Any]],
    *,
    base_scores: list[dict[str, Any]],
    teacher_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = {str(row["item_id"]): row for row in base_scores}
    teacher = {str(row["item_id"]): row for row in teacher_scores}
    eligible = [row for row in rows if row["partition"] in {"verified_train", "dev"}]
    expected = {str(row["item_id"]) for row in eligible}
    if set(base) != expected or set(teacher) != expected:
        raise RuntimeError(
            "P3.1 score coverage mismatch "
            f"base_missing={len(expected - set(base))} teacher_missing={len(expected - set(teacher))}"
        )
    merged = []
    for row in eligible:
        item_id = str(row["item_id"])
        if base[item_id]["partition"] != row["partition"] or teacher[item_id]["partition"] != row["partition"]:
            raise RuntimeError(f"P3.1 score partition changed for {item_id}")
        merged.append(
            {
                "battery": row["battery"],
                "battery_role": row["battery_role"],
                "partition": row["partition"],
                "document_id": row["document_id"],
                "item_id": item_id,
                "content_sha256": row["content_sha256"],
                "base_correct": bool(base[item_id]["correct"]),
                "teacher_14b_correct": bool(teacher[item_id]["correct"]),
            }
        )
    return merged


def build_completion(
    *,
    rows_path: Path,
    source_summary_path: Path,
    source_manifest_path: Path,
    base_scores_path: Path,
    teacher_scores_path: Path,
    model_score_receipts_path: Path,
    private_dir: Path,
    receipt_dir: Path,
) -> dict[str, Any]:
    rows = read_jsonl(rows_path)
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    ledger = source_summary["ledger"]
    model_score_receipts = json.loads(
        model_score_receipts_path.read_text(encoding="utf-8")
    )
    seals = seal_confirm_membership(
        ledger,
        output_dir=receipt_dir / "confirm_seals",
        source_rows_sha256=sha256_file(rows_path),
        source_manifest_sha256=sha256_file(source_manifest_path),
    )
    merged = merge_scores(
        rows,
        base_scores=read_jsonl(base_scores_path),
        teacher_scores=read_jsonl(teacher_scores_path),
    )
    merged_path = private_dir / "p31_merged_dev_verified_scores.jsonl"
    write_jsonl(merged_path, merged)
    references = reference_score_table([row for row in merged if row["partition"] == "dev"])
    verified = verified_stratum_counts(
        [row for row in merged if row["partition"] == "verified_train"]
    )
    panel, sentinel = build_sentinel_panel(rows, scored_rows=merged)
    panel_path = private_dir / "p31_sentinel_panel.jsonl"
    write_jsonl(panel_path, panel)
    sentinel.update({"path": str(panel_path), "file_sha256": sha256_file(panel_path)})
    write_json(receipt_dir / "reference_scores.json", references)
    write_json(receipt_dir / "verified_stratum.json", verified)
    write_json(receipt_dir / "sentinel_panel.json", sentinel)
    result = {
        "kind": "paper2_phase3_p31_currency_completion_receipt_v1",
        "status": "complete_dev_scored_confirm_sealed_unscored",
        "confirm_seals": seals,
        "model_score_receipts": model_score_receipts,
        "reference_scores": references,
        "verified_stratum": verified,
        "sentinel_panel": sentinel,
        "private_merged_scores": {
            "path": str(merged_path),
            "sha256": sha256_file(merged_path),
            "rows": len(merged),
        },
        "assertions": {
            "confirm_sealed_before_scoring": seals["status"] == "sealed_before_model_scoring",
            "model_receipts_confirm_unscored": (
                not model_score_receipts["confirm_scoring_spent"]
                and all(row["confirm_rows"] == 0 for row in model_score_receipts["models"])
            ),
            "model_receipts_share_confirm_seal": all(
                row["confirm_seal_sha256"]
                == model_score_receipts["confirm_seal_sha256"]
                for row in model_score_receipts["models"]
            ),
            "confirm_unscored": not references["confirm_scoring_spent"],
            "floor_excluded_from_headline": len(
                references["floor_batteries_excluded_from_headline_numerator"]
            ) == 3,
            "verified_labels_complete": verified["counts_all"]["total"] > 0,
            "sentinel_exactly_2048": sentinel["rows"] == 2_048,
            "sentinel_confirm_free": sentinel["confirm_rows"] == 0,
            "optimizer_absent": True,
            "training_steps_zero": True,
            "p33_training_unauthorized": True,
        },
        "p33_training_authorized": False,
        "optimizer_steps": 0,
    }
    failed = [name for name, passed in result["assertions"].items() if not passed]
    if failed:
        raise RuntimeError(f"P3.1 completion assertions failed: {failed}")
    write_json(receipt_dir / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows_jsonl", type=Path, required=True)
    parser.add_argument("--source_summary", type=Path, required=True)
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--base_scores", type=Path, required=True)
    parser.add_argument("--teacher_scores", type=Path, required=True)
    parser.add_argument("--model_score_receipts", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--receipt_dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_completion(
        rows_path=args.rows_jsonl,
        source_summary_path=args.source_summary,
        source_manifest_path=args.source_manifest,
        base_scores_path=args.base_scores,
        teacher_scores_path=args.teacher_scores,
        model_score_receipts_path=args.model_score_receipts,
        private_dir=args.private_dir,
        receipt_dir=args.receipt_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
