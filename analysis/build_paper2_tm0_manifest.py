"""Freeze the score-blind TM-0 base plus extension panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from training.paper2_tm0 import (
    atomic_json,
    build_tm0_panel,
    load_lock,
    read_jsonl,
    sha256_file,
    write_jsonl,
)


def _cka_calibration_rows(
    panel: list[dict[str, object]], *, rows: int, seed: int
) -> list[dict[str, object]]:
    """Freeze a battery-stratified CKA subset before any model forward."""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in panel:
        grouped[str(row["battery"])].append(row)
    total = len(panel)
    quotas = {
        battery: max(1, int(rows * len(values) // total))
        for battery, values in grouped.items()
    }
    remainder = rows - sum(quotas.values())
    fractional = sorted(
        grouped,
        key=lambda battery: (
            -(rows * len(grouped[battery]) / total - quotas[battery]),
            battery,
        ),
    )
    if remainder >= 0:
        for battery in fractional[:remainder]:
            quotas[battery] += 1
    else:
        for battery in reversed(fractional):
            while remainder < 0 and quotas[battery] > 1:
                quotas[battery] -= 1
                remainder += 1
    selected: list[dict[str, object]] = []
    for battery, values in sorted(grouped.items()):
        ordered = sorted(
            values,
            key=lambda row: hashlib.sha256(
                f"{seed}:tm0_cka:{battery}:{row['item_id']}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ordered[: quotas[battery]])
    if len(selected) != rows:
        raise RuntimeError("TM-0 CKA calibration subset cardinality changed")
    return sorted(selected, key=lambda row: str(row["item_id"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_rows", type=Path, required=True)
    parser.add_argument("--dev2_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hermetic_screen_result", type=Path)
    args = parser.parse_args()
    lock = load_lock()
    if sha256_file(args.source_rows) != lock["panels"]["source_rows_sha256"]:
        raise RuntimeError("TM-0 source-row hash mismatch")
    if sha256_file(args.dev2_manifest) != lock["panels"]["dev2_manifest_sha256"]:
        raise RuntimeError("TM-0 DEV-2 manifest hash mismatch")
    panel, extension, receipt = build_tm0_panel(
        read_jsonl(args.source_rows),
        read_jsonl(args.dev2_manifest),
        extension_size=int(lock["panels"]["extension_rows"]),
        seed=int(lock["panels"]["extension_seed"]),
        threshold=float(lock["panels"]["near_duplicate"]["threshold"]),
    )
    screen = None
    if args.hermetic_screen_result is not None:
        screen = json.loads(args.hermetic_screen_result.read_text(encoding="utf-8"))
        if screen.get("status") != "PASS" or screen.get("eval_e_scored") is not False:
            raise RuntimeError("TM-0 hermetic EVAL-E screen did not pass score-blind")
        dropped = set(map(str, screen["dropped_panel_row_ids"]))
        panel = [row for row in panel if str(row["item_id"]) not in dropped]
        extension = [row for row in extension if str(row["item_id"]) not in dropped]
        receipt["kind"] = "paper2_tm0_panel_freeze_v2" if dropped else receipt["kind"]
        receipt["eval_e_screen"] = {
            "result_sha256": sha256_file(args.hermetic_screen_result),
            "sealed_index_sha256": screen["sealed_index_sha256"],
            "dropped_row_count": len(dropped),
            "no_backfill": True,
            "eval_e_scored": False,
            "membership_materialized": screen["membership_materialized"],
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "tm0_panel.jsonl"
    extension_path = args.output_dir / "tm0_extension_manifest.jsonl"
    rejected_path = args.output_dir / "tm0_rejected_candidates.jsonl"
    cka_paths = [
        args.output_dir / "tm0_cka_calibration_manifest_a.jsonl",
        args.output_dir / "tm0_cka_calibration_manifest_b.jsonl",
    ]
    rejected = receipt.pop("rejected")
    cka_rows: list[list[dict[str, object]]] = []
    excluded_ids: set[str] = set()
    for seed in lock["tm1"]["cka_calibration_seeds"]:
        available = [row for row in panel if str(row["item_id"]) not in excluded_ids]
        selected = _cka_calibration_rows(
            available,
            rows=int(lock["tm1"]["cka_calibration_rows"]),
            seed=int(seed),
        )
        cka_rows.append(selected)
        excluded_ids.update(str(row["item_id"]) for row in selected)
    receipt["authority"] = lock["authority"]
    receipt["source_rows"] = {
        "bytes": args.source_rows.stat().st_size,
        "sha256": sha256_file(args.source_rows),
    }
    receipt["dev2_manifest"] = {
        "bytes": args.dev2_manifest.stat().st_size,
        "sha256": sha256_file(args.dev2_manifest),
    }
    receipt["panel"] = {"rows": len(panel), "sha256": write_jsonl(panel_path, panel)}
    receipt["extension_manifest"] = {
        "rows": len(extension),
        "sha256": write_jsonl(extension_path, extension),
    }
    receipt["rejected_candidates"] = {
        "rows": len(rejected),
        "sha256": write_jsonl(rejected_path, rejected),
    }
    receipt["cka_calibration_manifests"] = []
    for path, selected, seed in zip(
        cka_paths, cka_rows, lock["tm1"]["cka_calibration_seeds"]
    ):
        receipt["cka_calibration_manifests"].append(
            {
                "seed": seed,
                "rows": len(selected),
                "battery_counts": {
                    battery: sum(str(row["battery"]) == battery for row in selected)
                    for battery in sorted({str(row["battery"]) for row in selected})
                },
                "sha256": write_jsonl(path, selected),
            }
        )
    receipt["cka_calibration_disjoint"] = not (
        {str(row["item_id"]) for row in cka_rows[0]}
        & {str(row["item_id"]) for row in cka_rows[1]}
    )
    receipt_path = args.output_dir / "tm0_panel_freeze_receipt.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
