"""Audit archived Phase A dense continuations with a first-response reader."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.dense_response_reader import extract_first_completed_symbol  # noqa: E402


DEFAULT_SOURCE = ROOT / "outputs/stage5/stage5_synthetic_depth_frozen_eval_v2_depth14/data/test_chain_mcq.jsonl"
DEFAULT_COMPARISON = ROOT / "outputs/stage5/stage5_phase_a_checkpoint_comparison_20260713"
DEFAULT_RECURRENT_ROWS = (
    ROOT
    / "outputs/stage5/stage5_same_reader_final_symbol_20260707_021010/eval/same_reader_final_rows.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/stage5/stage5_phase_a_dense_reader_audit_20260722"
ARM_SPECS = {
    "B_step4000": "direct",
    "C_step4000": "serialized_orbit_scratchpad",
    "D_step4000": "direct",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidates_for_row(row: dict[str, Any]) -> list[str]:
    mapping = row.get("mapping") or {}
    candidates = sorted(
        {str(value).strip().upper() for value in mapping}
        | {str(value).strip().upper() for value in mapping.values()}
    )
    if not candidates:
        candidates = sorted({str(value).strip().upper() for value in row.get("orbit") or []})
    return candidates


def exact_sign_test_two_sided(helped: int, hurt: int) -> float:
    non_ties = helped + hurt
    if non_ties == 0:
        return 1.0
    tail = sum(math.comb(non_ties, k) for k in range(min(helped, hurt) + 1)) / (2**non_ties)
    return min(1.0, 2.0 * tail)


def paired_against_recurrent(
    corrected: dict[str, bool], recurrent: dict[str, bool]
) -> dict[str, Any]:
    if set(corrected) != set(recurrent):
        raise RuntimeError("Dense and recurrent row IDs differ")
    helped = sum(recurrent[row_id] and not corrected[row_id] for row_id in corrected)
    hurt = sum(corrected[row_id] and not recurrent[row_id] for row_id in corrected)
    return {
        "recurrent_helped": helped,
        "recurrent_hurt": hurt,
        "recurrent_net_correct": helped - hurt,
        "two_sided_exact_sign_p": exact_sign_test_two_sided(helped, hurt),
    }


def audit_arm(
    rows: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
    recurrent: dict[str, bool],
    surface: str,
) -> dict[str, Any]:
    by_depth: dict[str, dict[str, int | float]] = {}
    corrected_by_id: dict[str, bool] = {}
    examples: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row["id"])
        source = source_by_id[row_id]
        prediction = extract_first_completed_symbol(
            str(row.get("continuation") or ""),
            candidates_for_row(source),
        )
        target = str(row["target"]).strip().upper()
        corrected = prediction == target
        corrected_by_id[row_id] = corrected
        depth = str(int(row["depth"]))
        bucket = by_depth.setdefault(
            depth,
            {
                "total": 0,
                "registered_correct": 0,
                "corrected_correct": 0,
                "registered_parse_failures": 0,
                "corrected_parse_failures": 0,
                "prediction_changed": 0,
            },
        )
        bucket["total"] += 1
        bucket["registered_correct"] += int(bool(row.get("correct")))
        bucket["corrected_correct"] += int(corrected)
        bucket["registered_parse_failures"] += int(row.get("prediction") is None)
        bucket["corrected_parse_failures"] += int(prediction is None)
        changed = prediction != row.get("prediction")
        bucket["prediction_changed"] += int(changed)
        if changed and len(examples) < 12:
            examples.append(
                {
                    "id": row_id,
                    "depth": int(row["depth"]),
                    "target": target,
                    "registered_prediction": row.get("prediction"),
                    "corrected_prediction": prediction,
                    "continuation_prefix": str(row.get("continuation") or "")[:240],
                }
            )

    for bucket in by_depth.values():
        total = int(bucket["total"])
        bucket["registered_accuracy"] = int(bucket["registered_correct"]) / total
        bucket["corrected_accuracy"] = int(bucket["corrected_correct"]) / total
        bucket["correct_delta"] = int(bucket["corrected_correct"]) - int(bucket["registered_correct"])

    registered_total = sum(int(bucket["registered_correct"]) for bucket in by_depth.values())
    corrected_total = sum(int(bucket["corrected_correct"]) for bucket in by_depth.values())
    return {
        "surface": surface,
        "rows": len(rows),
        "registered_correct": registered_total,
        "corrected_correct": corrected_total,
        "correct_delta": corrected_total - registered_total,
        "corrected_parse_failures": sum(
            int(bucket["corrected_parse_failures"]) for bucket in by_depth.values()
        ),
        "by_depth": dict(sorted(by_depth.items(), key=lambda item: int(item[0]))),
        "paired_against_full_block_recurrent": paired_against_recurrent(corrected_by_id, recurrent),
        "changed_examples": examples,
    }


def build_audit(
    source_path: Path,
    comparison_dir: Path,
    recurrent_path: Path,
) -> dict[str, Any]:
    source_rows = read_jsonl(source_path)
    source_by_id = {str(row.get("id") or row.get("instance_id")): row for row in source_rows}
    recurrent_rows = read_jsonl(recurrent_path)
    recurrent = {str(row["id"]): bool(row["same_reader_final_hit"]) for row in recurrent_rows}
    arms: dict[str, Any] = {}
    source_receipts: dict[str, Any] = {
        "frozen_rows": source_path.relative_to(ROOT).as_posix(),
        "frozen_rows_sha256": sha256_file(source_path),
        "full_block_recurrent_rows": recurrent_path.relative_to(ROOT).as_posix(),
        "full_block_recurrent_rows_sha256": sha256_file(recurrent_path),
    }
    for label, surface in ARM_SPECS.items():
        rows_path = comparison_dir / "eval" / label / "rows.jsonl.gz"
        rows = read_gzip_jsonl(rows_path)
        if {str(row["id"]) for row in rows} != set(source_by_id):
            raise RuntimeError(f"{label} row IDs do not match the frozen source")
        arms[label] = audit_arm(rows, source_by_id, recurrent, surface)
        source_receipts[label] = {
            "path": rows_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(rows_path),
        }
    return {
        "kind": "stage5_phase_a_dense_first_response_reader_audit",
        "status": "corrected_reader_required",
        "registered_reader": "last_answer_marker_else_first_valid_full_symbol",
        "corrected_reader": "leading_symbol_else_first_answer_else_first_valid_full_symbol",
        "finding": (
            "The registered reader can overwrite a completed response with later untrained continuation. "
            "Dense accuracy and figures must use the corrected first-response readout."
        ),
        "sources": source_receipts,
        "arms": arms,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase A Dense First-Response Reader Audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Finding: {payload['finding']}",
        "",
        "| Arm | Registered | Corrected | Delta | D1 | D2 | D4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, arm in payload["arms"].items():
        by_depth = arm["by_depth"]
        lines.append(
            f"| {label} | {arm['registered_correct']} | {arm['corrected_correct']} | "
            f"{arm['correct_delta']:+d} | {by_depth['1']['corrected_correct']}/128 | "
            f"{by_depth['2']['corrected_correct']}/128 | {by_depth['4']['corrected_correct']}/128 |"
        )
    lines.extend(
        [
            "",
            "The correction is evaluation-only. It does not alter checkpoints, frozen rows, or model outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--recurrent-rows", type=Path, default=DEFAULT_RECURRENT_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    payload = build_audit(args.source, args.comparison_dir, args.recurrent_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
