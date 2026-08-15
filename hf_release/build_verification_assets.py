"""Build frozen, receipt-linked round-trip evaluation assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "hf_release" / "verification_assets"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def depth_rows(path: Path, *, depths: set[int], per_depth: int | None = None) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {depth: [] for depth in depths}
    for row in read_jsonl(path):
        depth = int(row["depth"])
        if depth in grouped and (per_depth is None or len(grouped[depth]) < per_depth):
            grouped[depth].append(row)
    rows = [row for depth in sorted(grouped) for row in grouped[depth]]
    expected = None if per_depth is None else per_depth
    for depth, values in grouped.items():
        if expected is not None and len(values) != expected:
            raise RuntimeError(f"Depth {depth} has {len(values)} rows, expected {expected}")
    return rows


def main() -> int:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    full_source = ROOT / "outputs/stage5/stage5_synthetic_depth_frozen_eval_v3_depth22_n24/data/test_chain_mcq.jsonl"
    adapter_source = ROOT / "outputs/stage5/stage5_synthetic_depth_frozen_eval_v2_depth14/data/test_chain_mcq.jsonl"
    natural_source = ROOT / "outputs/stage5/stage5_natural_surface_receipts_20260709_210151/data/robust_relay_fronted_d1_12.jsonl"
    natural_receipt_rows = read_jsonl(
        ROOT
        / "outputs/stage5/stage5_natural_surface_followups_2_3_20260710/active/step_2000/"
        "step_2000_robust_relay_fronted_d1_12_active_rows_sample.jsonl"
    )

    full_rows = depth_rows(full_source, depths={1, 2, 3, 4})
    adapter_rows = depth_rows(adapter_source, depths={1, 2, 3, 4})
    # The durable sample receipt preserves the first depth-1 rows. Use eight of
    # those exact rows as the small natural-surface release gate.
    natural_candidates = depth_rows(natural_source, depths={1}, per_depth=8)
    natural_ids = {str(row["id"]) for row in natural_candidates}
    natural_diagonal = [
        row
        for row in natural_receipt_rows
        if str(row["id"]) in natural_ids
        and bool(row["active_cell"])
        and int(row["forced_loop_count"]) == int(row["depth"])
    ]
    if len(natural_diagonal) != len(natural_candidates):
        raise RuntimeError("Natural verification slice is not fully covered by the frozen receipt sample")

    specs = {
        "recurrent-qwen2.5-0.5b-full-block": {
            "rows": full_rows,
            "expected_correct_by_depth": {"1": 128, "2": 127, "3": 127, "4": 128},
            "source_data": str(full_source.relative_to(ROOT)).replace("\\", "/"),
            "source_receipt": "outputs/stage5/stage5_n24_support12_rung_20260707_140139/eval/frozen_depth22_step_6000/active_summary.json",
            "identity_check": False,
        },
        "recurrent-qwen2.5-0.5b-r16-adapter": {
            "rows": adapter_rows,
            "expected_correct_by_depth": {"1": 128, "2": 128, "3": 127, "4": 128},
            "source_data": str(adapter_source.relative_to(ROOT)).replace("\\", "/"),
            "source_receipt": "outputs/stage5/stage5_adapter_budget_arm_e_20260718/eval/final_phase_a_1792/summary.json",
            "identity_check": False,
        },
        "recurrent-qwen2.5-0.5b-natural-keeper": {
            "rows": natural_candidates,
            "expected_correct_by_depth": {
                "1": sum(int(bool(row["hit"])) for row in natural_diagonal)
            },
            "source_data": str(natural_source.relative_to(ROOT)).replace("\\", "/"),
            "source_receipt": (
                "outputs/stage5/stage5_natural_surface_followups_2_3_20260710/active/step_2000/"
                "step_2000_robust_relay_fronted_d1_12_active_rows_sample.jsonl"
            ),
            "identity_check": True,
        },
    }
    summary: dict[str, Any] = {"schema_version": 1, "repos": {}}
    for repo_name, spec in specs.items():
        rows = spec.pop("rows")
        data_path = DESTINATION / f"{repo_name}.jsonl"
        data_sha = write_jsonl(data_path, rows)
        summary["repos"][repo_name] = {
            **spec,
            "verification_data": str(data_path.relative_to(ROOT)).replace("\\", "/"),
            "verification_data_sha256": data_sha,
            "row_count": len(rows),
            "rows_by_depth": {
                str(depth): sum(int(row["depth"]) == depth for row in rows)
                for depth in sorted({int(row["depth"]) for row in rows})
            },
        }
    (DESTINATION / "verification_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
