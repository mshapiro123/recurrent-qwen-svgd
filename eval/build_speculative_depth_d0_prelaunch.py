"""Build the read-only D0 target-policy and teacher-demand prelaunch receipts."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import read_jsonl
from eval.eval_speculative_depth_d0_floor import load_partition_cache
from eval.eval_speculative_depth_router_feasibility import summarize_teacher_demand
from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_postlock import build_training_schedule, validate_cache_summary
from training.speculative_depth_d0_spec import summarize_registered_target_schedule


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def load_floor_predictions(
    resume_dir: Path, *, expected_checkpoint_sha256: str, expected_rows: int
) -> dict[int, list[list[int]]]:
    predictions: dict[int, list[list[int]]] = {}
    for row_index in range(expected_rows):
        path = resume_dir / f"row_{row_index:06d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"D0 prelaunch is missing floor row cache: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            int(payload.get("row_index", -1)) != row_index
            or payload.get("forced_depths") != [1, 2, 3, 4, 5, 6]
            or payload.get("checkpoint_sha256") != expected_checkpoint_sha256
        ):
            raise RuntimeError(f"D0 floor row-cache lineage mismatch: {path}")
        predictions[row_index] = [
            [int(token) for token in position] for position in payload["predictions"]
        ]
    return predictions


def teacher_rows(
    cache_7b: dict[int, dict[str, Any]],
    cache_14b: dict[int, dict[str, Any]],
    predictions: dict[int, list[list[int]]],
) -> list[dict[str, Any]]:
    if set(cache_7b) != set(cache_14b) or set(cache_7b) != set(predictions):
        raise RuntimeError("D0 prelaunch teacher caches and floor predictions are not row-aligned")
    rows: list[dict[str, Any]] = []
    for row_index in sorted(cache_7b):
        seven = cache_7b[row_index]
        fourteen = cache_14b[row_index]
        if seven["row_sha256"] != fourteen["row_sha256"]:
            raise RuntimeError(f"D0 prelaunch teacher row hash mismatch at row {row_index}")
        seven_ids = seven["teacher_greedy_token_id"].tolist()
        fourteen_ids = fourteen["teacher_greedy_token_id"].tolist()
        accepted_7b = seven["accepted"].tolist()
        accepted_14b = fourteen["accepted"].tolist()
        row_predictions = predictions[row_index]
        if not (len(seven_ids) == len(fourteen_ids) == len(row_predictions)):
            raise RuntimeError(f"D0 prelaunch position count mismatch at row {row_index}")
        for local_position, predicted in enumerate(row_predictions):
            rows.append(
                {
                    "row_index": row_index,
                    "local_position": local_position,
                    "predictions": predicted,
                    "teacher_7b": int(seven_ids[local_position]),
                    "teacher_14b": int(fourteen_ids[local_position]),
                    "cache_rejected_7b": not bool(accepted_7b[local_position]),
                    "cache_rejected_14b": not bool(accepted_14b[local_position]),
                }
            )
    return rows


def cached_rejection_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["cache_rejected_7b"]]
    endorsed = sum(
        int(row["predictions"][0]) == int(row["teacher_14b"]) for row in selected
    )
    return {
        "population": "cached 7B rejection labels used by the landed floor",
        "positions": len(selected),
        "fourteen_endorses_floor_loop1": endorsed,
        "share": endorsed / len(selected) if selected else None,
        "scope": "teacher disagreement within the agreement target; not semantic correctness",
    }


def scheduled_positions(
    label_rows: list[dict[str, Any]],
    cache_rows: dict[int, dict[str, Any]],
    schedule: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    offsets = [0]
    for row in label_rows:
        offsets.append(offsets[-1] + len(row["input_ids"]) - 1)
    result: list[dict[str, Any]] = []
    for item in schedule:
        if item["kind"] != "natural":
            continue
        global_position = int(item["position_index"])
        row_index = bisect.bisect_right(offsets, global_position) - 1
        local_position = global_position - offsets[row_index]
        cache = cache_rows[row_index]
        result.append(
            {
                "accepted": bool(cache["accepted"][local_position]),
                "kl": float(cache["teacher_to_plain_drafter_kl"][local_position]),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_summary", required=True)
    parser.add_argument("--floor_summary", required=True)
    parser.add_argument("--floor_resume_dir", required=True)
    parser.add_argument("--label_train_jsonl", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--target_policy_output", required=True)
    args = parser.parse_args()

    cache_summary = read_json(args.cache_summary)
    validate_cache_summary(cache_summary)
    floor = read_json(args.floor_summary)
    if floor.get("status") != "complete" or floor.get("training_started") is not False:
        raise RuntimeError("D0 prelaunch requires the completed pre-training floor")

    calibration_7b = load_partition_cache(cache_summary, "teacher_7b", "calibration")
    calibration_14b = load_partition_cache(cache_summary, "teacher_14b", "calibration")
    predictions = load_floor_predictions(
        Path(args.floor_resume_dir),
        expected_checkpoint_sha256=str(floor["checkpoint_sha256"]),
        expected_rows=len(calibration_7b),
    )
    rows = teacher_rows(calibration_7b, calibration_14b, predictions)
    demand = summarize_teacher_demand(rows)
    demand["cached_7b_rejection_overlap"] = cached_rejection_overlap(rows)

    label_rows = read_jsonl(args.label_train_jsonl)
    label_cache = load_partition_cache(cache_summary, "teacher_7b", "label_train")
    natural_positions = sum(len(row["input_ids"]) - 1 for row in label_rows)
    schedule = build_training_schedule(total_steps=4000, natural_positions=natural_positions, seed=0)
    stored_schedule_path = Path(cache_summary["cache_root"]) / "registered_training_schedule.json"
    stored_schedule = read_json(stored_schedule_path)
    if stored_schedule["schedule"] != schedule:
        raise RuntimeError("D0 prelaunch schedule differs from the teacher-pass schedule")
    target_policy = summarize_registered_target_schedule(
        floor=floor,
        scheduled_positions=scheduled_positions(label_rows, label_cache, schedule),
    )
    target_policy.update(
        {
            "floor_summary_sha256": sha256_file(args.floor_summary),
            "label_train_sha256": sha256_file(args.label_train_jsonl),
            "registered_training_schedule_sha256": sha256_file(stored_schedule_path),
            "evaluation_partition_touched": False,
            "optimizer_steps": 0,
        }
    )
    write_json(args.target_policy_output, target_policy)

    summary = {
        "kind": "paper2_d0_prelaunch_receipts",
        "status": "complete",
        "floor_summary_sha256": sha256_file(args.floor_summary),
        "floor_checkpoint_sha256": floor["checkpoint_sha256"],
        "teacher_demand": demand,
        "target_policy_receipt": str(args.target_policy_output),
        "target_policy_receipt_sha256": sha256_file(args.target_policy_output),
        "measurement_amendment_timing": (
            "clarified after the floor landed and before any D0-trained model existed; "
            "the 7B floor distribution was visible and the clean 14B own-rejection "
            "distribution had not been computed"
        ),
        "trained_layer_caveat": (
            "binary depth-2 targets may compress both trained teacher-demand distributions "
            "toward depth 2; the floor layer is less confounded"
        ),
        "evaluation_partition_touched": False,
        "optimizer_steps": 0,
        "model_loaded": False,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
