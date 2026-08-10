"""Recover empirical DEV noise and size the Phase 3 sequential floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from eval.prepare_paper2_phase3_p31 import calibrate_false_stop
from training.paper2_phase3_p31 import estimate_empirical_paired_design


ROW_PATTERN = re.compile(r"rows_fixed_evaluation_step_(\d{5})\.pt$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _step(path: Path) -> int:
    match = ROW_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"unexpected Option B row filename: {path.name}")
    return int(match.group(1))


def load_seed_trajectory(directory: Path, *, expected_looks: int = 20) -> dict[str, Any]:
    paths = sorted(directory.glob("rows_fixed_evaluation_step_*.pt"), key=_step)
    paths = [path for path in paths if _step(path) > 0]
    if len(paths) != expected_looks:
        raise RuntimeError(
            f"empirical DEV trajectory requires {expected_looks} positive-step looks; "
            f"observed {len(paths)} in {directory}"
        )
    steps = [_step(path) for path in paths]
    expected_steps = list(range(1_000, expected_looks * 1_000 + 1, 1_000))
    if steps != expected_steps:
        raise RuntimeError(f"empirical DEV checkpoint schedule mismatch: {steps}")

    base_reference: torch.Tensor | None = None
    differences: list[np.ndarray] = []
    file_receipts: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        base = payload["base_correct_by_horizon"].bool().reshape(-1)
        augmented = payload["bridge_correct_by_horizon"].bool().reshape(-1)
        if base.shape != augmented.shape:
            raise RuntimeError(f"paired row shape mismatch at {path}")
        if base_reference is None:
            base_reference = base
        elif not torch.equal(base_reference, base):
            raise RuntimeError("frozen-base correctness changed across DEV looks")
        difference = augmented.to(torch.int8) - base.to(torch.int8)
        differences.append(difference.numpy().astype(np.float64, copy=False))
        file_receipts.append(
            {
                "path": str(path),
                "step": _step(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    matrix = np.stack(differences)
    estimate = estimate_empirical_paired_design(matrix)
    return {
        "kind": "paper2_phase3_empirical_dev_seed_trajectory_v1",
        "source": "option_b_fixed_evaluation_document_disjoint_dev_rows",
        "steps": steps,
        "paired_cells": int(matrix.shape[1]),
        "differences": matrix,
        "estimate": estimate,
        "file_receipts": file_receipts,
        "confirm_scoring_spent": False,
    }


def conservative_design(trajectories: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(trajectories)
    if len(rows) < 2:
        raise ValueError("the empirical design requires both registered seeds")
    discordance = max(
        float(row["estimate"]["paired_discordant_probability"]) for row in rows
    )
    correlation = max(
        float(row["estimate"]["adjacent_checkpoint_autocorrelation"]) for row in rows
    )
    return {
        "selection_rule": (
            "maximum seed-wise paired discordance and maximum seed-wise adjacent "
            "checkpoint autocorrelation; no cross-seed pseudo-replication"
        ),
        "paired_discordant_probability": discordance,
        "adjacent_checkpoint_autocorrelation": correlation,
        "looks": 20,
    }


def build_receipt(
    *,
    seed_directories: list[Path],
    candidate_rows: list[int],
    candidate_alphas: list[float],
    campaigns: int,
    seed: int,
) -> dict[str, Any]:
    trajectories = [load_seed_trajectory(path) for path in seed_directories]
    design = conservative_design(trajectories)
    calibration = calibrate_false_stop(
        rows=candidate_rows,
        alphas=candidate_alphas,
        looks=int(design["looks"]),
        campaigns=campaigns,
        seed=seed,
        discordant_probability=float(design["paired_discordant_probability"]),
        adjacent_correlation=float(design["adjacent_checkpoint_autocorrelation"]),
        power_drops=[0.03, 0.05],
    )
    public_trajectories = []
    for index, trajectory in enumerate(trajectories):
        public_trajectories.append(
            {
                key: value
                for key, value in trajectory.items()
                if key != "differences"
            }
            | {"seed": index}
        )
    return {
        "kind": "paper2_phase3_empirical_false_stop_calibration_receipt_v1",
        "status": "complete_empirical_dev_calibration_no_training",
        "seed_trajectories": public_trajectories,
        "binding_noise_model_candidate": design,
        "calibration": calibration,
        "assertions": {
            "both_registered_seeds_present": len(trajectories) == 2,
            "twenty_positive_step_looks_per_seed": all(
                len(row["steps"]) == 20 for row in trajectories
            ),
            "dev_only": all(row["estimate"]["source_is_dev"] for row in trajectories),
            "confirm_unscored": all(
                not row["confirm_scoring_spent"] for row in trajectories
            ),
            "power_3_and_5_points_reported": all(
                set(candidate["power"]) == {"drop_3_points", "drop_5_points"}
                for candidate in calibration["candidates"]
            ),
            "optimizer_absent": True,
            "training_steps_zero": True,
        },
        "selection_is_proposed_for_p33_lock": True,
        "p33_training_authorized": False,
        "optimizer_steps": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_row_dir", type=Path, action="append", required=True)
    parser.add_argument("--candidate_rows", type=int, nargs="+", default=[256, 512, 768, 1024])
    parser.add_argument(
        "--candidate_alphas",
        type=float,
        nargs="+",
        default=[0.0005, 0.0001, 0.00005, 0.00001],
    )
    parser.add_argument("--campaigns", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output_summary", type=Path, required=True)
    args = parser.parse_args()
    result = build_receipt(
        seed_directories=args.seed_row_dir,
        candidate_rows=args.candidate_rows,
        candidate_alphas=args.candidate_alphas,
        campaigns=args.campaigns,
        seed=args.seed,
    )
    failed = [name for name, passed in result["assertions"].items() if not passed]
    if failed:
        raise RuntimeError(f"empirical calibration assertions failed: {failed}")
    write_json(args.output_summary, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
