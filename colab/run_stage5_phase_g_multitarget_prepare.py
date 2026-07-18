"""Prepare the repeated-prompt Phase G correction without allocating a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.branching_relations_task import BranchingRelationsConfig, row_manifest
from training.phase_g_multitarget_spec import (
    assert_multitarget_curriculum,
    preregistration_payload,
)
from training.phase_g_multitarget_task import build_multitarget_rows, write_jsonl


ORIGINAL_COVERAGE_TEST = (
    "outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/data/test.jsonl"
)
ORIGINAL_COVERAGE_CALIBRATION = (
    "outputs/stage5/stage5_phase_g_alpha_guided_width_20260717/data/calibration.jsonl"
)


def base_question_sha256(rows: list[dict[str, Any]]) -> str:
    by_base = {
        str(row["base_problem_id"]): str(row["question"])
        for row in rows
    }
    material = "\n".join(
        f"{base_id}\t{question}" for base_id, question in sorted(by_base.items())
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _split_payload(
    *,
    output_dir: Path,
    filename: str,
    split: str,
    rows_per_depth: int,
    n_symbols: int,
    max_depth: int,
    targets_per_prompt: int | None,
) -> dict[str, Any]:
    rows = build_multitarget_rows(
        BranchingRelationsConfig(
            rows_per_depth=rows_per_depth,
            max_depth=max_depth,
        ),
        split=split,
        rendering="verbal",
        n_symbols=n_symbols,
        targets_per_prompt=targets_per_prompt,
    )
    validation = assert_multitarget_curriculum(
        rows,
        require_all_reachable_targets=targets_per_prompt is None,
    )
    path = write_jsonl(output_dir / filename, rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "base_question_sha256": base_question_sha256(rows),
        "manifest": row_manifest(rows),
        "validation": validation,
    }


def prepare_data(
    output_dir: str | Path,
    *,
    train_rows_per_depth: int = 128,
    control_rows_per_depth: int = 64,
    n_symbols: int = 20,
    max_depth: int = 4,
    targets_per_prompt: int | None = None,
) -> dict[str, Any]:
    """Build disjoint training and posterior-control repeated-prompt splits."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _split_payload(
        output_dir=output_dir,
        filename="train.jsonl",
        split="phase_g_multitarget_train",
        rows_per_depth=train_rows_per_depth,
        n_symbols=n_symbols,
        max_depth=max_depth,
        targets_per_prompt=targets_per_prompt,
    )
    control = _split_payload(
        output_dir=output_dir,
        filename="posterior_control.jsonl",
        split="phase_g_multitarget_control",
        rows_per_depth=control_rows_per_depth,
        n_symbols=n_symbols,
        max_depth=max_depth,
        targets_per_prompt=targets_per_prompt,
    )
    if train["base_question_sha256"] == control["base_question_sha256"]:
        raise AssertionError("Multi-target train and posterior-control prompts overlap")
    preregistration = preregistration_payload()
    (output_dir / "preregistration.json").write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "kind": "stage5_phase_g_multitarget_prepare",
        "status": "prepared",
        "train": train,
        "control": control,
        "targets_per_prompt": targets_per_prompt,
        "coverage_evaluation_reuse": {
            "calibration": ORIGINAL_COVERAGE_CALIBRATION,
            "test": ORIGINAL_COVERAGE_TEST,
        },
        "sampling_policy": "base_problem_uniform",
        "preregistration": str(output_dir / "preregistration.json"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default="outputs/stage5/stage5_phase_g_multitarget_prepare_20260718/data",
    )
    parser.add_argument("--train_rows_per_depth", type=int, default=128)
    parser.add_argument("--control_rows_per_depth", type=int, default=64)
    parser.add_argument("--n_symbols", type=int, default=20)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument(
        "--targets_per_prompt",
        type=int,
        default=0,
        help="Zero requires all exact reachable targets; positive values are smoke-only caps.",
    )
    args = parser.parse_args()
    if args.targets_per_prompt < 0:
        raise ValueError("targets_per_prompt must be nonnegative")
    targets_per_prompt = args.targets_per_prompt or None
    summary = prepare_data(
        ROOT / args.output_dir,
        train_rows_per_depth=args.train_rows_per_depth,
        control_rows_per_depth=args.control_rows_per_depth,
        n_symbols=args.n_symbols,
        max_depth=args.max_depth,
        targets_per_prompt=targets_per_prompt,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
