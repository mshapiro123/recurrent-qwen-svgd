"""Build the Stage 2B-S stopped-preflight receipt and diagnostic figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


EXPECTED = {0: [162, 10, 2, 2], 1: [162, 9, 5, 2]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def generation_dir(root: Path, seed: int) -> Path:
    matches = list((root / f"private/seed_{seed}/preflight").glob("*/generation"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one seed-{seed} generation directory, found {matches}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()

    seeds: dict[str, Any] = {}
    for seed in (0, 1):
        directory = generation_dir(args.artifact_root, seed)
        cells = []
        completion_times = []
        for k in range(1, 5):
            path = directory / f"native_interleaved__k{k}__gamma_0p05.jsonl"
            rows = read_jsonl(path)
            cells.append(
                {
                    "k": k,
                    "rows": len(rows),
                    "correct": sum(bool(row["augmented_correct"]) for row in rows),
                    "sha256": sha256_file(path),
                    "completion_time_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "correct_item_ids": [
                        row["item_id"] for row in rows if row["augmented_correct"]
                    ],
                }
            )
            completion_times.append(path.stat().st_mtime)
        seeds[str(seed)] = {
            "expected": EXPECTED[seed],
            "observed": [cell["correct"] for cell in cells],
            "cells": cells,
            "incremental_minutes_after_k1": [
                round((completion_times[index] - completion_times[index - 1]) / 60.0, 3)
                for index in range(1, 4)
            ],
            "preflight_receipt_sha256": sha256_file(
                args.artifact_root / f"receipts/seed_{seed}/preflight.json"
            ),
            "checkpoint_provenance_sha256": sha256_file(
                args.artifact_root / f"receipts/seed_{seed}/checkpoint_provenance.json"
            ),
        }

    variant_files = [
        path
        for path in args.artifact_root.glob("private/seed_*/preflight/*/generation/*.jsonl")
        if not path.name.startswith("native_interleaved__")
    ]
    result = {
        "kind": "paper2_stage2bs_depth_preflight_stop_analysis_v1",
        "status": "STOP_EXPECTATION_GAP_AND_COST_ESCALATION",
        "seeds": seeds,
        "variant_cells_scored": len(variant_files),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
        "interpretation": {
            "seed_0": "exact reproduction pass",
            "seed_1": (
                "first complete native matched-graph curve; locked K4=2 expectation lacked "
                "a prior canonical seed-1 native receipt"
            ),
            "scientific_scale": (
                "both seeds show the same severe K1-to-K2/K3/K4 collapse; the one-row "
                "seed-1 K4 difference is immaterial to the registered +20-row additivity floor"
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.6))
    colors = {0: "#13795b", 1: "#d05a3a"}
    for seed in (0, 1):
        values = seeds[str(seed)]["observed"]
        left.plot(range(1, 5), values, marker="o", linewidth=2.2, color=colors[seed], label=f"Seed {seed}")
    left.set_yscale("log")
    left.set_xticks(range(1, 5))
    left.set_xlabel("Native recurrent depth K")
    left.set_ylabel("Correct rows of 461 (log scale)")
    left.set_title("Matched-graph native preflight")
    left.grid(axis="y", alpha=0.25)
    left.legend(frameon=False)

    labels = ["K2", "K3", "K4"]
    x = range(len(labels))
    width = 0.36
    for offset, seed in ((-width / 2, 0), (width / 2, 1)):
        right.bar(
            [index + offset for index in x],
            seeds[str(seed)]["incremental_minutes_after_k1"],
            width=width,
            color=colors[seed],
            label=f"Seed {seed}",
        )
    right.set_xticks(list(x), labels)
    right.set_ylabel("Incremental cell runtime (minutes)")
    right.set_title("Observed generative-cell cost")
    right.grid(axis="y", alpha=0.25)
    right.legend(frameon=False)
    figure.suptitle("Stage 2B-S depth study stopped before variants", fontsize=13)
    figure.tight_layout()
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_svg, bbox_inches="tight")
    figure.savefig(args.output_png, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
