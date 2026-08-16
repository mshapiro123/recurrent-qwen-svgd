"""Post-hoc within-battery row-permutation sanity check for teacher CKA."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def centered_gram(features: torch.Tensor) -> np.ndarray:
    values = features.float()
    values = values - values.mean(dim=0, keepdim=True)
    return (values @ values.T).numpy()


def cka(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(left * right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def permutation_indexes(batteries: list[str], *, draws: int, seed: int) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, battery in enumerate(batteries):
        grouped[battery].append(index)
    generator = random.Random(int(seed))
    result = []
    for _ in range(int(draws)):
        indexes = list(range(len(batteries)))
        for members in grouped.values():
            shuffled = list(members)
            generator.shuffle(shuffled)
            for destination, source in zip(members, shuffled):
                indexes[destination] = source
        result.append(np.asarray(indexes, dtype=np.int64))
    return result


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student_cache", type=Path, required=True)
    parser.add_argument("--teacher_cache", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    student = torch.load(args.student_cache, map_location="cpu", weights_only=False)
    teacher = torch.load(args.teacher_cache, map_location="cpu", weights_only=False)
    panel = read_jsonl(args.panel)
    item_ids = [str(row["item_id"]) for row in panel]
    if [str(value) for value in student["item_ids"]] != item_ids:
        raise RuntimeError("student fingerprint cache order differs from panel")
    if [str(value) for value in teacher["item_ids"]] != item_ids:
        raise RuntimeError("teacher fingerprint cache order differs from panel")
    batteries = [str(row["battery"]) for row in panel]
    permutations = permutation_indexes(batteries, draws=args.draws, seed=args.seed)
    teacher_gram = centered_gram(teacher["teacher_features"][12])
    rows = []
    for checkpoint in ("p35_seed_0_ema_step_4400", "p35_seed_1_ema_step_4400"):
        core = student["core_cells"][checkpoint]
        surfaces = {
            "layer_24_cell": core[:, 43],
            "loop_4_pool": core[:, 32:40].mean(dim=1),
        }
        for name, values in surfaces.items():
            student_gram = centered_gram(values)
            observed = cka(student_gram, teacher_gram)
            null = [cka(student_gram, teacher_gram[index][:, index]) for index in permutations]
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "student_surface": name,
                    "teacher_layer": 12,
                    "matched_linear_cka": observed,
                    "within_battery_permutation_mean": float(np.mean(null)),
                    "within_battery_permutation_95ci": [
                        float(np.quantile(null, 0.025)),
                        float(np.quantile(null, 0.975)),
                    ],
                    "within_battery_permutation_maximum": float(max(null)),
                    "one_sided_p_value": (1 + sum(value >= observed for value in null))
                    / (len(null) + 1),
                }
            )
    payload = {
        "kind": "paper2_phase3_teacher_fingerprint_within_battery_null_v1",
        "status": "complete_posthoc_sanity_check",
        "draws": int(args.draws),
        "seed": int(args.seed),
        "rows": rows,
        "sources": {
            "student_cache_sha256": sha256_file(args.student_cache),
            "teacher_cache_sha256": sha256_file(args.teacher_cache),
            "panel_sha256": sha256_file(args.panel),
        },
        "scope": "post-hoc descriptive null; does not alter registered primary metrics",
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
