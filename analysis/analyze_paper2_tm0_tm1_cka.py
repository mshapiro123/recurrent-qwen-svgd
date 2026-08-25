"""Compute the ratified dual-subset TM-1 debiased CKA calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from training.paper2_tm0 import atomic_json, read_jsonl, sha256_file


def cache_shards(model_dir: Path) -> Iterable[dict[str, Any]]:
    index_path = next(model_dir.glob("*_cache_index.json"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("dry_run") or int(index.get("rows", -1)) != 6144:
        raise RuntimeError(f"TM-1 requires a complete full-panel cache: {index_path}")
    for receipt in index["shards"]:
        path = model_dir / "shards" / receipt["path"]
        if sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"TM-1 cache shard SHA mismatch: {path}")
        yield torch.load(path, map_location="cpu", weights_only=False)


def selected_cache(
    model_dir: Path, item_ids: list[str], *, key: str
) -> torch.Tensor:
    wanted = set(item_ids)
    found: dict[str, torch.Tensor] = {}
    for shard in cache_shards(model_dir):
        values = shard[key]
        for index, item_id in enumerate(shard["item_ids"]):
            if item_id in wanted:
                found[str(item_id)] = values[index].clone()
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"TM-1 selected cache misses {len(missing)} rows")
    return torch.stack([found[item_id] for item_id in item_ids]).float()


def unbiased_hsic(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 4:
        raise ValueError("unbiased HSIC requires matched square Gram matrices, n >= 4")
    n = left.shape[0]
    k = left.astype(np.float64, copy=True)
    l = right.astype(np.float64, copy=True)
    np.fill_diagonal(k, 0.0)
    np.fill_diagonal(l, 0.0)
    trace = float(np.sum(k * l))
    total = float(k.sum() * l.sum()) / ((n - 1) * (n - 2))
    rows = 2.0 * float(np.dot(k.sum(axis=1), l.sum(axis=1))) / (n - 2)
    return (trace + total - rows) / (n * (n - 3))


def debiased_linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    left = (x.double() @ x.double().T).numpy()
    right = (y.double() @ y.double().T).numpy()
    numerator = unbiased_hsic(left, right)
    denominator = math.sqrt(
        max(unbiased_hsic(left, left), 0.0)
        * max(unbiased_hsic(right, right), 0.0)
    )
    return numerator / denominator if denominator > 0.0 else float("nan")


def analyze_subset(
    *,
    manifest: Path,
    student_dir: Path,
    teacher_dirs: dict[str, Path],
) -> dict[str, Any]:
    rows = read_jsonl(manifest)
    item_ids = [str(row["item_id"]) for row in rows]
    student = selected_cache(student_dir, item_ids, key="h0")
    result: dict[str, Any] = {
        "manifest": {
            "path": str(manifest),
            "rows": len(rows),
            "sha256": sha256_file(manifest),
        },
        "teachers": {},
    }
    for teacher, directory in teacher_dirs.items():
        layers = selected_cache(directory, item_ids, key="layers")
        curves: dict[str, list[float]] = {}
        for pool_index, pool_name in enumerate(("last_active_token", "active_token_mean")):
            x = student[:, pool_index]
            curves[pool_name] = [
                debiased_linear_cka(x, layers[:, layer, pool_index])
                for layer in range(layers.shape[1])
            ]
        arithmetic_mean = np.mean(np.asarray(list(curves.values())), axis=0)
        selected = int(np.nanargmax(arithmetic_mean))
        result["teachers"][teacher] = {
            "layers": int(layers.shape[1]),
            "curves": curves,
            "arithmetic_mean_curve": arithmetic_mean.tolist(),
            "selected_layer_zero_based": selected,
            "selected_layer_one_based": selected + 1,
            "selected_mean_cka": float(arithmetic_mean[selected]),
        }
    return result


def write_figure(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    subsets = summary["subsets"]
    colors = {"teacher_7b": "#176B87", "teacher_14b": "#C54F3D"}
    for axis, (teacher, color) in zip(axes, colors.items()):
        for subset_name, linestyle in (("a", "-"), ("b", "--")):
            payload = subsets[subset_name]["teachers"][teacher]
            values = payload["arithmetic_mean_curve"]
            axis.plot(range(1, len(values) + 1), values, linestyle, color=color,
                      label=f"subset {subset_name.upper()}")
            axis.axvline(payload["selected_layer_one_based"], color=color,
                         linestyle=linestyle, alpha=0.35)
        axis.set_title(teacher.replace("teacher_", "Qwen2.5 ").upper())
        axis.set_xlabel("Teacher layer")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Debiased linear CKA with student h0")
    figure.suptitle("TM-1 interface alignment is calibrated on two disjoint subsets")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"tm1_cka_calibration.{suffix}", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--manifest_a", type=Path, required=True)
    parser.add_argument("--manifest_b", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    teacher_dirs = {
        "teacher_7b": args.cache_root / "teacher_7b",
        "teacher_14b": args.cache_root / "teacher_14b",
    }
    subsets = {
        name: analyze_subset(
            manifest=manifest,
            student_dir=args.cache_root / "student",
            teacher_dirs=teacher_dirs,
        )
        for name, manifest in (("a", args.manifest_a), ("b", args.manifest_b))
    }
    stability = {}
    for teacher in teacher_dirs:
        left = subsets["a"]["teachers"][teacher]["selected_layer_one_based"]
        right = subsets["b"]["teachers"][teacher]["selected_layer_one_based"]
        stability[teacher] = {
            "subset_a_layer": left,
            "subset_b_layer": right,
            "absolute_difference": abs(left - right),
            "passes_plus_or_minus_one": abs(left - right) <= 1,
        }
    summary = {
        "kind": "paper2_tm0_tm1_cka_calibration_v1",
        "estimator": "exact_debiased_linear_cka_unbiased_hsic",
        "subsets": subsets,
        "stability": stability,
        "status": (
            "PASS_STABLE_LAYER_SELECTION"
            if all(row["passes_plus_or_minus_one"] for row in stability.values())
            else "ESCALATE_UNSTABLE_LAYER_SELECTION"
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / "tm1_cka_calibration.json", summary)
    write_figure(summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
