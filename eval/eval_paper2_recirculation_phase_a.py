"""Run the locked, score-only paper-native recirculation Phase-A sweep."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import torch
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.eval_paper2_phase3_p34_task_trajectory import score_generation
from eval.eval_paper2_recirculation_phase0 import (
    file_receipt,
    load_model,
    read_jsonl,
    score_nll,
    sha256_file,
    validate_runtime,
    write_json,
    write_jsonl,
)
from eval.eval_paper2_stage2bs_depth_study import _generation_rows
from models.recirculation import PaperNativeRecirculationEvaluator, RecirculationConfig


KIND = "paper2_recirculation_phase_a_v1"
LOCK_KIND = "paper2_recirculation_phase_a_lock_v1"
STAGE = "stage5_paper2_recirculation_phase_a_20260827"


@dataclass(frozen=True)
class CellSpec:
    source_layer: int
    destination_layer: int
    alpha: float
    beta_mode: str = "convex"
    ramp_tokens: int | None = None
    normalization_mode: str = "norm_matched"

    @property
    def offset(self) -> int:
        return self.source_layer - self.destination_layer

    def key(self) -> tuple[int, int, float, str, int | None, str]:
        return (
            self.destination_layer,
            self.source_layer,
            float(self.alpha),
            self.beta_mode,
            self.ramp_tokens,
            self.normalization_mode,
        )

    def slug(self) -> str:
        alpha = f"{self.alpha:.2f}".replace(".", "p")
        ramp = "none" if self.ramp_tokens is None else str(self.ramp_tokens)
        return (
            f"d{self.destination_layer:02d}_s{self.source_layer:02d}_a{alpha}_"
            f"{self.beta_mode}_r{ramp}_{self.normalization_mode}"
        )


def coarse_pairs(lock: Mapping[str, Any]) -> list[tuple[int, int]]:
    grid = lock["coarse_grid"]
    pairs = [
        (int(destination), int(destination) + int(offset))
        for destination in grid["destinations"]
        for offset in grid["source_offsets"]
        if int(destination) + int(offset) <= int(grid["maximum_source"])
    ]
    if len(pairs) != int(grid["expected_pairs"]):
        raise RuntimeError(f"registered coarse pair count changed: {len(pairs)}")
    return pairs


def coarse_specs(lock: Mapping[str, Any]) -> list[CellSpec]:
    grid = lock["coarse_grid"]
    specs = [
        CellSpec(
            source_layer=source,
            destination_layer=destination,
            alpha=float(alpha),
            beta_mode=str(grid["beta_mode"]),
            ramp_tokens=grid["ramp_tokens"],
            normalization_mode=str(grid["normalization_mode"]),
        )
        for destination, source in coarse_pairs(lock)
        for alpha in grid["alphas"]
    ]
    if len(specs) != int(grid["expected_cells"]):
        raise RuntimeError(f"registered coarse cell count changed: {len(specs)}")
    return specs


def _pair_scores(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], float]:
    scores: dict[tuple[int, int], float] = {}
    for row in rows:
        config = row["config"]
        pair = (int(config["destination_layer"]), int(config["source_layer"]))
        scores[pair] = max(scores.get(pair, -math.inf), float(row["perplexity_reduction_percent"]))
    return scores


def _connected(
    pairs: Sequence[tuple[int, int]],
    *,
    destinations: Sequence[int],
    offsets: Sequence[int],
) -> bool:
    coordinates = {
        pair: (
            destinations.index(pair[0]),
            offsets.index(pair[1] - pair[0]),
        )
        for pair in pairs
    }
    seen = {pairs[0]}
    frontier = [pairs[0]]
    while frontier:
        current = frontier.pop()
        ci, cj = coordinates[current]
        for candidate in pairs:
            if candidate in seen:
                continue
            ni, nj = coordinates[candidate]
            if abs(ci - ni) + abs(cj - nj) == 1:
                seen.add(candidate)
                frontier.append(candidate)
    return len(seen) == len(pairs)


def select_contiguous_region(
    rows: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the pre-data connected-three selection rule and its tie breaks."""

    scores = _pair_scores(rows)
    expected = set(coarse_pairs(lock))
    if set(scores) != expected:
        raise RuntimeError("coarse results do not cover the registered pair set")
    grid = lock["coarse_grid"]
    destinations = [int(value) for value in grid["destinations"]]
    offsets = [int(value) for value in grid["source_offsets"]]
    region_size = int(lock["selection_policy"]["region_size"])
    candidates: list[tuple[tuple[Any, ...], tuple[tuple[int, int], ...]]] = []
    for region in itertools.combinations(sorted(expected), region_size):
        if not _connected(region, destinations=destinations, offsets=offsets):
            continue
        values = [scores[pair] for pair in region]
        # Minimize this key: the first three terms implement the registered
        # descending metrics; the last term is the lexicographic tie break.
        key = (-sum(values) / len(values), -min(values), -max(values), region)
        candidates.append((key, region))
    if not candidates:
        raise RuntimeError("registered coarse grid has no connected three-pair region")
    _, selected = min(candidates, key=lambda item: item[0])
    best_pair = min(scores, key=lambda pair: (-scores[pair], pair))
    return {
        "kind": "paper2_recirculation_phase_a_region_selection_v1",
        "coordinate": "destination_and_source_offset",
        "selected_pairs": [
            {
                "destination_layer": destination,
                "source_layer": source,
                "source_offset": source - destination,
                "pair_score_percent": scores[(destination, source)],
            }
            for destination, source in selected
        ],
        "selected_region_mean_percent": sum(scores[pair] for pair in selected)
        / len(selected),
        "selected_region_minimum_percent": min(scores[pair] for pair in selected),
        "best_pair": {
            "destination_layer": best_pair[0],
            "source_layer": best_pair[1],
            "source_offset": best_pair[1] - best_pair[0],
            "pair_score_percent": scores[best_pair],
        },
        "candidate_regions": len(candidates),
        "configured_before_first_phase_a_score": True,
    }


def refinement_specs(
    coarse_rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> list[tuple[str, CellSpec]]:
    """Construct the thirteen registered A3 NLL cells deterministically."""

    refinement = lock["refinement"]
    best = selection["best_pair"]
    best_pair = (int(best["destination_layer"]), int(best["source_layer"]))
    fine = [
        (
            "fine_alpha",
            CellSpec(
                source_layer=best_pair[1],
                destination_layer=best_pair[0],
                alpha=float(alpha),
                beta_mode=str(refinement["fine_beta_mode"]),
                normalization_mode=str(refinement["fine_normalization_mode"]),
            ),
        )
        for alpha in refinement["fine_alphas"]
    ]

    pair_best = _best_row_per_pair(coarse_rows)
    selected_pairs = [
        (int(item["destination_layer"]), int(item["source_layer"]))
        for item in selection["selected_pairs"]
    ]
    selected_pairs.sort(key=lambda pair: (-float(pair_best[pair]["perplexity_reduction_percent"]), pair))
    additive = []
    for pair in selected_pairs[: int(refinement["additive_pairs"])]:
        # Fine results do not exist yet. The selected best pair's placeholder
        # is replaced after the fine sweep by build_post_fine_specs.
        alpha = (
            float(refinement["fine_alphas"][0])
            if pair == best_pair
            else float(pair_best[pair]["config"]["alpha"])
        )
        additive.append(
            (
                "additive",
                CellSpec(
                    source_layer=pair[1],
                    destination_layer=pair[0],
                    alpha=alpha,
                    beta_mode=str(refinement["additive_beta_mode"]),
                    normalization_mode=str(refinement["fine_normalization_mode"]),
                ),
            )
        )
    # The last five cells are rebuilt after the fine sweep. Returning placeholders
    # here keeps the 8+3+1+1 contract inspectable before any Phase-A score.
    return fine + additive + [
        (
            "ramp",
            CellSpec(
                source_layer=best_pair[1],
                destination_layer=best_pair[0],
                alpha=float(refinement["fine_alphas"][0]),
                ramp_tokens=int(refinement["ramp_tokens"]),
            ),
        ),
        (
            "identity_normalization",
            CellSpec(
                source_layer=best_pair[1],
                destination_layer=best_pair[0],
                alpha=float(refinement["fine_alphas"][0]),
                normalization_mode="identity",
            ),
        ),
    ]


def build_post_fine_specs(
    *,
    coarse_rows: Sequence[Mapping[str, Any]],
    fine_rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> list[tuple[str, CellSpec]]:
    refinement = lock["refinement"]
    best = selection["best_pair"]
    best_pair = (int(best["destination_layer"]), int(best["source_layer"]))
    candidate_best_rows = list(fine_rows) + [
        row
        for row in coarse_rows
        if int(row["config"]["destination_layer"]) == best_pair[0]
        and int(row["config"]["source_layer"]) == best_pair[1]
    ]
    best_fine = min(
        candidate_best_rows,
        key=lambda row: (-float(row["perplexity_reduction_percent"]), float(row["config"]["alpha"])),
    )
    best_alpha = float(best_fine["config"]["alpha"])

    pair_best = _best_row_per_pair(coarse_rows)
    selected_pairs = [
        (int(item["destination_layer"]), int(item["source_layer"]))
        for item in selection["selected_pairs"]
    ]
    selected_pairs.sort(key=lambda pair: (-float(pair_best[pair]["perplexity_reduction_percent"]), pair))
    specs: list[tuple[str, CellSpec]] = []
    for pair in selected_pairs[: int(refinement["additive_pairs"])]:
        alpha = best_alpha if pair == best_pair else float(pair_best[pair]["config"]["alpha"])
        specs.append(
            (
                "additive",
                CellSpec(
                    source_layer=pair[1],
                    destination_layer=pair[0],
                    alpha=alpha,
                    beta_mode=str(refinement["additive_beta_mode"]),
                    normalization_mode=str(refinement["fine_normalization_mode"]),
                ),
            )
        )
    specs.extend(
        [
            (
                "ramp",
                CellSpec(
                    source_layer=best_pair[1],
                    destination_layer=best_pair[0],
                    alpha=best_alpha,
                    beta_mode=str(refinement["fine_beta_mode"]),
                    ramp_tokens=int(refinement["ramp_tokens"]),
                    normalization_mode=str(refinement["fine_normalization_mode"]),
                ),
            ),
            (
                "identity_normalization",
                CellSpec(
                    source_layer=best_pair[1],
                    destination_layer=best_pair[0],
                    alpha=best_alpha,
                    beta_mode=str(refinement["fine_beta_mode"]),
                    normalization_mode="identity",
                ),
            ),
        ]
    )
    return specs


def _best_row_per_pair(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], Mapping[str, Any]]:
    best: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        config = row["config"]
        pair = (int(config["destination_layer"]), int(config["source_layer"]))
        if pair not in best or (
            -float(row["perplexity_reduction_percent"]), float(config["alpha"])
        ) < (
            -float(best[pair]["perplexity_reduction_percent"]),
            float(best[pair]["config"]["alpha"]),
        ):
            best[pair] = row
    return best


def rank_unique_configurations(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    unique: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        spec = CellSpec(**row["config"])
        key = spec.key()
        if key in unique:
            prior = unique[key]
            if float(prior["recirculated"]["mean_nll"]) != float(row["recirculated"]["mean_nll"]):
                raise RuntimeError(f"duplicate deterministic cell changed: {spec.slug()}")
            continue
        unique[key] = row
    return sorted(
        unique.values(),
        key=lambda row: (float(row["recirculated"]["mean_nll"]), CellSpec(**row["config"]).key()),
    )


def expected_cell_seconds(lock: Mapping[str, Any], destination_layer: int) -> float:
    pilot_seconds = 63.515999336999926
    layers = int(lock["model"]["layers"])
    pilot_destination = 8
    return pilot_seconds * (1.0 + (layers - destination_layer) / layers) / (
        1.0 + (layers - pilot_destination) / layers
    )


def expected_total_seconds(lock: Mapping[str, Any], completed: int) -> float:
    specs = coarse_specs(lock)
    phase0 = float(lock["phase0"]["phase0_elapsed_seconds"])
    nll_expected = [expected_cell_seconds(lock, spec.destination_layer) for spec in specs]
    # The registered projection prices all thirteen refinement cells at the
    # worst destination and each battery cell from the banked anchor time.
    nll_expected.extend(
        [max(nll_expected)] * int(lock["refinement"]["expected_nll_cells"])
    )
    battery_each = (
        1705.1757169150003 * max(nll_expected) / 63.515999336999926
    )
    planned = nll_expected + [battery_each] * int(lock["refinement"]["battery_cells"])
    return phase0 + sum(planned[:completed])


def checkpoint_overrun(
    *,
    completed: int,
    resume_completed: int,
    actual_total_seconds: float,
    expected_total_seconds_at_checkpoint: float,
    checkpoint_set: set[int],
    overrun_multiplier: float,
    cost_ceiling_seconds: float,
) -> bool:
    if actual_total_seconds > cost_ceiling_seconds:
        return True
    return (
        completed > resume_completed
        and completed in checkpoint_set
        and actual_total_seconds / expected_total_seconds_at_checkpoint
        > overrun_multiplier
    )


def _receipt_path(root: Path, *, index: int, stage: str, spec: CellSpec) -> Path:
    return root / "cells" / f"{index:03d}_{stage}_{spec.slug()}.json"


def score_cell(
    *,
    model: Any,
    windows: torch.Tensor,
    baseline: Mapping[str, Any],
    spec: CellSpec,
    stage: str,
    index: int,
    root: Path,
    batch_size: int,
) -> dict[str, Any]:
    path = _receipt_path(root, index=index, stage=stage, spec=spec)
    if path.is_file():
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("config") != asdict(spec) or int(row.get("run_index", -1)) != index:
            raise RuntimeError(f"durable cell identity changed: {path.name}")
        print(f"recirculation_phase_a_resume index={index} cell={spec.slug()}", flush=True)
        return row
    evaluator = PaperNativeRecirculationEvaluator(
        model,
        RecirculationConfig(
            source_layer=spec.source_layer,
            destination_layer=spec.destination_layer,
            alpha=spec.alpha,
            beta_mode=spec.beta_mode,
            ramp_tokens=spec.ramp_tokens,
            normalization_mode=spec.normalization_mode,
        ),
    )
    result = score_nll(
        evaluator,
        windows,
        recirculate=True,
        batch_size=batch_size,
        label=f"phase_a_{index:03d}_{stage}_{spec.slug()}",
    )
    row = {
        "kind": "paper2_recirculation_phase_a_nll_cell_v1",
        "run_index": index,
        "stage": stage,
        "config": asdict(spec),
        "evaluator": "paper_native_serial_first_iteration_readout_v1",
        "runtime": "A100-SXM4-40GB_bfloat16_sdpa",
        "weights_frozen": True,
        "baseline": dict(baseline),
        "recirculated": result,
        "perplexity_reduction_percent": 100.0
        * (float(baseline["perplexity"]) - float(result["perplexity"]))
        / float(baseline["perplexity"]),
    }
    write_json(path, row)
    return row


def battery_summary(
    *,
    current: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    spec: CellSpec,
    rows_path: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if [row["item_id"] for row in current] != [row["item_id"] for row in baseline]:
        raise RuntimeError("battery row order changed against the matched comparator")
    per_battery: dict[str, dict[str, int]] = {}
    fixes = regressions = 0
    for now, before in zip(current, baseline):
        battery = str(now["battery"])
        stats = per_battery.setdefault(
            battery, {"rows": 0, "baseline_correct": 0, "current_correct": 0, "fixes": 0, "regressions": 0}
        )
        stats["rows"] += 1
        old = bool(before["augmented_correct"])
        new = bool(now["augmented_correct"])
        stats["baseline_correct"] += int(old)
        stats["current_correct"] += int(new)
        stats["fixes"] += int(new and not old)
        stats["regressions"] += int(old and not new)
        fixes += int(new and not old)
        regressions += int(old and not new)
    correct = sum(bool(row["augmented_correct"]) for row in current)
    return {
        "kind": "paper2_recirculation_phase_a_battery_cell_v1",
        "config": asdict(spec),
        "evaluator": "paper_native_serial_first_iteration_readout_v1",
        "rows": len(current),
        "correct": correct,
        "baseline_correct": sum(bool(row["augmented_correct"]) for row in baseline),
        "delta_rows": correct - sum(bool(row["augmented_correct"]) for row in baseline),
        "fixes": fixes,
        "regressions": regressions,
        "per_battery": per_battery,
        "elapsed_seconds": elapsed_seconds,
        "row_receipt": file_receipt(rows_path),
        "weights_frozen": True,
    }


def qualifying_components(
    rows: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]
) -> list[dict[str, Any]]:
    floor = float(lock["gates"]["perplexity_materiality_percent"])
    destinations = [int(value) for value in lock["coarse_grid"]["destinations"]]
    offsets = [int(value) for value in lock["coarse_grid"]["source_offsets"]]
    results: list[dict[str, Any]] = []
    for alpha in sorted({float(row["config"]["alpha"]) for row in rows}):
        eligible = sorted(
            (
                int(row["config"]["destination_layer"]),
                int(row["config"]["source_layer"]),
            )
            for row in rows
            if float(row["config"]["alpha"]) == alpha
            and float(row["perplexity_reduction_percent"]) >= floor
        )
        remaining = set(eligible)
        components: list[list[tuple[int, int]]] = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            component = [seed]
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                ci = destinations.index(current[0])
                cj = offsets.index(current[1] - current[0])
                neighbors = [
                    pair
                    for pair in remaining
                    if abs(destinations.index(pair[0]) - ci)
                    + abs(offsets.index(pair[1] - pair[0]) - cj)
                    == 1
                ]
                for neighbor in sorted(neighbors):
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    frontier.append(neighbor)
            components.append(sorted(component))
        results.append(
            {
                "alpha": alpha,
                "materiality_floor_percent": floor,
                "components": [
                    [{"destination_layer": d, "source_layer": s} for d, s in component]
                    for component in components
                ],
                "largest_component_cells": max((len(component) for component in components), default=0),
            }
        )
    return results


def write_nll_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_index",
                "stage",
                "destination_layer",
                "source_layer",
                "source_offset",
                "alpha",
                "beta_mode",
                "ramp_tokens",
                "normalization_mode",
                "mean_nll",
                "perplexity",
                "perplexity_reduction_percent",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for row in rows:
            config = row["config"]
            writer.writerow(
                {
                    "run_index": row["run_index"],
                    "stage": row["stage"],
                    "destination_layer": config["destination_layer"],
                    "source_layer": config["source_layer"],
                    "source_offset": int(config["source_layer"]) - int(config["destination_layer"]),
                    "alpha": config["alpha"],
                    "beta_mode": config["beta_mode"],
                    "ramp_tokens": config["ramp_tokens"],
                    "normalization_mode": config["normalization_mode"],
                    "mean_nll": row["recirculated"]["mean_nll"],
                    "perplexity": row["recirculated"]["perplexity"],
                    "perplexity_reduction_percent": row["perplexity_reduction_percent"],
                    "elapsed_seconds": row["recirculated"]["elapsed_seconds"],
                }
            )


def render_heatmaps(
    *, rows: Sequence[Mapping[str, Any]], lock: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    destinations = [int(value) for value in lock["coarse_grid"]["destinations"]]
    sources = sorted({source for _, source in coarse_pairs(lock)})
    alphas = [float(value) for value in lock["coarse_grid"]["alphas"]]
    maximum = max(abs(float(row["perplexity_reduction_percent"])) for row in rows)
    limit = max(maximum, 0.25)
    figure, axes = plt.subplots(1, len(alphas), figsize=(15, 5), constrained_layout=True)
    image = None
    for axis, alpha in zip(axes, alphas):
        matrix = np.full((len(destinations), len(sources)), np.nan)
        for row in rows:
            config = row["config"]
            if float(config["alpha"]) != alpha:
                continue
            matrix[destinations.index(int(config["destination_layer"])), sources.index(int(config["source_layer"]))] = float(
                row["perplexity_reduction_percent"]
            )
        image = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_title(f"alpha = {alpha:.2f}")
        axis.set_xlabel("Source layer s")
        axis.set_xticks(range(len(sources)), sources)
        axis.set_ylabel("Destination layer d")
        axis.set_yticks(range(len(destinations)), destinations)
        for y in range(len(destinations)):
            for x in range(len(sources)):
                if not np.isnan(matrix[y, x]):
                    axis.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=7)
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.82, label="Perplexity reduction (%)")
    figure.suptitle("Qwen2.5-0.5B paper-native recirculation coarse sweep")
    png = output_dir / "phase_a_coarse_heatmaps.png"
    svg = output_dir / "phase_a_coarse_heatmaps.svg"
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return {"png": file_receipt(png), "svg": file_receipt(svg), "common_scale_limit_percent": limit}


def archive_progress(artifact_root: Path, export_path: Path) -> dict[str, Any]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = export_path.with_suffix(export_path.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(artifact_root, arcname=STAGE)
    temporary.replace(export_path)
    receipt = file_receipt(export_path)
    print(
        f"recirculation_phase_a_checkpoint_archive path={export_path} "
        f"bytes={receipt['bytes']} sha256={receipt['sha256']}",
        flush=True,
    )
    return receipt


def _verify_authorities(lock: Mapping[str, Any], repo_root: Path) -> None:
    for authority in lock["authorities"]:
        path = repo_root / "docs" / authority["filename"]
        if not path.is_file() or file_receipt(path) != {
            "bytes": int(authority["bytes"]),
            "sha256": str(authority["sha256"]),
        }:
            raise RuntimeError(f"Phase-A authority identity changed: {authority['filename']}")


def canonical_lf_receipt(path: Path) -> dict[str, Any]:
    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in normalized:
        raise RuntimeError(f"unauthorized carriage return in {path}")
    return {
        "bytes": len(normalized),
        "sha256": hashlib.sha256(normalized).hexdigest(),
    }


def _status_elapsed(status: Mapping[str, Any]) -> float:
    return float(status.get("phase_a_elapsed_seconds", 0.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--phase0_root", type=Path, required=True)
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--artifact_root", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--progress_archive", type=Path, required=True)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--nll_batch_size", type=int, default=32)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("kind") != LOCK_KIND or not bool(lock.get("phase_a_authorized")):
        raise RuntimeError("recirculation Phase-A lock is invalid or not authorized")
    if lock.get("phase_b_training_authorized") is not False or int(lock.get("optimizer_steps_allowed", -1)) != 0:
        raise RuntimeError("recirculation Phase-A lock is over-authorized")
    _verify_authorities(lock, args.repo_root)
    public_summary_path = args.repo_root / lock["phase0"]["public_summary"]["path"]
    if canonical_lf_receipt(public_summary_path) != {
        "bytes": int(lock["phase0"]["public_summary"]["canonical_lf_bytes"]),
        "sha256": str(lock["phase0"]["public_summary"]["canonical_lf_sha256"]),
    }:
        raise RuntimeError("banked Phase-0 public summary canonical identity changed")

    receipts_dir = args.artifact_root / "receipts" / "phase_a"
    private_dir = args.artifact_root / "private" / "phase_a"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.artifact_root / "receipts" / "status.json"
    existing_status = (
        json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    )
    resume_completed = int(existing_status.get("completed_measurements", 0))
    prior_elapsed = _status_elapsed(existing_status)
    session_started = time.perf_counter()
    runtime = validate_runtime(lock)

    phase0_private = args.phase0_root / "private" / "phase0"
    phase0_receipts = args.phase0_root / "receipts" / "phase0"
    corpus_path = phase0_private / "corpus_token_windows.pt"
    baseline_rows_path = phase0_private / "battery_anchor_rows.jsonl"
    if file_receipt(corpus_path) != dict(lock["phase0"]["corpus_token_windows"]):
        raise RuntimeError("Phase-0 corpus token cache identity changed")
    if file_receipt(baseline_rows_path) != dict(lock["phase0"]["battery_row_receipt"]):
        raise RuntimeError("matched battery comparator rows changed")
    graph_path = phase0_receipts / "graph_receipt.json"
    if sha256_file(graph_path) != lock["phase0"]["graph_receipt_sha256"]:
        raise RuntimeError("paper-native evaluator graph receipt changed")
    phase0_status = json.loads((phase0_receipts / "phase0_status.json").read_text(encoding="utf-8"))
    if phase0_status.get("status") != lock["phase0"]["status"]:
        raise RuntimeError("banked Phase-0 status changed")
    battery_anchor = json.loads((phase0_receipts / "battery_anchor_v2_adjudicated.json").read_text(encoding="utf-8"))
    if int(battery_anchor["correct"]) != int(lock["gates"]["battery_baseline_correct"]):
        raise RuntimeError("matched paper-native battery comparator changed")

    saved = torch.load(corpus_path, map_location="cpu", weights_only=False)
    windows = saved["qwen"]
    if tuple(windows.shape) != (int(lock["corpus"]["windows"]), int(lock["corpus"]["window_tokens"])):
        raise RuntimeError(f"registered Qwen corpus shape changed: {tuple(windows.shape)}")
    panel = read_jsonl(args.panel)
    generation_panel = list(_generation_rows(panel))
    baseline_rows = read_jsonl(baseline_rows_path)
    if len(generation_panel) != int(lock["gates"]["battery_rows"]) or len(baseline_rows) != len(generation_panel):
        raise RuntimeError("registered 461-row battery population changed")
    if [row["item_id"] for row in generation_panel] != [row["item_id"] for row in baseline_rows]:
        raise RuntimeError("battery population order changed against matched comparator")

    status: dict[str, Any] = {
        "kind": KIND,
        "status": "running_phase_a",
        "runtime": runtime,
        "phase_a_authorized": True,
        "phase_b_training_authorized": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
        "completed_measurements": resume_completed,
        "phase_a_elapsed_seconds": prior_elapsed,
    }
    write_json(status_path, status)

    tokenizer = AutoTokenizer.from_pretrained(
        lock["model"]["id"], revision=lock["model"]["revision"], cache_dir=args.model_cache
    )
    model = load_model(lock["model"], cache_dir=args.model_cache)
    identity_tokens = windows[:1, :32].to("cuda")
    identity = PaperNativeRecirculationEvaluator(
        model,
        RecirculationConfig(source_layer=16, destination_layer=8, alpha=0.0),
    ).identity_receipt(input_ids=identity_tokens, attention_mask=torch.ones_like(identity_tokens))
    if not identity["bit_exact"]:
        raise RuntimeError(f"Phase-A identity drift: {identity}")
    write_json(receipts_dir / "identity_recheck.json", identity)
    baseline = {
        "mean_nll": float(lock["phase0"]["qwen_baseline_mean_nll"]),
        "perplexity": float(lock["phase0"]["qwen_baseline_perplexity"]),
        "predicted_tokens": int(lock["corpus"]["predicted_tokens"]),
        "windows": int(lock["corpus"]["windows"]),
        "window_tokens": int(lock["corpus"]["window_tokens"]),
        "source": "banked_phase0_same_corpus_same_evaluator",
    }

    checkpoint_set = {int(value) for value in lock["gates"]["overrun_checkpoints"]}
    completed = 0

    def update_progress(*, outcome: str = "running_phase_a") -> bool:
        nonlocal completed, status
        elapsed = prior_elapsed + (time.perf_counter() - session_started)
        expected = expected_total_seconds(lock, completed)
        actual_total = float(lock["phase0"]["phase0_elapsed_seconds"]) + elapsed
        multiplier = actual_total / expected if expected else 0.0
        overrun = checkpoint_overrun(
            completed=completed,
            resume_completed=resume_completed,
            actual_total_seconds=actual_total,
            expected_total_seconds_at_checkpoint=expected,
            checkpoint_set=checkpoint_set,
            overrun_multiplier=float(lock["gates"]["overrun_multiplier"]),
            cost_ceiling_seconds=float(lock["runtime"]["cost_ceiling_a100_hours"])
            * 3600.0,
        )
        status.update(
            status=("overrun_stop_awaiting_relay" if overrun else outcome),
            completed_measurements=completed,
            phase_a_elapsed_seconds=elapsed,
            actual_total_a100_hours=actual_total / 3600.0,
            expected_total_a100_hours_at_checkpoint=expected / 3600.0,
            actual_to_expected_multiplier=multiplier,
            cost_ceiling_a100_hours=float(lock["runtime"]["cost_ceiling_a100_hours"]),
        )
        write_json(status_path, status)
        if completed in checkpoint_set or overrun or outcome != "running_phase_a":
            status["checkpoint_archive"] = archive_progress(args.artifact_root, args.progress_archive)
            write_json(status_path, status)
        return overrun

    coarse_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(coarse_specs(lock), start=1):
        row = score_cell(
            model=model,
            windows=windows,
            baseline=baseline,
            spec=spec,
            stage="coarse",
            index=index,
            root=receipts_dir,
            batch_size=args.nll_batch_size,
        )
        coarse_rows.append(row)
        completed = index
        if update_progress():
            return 0
    write_nll_table(receipts_dir / "phase_a_coarse_cells.csv", coarse_rows)
    write_json(receipts_dir / "phase_a_coarse_cells.json", {"kind": "paper2_recirculation_phase_a_coarse_table_v1", "cells": coarse_rows})
    heatmaps = render_heatmaps(rows=coarse_rows, lock=lock, output_dir=receipts_dir)
    selection = select_contiguous_region(coarse_rows, lock)
    write_json(receipts_dir / "phase_a_region_selection.json", selection)
    write_json(
        receipts_dir / "phase_a_coarse_components.json",
        {"kind": "paper2_recirculation_phase_a_components_v1", "by_alpha": qualifying_components(coarse_rows, lock)},
    )

    prebuilt = refinement_specs(coarse_rows, selection, lock)
    if len(prebuilt) != int(lock["refinement"]["expected_nll_cells"]):
        raise RuntimeError("registered A3 refinement count changed before scoring")
    fine_rows: list[dict[str, Any]] = []
    for offset, (stage, spec) in enumerate(prebuilt[:8], start=97):
        row = score_cell(
            model=model,
            windows=windows,
            baseline=baseline,
            spec=spec,
            stage=stage,
            index=offset,
            root=receipts_dir,
            batch_size=args.nll_batch_size,
        )
        fine_rows.append(row)
        completed = offset
        update_progress()
    post_fine = build_post_fine_specs(
        coarse_rows=coarse_rows, fine_rows=fine_rows, selection=selection, lock=lock
    )
    if len(post_fine) != 5:
        raise RuntimeError("registered post-fine refinement count changed")
    post_rows: list[dict[str, Any]] = []
    for index, (stage, spec) in enumerate(post_fine, start=105):
        row = score_cell(
            model=model,
            windows=windows,
            baseline=baseline,
            spec=spec,
            stage=stage,
            index=index,
            root=receipts_dir,
            batch_size=args.nll_batch_size,
        )
        post_rows.append(row)
        completed = index
        if update_progress():
            return 0
    refinement_rows = fine_rows + post_rows
    write_nll_table(receipts_dir / "phase_a_refinement_cells.csv", refinement_rows)
    write_json(receipts_dir / "phase_a_refinement_cells.json", {"kind": "paper2_recirculation_phase_a_refinement_table_v1", "cells": refinement_rows})

    ranked = rank_unique_configurations(coarse_rows + refinement_rows)
    top_two = ranked[: int(lock["refinement"]["battery_cells"])]
    write_json(
        receipts_dir / "phase_a_battery_selection.json",
        {
            "kind": "paper2_recirculation_phase_a_battery_selection_v1",
            "rule": lock["refinement"]["battery_selection"],
            "selected": [
                {"rank": rank, "config": row["config"], "mean_nll": row["recirculated"]["mean_nll"], "perplexity_reduction_percent": row["perplexity_reduction_percent"]}
                for rank, row in enumerate(top_two, start=1)
            ],
        },
    )

    battery_receipts: list[dict[str, Any]] = []
    for rank, selected in enumerate(top_two, start=1):
        index = 109 + rank
        spec = CellSpec(**selected["config"])
        rows_path = private_dir / f"battery_rank_{rank}_{spec.slug()}_rows.jsonl"
        receipt_path = receipts_dir / f"battery_rank_{rank}_{spec.slug()}.json"
        if rows_path.is_file() and receipt_path.is_file():
            rows = read_jsonl(rows_path)
            battery = json.loads(receipt_path.read_text(encoding="utf-8"))
            if battery.get("config") != asdict(spec) or battery.get("row_receipt") != file_receipt(rows_path):
                raise RuntimeError(f"durable battery cell identity changed: rank {rank}")
            print(f"recirculation_phase_a_resume index={index} battery_rank={rank}", flush=True)
        else:
            graph = PaperNativeRecirculationEvaluator(
                model,
                RecirculationConfig(
                    source_layer=spec.source_layer,
                    destination_layer=spec.destination_layer,
                    alpha=spec.alpha,
                    beta_mode=spec.beta_mode,
                    ramp_tokens=spec.ramp_tokens,
                    normalization_mode=spec.normalization_mode,
                ),
            )
            battery_started = time.perf_counter()
            rows = score_generation(
                graph, tokenizer, generation_panel, batch_size=args.generation_batch_size
            )
            battery_elapsed = time.perf_counter() - battery_started
            write_jsonl(rows_path, rows)
            battery = battery_summary(
                current=rows,
                baseline=baseline_rows,
                spec=spec,
                rows_path=rows_path,
                elapsed_seconds=battery_elapsed,
            )
            write_json(receipt_path, battery)
        if int(battery["rows"]) != int(lock["gates"]["battery_rows"]):
            raise RuntimeError("battery cell did not score the complete registered population")
        battery_receipts.append(battery)
        completed = index
        if update_progress():
            return 0

    del model
    gc.collect()
    torch.cuda.empty_cache()

    summary = {
        "kind": "paper2_recirculation_phase_a_result_v1",
        "status": "phase_a_complete_awaiting_strategy_adjudication",
        "runtime": runtime,
        "phase0_status": lock["phase0"]["status"],
        "identity_recheck": identity,
        "baseline": baseline,
        "coarse_cells": len(coarse_rows),
        "refinement_cells": len(refinement_rows),
        "battery_cells": battery_receipts,
        "selection": selection,
        "best_nll_configuration": {
            "config": ranked[0]["config"],
            "perplexity_reduction_percent": ranked[0]["perplexity_reduction_percent"],
            "mean_nll": ranked[0]["recirculated"]["mean_nll"],
        },
        "qualifying_components": qualifying_components(coarse_rows, lock),
        "heatmaps": heatmaps,
        "registered_bars": dict(lock["gates"]),
        "strategy_key_resolved": False,
        "strategy_only_keys": list(lock["interpretation"]["strategy_only_keys"]),
        "weights_frozen": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
        "phase_b_training_authorized": False,
    }
    write_json(receipts_dir / "phase_a_summary.json", summary)
    completed = 111
    update_progress(outcome="phase_a_complete_awaiting_strategy_adjudication")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
