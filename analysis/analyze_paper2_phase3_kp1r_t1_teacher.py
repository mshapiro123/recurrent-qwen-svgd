"""Reconstruct and summarize the landed KP-1R and teacher-fingerprint wave."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from eval.eval_paper2_phase3_kp1r_t1_teacher import benjamini_hochberg
from training.paper2_phase3_kp1r_t1_teacher import summarize_margin


PRIMARY_SURFACES = (
    "substrate_layer_24",
    "p35_seed_0_loop_4_recurrent_cell_set",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_surface(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[float], list[str]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["item_id"])].append(row)
    aggregates: list[dict[str, Any]] = []
    margins: list[float] = []
    batteries: list[str] = []
    for item_id, values in grouped.items():
        probe = sum(bool(row["probe_correct"]) for row in values) / len(values)
        control = sum(bool(row["frequency_correct"]) for row in values) / len(values)
        battery = str(values[0]["battery"])
        result = {
            "item_id": item_id,
            "battery": battery,
            "tokens": len(values),
            "probe_token_accuracy": probe,
            "frequency_token_accuracy": control,
            "margin": probe - control,
            "probe_mean_log_probability": float(
                np.mean([float(row["probe_target_log_probability"]) for row in values])
            ),
            "native_base_mean_log_probability": float(
                np.mean([float(row["native_base_log_probability"]) for row in values])
            ),
        }
        aggregates.append(result)
        margins.append(probe - control)
        batteries.append(battery)
    return aggregates, margins, batteries


def row_estimator_permutation(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    observed_rows, observed_margins, observed_batteries = aggregate_surface(rows)
    observed_pooled = float(np.mean(observed_margins))
    observed_macro = float(
        np.mean(
            [
                np.mean(
                    [margin for margin, local in zip(observed_margins, observed_batteries) if local == battery]
                )
                for battery in sorted(set(observed_batteries))
            ]
        )
    )
    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(str(row["battery"]), int(row["token_position"]))].append(index)
    generator = random.Random(int(seed))
    pooled_exceed = 0
    macro_exceed = 0
    pooled_sum = 0.0
    macro_sum = 0.0
    original_targets = [int(row["target_id"]) for row in rows]
    for _ in range(int(draws)):
        targets = list(original_targets)
        for indexes in strata.values():
            local = [targets[index] for index in indexes]
            generator.shuffle(local)
            for index, value in zip(indexes, local):
                targets[index] = value
        replaced = [
            {
                **dict(row),
                "probe_correct": int(row["probe_prediction_id"]) == targets[index],
                "frequency_correct": int(row["frequency_prediction_id"]) == targets[index],
            }
            for index, row in enumerate(rows)
        ]
        _summary, margins, batteries = aggregate_surface(replaced)
        pooled = float(np.mean(margins))
        macro = float(
            np.mean(
                [
                    np.mean([margin for margin, local in zip(margins, batteries) if local == battery])
                    for battery in sorted(set(batteries))
                ]
            )
        )
        pooled_sum += pooled
        macro_sum += macro
        pooled_exceed += int(pooled >= observed_pooled)
        macro_exceed += int(macro >= observed_macro)
    return {
        "kind": "fixed-prediction_within-battery-and-position_row-estimator_permutation",
        "rows": len(observed_rows),
        "draws": int(draws),
        "seed": int(seed),
        "observed_pooled_margin": observed_pooled,
        "observed_battery_macro_margin": observed_macro,
        "null_mean_pooled_margin": pooled_sum / draws,
        "null_mean_battery_macro_margin": macro_sum / draws,
        "pooled_one_sided_p_value": (1 + pooled_exceed) / (draws + 1),
        "battery_macro_one_sided_p_value": (1 + macro_exceed) / (draws + 1),
    }


def surface_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregates, margins, batteries = aggregate_surface(rows)
    probe_accuracy = float(np.mean([row["probe_token_accuracy"] for row in aggregates]))
    frequency_accuracy = float(np.mean([row["frequency_token_accuracy"] for row in aggregates]))
    logp_delta = [
        float(row["probe_mean_log_probability"] - row["native_base_mean_log_probability"])
        for row in aggregates
    ]
    return {
        "eval_token_positions": len(rows),
        "eval_rows": len(aggregates),
        "probe_row_mean_token_accuracy": probe_accuracy,
        "frequency_row_mean_token_accuracy": frequency_accuracy,
        "knowledge_presence_margin": summarize_margin(margins, batteries, seed=20260817, draws=10_000),
        "row_estimator_label_permutation_control": row_estimator_permutation(
            rows, draws=10_000, seed=20260818
        ),
        "probe_minus_native_base_log_probability": summarize_margin(
            logp_delta, batteries, seed=20260820, draws=10_000
        ),
        "row_summary": aggregates,
    }


def cell_group(index: int) -> str:
    if index < 8:
        return "prelude"
    if index < 40:
        return f"loop_{1 + (index - 8) // 8}"
    return {40: "layer_6", 41: "layer_12", 42: "layer_18", 43: "layer_24"}[index]


def aggregate_geometry(
    comparisons: Sequence[Mapping[str, Any]], transport: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[(str(row["checkpoint"]), cell_group(int(row["student_cell_index"])), int(row["teacher_layer"]))].append(row)
    group_rows = []
    for (checkpoint, group, teacher_layer), rows in sorted(grouped.items()):
        group_rows.append(
            {
                "checkpoint": checkpoint,
                "student_group": group,
                "teacher_layer": teacher_layer,
                "cells": len(rows),
                "mean_linear_cka": float(np.mean([row["linear_cka"] for row in rows])),
                "mean_principal_angle_degrees": float(
                    np.mean([row["principal_angles"]["mean_angle_degrees"] for row in rows])
                ),
                "mean_principal_cosine": float(
                    np.mean([row["principal_angles"]["mean_cosine"] for row in rows])
                ),
            }
        )
    max_cka = max(comparisons, key=lambda row: float(row["linear_cka"]))
    max_transport_top1 = max(transport, key=lambda row: float(row["top1_retrieval_accuracy"]))
    max_transport_top10 = max(transport, key=lambda row: float(row["top10_retrieval_accuracy"]))
    return {
        "comparison_rows": len(comparisons),
        "transport_rows": len(transport),
        "group_summary": group_rows,
        "maximum_cell_cka": max_cka,
        "maximum_transport_top1": max_transport_top1,
        "maximum_transport_top10": max_transport_top10,
        "chance_retrieval": {"top1": 1 / 410, "top10": 10 / 410},
    }


def render_figure(
    output: Path,
    cached: Mapping[str, Any],
    strong: Mapping[str, Any],
    geometry: Mapping[str, Any],
    transport: Sequence[Mapping[str, Any]],
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)

    labels = ["Cached\nlayer-24 proxy", "Cached\nloop 4", "Strong\nlayer 24", "Strong\nloop 4"]
    cached_reads = cached["surfaces"]
    reads = [
        cached_reads["cached_projected_substrate_layer_24_proxy"],
        cached_reads["p35_seed_0_loop_4_recurrent_cell_set"],
        strong["substrate_layer_24"],
        strong["p35_seed_0_loop_4_recurrent_cell_set"],
    ]
    margins = [read["knowledge_presence_margin"]["pooled_margin"] * 100 for read in reads]
    low = [read["knowledge_presence_margin"]["pooled_bootstrap_95ci"][0] * 100 for read in reads]
    high = [read["knowledge_presence_margin"]["pooled_bootstrap_95ci"][1] * 100 for read in reads]
    axes[0].bar(range(4), margins, color=["#9aa0a6", "#6f7782", "#1f77b4", "#2ca02c"])
    axes[0].errorbar(
        range(4), margins, yerr=[[m - l for m, l in zip(margins, low)], [h - m for h, m in zip(high, margins)]],
        fmt="none", ecolor="#202124", capsize=3, linewidth=1,
    )
    axes[0].axhline(0, color="#202124", linewidth=0.8)
    axes[0].set_xticks(range(4), labels)
    axes[0].set_ylabel("Probe minus frequency control (points)")
    axes[0].set_title("A. Knowledge-presence margin")

    groups = ["prelude", "loop_1", "loop_2", "loop_3", "loop_4", "layer_6", "layer_12", "layer_18", "layer_24"]
    teacher_layers = [12, 24, 36, 48]
    lookup = {
        (row["student_group"], row["teacher_layer"]): row["mean_linear_cka"]
        for row in geometry["group_summary"]
        if row["checkpoint"] == "p35_seed_0_ema_step_4400"
    }
    matrix = np.array([[lookup[(group, layer)] for layer in teacher_layers] for group in groups])
    image = axes[1].imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.65, matrix.max()))
    axes[1].set_xticks(range(4), [f"T{layer}" for layer in teacher_layers])
    axes[1].set_yticks(range(len(groups)), [value.replace("_", " ") for value in groups])
    axes[1].set_title("B. Student-teacher linear CKA")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    selected = [
        row for row in transport
        if row["checkpoint"] == "p35_seed_0_ema_step_4400" and int(row["teacher_layer"]) == 12
    ]
    selected.sort(key=lambda row: str(row["student_surface"]))
    names = [str(row["student_surface"]).replace("_", " ") for row in selected]
    top1 = [100 * float(row["top1_retrieval_accuracy"]) for row in selected]
    order = np.argsort(top1)
    axes[2].barh(np.arange(len(order)), np.array(top1)[order], color="#d55e00")
    axes[2].axvline(100 / 410, color="#202124", linestyle="--", linewidth=1, label="chance")
    axes[2].set_yticks(np.arange(len(order)), np.array(names)[order])
    axes[2].set_xlabel("Held-out matching-item retrieval (%)")
    axes[2].set_title("C. Split-fit transport to teacher layer 12")
    axes[2].legend(frameon=False, loc="lower right")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    cached = read_json(args.artifact_root / "cached/receipts/summary.json")
    strong_status = read_json(args.artifact_root / "strong/receipts/status.json")
    predictions = read_jsonl(
        args.artifact_root / "strong/private/kp1r_teacher_forced_row_predictions.jsonl"
    )
    comparisons = read_jsonl(
        args.artifact_root / "strong/private/teacher_fingerprint_comparisons.jsonl"
    )
    transport = read_jsonl(
        args.artifact_root / "strong/private/teacher_fingerprint_transport.jsonl"
    )
    by_surface: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_surface[str(row["surface"])].append(row)
    surface_reads = {name: surface_summary(rows) for name, rows in sorted(by_surface.items())}
    secondary_p = {
        name: read["row_estimator_label_permutation_control"]["pooled_one_sided_p_value"]
        for name, read in surface_reads.items()
        if name not in PRIMARY_SURFACES
    }
    secondary_q = benjamini_hochberg(secondary_p)
    for name, value in secondary_q.items():
        surface_reads[name]["secondary_bh_fdr_q_value"] = value
    geometry = aggregate_geometry(comparisons, transport)
    recovered = {
        "kind": "paper2_phase3_kp1r_t1_teacher_recovered_summary_v2",
        "status": "complete_score_only_recovered_from_durable_rows",
        "recovery_reason": "public summary copy did not flush before VM teardown; all primary row and geometry receipts were durable",
        "scope": {
            "dev_only": True,
            "confirm_scored": False,
            "eval_e_scored": False,
            "optimizer_steps": 0,
            "mbpp_exploratory_sequence_log_likelihood_not_recovered": True,
        },
        "authority": cached["authority"],
        "target_entropy_audit": cached["target_entropy_audit"],
        "publication_recovery": {
            "original_summary_expected_sha256": strong_status["summary_sha256"],
            "original_summary_missing_from_drive": True,
            "primary_and_geometry_receipts_present": True,
            "row_permutation_recomputed_on_registered_row_estimator": True,
        },
        "kp1r": {
            "primary_surfaces": {name: surface_reads[name] for name in PRIMARY_SURFACES},
            "secondary_surfaces": {
                name: read for name, read in surface_reads.items() if name not in PRIMARY_SURFACES
            },
        },
        "teacher_fingerprints": geometry,
        "durable_inputs": {
            "row_predictions_sha256": sha256_file(
                args.artifact_root / "strong/private/kp1r_teacher_forced_row_predictions.jsonl"
            ),
            "comparisons_sha256": sha256_file(
                args.artifact_root / "strong/private/teacher_fingerprint_comparisons.jsonl"
            ),
            "transport_sha256": sha256_file(
                args.artifact_root / "strong/private/teacher_fingerprint_transport.jsonl"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mbpp_path = args.output_dir / "mbpp_recovery/mbpp_recovery_summary.json"
    if mbpp_path.is_file():
        recovered["kp1r"]["mbpp_exploratory"] = read_json(mbpp_path)
        recovered["scope"]["mbpp_exploratory_sequence_log_likelihood_not_recovered"] = False
        recovered["publication_recovery"]["mbpp_recovery_summary_sha256"] = sha256_file(
            mbpp_path
        )
    fingerprint_null_path = args.output_dir / "teacher_fingerprint_null.json"
    if fingerprint_null_path.is_file():
        recovered["teacher_fingerprints"]["within_battery_permutation_sanity"] = read_json(
            fingerprint_null_path
        )
        recovered["publication_recovery"]["teacher_fingerprint_null_sha256"] = sha256_file(
            fingerprint_null_path
        )
    pre_model = {
        "kind": "paper2_phase3_kp1r_t1_teacher_pre_model_audit_v1",
        "target_entropy": cached["target_entropy_audit"],
        "confirm_scored": False,
        "eval_e_scored": False,
        "model_loaded": False,
        "optimizer_constructed": False,
    }
    write_json(args.output_dir / "pre_model_target_audit_recovered.json", pre_model)
    recovered["publication_recovery"]["pre_model_target_audit_recovered_sha256"] = sha256_file(
        args.output_dir / "pre_model_target_audit_recovered.json"
    )
    recovered["publication_recovery"]["pre_model_target_audit_original_sha256"] = strong_status[
        "pre_model_target_audit_sha256"
    ]
    write_json(args.output_dir / "recovered_summary.json", recovered)
    recovery_receipt = {
        "kind": "paper2_phase3_kp1r_t1_teacher_publication_recovery_v1",
        "status": "complete",
        "reason": recovered["recovery_reason"],
        "scientific_status_before_recovery": strong_status["status"],
        "original_summary_expected_sha256": strong_status["summary_sha256"],
        "recovered_summary_sha256": sha256_file(args.output_dir / "recovered_summary.json"),
        "pre_model_audit": {
            "original_expected_sha256": strong_status["pre_model_target_audit_sha256"],
            "recovered_sha256": sha256_file(
                args.output_dir / "pre_model_target_audit_recovered.json"
            ),
            "exact_hash_match": (
                strong_status["pre_model_target_audit_sha256"]
                == sha256_file(args.output_dir / "pre_model_target_audit_recovered.json")
            ),
        },
        "durable_inputs": recovered["durable_inputs"],
        "recovered_companions": {
            key: value
            for key, value in recovered["publication_recovery"].items()
            if key.endswith("_sha256")
        },
        "sealed_partition_assertions": {
            "confirm_scored": bool(strong_status["confirm_scored"]),
            "eval_e_scored": bool(strong_status["eval_e_scored"]),
            "optimizer_steps": int(strong_status["optimizer_steps"]),
        },
        "disclosure": (
            "The score-only job completed, but the top-level JSON did not flush from the "
            "Drive FUSE mount before VM teardown. This receipt reconstructs the publication "
            "layer from durable row-level predictions and geometry tables; it does not rerun "
            "or change model outputs."
        ),
    }
    write_json(args.output_dir / "publication_recovery_receipt.json", recovery_receipt)
    render_figure(
        args.output_dir / "paper2_kp1r_t1_teacher_wave_20260816",
        cached,
        surface_reads,
        geometry,
        transport,
    )
    print(json.dumps(recovered, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
