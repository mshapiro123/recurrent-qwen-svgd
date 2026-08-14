"""Build the CPU-only P3.4 controller autopsy and task stratification receipt.

The analysis reads landed DEV artifacts only. It does not load a model, mutate a
checkpoint, or touch CONFIRM/EVAL-E. Counterfactual loss shares hold the
observed combined group-clip geometry fixed because per-window gradient bundles
were not persisted; the receipt labels that limitation explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.stats import binom, norm

from training.paper2_phase3_p34 import classify_loss_shares


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPTS = ROOT / ".codex_p34_final_download"
DEFAULT_CAMPAIGN = ROOT / (
    "outputs/stage5/stage5_paper2_phase3_p34_analysis_20260813/summary.json"
)
DEFAULT_LOCK = ROOT / "training/paper2_phase3_p34_preregistration.json"
DEFAULT_PANEL = ROOT / (
    "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/"
    "p34_task_panel.jsonl"
)
DEFAULT_BASE_SCORES = ROOT / (
    "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/"
    "p34_panel_base_scores.jsonl"
)
DEFAULT_P31 = ROOT / ".codex_p31_reference/summary.json"
DEFAULT_OUTPUT = ROOT / (
    "outputs/stage5/stage5_paper2_phase3_p34_a2_autopsy_20260814"
)
DEFAULT_FIGURES = ROOT / "docs/figures"

MAIN_CONDITIONS = ("main_seed_0", "main_seed_1")
ALL_CONDITIONS = (*MAIN_CONDITIONS, "slot_seed_0")
LOCK_WEIGHT_KEYS = {
    "main_seed_0": "seed_0_main",
    "main_seed_1": "seed_1_main",
    "slot_seed_0": "seed_0_slot",
}
I1_SEEDS = {"main_seed_0": 0, "main_seed_1": 1, "slot_seed_0": 0}
BATTERIES = ("arc_challenge", "arc_easy", "gsm8k", "mbpp", "mmlu", "tier1")
DISPLAY = {
    "arc_challenge": "ARC-C",
    "arc_easy": "ARC-E",
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
    "mmlu": "MMLU",
    "tier1": "Tier-1",
}
CONTROLLER_GAIN = 0.5
CONTROLLER_LOG_CLIP = 0.5
POWER_ALPHA = 0.05
POWER_TARGET = 0.80


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalize_svg(path: Path) -> None:
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )


def reconstruct_window_rungs(summary: Mapping[str, Any], *, initial_rung: int = 1) -> list[int]:
    """Recover the rung that generated each share window, before any transition."""

    task_controller = {int(item["step"]): item["controller"] for item in summary["history"]}
    rung = int(initial_rung)
    output: list[int] = []
    for event in summary["share_contract_events"]:
        output.append(rung)
        if event.get("controller"):
            if int(event["controller"]["rung_before"]) != rung:
                raise AssertionError("share-controller rung chain is discontinuous")
            rung = int(event["controller"]["rung_after"])
        controller = task_controller.get(int(event["step"]))
        if controller is not None:
            if int(controller["rung_before"]) != rung:
                raise AssertionError("task-controller rung chain is discontinuous")
            rung = int(controller["rung_after"])
    return output


def inferred_unit_masses(
    shares: Mapping[str, float], weights: Mapping[str, float], names: Iterable[str]
) -> dict[str, float]:
    """Invert scalar weights while holding the observed clip scales fixed."""

    return {name: float(shares[name]) / float(weights[name]) for name in names}


def shares_for_weights(
    unit_masses: Mapping[str, float], weights: Mapping[str, float]
) -> dict[str, float]:
    mass = {name: float(unit_masses[name]) * float(weights[name]) for name in unit_masses}
    total = sum(mass.values())
    return {name: value / total for name, value in mass.items()}


def rung_targets(*, rung: int, names: Iterable[str], preserve_target: float) -> dict[str, float]:
    """Allocate non-preservation mass in the registered primary-floor ratios."""

    names = tuple(names)
    if not 0.0 <= preserve_target <= 0.25:
        raise ValueError("rung preservation target must stay inside its registered ceiling")
    primary = {"kl": 0.35, "aim": 0.15, "ce": 0.10, "gate": 0.03}
    if "slot" in names:
        primary["slot"] = 0.10
    scale = (1.0 - preserve_target) / sum(primary.values())
    targets = {name: value * scale for name, value in primary.items()}
    targets["preserve"] = preserve_target
    if set(targets) != set(names):
        raise AssertionError(f"rung {rung}: target names changed")
    return targets


def projected_log_weight_update(
    *,
    weights: Mapping[str, float],
    observed_shares: Mapping[str, float],
    target_shares: Mapping[str, float],
    gain: float = CONTROLLER_GAIN,
    max_abs_log_update: float = CONTROLLER_LOG_CLIP,
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply clipped proportional feedback in log-weight space."""

    names = tuple(weights)
    if set(observed_shares) != set(names) or set(target_shares) != set(names):
        raise ValueError("controller mappings must cover the same losses")
    deltas = {
        name: float(
            np.clip(
                gain
                * math.log(float(target_shares[name]) / max(float(observed_shares[name]), 1e-30)),
                -max_abs_log_update,
                max_abs_log_update,
            )
        )
        for name in names
    }
    updated = {name: float(weights[name]) * math.exp(deltas[name]) for name in names}
    anchor = updated["kl"]
    updated = {name: value / anchor for name, value in updated.items()}
    return updated, deltas


def maximum_miss_streak(classifications: Iterable[Mapping[str, Any]]) -> int:
    streak = maximum = 0
    for item in classifications:
        streak = streak + 1 if item["failed_contracts"] else 0
        maximum = max(maximum, streak)
    return maximum


def static_hard_floor_feasibility(
    windows: list[dict[str, float]], *, names: tuple[str, ...]
) -> dict[str, Any]:
    """Solve the registered inequalities under fixed observed clip geometry."""

    rows: list[np.ndarray] = []
    bounds: list[float] = []
    floor = {"kl": 0.35, "aim": 0.15, "ce": 0.10, "gate": 0.03, "slot": 0.10}
    for window in windows:
        unit = np.asarray([window[name] for name in names], dtype=np.float64)
        for index, name in enumerate(names):
            if name == "preserve":
                threshold = 0.25
                row = np.asarray([-threshold * value for value in unit])
                row[index] = (1.0 - threshold) * unit[index]
            else:
                threshold = floor[name]
                row = np.asarray([threshold * value for value in unit])
                row[index] = -(1.0 - threshold) * unit[index]
            rows.append(row)
            bounds.append(0.0)
    variable_bounds = [(1.0, 1.0) if name == "kl" else (1e-8, 1e8) for name in names]
    result = linprog(
        np.zeros(len(names)),
        A_ub=np.stack(rows),
        b_ub=np.asarray(bounds),
        bounds=variable_bounds,
        method="highs",
    )
    payload: dict[str, Any] = {
        "feasible_under_fixed_clip_geometry": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
    }
    if result.success:
        weights = {name: float(result.x[index]) for index, name in enumerate(names)}
        realized = [shares_for_weights(window, weights) for window in windows]
        payload.update(
            {
                "weights_kl_normalized": weights,
                "share_ranges": {
                    name: [
                        min(item[name] for item in realized),
                        max(item[name] for item in realized),
                    ]
                    for name in names
                },
                "preserve_weight_degenerate": weights["preserve"] <= 1e-6,
            }
        )
    return payload


def exact_one_sided_sign_power(*, rows: int, discordance: float, delta: float) -> float:
    """Power of a one-sided paired sign test under a binomial discordance model."""

    if not 0.0 < discordance <= 1.0 or not 0.0 <= delta <= discordance:
        raise ValueError("delta must lie between zero and the discordance rate")
    discordant = np.arange(rows + 1)
    discordant_mass = binom.pmf(discordant, rows, discordance)
    critical = binom.ppf(1.0 - POWER_ALPHA, discordant, 0.5) + 1.0
    fix_probability = (discordance + delta) / (2.0 * discordance)
    rejection = binom.sf(critical - 1.0, discordant, fix_probability)
    rejection[critical > discordant] = 0.0
    return float(np.dot(discordant_mass, rejection))


def minimum_effect_for_power(*, rows: int, discordance: float) -> dict[str, Any]:
    for delta_rows in range(1, rows + 1):
        delta = delta_rows / rows
        power = exact_one_sided_sign_power(rows=rows, discordance=discordance, delta=delta)
        if power >= POWER_TARGET:
            return {
                "minimum_net_rows": delta_rows,
                "minimum_accuracy_delta": delta,
                "minimum_accuracy_points": 100.0 * delta,
                "power": power,
                "alpha_one_sided": POWER_ALPHA,
                "power_target": POWER_TARGET,
            }
    raise AssertionError("power target was not reached")


def condition_rows(receipts: Path, condition: str) -> tuple[list[dict[str, Any]], int]:
    summary = read_json(receipts / condition / "outputs/summary.json")
    look = int(summary["history"][-1]["look"])
    path = receipts / condition / "private" / f"task_rows_look_{look:02d}.jsonl"
    return read_jsonl(path), look


def task_delta(rows: Iterable[Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    fixes = regressions = 0
    for row in rows:
        before = bool(reference[row["item_id"]]["augmented_correct"])
        after = bool(row["augmented_correct"])
        fixes += int(not before and after)
        regressions += int(before and not after)
    return {"fixes": fixes, "regressions": regressions, "net": fixes - regressions}


def build_task_diagnostic(receipts: Path, panel_path: Path) -> dict[str, Any]:
    panel = {row["item_id"]: row for row in read_jsonl(panel_path)}
    output: dict[str, Any] = {"conditions": {}, "tier1_shared_regressions": []}
    lost_sets: list[set[str]] = []
    for condition in ALL_CONDITIONS:
        rows, look = condition_rows(receipts, condition)
        seed = I1_SEEDS[condition]
        i1_path = ROOT / (
            "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/"
            f"task_calibration/seed_{seed}/seed_{seed}_i1_rows.jsonl"
        )
        i1 = {row["item_id"]: row for row in read_jsonl(i1_path)}
        by_battery = {}
        for battery in BATTERIES:
            subset = [row for row in rows if row["battery"] == battery]
            base = {
                "fixes": sum(not row["base_correct"] and row["augmented_correct"] for row in subset),
                "regressions": sum(row["base_correct"] and not row["augmented_correct"] for row in subset),
            }
            base["net"] = base["fixes"] - base["regressions"]
            by_battery[battery] = {
                "endpoint_minus_frozen_base": base,
                "endpoint_minus_i1_training_start": task_delta(subset, i1),
            }
        target_rows = [row for row in rows if row["panel_group"] == "target"]
        pooled = {
            "endpoint_minus_frozen_base": {
                "fixes": sum(not row["base_correct"] and row["augmented_correct"] for row in rows),
                "regressions": sum(row["base_correct"] and not row["augmented_correct"] for row in rows),
            },
            "endpoint_minus_i1_training_start": task_delta(rows, i1),
        }
        pooled["endpoint_minus_frozen_base"]["net"] = (
            pooled["endpoint_minus_frozen_base"]["fixes"]
            - pooled["endpoint_minus_frozen_base"]["regressions"]
        )
        target = {
            "endpoint_minus_frozen_base": {
                "fixes": sum(
                    not row["base_correct"] and row["augmented_correct"] for row in target_rows
                ),
                "regressions": sum(
                    row["base_correct"] and not row["augmented_correct"] for row in target_rows
                ),
            },
            "endpoint_minus_i1_training_start": task_delta(target_rows, i1),
        }
        target["endpoint_minus_frozen_base"]["net"] = (
            target["endpoint_minus_frozen_base"]["fixes"]
            - target["endpoint_minus_frozen_base"]["regressions"]
        )
        tier1_lost = {
            row["item_id"]
            for row in rows
            if row["battery"] == "tier1" and row["base_correct"] and not row["augmented_correct"]
        }
        lost_sets.append(tier1_lost)
        output["conditions"][condition] = {
            "look": look,
            "by_battery": by_battery,
            "pooled": pooled,
            "target_half": target,
            "tier1_regressions_vs_base": sorted(tier1_lost),
            "tier1_regressions_already_present_at_i1": sorted(
                item_id for item_id in tier1_lost if not i1[item_id]["augmented_correct"]
            ),
            "source_i1": file_receipt(i1_path),
        }
    shared = set.intersection(*lost_sets)
    output["tier1_shared_regressions"] = [
        {
            "item_id": item_id,
            "prompt": panel[item_id]["prompt"],
            "answer": panel[item_id]["answer"],
            "inherited_from_i1_all_conditions": all(
                item_id
                in output["conditions"][condition]["tier1_regressions_already_present_at_i1"]
                for condition in ALL_CONDITIONS
            ),
        }
        for item_id in sorted(shared)
    ]
    output["gate_telemetry"] = {
        "available_in_cached_task_rows": False,
        "reason": (
            "task_rows contain correctness, predictions, generated text, and option scores, "
            "but omit position_gate and realized_writeback_ratio"
        ),
        "gate_by_battery_answered": False,
        "gsm8k_regression_gate_colocation_answered": False,
        "required_followup": (
            "add score-preserving telemetry to the resumed DEV task pass; do not use it for "
            "checkpoint selection"
        ),
    }
    return output


def controller_autopsy(receipts: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    locked_weights = lock["loss_share_contract"]["scalar_weights_by_seed"]
    conditions: dict[str, Any] = {}
    preserve_by_rung_all: dict[int, list[float]] = defaultdict(list)
    preserve_by_rung_main: dict[int, list[float]] = defaultdict(list)
    prepared: dict[str, Any] = {}
    for condition in ALL_CONDITIONS:
        summary = read_json(receipts / condition / "outputs/summary.json")
        rungs = reconstruct_window_rungs(summary)
        key = LOCK_WEIGHT_KEYS[condition]
        weights = {name: float(value) for name, value in locked_weights[key].items()}
        windows = []
        for event, rung in zip(summary["share_contract_events"], rungs):
            shares = {name: float(value) for name, value in event["read"]["shares"].items()}
            preserve_by_rung_all[rung].append(shares["preserve"])
            if condition in MAIN_CONDITIONS:
                preserve_by_rung_main[rung].append(shares["preserve"])
            windows.append(
                {
                    "step": int(event["step"]),
                    "rung": rung,
                    "observed_shares": shares,
                    "unit_masses_fixed_clip": inferred_unit_masses(shares, weights, weights),
                }
            )
        prepared[condition] = {"summary": summary, "weights": weights, "windows": windows}
    preserve_targets = {
        rung: min(0.25, float(np.median(values)))
        for rung, values in sorted(preserve_by_rung_main.items())
    }
    all_condition_medians = {
        rung: min(0.25, float(np.median(values)))
        for rung, values in sorted(preserve_by_rung_all.items())
    }
    main_names = tuple(prepared[MAIN_CONDITIONS[0]]["weights"])
    main_joint_by_rung: dict[str, Any] = {}
    for rung in sorted(preserve_targets):
        joint_windows = [
            window
            for condition in MAIN_CONDITIONS
            for window in prepared[condition]["windows"]
            if window["rung"] == rung
        ]
        unit_windows = [window["unit_masses_fixed_clip"] for window in joint_windows]
        target = rung_targets(
            rung=rung,
            names=main_names,
            preserve_target=preserve_targets[rung],
        )
        geometric = {
            name: float(
                math.exp(
                    np.mean([math.log(max(window[name], 1e-30)) for window in unit_windows])
                )
            )
            for name in main_names
        }
        target_weights = {name: target[name] / geometric[name] for name in main_names}
        anchor = target_weights["kl"]
        target_weights = {name: value / anchor for name, value in target_weights.items()}
        target_reads = [shares_for_weights(window, target_weights) for window in unit_windows]
        main_joint_by_rung[str(rung)] = {
            "condition_steps": [
                {"condition": condition, "step": window["step"]}
                for condition in MAIN_CONDITIONS
                for window in prepared[condition]["windows"]
                if window["rung"] == rung
            ],
            "preservation_target": preserve_targets[rung],
            "target_shares": target,
            "geometric_mean_target_fit_weights": target_weights,
            "target_fit_passing_windows": sum(
                not classify_loss_shares(read)["failed_contracts"] for read in target_reads
            ),
            "window_count": len(unit_windows),
            "hard_floor_feasibility": static_hard_floor_feasibility(
                unit_windows, names=main_names
            ),
        }
    for condition, item in prepared.items():
        names = tuple(item["weights"])
        by_rung: dict[str, Any] = {}
        for rung in sorted({window["rung"] for window in item["windows"]}):
            unit_windows = [
                window["unit_masses_fixed_clip"]
                for window in item["windows"]
                if window["rung"] == rung
            ]
            target = rung_targets(
                rung=rung,
                names=names,
                preserve_target=preserve_targets[rung],
            )
            geometric = {
                name: float(
                    math.exp(
                        np.mean([math.log(max(window[name], 1e-30)) for window in unit_windows])
                    )
                )
                for name in names
            }
            target_weights = {name: target[name] / geometric[name] for name in names}
            anchor = target_weights["kl"]
            target_weights = {name: value / anchor for name, value in target_weights.items()}
            target_reads = [shares_for_weights(window, target_weights) for window in unit_windows]
            by_rung[str(rung)] = {
                "steps": [
                    window["step"] for window in item["windows"] if window["rung"] == rung
                ],
                "preservation_target": preserve_targets[rung],
                "target_shares": target,
                "geometric_mean_target_fit_weights": target_weights,
                "target_fit_passing_windows": sum(
                    not classify_loss_shares(read)["failed_contracts"] for read in target_reads
                ),
                "window_count": len(unit_windows),
                "hard_floor_feasibility": static_hard_floor_feasibility(
                    unit_windows, names=names
                ),
            }
        current = dict(item["weights"])
        replay = []
        classifications = []
        for window in item["windows"]:
            shares = shares_for_weights(window["unit_masses_fixed_clip"], current)
            classification = classify_loss_shares(shares)
            target = rung_targets(
                rung=window["rung"],
                names=names,
                preserve_target=preserve_targets[window["rung"]],
            )
            updated, log_delta = projected_log_weight_update(
                weights=current,
                observed_shares=shares,
                target_shares=target,
            )
            counterfactual_next = shares_for_weights(window["unit_masses_fixed_clip"], updated)
            replay.append(
                {
                    "step": window["step"],
                    "rung": window["rung"],
                    "shares_before_update": shares,
                    "classification_before_update": classification,
                    "log_weight_delta": log_delta,
                    "weights_after_update": updated,
                    "same_window_counterfactual_shares_after_update": counterfactual_next,
                }
            )
            classifications.append(classification)
            current = updated
        history = item["summary"]["history"]
        healthy = [
            int(row["step"])
            for row in history
            if not row["trailing_shares"]["failed_contracts"]
        ]
        latest = max(healthy) if healthy else None
        checkpoint = (
            receipts / condition / "private" / f"checkpoint_step_{latest:04d}.pt"
            if latest is not None
            else None
        )
        conditions[condition] = {
            "fixed_locked_weights": item["weights"],
            "windows": item["windows"],
            "by_rung": by_rung,
            "dynamic_replay": {
                "gain": CONTROLLER_GAIN,
                "max_abs_log_update": CONTROLLER_LOG_CLIP,
                "maximum_consecutive_misses": maximum_miss_streak(classifications),
                "would_reach_four_miss_stop_under_fixed_clip_replay": (
                    maximum_miss_streak(classifications) >= 4
                ),
                "passing_windows": sum(not row["failed_contracts"] for row in classifications),
                "window_count": len(classifications),
                "events": replay,
            },
            "last_healthy_evaluation": {
                "step": latest,
                "checkpoint": file_receipt(checkpoint) if checkpoint is not None else None,
                "continuation_eligible": condition in MAIN_CONDITIONS and checkpoint is not None,
                "disposition": (
                    "resume_from_pinned_last_healthy"
                    if condition in MAIN_CONDITIONS and checkpoint is not None
                    else "shelved_restart_if_reopened"
                ),
            },
        }
    return {
        "estimator": (
            "local counterfactual shares from observed post-clip shares divided by the active "
            "scalar weights; combined group-clip scales held fixed"
        ),
        "exact_reclip_available": False,
        "exact_reclip_blocker": "per-window gradient bundles were not persisted",
        "preservation_targets_by_rung": {str(k): v for k, v in preserve_targets.items()},
        "descriptive_all_condition_preservation_medians_by_rung": {
            str(k): v for k, v in all_condition_medians.items()
        },
        "main_joint_static_analysis_by_rung": main_joint_by_rung,
        "controller_candidate": {
            "formula": "delta_log_w_i = clip(0.5 * log(target_share_i / observed_share_i), -0.5, 0.5)",
            "cadence": "non-overlapping 100-step boundaries only",
            "normalization": "KL weight normalized to 1 after every update",
            "counterfactual_logged_before_application": True,
            "rung_changes_reserved_for_task_and_causal_controller": True,
        },
        "conditions": conditions,
    }


def confirmation_power(
    *, receipts: Path, task: Mapping[str, Any], p31: Mapping[str, Any]
) -> dict[str, Any]:
    seals = p31["confirm_seals"]["seals"]
    counts = {row["battery"]: int(row["row_count"]) for row in seals}
    pooled_rows = sum(counts.values())
    target_rows = sum(counts[name] for name in ("gsm8k", "mbpp", "arc_challenge"))
    output: dict[str, Any] = {"confirm_rows": {"pooled": pooled_rows, "target": target_rows}}
    for group, rows in (("pooled", pooled_rows), ("target_half", target_rows)):
        key = "pooled" if group == "pooled" else "target_half"
        discordance = []
        deltas = []
        vectors = []
        for condition in MAIN_CONDITIONS:
            endpoint_rows, _ = condition_rows(receipts, condition)
            selected = endpoint_rows if group == "pooled" else [
                row for row in endpoint_rows if row["panel_group"] == "target"
            ]
            vector = np.asarray(
                [int(row["augmented_correct"]) - int(row["base_correct"]) for row in selected],
                dtype=np.float64,
            )
            vectors.append(vector)
            discordance.append(float(np.count_nonzero(vector) / len(vector)))
            deltas.append(float(vector.mean()))
        q = float(np.mean(discordance))
        single = minimum_effect_for_power(rows=rows, discordance=q)
        correlation = float(np.corrcoef(vectors[0], vectors[1])[0, 1])
        averaged = (vectors[0] + vectors[1]) / 2.0
        joint_delta = (
            (norm.ppf(1.0 - POWER_ALPHA) + norm.ppf(POWER_TARGET))
            * math.sqrt(float(averaged.var(ddof=1)) / rows)
        )
        dev_denominator = 292 if group == "pooled" else 157
        dev_rows = 1024 if group == "pooled" else 512
        projected_teacher_gap = rows * dev_denominator / dev_rows
        single["projected_gap_closed"] = single["minimum_net_rows"] / projected_teacher_gap
        output[key] = {
            "dev_discordance_by_seed": discordance,
            "dev_endpoint_delta_by_seed": deltas,
            "mean_discordance": q,
            "single_seed_exact_sign_test": single,
            "two_seed_joint_normal_approximation": {
                "row_delta_correlation": correlation,
                "minimum_accuracy_delta": joint_delta,
                "minimum_accuracy_points": 100.0 * joint_delta,
                "minimum_average_net_rows": int(math.ceil(joint_delta * rows)),
                "warning": (
                    "planning approximation only; the P3.6 lock must pin a cluster-level "
                    "joint estimator and simulation before spending CONFIRM"
                ),
            },
            "power_at_ceiling_consistent_effects": {
                str(delta): exact_one_sided_sign_power(rows=rows, discordance=q, delta=delta)
                for delta in (0.006, 0.008, 0.011)
            },
        }
    output["confirm_unspent"] = not bool(p31["confirm_seals"]["confirm_scoring_spent"])
    output["reading"] = (
        "The sealed panel is underpowered for the observed 0.6-1.1 point effect in a "
        "single-seed paired test; high cross-seed row-delta correlation means naive seed "
        "pooling adds little power."
    )
    return output


def write_figure(summary: Mapping[str, Any], figures: Path) -> list[Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    colors = {"main_seed_0": "#2878B5", "main_seed_1": "#D95F02", "slot_seed_0": "#708238"}
    for condition in ALL_CONDITIONS:
        actual = summary["controller_autopsy"]["conditions"][condition]["windows"]
        replay = summary["controller_autopsy"]["conditions"][condition]["dynamic_replay"]["events"]
        x = [row["step"] for row in actual]
        actual_failures = []
        replay_failures = []
        for row, counterfactual in zip(actual, replay):
            actual_read = classify_loss_shares(row["observed_shares"])
            replay_read = counterfactual["classification_before_update"]
            actual_failures.append(len(actual_read["failed_contracts"]))
            replay_failures.append(len(replay_read["failed_contracts"]))
        axes[0].plot(x, actual_failures, color=colors[condition], alpha=0.35,
                     linestyle="--")
        axes[0].plot(x, replay_failures, color=colors[condition], marker="o", markersize=2.5,
                     label=condition.replace("_", " "))
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].set(title="Actual versus replayed contract breaches", xlabel="Optimizer step",
                ylabel="Count of failed share bounds")
    axes[0].text(0.02, 0.96, "solid: dynamic replay\ndashed: landed fixed weights",
                 transform=axes[0].transAxes, va="top", fontsize=8, color="#444444")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    x = np.arange(len(BATTERIES))
    width = 0.18
    for index, condition in enumerate(MAIN_CONDITIONS):
        data = summary["task_diagnostic"]["conditions"][condition]["by_battery"]
        base = [data[b]["endpoint_minus_frozen_base"]["net"] for b in BATTERIES]
        train = [data[b]["endpoint_minus_i1_training_start"]["net"] for b in BATTERIES]
        offset = (-1.5 + index * 2.0) * width
        axes[1].bar(x + offset, base, width, color=colors[condition], alpha=0.35,
                    label=f"seed {index}: endpoint - base")
        axes[1].bar(x + offset + width, train, width, color=colors[condition],
                    label=f"seed {index}: endpoint - i1")
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_xticks(x, [DISPLAY[name] for name in BATTERIES], rotation=20)
    axes[1].set(title="Battery attribution changes after rebasing", ylabel="Net correct rows")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("P3.4 a2 CPU autopsy: controller replay and training-start attribution",
                 fontsize=13, fontweight="bold")
    figures.mkdir(parents=True, exist_ok=True)
    stem = figures / "p34_a2_controller_and_task_autopsy_20260814"
    paths = [stem.with_suffix(".svg"), stem.with_suffix(".png")]
    fig.savefig(paths[0], metadata={"Date": "2026-08-14"})
    normalize_svg(paths[0])
    fig.savefig(paths[1], dpi=190, metadata={"Creation Time": "2026-08-14"})
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--p31", type=Path, default=DEFAULT_P31)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = read_json(args.lock)
    campaign = read_json(args.campaign)
    p31 = read_json(args.p31)
    task = build_task_diagnostic(args.receipts, args.panel)
    summary: dict[str, Any] = {
        "kind": "paper2_phase3_p34_a2_cpu_autopsy_v1",
        "status": "complete_dev_only_no_model_no_optimizer",
        "analysis_date": "2026-08-14",
        "scope": "cached P3.4 DEV receipts only; CONFIRM and EVAL-E untouched",
        "controller_autopsy": controller_autopsy(args.receipts, lock),
        "task_diagnostic": task,
        "confirmation_power": confirmation_power(receipts=args.receipts, task=task, p31=p31),
        "integrity": {
            "campaign_verdict": campaign["verdict"],
            "confirm_scored": False,
            "eval_e_scored": False,
            "model_loaded": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
        },
        "source_receipts": {
            "campaign": file_receipt(args.campaign),
            "lock": file_receipt(args.lock),
            "panel": file_receipt(args.panel),
            "p31": file_receipt(args.p31),
        },
    }
    figure_paths = write_figure(summary, args.figures)
    summary["figure_receipts"] = [file_receipt(path) for path in figure_paths]
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": file_receipt(output_path), "figures": summary["figure_receipts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
