"""Build the CPU-only P3.4 directional-stability diagnostic.

The analysis is deliberately score-only and receipt-only. It reconstructs the
registered training lottery from the saved RNG contract, verifies every batch
hash, joins all twenty DEV looks to controller telemetry, and measures exact
row churn only where private row receipts survived transport. It never loads a
model, scores CONFIRM/EVAL-E, or treats score-curve smoothing as a weight EMA.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / ".codex_p34_final_download"
DEFAULT_PREFLIGHT = ROOT / "stage5_paper2_phase3_retention_preflight_20260811.zip"
DEFAULT_OUTPUT = ROOT / (
    "outputs/stage5/stage5_paper2_phase3_p34_directional_stability_20260815"
)
DEFAULT_FIGURES = ROOT / "docs/figures"

ROWS = 1024
LOOKS = 20
STEPS_PER_LOOK = 200
POSITIVE_PER_BATCH = 32
NEGATIVE_PER_BATCH = 96
GATE_CEILINGS = (0.02, 0.08, 0.20, 0.50)
STAGED_MEMBER = "drive/private/p33_prep/p33_staged_labels.jsonl"
GROUP_ORDER = ("evaluation_surface", "curriculum", "controller", "phase_proxy")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_svg(path: Path) -> None:
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )


def tar_json(path: Path, member: str) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        handle = archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError(f"{member} is absent from {path}")
        return json.loads(handle.read().decode("utf-8"))


def tar_task_summaries(path: Path) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.name.startswith("output/task_summary_look_"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            payload = json.loads(handle.read().decode("utf-8"))
            output[int(payload["look"])] = payload
    return output


def staged_records(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        lines = archive.read(STAGED_MEMBER).decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def outcome(row: Mapping[str, Any]) -> str:
    base = bool(row["base_correct"])
    augmented = bool(row["augmented_correct"])
    if augmented and not base:
        return "fix"
    if base and not augmented:
        return "regression"
    if base:
        return "stable_correct"
    return "stable_wrong"


def outcome_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(outcome(row) for row in rows)
    return {
        "fix": int(counts["fix"]),
        "regression": int(counts["regression"]),
        "unchanged": int(counts["stable_correct"] + counts["stable_wrong"]),
    }


def option_margin(row: Mapping[str, Any]) -> float | None:
    scores = sorted((float(value) for value in row.get("option_scores", {}).values()), reverse=True)
    return scores[0] - scores[1] if len(scores) >= 2 else None


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def adjacent_row_churn(
    prior: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    before = {str(row["item_id"]): row for row in prior}
    after = {str(row["item_id"]): row for row in current}
    if set(before) != set(after):
        raise RuntimeError("adjacent P3.4 row receipts do not cover the same panel")
    before_outcome = {item: outcome(row) for item, row in before.items()}
    after_outcome = {item: outcome(row) for item, row in after.items()}
    changed = {item for item in before if before_outcome[item] != after_outcome[item]}
    swaps = {
        item
        for item in changed
        if {before_outcome[item], after_outcome[item]} == {"fix", "regression"}
    }
    fix_before = {item for item, value in before_outcome.items() if value == "fix"}
    fix_after = {item for item, value in after_outcome.items() if value == "fix"}
    regression_before = {
        item for item, value in before_outcome.items() if value == "regression"
    }
    regression_after = {
        item for item, value in after_outcome.items() if value == "regression"
    }
    battery = defaultdict(lambda: {"rows": 0, "changed": 0})
    changed_margins: list[float] = []
    stable_margins: list[float] = []
    for item, row in after.items():
        key = str(row["battery"])
        battery[key]["rows"] += 1
        is_changed = item in changed
        battery[key]["changed"] += int(is_changed)
        margin = option_margin(row)
        if margin is not None:
            (changed_margins if is_changed else stable_margins).append(margin)
    return {
        "prior_look": int(prior[0]["look"]),
        "current_look": int(current[0]["look"]),
        "rows": len(before),
        "outcome_changed_rows": len(changed),
        "outcome_changed_fraction": len(changed) / len(before),
        "fix_set_jaccard": jaccard(fix_before, fix_after),
        "regression_set_jaccard": jaccard(regression_before, regression_after),
        "direct_fix_regression_swaps": len(swaps),
        "battery_change_rates": {
            key: {
                **value,
                "changed_fraction": value["changed"] / value["rows"],
            }
            for key, value in sorted(battery.items())
        },
        "option_margin": {
            "changed_mean": float(np.mean(changed_margins)) if changed_margins else None,
            "stable_mean": float(np.mean(stable_margins)) if stable_margins else None,
            "scope": "augmented option-score top1-minus-top2 margin",
        },
    }


def row_receipts_for_inherited_segment(source_dir: Path, cutoff_look: int) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for look in range(1, cutoff_look + 1):
        path = source_dir / "private" / f"task_rows_look_{look:02d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        output[look] = read_jsonl(path)
    return output


def summary_outcomes(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]] | None
) -> dict[str, int]:
    telemetry = summary.get("score_preserving_telemetry") or {}
    grouped = telemetry.get("by_battery_and_outcome") or {}
    if "outcome:fix" in grouped and "outcome:regression" in grouped:
        fixes = int(grouped["outcome:fix"]["rows"])
        regressions = int(grouped["outcome:regression"]["rows"])
        return {"fix": fixes, "regression": regressions, "unchanged": ROWS - fixes - regressions}
    if rows is None:
        raise RuntimeError(f"look {summary['look']} lacks both outcome telemetry and rows")
    return outcome_counts(rows)


def scored_rung(summary: Mapping[str, Any]) -> int:
    candidates = [
        item
        for item in summary.get("checkpoint_receipts", [])
        if item.get("label") == "p34"
    ]
    if not candidates or "controller_rung" not in candidates[-1]:
        raise RuntimeError(f"look {summary['look']} lacks scored controller rung")
    return int(candidates[-1]["controller_rung"])


def load_seed_material(
    *, seed: int, source_root: Path, final_receipts: Path
) -> dict[str, Any]:
    condition = f"main_seed_{seed}"
    source_dir = source_root / condition
    source_summary = read_json(source_dir / "outputs" / "summary.json")
    final_summary = tar_json(final_receipts, "output/summary.json")
    cutoff = int(final_summary["continuation"]["source_step"]) // STEPS_PER_LOOK
    if cutoff not in (2, 5):
        raise RuntimeError(f"unexpected P3.4 inherited cutoff for seed {seed}: {cutoff}")
    source_tasks = {
        look: read_json(source_dir / "outputs" / f"task_summary_look_{look:02d}.json")
        for look in range(1, cutoff + 1)
    }
    final_tasks = tar_task_summaries(final_receipts)
    tasks = {**source_tasks, **{look: value for look, value in final_tasks.items() if look > cutoff}}
    if set(tasks) != set(range(1, LOOKS + 1)):
        raise RuntimeError(f"seed {seed} task look coverage changed: {sorted(tasks)}")
    panel_shas = {str(value["panel_sha256"]) for value in tasks.values()}
    if len(panel_shas) != 1:
        raise RuntimeError(f"seed {seed} task panel changed across looks")
    inherited_rows = row_receipts_for_inherited_segment(source_dir, cutoff)
    history = {int(item["look"]): item for item in source_summary.get("history", [])}
    history.update({int(item["look"]): item for item in final_summary.get("history", [])})
    return {
        "seed": seed,
        "cutoff_look": cutoff,
        "tasks": tasks,
        "inherited_rows": inherited_rows,
        "source_summary": source_summary,
        "final_summary": final_summary,
        "history": history,
        "panel_sha256": panel_shas.pop(),
    }


def reconstruct_schedule(
    *, seed: int, resume_path: Path, preflight_zip: Path, expected_summary_sha: str
) -> list[dict[str, Any]]:
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    hashes = [str(value) for value in payload["schedule_hashes"]]
    if len(hashes) != LOOKS * STEPS_PER_LOOK:
        raise RuntimeError(f"seed {seed} schedule has {len(hashes)} steps")
    digest = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    if digest != expected_summary_sha:
        raise RuntimeError(f"seed {seed} schedule receipt digest mismatch")
    records = staged_records(preflight_zip)
    positives = [row for row in records if int(row["gate_label"]) == 1]
    negatives = [row for row in records if int(row["gate_label"]) == 0]
    if (len(positives), len(negatives)) != (34_521, 103_563):
        raise RuntimeError("P3.4 staged population counts changed")
    generator = torch.Generator().manual_seed(20260813 + seed)
    segments: list[dict[str, Any]] = []
    accumulator: dict[str, Any] = {}

    def reset() -> dict[str, Any]:
        return {
            "depth": Counter(),
            "stratum": Counter(),
            "source": Counter(),
            "horizon": Counter(),
            "teachability_sum": 0.0,
            "cross_scale": 0,
            "flip_candidate": 0,
            "rows": 0,
        }

    accumulator = reset()
    for index, expected_hash in enumerate(hashes):
        pos = torch.randint(len(positives), (POSITIVE_PER_BATCH,), generator=generator)
        neg = torch.randint(len(negatives), (NEGATIVE_PER_BATCH,), generator=generator)
        batch = [positives[int(value)] for value in pos]
        batch.extend(negatives[int(value)] for value in neg)
        permutation = torch.randperm(len(batch), generator=generator).tolist()
        batch = [batch[value] for value in permutation]
        observed_hash = hashlib.sha256(
            "\n".join(str(row["record_id"]) for row in batch).encode()
        ).hexdigest()
        if observed_hash != expected_hash:
            raise RuntimeError(f"seed {seed} schedule diverged at step {index + 1}")
        weights = torch.arange(1, 5, dtype=torch.float64)
        depth = int(torch.multinomial(weights, 1, generator=generator).item()) + 1
        accumulator["depth"][depth] += 1
        for row in batch:
            accumulator["stratum"][str(row["stratum"])] += 1
            accumulator["source"][str(row["source"])] += 1
            accumulator["horizon"][int(row["horizon"])] += 1
            accumulator["teachability_sum"] += float(row["teachability"])
            accumulator["cross_scale"] += int(bool(row["cross_scale_consistent"]))
            accumulator["flip_candidate"] += int(bool(row["flip_candidate_14b"]))
            accumulator["rows"] += 1
        if (index + 1) % STEPS_PER_LOOK == 0:
            look = (index + 1) // STEPS_PER_LOOK
            depth_total = sum(accumulator["depth"].values())
            row_total = int(accumulator["rows"])
            segments.append(
                {
                    "look": look,
                    "step_start": index + 2 - STEPS_PER_LOOK,
                    "step_end": index + 1,
                    "depth_counts": {
                        str(value): int(accumulator["depth"][value]) for value in range(1, 5)
                    },
                    "depth_proportions": {
                        str(value): accumulator["depth"][value] / depth_total
                        for value in range(1, 5)
                    },
                    "mean_depth": sum(
                        value * accumulator["depth"][value] for value in range(1, 5)
                    )
                    / depth_total,
                    "code_fraction": accumulator["stratum"]["code"] / row_total,
                    "new_source_fraction": accumulator["source"]["new"] / row_total,
                    "mean_horizon": sum(
                        value * count for value, count in accumulator["horizon"].items()
                    )
                    / row_total,
                    "mean_teachability": accumulator["teachability_sum"] / row_total,
                    "cross_scale_fraction": accumulator["cross_scale"] / row_total,
                    "flip_candidate_fraction": accumulator["flip_candidate"] / row_total,
                }
            )
            accumulator = reset()
    return segments


def build_seed_series(material: Mapping[str, Any], curriculum: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    curriculum_by_look = {int(item["look"]): item for item in curriculum}
    events = list(material["final_summary"]["share_contract_events"])
    rows_by_look = material["inherited_rows"]
    output: list[dict[str, Any]] = []
    prior_net: int | None = None
    prior_rung: int | None = None
    prior_post_score_transition = False
    for look in range(1, LOOKS + 1):
        task = material["tasks"][look]
        rows = rows_by_look.get(look)
        counts = summary_outcomes(task, rows)
        augmented = int(round(float(task["augmented_accuracy"]) * ROWS))
        base = int(round(float(task["base_accuracy"]) * ROWS))
        net = augmented - base
        rung = scored_rung(task)
        segment_events = [
            event
            for event in events
            if (look - 1) * STEPS_PER_LOOK < int(event["step"]) <= look * STEPS_PER_LOOK
        ]
        updates = [
            float(event["objective_controller"]["maximum_absolute_log_update"])
            for event in segment_events
            if event.get("objective_controller")
        ]
        history = material["history"].get(look, {})
        post_read_controller = history.get("controller") or {}
        post_read_reason = str(post_read_controller.get("reason", ""))
        post_read_transition = (
            post_read_controller.get("rung_before") != post_read_controller.get("rung_after")
        )
        transition_since_prior_score = bool(
            prior_post_score_transition
            or any(event.get("controller") is not None for event in segment_events)
        )
        output.append(
            {
                "seed": int(material["seed"]),
                "look": look,
                "step": look * STEPS_PER_LOOK,
                "base_correct": base,
                "augmented_correct": augmented,
                "net_correct": net,
                "signed_swing": None if prior_net is None else net - prior_net,
                "absolute_swing": None if prior_net is None else abs(net - prior_net),
                "fixes": counts["fix"],
                "regressions": counts["regression"],
                "discordant_rows": counts["fix"] + counts["regression"],
                "discordant_fraction": (counts["fix"] + counts["regression"]) / ROWS,
                "scored_rung": rung,
                "scored_gate_ceiling": GATE_CEILINGS[rung],
                "rung_changed_since_prior_score": (
                    False if prior_rung is None else rung != prior_rung
                ),
                "share_demotions_in_segment": sum(
                    int(event.get("controller") is not None) for event in segment_events
                ),
                "training_controller_transition_since_prior_score": (
                    False if prior_net is None else transition_since_prior_score
                ),
                "maximum_absolute_log_weight_update": max(updates) if updates else 0.0,
                "mean_absolute_log_weight_update": float(np.mean(updates)) if updates else 0.0,
                "post_read_controller": post_read_controller,
                "post_read_controller_transition": post_read_transition,
                "post_read_controller_reason": post_read_reason,
                "curriculum": dict(curriculum_by_look[look]),
            }
        )
        prior_net = net
        prior_rung = rung
        prior_post_score_transition = post_read_transition
    return output


def safe_correlation(left: Sequence[float], right: Sequence[float]) -> dict[str, float | None]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    if not all(math.isfinite(float(value)) for value in (pr.statistic, pr.pvalue, sr.statistic, sr.pvalue)):
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    return {
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def design_matrix(rows: Sequence[Mapping[str, Any]], groups: Iterable[str]) -> np.ndarray:
    selected = set(groups)
    columns: list[np.ndarray] = [np.ones(len(rows), dtype=np.float64)]
    columns.append(np.asarray([float(row["seed"]) for row in rows], dtype=np.float64))
    values: dict[str, list[np.ndarray]] = {
        "evaluation_surface": [
            np.asarray([float(row["discordant_fraction"]) for row in rows])
        ],
        "curriculum": [
            np.asarray([float(row["curriculum"]["mean_depth"]) for row in rows]),
            np.asarray([float(row["curriculum"]["code_fraction"]) for row in rows]),
        ],
        "controller": [
            np.asarray(
                [float(row["training_controller_transition_since_prior_score"]) for row in rows]
            ),
            np.asarray([float(row["maximum_absolute_log_weight_update"]) for row in rows]),
        ],
        "phase_proxy": [
            np.asarray([float(row["look"]) / LOOKS for row in rows])
        ],
    }
    for group in GROUP_ORDER:
        if group not in selected:
            continue
        for column in values[group]:
            standard = float(np.std(column))
            if standard > 0.0:
                columns.append((column - float(np.mean(column))) / standard)
    return np.column_stack(columns)


def in_sample_r2(y: np.ndarray, x: np.ndarray) -> float:
    fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    denominator = float(np.sum((y - float(np.mean(y))) ** 2))
    if denominator == 0.0:
        return 0.0
    return float(1.0 - np.sum((y - fitted) ** 2) / denominator)


def shapley_r2(rows: Sequence[Mapping[str, Any]], response: str) -> dict[str, Any]:
    y = np.asarray([float(row[response]) for row in rows], dtype=np.float64)
    cache: dict[frozenset[str], float] = {}

    def score(groups: Iterable[str]) -> float:
        key = frozenset(groups)
        if key not in cache:
            cache[key] = in_sample_r2(y, design_matrix(rows, key))
        return cache[key]

    contributions = {name: 0.0 for name in GROUP_ORDER}
    permutations = list(itertools.permutations(GROUP_ORDER))
    for permutation in permutations:
        active: set[str] = set()
        prior = score(active)
        for group in permutation:
            active.add(group)
            current = score(active)
            contributions[group] += current - prior
            prior = current
    contributions = {
        key: value / len(permutations) for key, value in contributions.items()
    }
    base = score(())
    full = score(GROUP_ORDER)
    return {
        "response": response,
        "rows": len(rows),
        "base_r2_seed_indicator_only": base,
        "full_in_sample_r2": full,
        "incremental_r2": full - base,
        "shapley_incremental_r2": contributions,
        "scope": (
            "descriptive pooled linear decomposition; evaluation uses discordant-row "
            "volume, phase is a constant-LR training-time proxy, and values are not "
            "causal shares"
        ),
    }


def score_curve_smoothing(values: Sequence[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, Any] = {
        "raw_endpoint": int(values[-1]),
        "late_3_mean": float(np.mean(array[-3:])),
        "late_5_mean": float(np.mean(array[-5:])),
        "late_10_mean": float(np.mean(array[-10:])),
    }
    for beta in (0.8, 0.9, 0.95):
        state = float(array[0])
        for value in array[1:]:
            state = beta * state + (1.0 - beta) * float(value)
        result[f"score_ema_beta_{str(beta).replace('.', 'p')}"] = state
    result["scope"] = (
        "score-curve telemetry only; this is not a weight EMA and cannot estimate an "
        "EMA checkpoint's accuracy"
    )
    return result


def correlations(intervals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    response = [float(row["signed_swing"]) for row in intervals]
    absolute = [float(row["absolute_swing"]) for row in intervals]
    return {
        "signed_swing_vs_mean_depth": safe_correlation(
            response, [float(row["curriculum"]["mean_depth"]) for row in intervals]
        ),
        "signed_swing_vs_code_fraction": safe_correlation(
            response, [float(row["curriculum"]["code_fraction"]) for row in intervals]
        ),
        "absolute_swing_vs_discordant_fraction": safe_correlation(
            absolute, [float(row["discordant_fraction"]) for row in intervals]
        ),
        "absolute_swing_vs_objective_update": safe_correlation(
            absolute,
            [float(row["maximum_absolute_log_weight_update"]) for row in intervals],
        ),
        "signed_swing_vs_gate_ceiling": safe_correlation(
            response, [float(row["scored_gate_ceiling"]) for row in intervals]
        ),
    }


def build_figure(receipt: Mapping[str, Any], png: Path, svg: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    colors = {0: "#1976a3", 1: "#c64b39"}
    for seed_payload in receipt["seeds"]:
        seed = int(seed_payload["seed"])
        series = seed_payload["series"]
        looks = [row["look"] for row in series]
        net = [row["net_correct"] for row in series]
        axes[0, 0].plot(looks, net, marker="o", linewidth=1.8, markersize=3.5,
                        color=colors[seed], label=f"seed {seed}")
        axes[0, 1].plot(looks, [row["fixes"] for row in series], linewidth=1.8,
                        color=colors[seed], label=f"seed {seed} fixes")
        axes[0, 1].plot(looks, [row["regressions"] for row in series], linewidth=1.4,
                        linestyle="--", color=colors[seed], label=f"seed {seed} regressions")
        axes[1, 0].scatter(
            [row["curriculum"]["mean_depth"] for row in series[1:]],
            [row["signed_swing"] for row in series[1:]],
            s=28,
            alpha=0.75,
            color=colors[seed],
            label=f"seed {seed}",
        )
    axes[0, 0].axhline(10, color="#2f6f44", linestyle=":", linewidth=1.5,
                       label="Trigger B (+10 mean rows)")
    axes[0, 0].set(title="A. DEV net gain remains a moving endpoint", xlabel="look",
                   ylabel="augmented minus base correct")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].set(title="B. Net effect is a difference of larger flows", xlabel="look",
                   ylabel="rows")
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 0].axhline(0, color="#555555", linewidth=1)
    axes[1, 0].set(title="C. Depth lottery versus next checkpoint swing",
                   xlabel="mean sampled depth in preceding 200 steps",
                   ylabel="signed change in net-correct rows")
    axes[1, 0].legend(fontsize=8)
    attribution = receipt["pooled_attribution"]["absolute_swing"]["shapley_incremental_r2"]
    labels = ["margin\nvolume", "curriculum", "controller", "training\nphase"]
    values = [attribution[key] for key in GROUP_ORDER]
    axes[1, 1].bar(labels, values, color=["#1976a3", "#6b8e23", "#c64b39", "#7a5aa6"])
    axes[1, 1].set(title="D. Descriptive variance attribution", ylabel="incremental in-sample R²")
    axes[1, 1].text(
        0.02, 0.98, "Not causal; late row identities and optimizer controls are unavailable",
        transform=axes[1, 1].transAxes, va="top", fontsize=8,
    )
    figure.suptitle("P3.4 directional stability: replicated signal on a knife-edge read", fontsize=15)
    png.parent.mkdir(parents=True, exist_ok=True)
    svg.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    normalize_svg(svg)


def build(args: argparse.Namespace) -> dict[str, Any]:
    receipt_paths = {0: args.seed0_receipts, 1: args.seed1_receipts}
    resume_paths = {0: args.seed0_resume, 1: args.seed1_resume}
    materials: list[dict[str, Any]] = []
    seed_receipts: list[dict[str, Any]] = []
    pooled_intervals: list[dict[str, Any]] = []
    for seed in (0, 1):
        material = load_seed_material(
            seed=seed, source_root=args.source_root, final_receipts=receipt_paths[seed]
        )
        curriculum = reconstruct_schedule(
            seed=seed,
            resume_path=resume_paths[seed],
            preflight_zip=args.preflight_zip,
            expected_summary_sha=str(material["final_summary"]["schedule_sha256"]),
        )
        series = build_seed_series(material, curriculum)
        intervals = [row for row in series if row["signed_swing"] is not None]
        pooled_intervals.extend(intervals)
        inherited = material["inherited_rows"]
        churn = [
            adjacent_row_churn(inherited[look - 1], inherited[look])
            for look in range(2, int(material["cutoff_look"]) + 1)
        ]
        seed_receipts.append(
            {
                "seed": seed,
                "series": series,
                "curriculum_hash_replay": {
                    "status": "all_4000_batch_hashes_exact",
                    "schedule_sha256": material["final_summary"]["schedule_sha256"],
                    "resume_sha256": sha256_file(resume_paths[seed]),
                },
                "exact_row_churn": {
                    "available_looks": sorted(inherited),
                    "adjacent_pairs": churn,
                    "late_window_status": (
                        "not_identifiable_from_cached rows; aggregate outcome counts remain complete"
                    ),
                },
                "correlations": correlations(intervals),
                "score_curve_smoothing": score_curve_smoothing(
                    [int(row["net_correct"]) for row in series]
                ),
                "endpoint": {
                    "net_correct": int(series[-1]["net_correct"]),
                    "fixes": int(series[-1]["fixes"]),
                    "regressions": int(series[-1]["regressions"]),
                    "discordant_rows": int(series[-1]["discordant_rows"]),
                    "scored_rung": int(series[-1]["scored_rung"]),
                    "scored_gate_ceiling": float(series[-1]["scored_gate_ceiling"]),
                },
            }
        )
        materials.append(material)
    pooled_attribution = {
        "signed_swing": shapley_r2(pooled_intervals, "signed_swing"),
        "absolute_swing": shapley_r2(pooled_intervals, "absolute_swing"),
    }
    joint_late_means = {
        window: float(
            np.mean(
                [
                    np.mean([row["net_correct"] for row in seed["series"][-window:]])
                    for seed in seed_receipts
                ]
            )
        )
        for window in (3, 5, 10)
    }
    all_churn = [
        pair
        for seed in seed_receipts
        for pair in seed["exact_row_churn"]["adjacent_pairs"]
    ]
    changed_margin = [
        float(pair["option_margin"]["changed_mean"])
        for pair in all_churn
        if pair["option_margin"]["changed_mean"] is not None
    ]
    stable_margin = [
        float(pair["option_margin"]["stable_mean"])
        for pair in all_churn
        if pair["option_margin"]["stable_mean"] is not None
    ]
    battery_changed: dict[str, int] = defaultdict(int)
    battery_rows: dict[str, int] = defaultdict(int)
    for pair in all_churn:
        for battery, values in pair["battery_change_rates"].items():
            battery_changed[battery] += int(values["changed"])
            battery_rows[battery] += int(values["rows"])
    rung_change = [
        row for row in pooled_intervals if row["training_controller_transition_since_prior_score"]
    ]
    rung_stable = [
        row for row in pooled_intervals if not row["training_controller_transition_since_prior_score"]
    ]
    mean_depths = [float(row["curriculum"]["mean_depth"]) for row in pooled_intervals]
    diagnostic_findings = {
        "evaluation_surface": {
            "endpoint_net_over_discordant": {
                str(seed["seed"]): abs(seed["endpoint"]["net_correct"])
                / seed["endpoint"]["discordant_rows"]
                for seed in seed_receipts
            },
            "exact_early_adjacent_pairs": len(all_churn),
            "exact_early_outcome_changed_rows": sum(
                int(pair["outcome_changed_rows"]) for pair in all_churn
            ),
            "changed_option_margin_mean_across_pairs": float(np.mean(changed_margin)),
            "stable_option_margin_mean_across_pairs": float(np.mean(stable_margin)),
            "battery_change_rates_on_exact_pairs": {
                battery: battery_changed[battery] / battery_rows[battery]
                for battery in sorted(battery_rows)
            },
            "reading": (
                "strong knife-edge evidence on the retained early rows: changed outcomes sit "
                "near zero option margin, while aggregate late identity churn remains unobserved"
            ),
        },
        "curriculum": {
            "mean_depth_range": [min(mean_depths), max(mean_depths)],
            "signed_swing_pearson_r": correlations(pooled_intervals)[
                "signed_swing_vs_mean_depth"
            ]["pearson_r"],
            "reading": (
                "all batch hashes reproduce; 200-step averaging keeps depth close to its "
                "registered mean and provides little evidence that the lottery drives swings"
            ),
        },
        "controller": {
            "training_transition_intervals": len(rung_change),
            "mean_absolute_swing_training_transition": (
                float(np.mean([row["absolute_swing"] for row in rung_change]))
                if rung_change
                else None
            ),
            "mean_absolute_swing_training_stable": float(
                np.mean([row["absolute_swing"] for row in rung_stable])
            ),
            "scored_gate_ceilings_by_seed": {
                str(seed["seed"]): sorted(
                    {float(row["scored_gate_ceiling"]) for row in seed["series"]}
                )
                for seed in seed_receipts
            },
            "signed_swing_vs_gate_ceiling": correlations(pooled_intervals)[
                "signed_swing_vs_gate_ceiling"
            ],
            "reading": (
                "the endpoint comparison uses different ceilings across seeds, but every "
                "within-seed score read used one fixed ceiling; training-time transitions "
                "remain a possible parameter-path effect, not a changing-ruler explanation"
            ),
        },
        "optimizer_and_landing": {
            "joint_late_score_means": joint_late_means,
            "phase_proxy_absolute_swing_incremental_r2": pooled_attribution[
                "absolute_swing"
            ]["shapley_incremental_r2"]["phase_proxy"],
            "reading": (
                "constant LR leaves the endpoint un-damped; score-space smoothing centers the "
                "joint late read near eight rows, but no weight-EMA counterfactual is available"
            ),
        },
    }
    receipt: dict[str, Any] = {
        "kind": "paper2_phase3_p34_directional_stability_v1",
        "status": "complete_cpu_dev_receipts_only",
        "registered_verdict_unchanged": "REPLICATED_POSITIVE_BELOW_TRIGGER_B",
        "sealed_data": {"confirm_scored": False, "eval_e_scored": False},
        "provenance": {
            "strategy_r2_drive_id": "18s_2BkOro0XFGxL1Hj4MBBf73Pl8-x6D",
            "strategy_r2_sha256_as_relayed": "c58fe95b...8943",
            "preflight_zip": str(args.preflight_zip),
            "preflight_zip_sha256": sha256_file(args.preflight_zip),
            "seed0_receipts_sha256": sha256_file(args.seed0_receipts),
            "seed1_receipts_sha256": sha256_file(args.seed1_receipts),
        },
        "seeds": seed_receipts,
        "pooled_correlations": correlations(pooled_intervals),
        "pooled_attribution": pooled_attribution,
        "diagnostic_findings": diagnostic_findings,
        "landing_protocol_telemetry": {
            "joint_late_score_means": joint_late_means,
            "registered_endpoint_mean": float(
                np.mean([seed["endpoint"]["net_correct"] for seed in seed_receipts])
            ),
            "interpretation": (
                "late score averaging prices a landing protocol near the +8-row region, "
                "but is not a weight-EMA counterfactual and does not meet Trigger B"
            ),
        },
        "identifiability": {
            "knife_edge_discordant_volume": "measured at every look",
            "same_row_flicker": (
                "measured exactly only for inherited looks 1-2 in seed 0 and 1-5 in seed 1"
            ),
            "curriculum_lottery": "exactly replayed and hash-verified for all 8000 steps",
            "controller_state": "measured at every look and every 100-step share window",
            "optimizer_causality": (
                "not identifiable because LR and averaging policy did not vary; phase proxy "
                "and score-space smoothing are descriptive only"
            ),
            "continuous_answer_margin": (
                "available only on inherited private row receipts; must be persisted at every "
                "P3.5 look"
            ),
        },
        "stability_recommendation": {
            "mandatory_p35_contracts": [
                "pin the evaluation ceiling independently of the training controller",
                "decay learning rate over the final 10 percent of steps",
                "freeze controller weights and pin the training rung during the landing window",
                "archive raw and EMA checkpoints and preregister which is primary",
                "persist per-row answer-token margin and outcome identity at every look",
                "report churn by battery, especially GSM8K versus MBPP",
            ],
            "next_measurements": [
                "authorized 2x2 fixed-ceiling score probe plus seed-1 target reconstruction",
                "oracle-direction refresh on both endpoints",
                "cross-token persistence probe if refresh leaves capture flat",
            ],
        },
        "do_not_claim": [
            "the descriptive Shapley-R2 decomposition is not causal variance attribution",
            "score-curve smoothing is not a weight EMA result",
            "late same-row churn is not recoverable from the retained cache",
            "the registered P3.4 verdict or sealed-data decision changed",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    manifest_path = args.output_dir / "artifact_manifest.json"
    figure_png = args.figures_dir / "p34_directional_stability_20260815.png"
    figure_svg = args.figures_dir / "p34_directional_stability_20260815.svg"
    build_figure(receipt, figure_png, figure_svg)
    receipt["artifacts"] = {
        "summary": {"path": str(summary_path)},
        "figure_png": {"path": str(figure_png), "sha256": sha256_file(figure_png)},
        "figure_svg": {"path": str(figure_svg), "sha256": sha256_file(figure_svg)},
        "artifact_manifest": {"path": str(manifest_path)},
    }
    summary_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "kind": "paper2_phase3_p34_directional_stability_artifact_manifest_v1",
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "figure_png": {"path": str(figure_png), "sha256": sha256_file(figure_png)},
        "figure_svg": {"path": str(figure_svg), "sha256": sha256_file(figure_svg)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed0_receipts", type=Path, required=True)
    parser.add_argument("--seed1_receipts", type=Path, required=True)
    parser.add_argument("--seed0_resume", type=Path, required=True)
    parser.add_argument("--seed1_resume", type=Path, required=True)
    parser.add_argument("--preflight_zip", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures_dir", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args()


def main() -> int:
    receipt = build(parse_args())
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
