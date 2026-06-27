"""Transfer conservative rescue selectors from discovery to held-out depth sweeps.

``analyze_rescue_predictability.py`` asks whether rescue is predictable on one
forced-depth sweep. This script turns that into an out-of-sample selector test:

1. fit/select simple threshold gates on the discovery sweep;
2. apply the exact thresholds to every benchmark in the held-out sweep;
3. report rescue captured, harm triggered, and oracle-gap capture.

The selector family is deliberately plain: default to loop 1, route only a
thresholded low-confidence slice to one deeper loop.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analyze_depth_sweep import joined_examples, load_loop_payloads, path_for_cli, resolve_path  # noqa: E402
from eval.analyze_rescue_predictability import (  # noqa: E402
    SELECTOR_FEATURES,
    binary_gate_sweep,
    discrimination_for_feature,
    evaluate_binary_gate,
    label_examples,
    quantile_thresholds,
    sign_test_p_value,
    feature_value,
)


EPS = 1e-12
DEFAULT_PROBE_SHRINKAGES = [0.1, 1.0, 10.0]


def examples_for_sweep(sweep_summary: Path, *, benchmark: str, score_target: str, aggregate: str) -> tuple[list[int], list[dict[str, Any]]]:
    _sweep, loop_payloads = load_loop_payloads(sweep_summary)
    loops = sorted(loop_payloads)
    examples = joined_examples(loop_payloads, benchmark, score_target, aggregate)
    return loops, label_examples(examples, loops)


def available_benchmarks(sweep_summary: Path) -> list[str]:
    _sweep, loop_payloads = load_loop_payloads(sweep_summary)
    loops = sorted(loop_payloads)
    return list(loop_payloads[loops[0]].get("benchmarks", []))


def select_policy(gates: list[dict[str, Any]], *, label: str, harm_budget: int | None) -> dict[str, Any] | None:
    candidates = gates
    if harm_budget is not None:
        candidates = [gate for gate in gates if int(gate.get("harm_triggered", 0)) <= harm_budget]
    if not candidates:
        return None
    best = sorted(
        candidates,
        key=lambda row: (
            int(row.get("delta_vs_loop1", 0)),
            int(row.get("rescue_captured", 0)),
            -int(row.get("harm_triggered", 0)),
            -int(row.get("routed_deep", 0)),
        ),
        reverse=True,
    )[0]
    return {
        "label": label,
        "harm_budget": harm_budget,
        "feature": best["feature"],
        "threshold": best["threshold"],
        "direction": best["direction"],
        "fallback_loop": best["fallback_loop"],
        "discovery_result": best,
    }


def policy_from_manual(
    *,
    label: str,
    feature: str,
    threshold: float,
    direction: str,
    fallback_loop: int,
) -> dict[str, Any]:
    return {
        "label": label,
        "harm_budget": None,
        "feature": feature,
        "threshold": threshold,
        "direction": direction,
        "fallback_loop": fallback_loop,
        "discovery_result": None,
    }


def apply_policy(examples: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_binary_gate(
        examples,
        feature=str(policy["feature"]),
        threshold=float(policy["threshold"]),
        direction=str(policy["direction"]),
        fallback_loop=int(policy["fallback_loop"]),
    )
    return {
        "policy_label": policy["label"],
        "feature": policy["feature"],
        "threshold": policy["threshold"],
        "direction": policy["direction"],
        "fallback_loop": policy["fallback_loop"],
        **result,
    }


def policy_key(policy: dict[str, Any]) -> tuple[str, float, str, int]:
    return (
        str(policy["feature"]),
        float(policy["threshold"]),
        str(policy["direction"]),
        int(policy["fallback_loop"]),
    )


def policy_from_gate(label: str, gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "harm_budget": None,
        "feature": gate["feature"],
        "threshold": gate["threshold"],
        "direction": gate["direction"],
        "fallback_loop": gate["fallback_loop"],
        "discovery_result": gate,
    }


def discovery_curve_policies(gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str, int]] = set()
    for index, gate in enumerate(gates):
        policy = policy_from_gate(f"curve_{index:04d}", gate)
        key = policy_key(policy)
        if key in seen:
            continue
        seen.add(key)
        policies.append(policy)
    return policies


def best_result(
    rows: list[dict[str, Any]],
    *,
    label: str,
    harm_budget: int | None = None,
) -> dict[str, Any] | None:
    candidates = rows
    if harm_budget is not None:
        candidates = [row for row in rows if int(row.get("harm_triggered", 0)) <= harm_budget]
    if not candidates:
        return None
    best = sorted(
        candidates,
        key=lambda row: (
            int(row.get("delta_vs_loop1", 0)),
            int(row.get("rescue_captured", 0)),
            -int(row.get("harm_triggered", 0)),
            -int(row.get("routed_deep", 0)),
        ),
        reverse=True,
    )[0]
    return {"label": label, **best}


def transferred_curve(
    examples: list[dict[str, Any]],
    policies: list[dict[str, Any]],
    *,
    loops: list[int],
) -> list[dict[str, Any]]:
    rows = []
    available_loops = set(loops)
    for policy in policies:
        if int(policy["fallback_loop"]) not in available_loops:
            continue
        result = apply_policy(examples, policy)
        discovery_result = policy.get("discovery_result") or {}
        rows.append(
            {
                **result,
                "discovery_delta_vs_loop1": discovery_result.get("delta_vs_loop1"),
                "discovery_rescue_captured": discovery_result.get("rescue_captured"),
                "discovery_harm_triggered": discovery_result.get("harm_triggered"),
                "discovery_routed_deep": discovery_result.get("routed_deep"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("delta_vs_loop1", 0)),
            int(row.get("rescue_captured", 0)),
            -int(row.get("harm_triggered", 0)),
            -int(row.get("routed_deep", 0)),
        ),
        reverse=True,
    )


def rescue_discrimination(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [discrimination_for_feature(examples, feature, positive_label="rescuable") for feature in SELECTOR_FEATURES],
        key=lambda row: (row.get("oriented_auc") or 0.0, row.get("fisher_ratio") or 0.0),
        reverse=True,
    )


def category_counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for example in examples:
        category = str(example["category"])
        out[category] = out.get(category, 0) + 1
    return dict(sorted(out.items()))


def finite_or_nan(value: Any) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float("nan")


def fit_feature_stats(examples: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    raw = torch.tensor(
        [
            [finite_or_nan(feature_value(example, feature)) for feature in features]
            for example in examples
        ],
        dtype=torch.float64,
    )
    means = []
    stds = []
    for col in range(raw.shape[1]):
        values = raw[:, col]
        finite = torch.isfinite(values)
        if not bool(finite.any()):
            means.append(0.0)
            stds.append(1.0)
            continue
        kept = values[finite]
        mean = float(kept.mean().item())
        std = float(kept.std(unbiased=False).item())
        means.append(mean)
        stds.append(std if std > EPS else 1.0)
    return {"features": list(features), "means": means, "stds": stds}


def transform_features(examples: list[dict[str, Any]], stats: dict[str, Any]) -> torch.Tensor:
    features = list(stats["features"])
    raw = torch.tensor(
        [
            [finite_or_nan(feature_value(example, feature)) for feature in features]
            for example in examples
        ],
        dtype=torch.float64,
    )
    means = torch.tensor(stats["means"], dtype=torch.float64)
    stds = torch.tensor(stats["stds"], dtype=torch.float64)
    finite = torch.isfinite(raw)
    raw = torch.where(finite, raw, means.unsqueeze(0).expand_as(raw))
    return (raw - means) / stds.clamp_min(EPS)


def safe_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def covariance_matrix(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.empty((0, 0), dtype=torch.float64)
    centered = x - x.mean(dim=0, keepdim=True)
    denom = max(1, x.shape[0] - 1)
    return centered.T @ centered / denom


def eigen_descending(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cov.numel() == 0:
        return torch.empty(0, dtype=torch.float64), torch.empty((0, 0), dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    return evals[order].clamp_min(0.0), evecs[:, order]


def spectral_localization(
    examples: list[dict[str, Any]],
    *,
    features: list[str],
    positive_category: str,
    negative_categories: set[str],
) -> dict[str, Any]:
    stats = fit_feature_stats(examples, features)
    x = transform_features(examples, stats)
    pos_idx = [idx for idx, example in enumerate(examples) if example["category"] == positive_category]
    neg_idx = [idx for idx, example in enumerate(examples) if example["category"] in negative_categories]
    if not pos_idx or not neg_idx or x.shape[1] == 0:
        return {
            "available": False,
            "positive_category": positive_category,
            "negative_categories": sorted(negative_categories),
            "positive": len(pos_idx),
            "negative": len(neg_idx),
        }
    cov = covariance_matrix(x)
    evals, evecs = eigen_descending(cov)
    diff = x[pos_idx].mean(dim=0) - x[neg_idx].mean(dim=0)
    diff_norm = torch.linalg.vector_norm(diff)
    if float(diff_norm.item()) <= EPS:
        energy = torch.zeros_like(evals)
    else:
        projections = evecs.T @ (diff / diff_norm)
        energy = projections.pow(2)
        energy = energy / energy.sum().clamp_min(EPS)
    dim = int(energy.numel())
    top = {
        str(k): safe_float(energy[: min(k, dim)].sum())
        for k in [1, 2, 4, 8]
        if dim > 0
    }
    half = max(1, dim // 2)
    return {
        "available": True,
        "feature_space": "selector_telemetry",
        "features": features,
        "positive_category": positive_category,
        "negative_categories": sorted(negative_categories),
        "positive": len(pos_idx),
        "negative": len(neg_idx),
        "dimension": dim,
        "diff_norm": safe_float(diff_norm),
        "eigenvalues": [safe_float(value) for value in evals],
        "energy_by_eigen_index": [safe_float(value) for value in energy],
        "top_energy": top,
        "tail_half_energy": safe_float(energy[half:].sum()) if dim > 1 else 0.0,
    }


def normalized_direction(direction: torch.Tensor | None) -> torch.Tensor | None:
    if direction is None:
        return None
    norm = torch.linalg.vector_norm(direction)
    if float(norm.item()) <= EPS:
        return None
    return direction / norm


def fit_whitened_direction(
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    shrinkage: float,
) -> torch.Tensor | None:
    pos = labels.bool()
    neg = ~pos
    if int(pos.sum().item()) < 1 or int(neg.sum().item()) < 1:
        return None
    diff = x[pos].mean(dim=0) - x[neg].mean(dim=0)
    cov = covariance_matrix(x)
    evals, _evecs = eigen_descending(cov)
    positive_evals = evals[evals > EPS]
    scale = float(torch.median(positive_evals).item()) if positive_evals.numel() else 1.0
    reg = max(EPS, float(shrinkage) * scale)
    eye = torch.eye(cov.shape[0], dtype=torch.float64)
    try:
        direction = torch.linalg.solve(cov + reg * eye, diff)
    except torch.linalg.LinAlgError:
        direction = torch.linalg.pinv(cov + reg * eye) @ diff
    return normalized_direction(direction)


def pairwise_alignment(directions: list[torch.Tensor]) -> float | None:
    if len(directions) < 2:
        return None
    values = []
    for i, left in enumerate(directions):
        for right in directions[i + 1 :]:
            values.append(abs(float(torch.dot(left, right).item())))
    return sum(values) / len(values) if values else None


def fit_subsample_directions(
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    shrinkage: float,
    repeats: int,
    sample_fraction: float,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    take = min(n, max(4, int(round(n * sample_fraction))))
    directions: list[torch.Tensor] = []
    attempts = 0
    while len(directions) < repeats and attempts < repeats * 10:
        attempts += 1
        indices = torch.randperm(n, generator=generator)[:take]
        direction = fit_whitened_direction(x[indices], labels[indices], shrinkage=shrinkage)
        if direction is not None:
            directions.append(direction)
    return directions


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def diverse_probe_detectability(
    examples: list[dict[str, Any]],
    *,
    features: list[str],
    shrinkage: float = 1.0,
    repeats: int = 48,
    permutations: int = 64,
    sample_fraction: float = 0.7,
    seed: int = 13,
) -> dict[str, Any]:
    stats = fit_feature_stats(examples, features)
    x = transform_features(examples, stats)
    labels = torch.tensor([example["category"] == "rescuable" for example in examples], dtype=torch.bool)
    if int(labels.sum().item()) < 2 or int((~labels).sum().item()) < 2:
        return {
            "available": False,
            "reason": "insufficient_positive_or_negative_examples",
            "positive": int(labels.sum().item()),
            "negative": int((~labels).sum().item()),
        }
    observed_directions = fit_subsample_directions(
        x,
        labels,
        shrinkage=shrinkage,
        repeats=repeats,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    observed = pairwise_alignment(observed_directions)
    generator = torch.Generator().manual_seed(seed + 1)
    null_values: list[float] = []
    for index in range(permutations):
        permuted = labels[torch.randperm(labels.numel(), generator=generator)]
        directions = fit_subsample_directions(
            x,
            permuted,
            shrinkage=shrinkage,
            repeats=max(8, repeats // 3),
            sample_fraction=sample_fraction,
            seed=seed + 101 + index,
        )
        value = pairwise_alignment(directions)
        if value is not None:
            null_values.append(value)
    null_p95 = percentile(null_values, 0.95)
    return {
        "available": observed is not None and null_p95 is not None,
        "feature_space": "selector_telemetry",
        "features": features,
        "positive": int(labels.sum().item()),
        "negative": int((~labels).sum().item()),
        "shrinkage": shrinkage,
        "repeats": repeats,
        "permutations": permutations,
        "sample_fraction": sample_fraction,
        "observed_alignment": observed,
        "null_mean_alignment": (sum(null_values) / len(null_values)) if null_values else None,
        "null_p95_alignment": null_p95,
        "clears_null_p95": bool(observed is not None and null_p95 is not None and observed > null_p95),
    }


def evaluate_score_gate(
    examples: list[dict[str, Any]],
    scores: list[float],
    *,
    feature: str,
    threshold: float,
    direction: str,
    fallback_loop: int,
) -> dict[str, Any]:
    loop1 = min(examples[0]["loop_hits"]) if examples else 1
    loop1_correct = sum(1 for example in examples if example["loop_hits"][loop1])
    oracle_correct = sum(
        1 for example in examples if example["loop_hits"][loop1] or example["loop_hits"].get(fallback_loop, False)
    )
    correct = 0
    routed_deep = 0
    wins = 0
    losses = 0
    rescue_captured = 0
    harm_triggered = 0
    eligible = 0
    for example, score in zip(examples, scores, strict=True):
        if not math.isfinite(float(score)):
            use_deep = False
        else:
            eligible += 1
            use_deep = score >= threshold if direction == "high" else score <= threshold
        chosen_loop = fallback_loop if use_deep else loop1
        routed_deep += int(use_deep)
        hit = bool(example["loop_hits"][chosen_loop])
        correct += int(hit)
        loop1_hit = bool(example["loop_hits"][loop1])
        if use_deep and example["rescuable"] and hit:
            rescue_captured += 1
        if use_deep and example["harmable"] and loop1_hit and not hit:
            harm_triggered += 1
        if hit and not loop1_hit:
            wins += 1
        elif loop1_hit and not hit:
            losses += 1
    gap = max(0, oracle_correct - loop1_correct)
    return {
        "feature": feature,
        "threshold": threshold,
        "direction": direction,
        "fallback_loop": fallback_loop,
        "eligible": eligible,
        "total": len(examples),
        "correct": correct,
        "loop1_correct": loop1_correct,
        "oracle_correct": oracle_correct,
        "delta_vs_loop1": correct - loop1_correct,
        "oracle_gap_capture": None if gap == 0 else (correct - loop1_correct) / gap,
        "routed_deep": routed_deep,
        "wins_vs_loop1": wins,
        "losses_vs_loop1": losses,
        "sign_test_p": sign_test_p_value(wins, losses),
        "rescue_captured": rescue_captured,
        "harm_triggered": harm_triggered,
    }


def score_gate_sweep(
    examples: list[dict[str, Any]],
    scores: list[float],
    loops: list[int],
    *,
    feature: str,
) -> list[dict[str, Any]]:
    thresholds = quantile_thresholds([score for score in scores if math.isfinite(float(score))])
    rows: list[dict[str, Any]] = []
    for fallback_loop in [loop for loop in loops if loop != min(loops)]:
        for threshold in thresholds:
            rows.append(
                evaluate_score_gate(
                    examples,
                    scores,
                    feature=feature,
                    threshold=threshold,
                    direction="high",
                    fallback_loop=fallback_loop,
                )
            )
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("delta_vs_loop1", 0)),
            int(row.get("rescue_captured", 0)),
            -int(row.get("harm_triggered", 0)),
            -int(row.get("routed_deep", 0)),
        ),
        reverse=True,
    )


def train_supervised_probes(
    examples: list[dict[str, Any]],
    loops: list[int],
    *,
    features: list[str],
    shrinkages: list[float] | None = None,
) -> list[dict[str, Any]]:
    shrinkages = shrinkages or DEFAULT_PROBE_SHRINKAGES
    stats = fit_feature_stats(examples, features)
    x = transform_features(examples, stats)
    labels = torch.tensor([example["category"] == "rescuable" for example in examples], dtype=torch.bool)
    probes: list[dict[str, Any]] = []
    for shrinkage in shrinkages:
        direction = fit_whitened_direction(x, labels, shrinkage=shrinkage)
        if direction is None:
            continue
        scores = (x @ direction).tolist()
        feature_name = f"whitened_rescue_score_shrinkage_{shrinkage:g}"
        curve = score_gate_sweep(examples, [float(score) for score in scores], loops, feature=feature_name)
        probes.append(
            {
                "feature": feature_name,
                "kind": "regularized_whitened_rescue_score",
                "shrinkage": shrinkage,
                "stats": stats,
                "direction": [float(value) for value in direction.tolist()],
                "discovery_curve_summary": {
                    "points": len(curve),
                    "max_net": best_result(curve, label="max_net"),
                    "zero_harm": best_result(curve, label="zero_harm", harm_budget=0),
                    "harm_budget_1": best_result(curve, label="harm_budget_1", harm_budget=1),
                    "harm_budget_2": best_result(curve, label="harm_budget_2", harm_budget=2),
                },
                "discovery_curve": curve,
                "discovery_curve_top25": curve[:25],
            }
        )
    return probes


def apply_supervised_probe(
    examples: list[dict[str, Any]],
    loops: list[int],
    probe: dict[str, Any],
) -> dict[str, Any]:
    x = transform_features(examples, probe["stats"])
    direction = torch.tensor(probe["direction"], dtype=torch.float64)
    scores = [float(score) for score in (x @ direction).tolist()]
    rows = []
    for discovery_row in probe["discovery_curve"]:
        fallback_loop = int(discovery_row["fallback_loop"])
        if fallback_loop not in loops:
            continue
        rows.append(
            {
                **evaluate_score_gate(
                    examples,
                    scores,
                    feature=str(probe["feature"]),
                    threshold=float(discovery_row["threshold"]),
                    direction=str(discovery_row["direction"]),
                    fallback_loop=fallback_loop,
                ),
                "discovery_delta_vs_loop1": discovery_row.get("delta_vs_loop1"),
                "discovery_rescue_captured": discovery_row.get("rescue_captured"),
                "discovery_harm_triggered": discovery_row.get("harm_triggered"),
                "discovery_routed_deep": discovery_row.get("routed_deep"),
            }
        )
    return {
        "feature": probe["feature"],
        "kind": probe["kind"],
        "shrinkage": probe["shrinkage"],
        "heldout_curve_summary": {
            "points": len(rows),
            "max_net": best_result(rows, label="max_net"),
            "zero_harm": best_result(rows, label="zero_harm", harm_budget=0),
            "harm_budget_1": best_result(rows, label="harm_budget_1", harm_budget=1),
            "harm_budget_2": best_result(rows, label="harm_budget_2", harm_budget=2),
        },
        "heldout_curve_top25": rows[:25],
    }


def analyze_transfer(
    *,
    discovery_sweep_summary: Path,
    heldout_sweep_summary: Path,
    discovery_benchmark: str,
    score_target: str,
    aggregate: str,
    include_manual_base_margin_thresholds: list[float],
    run_id: str | None = None,
) -> dict[str, Any]:
    discovery_loops, discovery_examples = examples_for_sweep(
        discovery_sweep_summary,
        benchmark=discovery_benchmark,
        score_target=score_target,
        aggregate=aggregate,
    )
    gates = binary_gate_sweep(discovery_examples, discovery_loops, SELECTOR_FEATURES)
    curve_policies = discovery_curve_policies(gates)
    spectral = {
        "rescuable_vs_harmable": spectral_localization(
            discovery_examples,
            features=SELECTOR_FEATURES,
            positive_category="rescuable",
            negative_categories={"harmable"},
        ),
        "rescuable_vs_rest": spectral_localization(
            discovery_examples,
            features=SELECTOR_FEATURES,
            positive_category="rescuable",
            negative_categories={"harmable", "stable_correct", "stable_wrong"},
        ),
    }
    detectability = diverse_probe_detectability(discovery_examples, features=SELECTOR_FEATURES)
    supervised_probes = train_supervised_probes(discovery_examples, discovery_loops, features=SELECTOR_FEATURES)
    selected: list[dict[str, Any]] = []
    for label, harm_budget in [
        ("max_net", None),
        ("zero_harm", 0),
        ("harm_budget_1", 1),
        ("harm_budget_2", 2),
    ]:
        policy = select_policy(gates, label=label, harm_budget=harm_budget)
        if policy is not None:
            selected.append(policy)
    for threshold in include_manual_base_margin_thresholds:
        selected.append(
            policy_from_manual(
                label=f"manual_base_margin_low_{threshold:g}",
                feature="base_predicted_margin",
                threshold=threshold,
                direction="low",
                fallback_loop=3 if 3 in discovery_loops else max(discovery_loops),
            )
        )

    heldout: dict[str, Any] = {}
    for benchmark in available_benchmarks(heldout_sweep_summary):
        loops, examples = examples_for_sweep(
            heldout_sweep_summary,
            benchmark=benchmark,
            score_target=score_target,
            aggregate=aggregate,
        )
        applied = []
        for policy in selected:
            # Skip policies whose selected fallback loop does not exist in this sweep.
            if int(policy["fallback_loop"]) not in loops:
                continue
            applied.append(apply_policy(examples, policy))
        curve = transferred_curve(examples, curve_policies, loops=loops)
        supervised_results = [
            apply_supervised_probe(examples, loops, probe)
            for probe in supervised_probes
        ]
        heldout[benchmark] = {
            "total": len(examples),
            "loops": loops,
            "category_counts": category_counts(examples),
            "rescue_discrimination": rescue_discrimination(examples),
            "policy_results": applied,
            "transferred_curve_summary": {
                "points": len(curve),
                "max_net": best_result(curve, label="max_net"),
                "zero_harm": best_result(curve, label="zero_harm", harm_budget=0),
                "harm_budget_1": best_result(curve, label="harm_budget_1", harm_budget=1),
                "harm_budget_2": best_result(curve, label="harm_budget_2", harm_budget=2),
            },
            "transferred_curve_top25": curve[:25],
            "supervised_probe_results": supervised_results,
        }

    return {
        "kind": "stage5_rescue_selector_transfer",
        "run_id": run_id or "stage5_rescue_selector_transfer_" + time.strftime("%Y%m%d_%H%M%S"),
        "discovery_sweep_summary": path_for_cli(discovery_sweep_summary),
        "heldout_sweep_summary": path_for_cli(heldout_sweep_summary),
        "discovery_benchmark": discovery_benchmark,
        "score_target": score_target,
        "aggregate": aggregate,
        "discovery": {
            "total": len(discovery_examples),
            "loops": discovery_loops,
            "category_counts": category_counts(discovery_examples),
            "rescue_discrimination": rescue_discrimination(discovery_examples),
            "spectral_localization": spectral,
            "diverse_probe_detectability": detectability,
            "selected_policies": selected,
            "curve_points": len(gates),
            "top_curve_points": gates[:25],
            "supervised_probes": [
                {
                    key: value
                    for key, value in probe.items()
                    if key not in {"stats", "direction"}
                }
                for probe in supervised_probes
            ],
        },
        "heldout": heldout,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Rescue Selector Transfer - {payload['run_id']}",
        "",
        f"- Discovery: `{payload['discovery_sweep_summary']}`",
        f"- Held-out: `{payload['heldout_sweep_summary']}`",
        f"- Discovery benchmark: `{payload['discovery_benchmark']}`",
        f"- Score target / aggregate: `{payload['score_target']}` / `{payload['aggregate']}`",
        "",
    ]
    spectral = payload["discovery"].get("spectral_localization", {})
    detectability = payload["discovery"].get("diverse_probe_detectability", {})
    if spectral:
        lines.extend(["## Discovery Spectral Diagnostics", ""])
        for label, row in spectral.items():
            if not row.get("available"):
                lines.append(f"- `{label}`: unavailable `{row}`")
                continue
            lines.append(
                f"- `{label}`: dim `{row['dimension']}`, diff norm `{row['diff_norm']}`, "
                f"top energy `{row['top_energy']}`, tail-half energy `{row['tail_half_energy']}`"
            )
        lines.append("")
    if detectability:
        lines.extend(
            [
                "## Diverse-Probe Detectability",
                "",
                f"- Available: `{detectability.get('available')}`",
                f"- Observed alignment: `{detectability.get('observed_alignment')}`",
                f"- Null p95 alignment: `{detectability.get('null_p95_alignment')}`",
                f"- Clears null p95: `{detectability.get('clears_null_p95')}`",
                "",
            ]
        )
    if payload["discovery"].get("supervised_probes"):
        lines.extend(["## Discovery Supervised Probe Candidates", ""])
        for probe in payload["discovery"]["supervised_probes"]:
            summary = probe["discovery_curve_summary"]
            best = summary.get("max_net") or {}
            zero = summary.get("zero_harm") or {}
            lines.append(
                f"- `{probe['feature']}`: discovery max-net delta `{best.get('delta_vs_loop1')}`, "
                f"rescue/harm `{best.get('rescue_captured')}/{best.get('harm_triggered')}`; "
                f"zero-harm delta `{zero.get('delta_vs_loop1')}`, "
                f"rescue/harm `{zero.get('rescue_captured')}/{zero.get('harm_triggered')}`"
            )
        lines.append("")
    lines.extend(["## Discovery Policies", ""])
    for policy in payload["discovery"]["selected_policies"]:
        result = policy.get("discovery_result") or {}
        lines.append(
            f"- `{policy['label']}`: `{policy['feature']}` {policy['direction']} `{policy['threshold']}` -> "
            f"loop `{policy['fallback_loop']}`; discovery delta `{result.get('delta_vs_loop1')}`, "
            f"rescue/harm `{result.get('rescue_captured')}/{result.get('harm_triggered')}`, "
            f"routed `{result.get('routed_deep')}`"
        )
    lines.extend(["", "## Held-Out Results", ""])
    for benchmark, result in payload["heldout"].items():
        lines.extend(
            [
                f"### {benchmark}",
                "",
                f"- Categories: `{result['category_counts']}`",
                "- Rescue AUCs: "
                + ", ".join(
                    f"`{row['feature']}` {row.get('oriented_auc')}"
                    for row in result["rescue_discrimination"][:3]
                ),
                "",
            ]
        )
        for row in result["policy_results"]:
            lines.append(
                f"- `{row['policy_label']}`: correct `{row['correct']}/{row['total']}`, "
                f"delta `{row['delta_vs_loop1']}`, gap capture `{row['oracle_gap_capture']}`, "
                f"routed `{row['routed_deep']}`, W/L `{row['wins_vs_loop1']}/{row['losses_vs_loop1']}`, "
                f"rescue/harm `{row['rescue_captured']}/{row['harm_triggered']}`"
            )
        lines.extend(["", "#### Transferred Curve Summary", ""])
        for label, row in result["transferred_curve_summary"].items():
            if label == "points":
                continue
            if row is None:
                lines.append(f"- `{label}`: no eligible curve point")
                continue
            lines.append(
                f"- `{label}`: `{row['feature']}` {row['direction']} `{row['threshold']}` -> "
                f"loop `{row['fallback_loop']}`; correct `{row['correct']}/{row['total']}`, "
                f"delta `{row['delta_vs_loop1']}`, gap capture `{row['oracle_gap_capture']}`, "
                f"routed `{row['routed_deep']}`, W/L `{row['wins_vs_loop1']}/{row['losses_vs_loop1']}`, "
                f"rescue/harm `{row['rescue_captured']}/{row['harm_triggered']}`"
            )
        if result.get("supervised_probe_results"):
            lines.extend(["", "#### Supervised Probe Transfer", ""])
            for probe in result["supervised_probe_results"]:
                best = probe["heldout_curve_summary"].get("max_net") or {}
                zero = probe["heldout_curve_summary"].get("zero_harm") or {}
                lines.append(
                    f"- `{probe['feature']}`: max-net delta `{best.get('delta_vs_loop1')}`, "
                    f"rescue/harm `{best.get('rescue_captured')}/{best.get('harm_triggered')}`; "
                    f"zero-harm delta `{zero.get('delta_vs_loop1')}`, "
                    f"rescue/harm `{zero.get('rescue_captured')}/{zero.get('harm_triggered')}`"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery_sweep_summary", required=True)
    parser.add_argument("--heldout_sweep_summary", required=True)
    parser.add_argument("--discovery_benchmark", default="arc_challenge")
    parser.add_argument("--score_target", default="content_question_only")
    parser.add_argument("--aggregate", default="mean")
    parser.add_argument("--manual_base_margin_thresholds", default="")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    manual_thresholds = [
        float(item.strip())
        for item in args.manual_base_margin_thresholds.split(",")
        if item.strip()
    ]
    payload = analyze_transfer(
        discovery_sweep_summary=resolve_path(args.discovery_sweep_summary),
        heldout_sweep_summary=resolve_path(args.heldout_sweep_summary),
        discovery_benchmark=args.discovery_benchmark,
        score_target=args.score_target,
        aggregate=args.aggregate,
        include_manual_base_margin_thresholds=manual_thresholds,
        run_id=args.run_id.strip() or None,
    )
    output_dir = resolve_path(args.output_dir) if args.output_dir else ROOT / "outputs" / "stage5" / payload["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, output_dir / "summary.md")
    print((output_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
