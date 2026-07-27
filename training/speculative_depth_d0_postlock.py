"""Locked post-registration helpers for Paper Two speculative-depth D0."""

from __future__ import annotations

import random
import bisect
import math
import statistics
from typing import Any, Sequence

import torch

from training.speculative_depth_d0_spec import calibrated_depth_targets


D0_LOCK_COMMIT = "90cbc48c9aa749cb2e53dfef35bb2af9a24d9ae3"
D0_LOCK_RUN_ID = "stage5_paper2_d0_preregistration_20260726"
D0_RUN_ID = "stage5_paper2_d0_20260726"


def cache_plan() -> dict[str, Any]:
    """Return the immutable teacher/partition plan established by Draft 7."""

    return {
        "teacher_7b": {
            "partitions": ["label_train", "calibration", "evaluation", "in_era_contrast"],
            "full_logit_scope": "registered_natural_training_positions",
        },
        "teacher_14b": {
            "partitions": ["calibration"],
            "full_logit_scope": "none",
        },
    }


def build_training_schedule(
    *,
    total_steps: int,
    natural_positions: int,
    seed: int,
    rehearsal_fraction: float = 0.30,
) -> list[dict[str, Any]]:
    """Freeze the batch-1 70/30 schedule before teacher labeling.

    The ten-step cadence makes the exact mixture inspectable while the seeded
    natural-position draw fixes which teacher distributions must be cached.
    """

    if total_steps <= 0 or natural_positions <= 0:
        raise ValueError("D0 schedule requires positive steps and natural positions")
    if total_steps % 10 != 0 or rehearsal_fraction != 0.30:
        raise ValueError("D0 locks a 30 percent rehearsal cadence over ten-step blocks")
    generator = random.Random(int(seed))
    schedule: list[dict[str, Any]] = []
    natural_index = 0
    rehearsal_index = 0
    for step in range(1, int(total_steps) + 1):
        within_block = (step - 1) % 10
        if within_block < 7:
            schedule.append(
                {
                    "step": step,
                    "kind": "natural",
                    "natural_index": natural_index,
                    "position_index": generator.randrange(int(natural_positions)),
                }
            )
            natural_index += 1
        else:
            schedule.append(
                {
                    "step": step,
                    "kind": "rehearsal",
                    "rehearsal_index": rehearsal_index,
                }
            )
            rehearsal_index += 1
    return schedule


def rejection_run_lengths(rejected: Sequence[bool]) -> list[int]:
    """Assign every rejected position the length of its contiguous run."""

    result = [0] * len(rejected)
    start = 0
    while start < len(rejected):
        if not bool(rejected[start]):
            start += 1
            continue
        stop = start + 1
        while stop < len(rejected) and bool(rejected[stop]):
            stop += 1
        run = stop - start
        result[start:stop] = [run] * run
        start = stop
    return result


def score_teacher_signals(
    teacher_logits: torch.Tensor,
    drafter_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the locked compact signals on the shared original vocabulary."""

    if teacher_logits.ndim != 2 or drafter_logits.ndim != 2:
        raise ValueError("D0 signal logits must be [positions, vocabulary]")
    if teacher_logits.shape[0] != drafter_logits.shape[0]:
        raise ValueError("D0 teacher and drafter position counts differ")
    original_vocab = teacher_logits.shape[-1]
    if drafter_logits.shape[-1] < original_vocab:
        raise ValueError("D0 drafter vocabulary is smaller than the teacher vocabulary")
    aligned_drafter = drafter_logits[..., :original_vocab].float()
    teacher = teacher_logits.float()
    teacher_log_probs = torch.log_softmax(teacher, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    drafter_log_probs = torch.log_softmax(aligned_drafter, dim=-1)
    teacher_greedy = teacher.argmax(dim=-1)
    drafter_greedy = aligned_drafter.argmax(dim=-1)
    selected_teacher_logits = teacher.gather(-1, drafter_greedy.unsqueeze(-1)).squeeze(-1)
    ranks = teacher.gt(selected_teacher_logits.unsqueeze(-1)).sum(dim=-1).to(torch.int32) + 1
    return {
        "teacher_greedy_token_id": teacher_greedy.to(torch.int32),
        "drafter_greedy_token_id": drafter_greedy.to(torch.int32),
        "target_token_id": target_token_ids.to(torch.int32),
        "accepted": teacher_greedy.eq(drafter_greedy),
        "drafter_token_logprob_under_teacher": teacher_log_probs
        .gather(-1, drafter_greedy.unsqueeze(-1))
        .squeeze(-1),
        "drafter_token_rank_under_teacher": ranks,
        "teacher_entropy": -(teacher_probs * teacher_log_probs).sum(dim=-1),
        "teacher_to_plain_drafter_kl": (
            teacher_probs * (teacher_log_probs - drafter_log_probs)
        ).sum(dim=-1),
    }


def calibration_verdict(bin_curves: dict[str, list[float]]) -> dict[str, Any]:
    """Apply the Draft 7 branch rule to six-depth diagnostic curves."""

    if set(bin_curves) != {"q1", "q2", "q3", "q4"}:
        raise ValueError("D0 calibration requires q1 through q4")
    if any(len(values) != 6 for values in bin_curves.values()):
        raise ValueError("D0 floor curves must contain depths 1 through 6")
    locked = calibrated_depth_targets(
        {name: [float(value) for value in values[:4]] for name, values in bin_curves.items()}
    )
    return {**locked, "forced_depths": [1, 2, 3, 4, 5, 6], "bin_curves": bin_curves}


def validate_cache_summary(summary: dict[str, Any]) -> None:
    if summary.get("status") != "complete":
        raise AssertionError("D0 teacher cache is not complete")
    if summary.get("lock_commit") != D0_LOCK_COMMIT:
        raise AssertionError("D0 teacher cache references the wrong lock commit")
    if summary.get("teacher_reloaded_after_completed_cache") is not False:
        raise AssertionError("D0 teacher was reloaded after its completed cache")
    observed = summary.get("caches") or {}
    for teacher, plan in cache_plan().items():
        partitions = observed.get(teacher) or {}
        for partition in plan["partitions"]:
            if (partitions.get(partition) or {}).get("status") != "complete":
                raise AssertionError(f"D0 cache is missing {teacher}/{partition}")


def fit_isotonic(values: Sequence[float], targets: Sequence[float]) -> dict[str, Any]:
    """Fit a nondecreasing piecewise-constant mapping with weighted PAVA."""

    if len(values) != len(targets) or not values:
        raise ValueError("isotonic fit requires aligned nonempty values and targets")
    grouped: list[dict[str, float]] = []
    for value, target in sorted(zip(values, targets, strict=True)):
        if grouped and float(value) == grouped[-1]["x_max"]:
            block = grouped[-1]
            total = block["prediction"] * block["weight"] + float(target)
            block["weight"] += 1.0
            block["prediction"] = total / block["weight"]
        else:
            grouped.append(
                {"x_min": float(value), "x_max": float(value), "weight": 1.0, "prediction": float(target)}
            )
    index = 0
    while index < len(grouped) - 1:
        if grouped[index]["prediction"] <= grouped[index + 1]["prediction"]:
            index += 1
            continue
        left, right = grouped[index], grouped[index + 1]
        weight = left["weight"] + right["weight"]
        merged = {
            "x_min": left["x_min"],
            "x_max": right["x_max"],
            "weight": weight,
            "prediction": (
                left["prediction"] * left["weight"] + right["prediction"] * right["weight"]
            )
            / weight,
        }
        grouped[index : index + 2] = [merged]
        index = max(index - 1, 0)
    return {"kind": "monotone_isotonic", "blocks": grouped, "minimum": 1, "maximum": 4}


def predict_isotonic(model: dict[str, Any], value: float) -> int:
    blocks = list(model["blocks"])
    if not blocks:
        raise ValueError("isotonic model has no blocks")
    index = bisect.bisect_left([float(block["x_max"]) for block in blocks], float(value))
    prediction = float(blocks[min(index, len(blocks) - 1)]["prediction"])
    return max(int(model.get("minimum", 1)), min(int(model.get("maximum", 4)), int(round(prediction))))


def _linear_fit(values: Sequence[float], targets: Sequence[float]) -> dict[str, float]:
    x_mean = statistics.fmean(values)
    y_mean = statistics.fmean(targets)
    denominator = sum((value - x_mean) ** 2 for value in values)
    slope = max(0.0, sum((x - x_mean) * (y - y_mean) for x, y in zip(values, targets, strict=True)) / denominator) if denominator else 0.0
    return {"intercept": y_mean - slope * x_mean, "slope": slope}


def _clamped(value: float) -> float:
    return max(1.0, min(4.0, float(value)))


def fit_depth_mapping(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fit the locked primary and comparison forms on deterministic split halves."""

    eligible = [row for row in examples if int(row["run_length"]) <= 8]
    tail = len(examples) - len(eligible)
    fit_rows = [row for index, row in enumerate(eligible) if index % 2 == 0]
    heldout = [row for index, row in enumerate(eligible) if index % 2 == 1]
    if len(fit_rows) < 4 or not heldout:
        raise ValueError("D0 depth mapping needs at least four fit rows and one held-out row")
    targets = [float(row["required_depth"]) for row in fit_rows]
    isotonic_features = {
        "teacher_to_plain_drafter_kl": "kl",
        "drafter_token_rank_under_teacher": "rank",
        "rejection_run_length": "run_length",
        "teacher_entropy": "teacher_entropy",
        "negative_drafter_logprob_under_teacher": "negative_drafter_logprob_under_teacher",
    }
    isotonic_models = {
        name: fit_isotonic([float(row[field]) for row in fit_rows], targets)
        for name, field in isotonic_features.items()
    }
    isotonic = isotonic_models["teacher_to_plain_drafter_kl"]
    run_linear = _linear_fit([float(row["run_length"]) for row in fit_rows], targets)
    log_kl_values = [math.log1p(max(0.0, float(row["kl"]))) for row in fit_rows]
    log_kl_linear = _linear_fit(log_kl_values, targets)
    maximum_kl = max(float(row["kl"]) for row in fit_rows) or 1.0
    saturating_candidates: list[tuple[float, float]] = []
    for rate in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        errors = []
        for row in fit_rows:
            scaled = max(0.0, float(row["kl"])) / maximum_kl
            prediction = 1.0 + 3.0 * (1.0 - math.exp(-rate * scaled))
            errors.append((prediction - float(row["required_depth"])) ** 2)
        saturating_candidates.append((statistics.fmean(errors), rate))
    _, saturating_rate = min(saturating_candidates)

    def score(name: str) -> dict[str, Any]:
        squared: list[float] = []
        absolute: list[float] = []
        exact = 0
        for row in heldout:
            if name == "isotonic":
                prediction = float(predict_isotonic(isotonic, float(row["kl"])))
            elif name == "linear_run_length":
                prediction = _clamped(run_linear["intercept"] + run_linear["slope"] * float(row["run_length"]))
            elif name == "linear_log_kl":
                value = math.log1p(max(0.0, float(row["kl"])))
                prediction = _clamped(log_kl_linear["intercept"] + log_kl_linear["slope"] * value)
            else:
                scaled = max(0.0, float(row["kl"])) / maximum_kl
                prediction = 1.0 + 3.0 * (1.0 - math.exp(-saturating_rate * scaled))
            target = float(row["required_depth"])
            squared.append((prediction - target) ** 2)
            absolute.append(abs(prediction - target))
            exact += int(round(prediction) == int(target))
        return {
            "heldout_mse": statistics.fmean(squared),
            "heldout_mae": statistics.fmean(absolute),
            "heldout_exact": exact,
            "heldout_total": len(heldout),
        }

    def score_isotonic_feature(name: str, field: str) -> dict[str, Any]:
        model = isotonic_models[name]
        predictions = [float(predict_isotonic(model, float(row[field]))) for row in heldout]
        actual = [float(row["required_depth"]) for row in heldout]
        return {
            "heldout_mse": statistics.fmean((a - b) ** 2 for a, b in zip(predictions, actual, strict=True)),
            "heldout_mae": statistics.fmean(abs(a - b) for a, b in zip(predictions, actual, strict=True)),
            "heldout_exact": sum(round(a) == int(b) for a, b in zip(predictions, actual, strict=True)),
            "heldout_total": len(heldout),
        }

    return {
        "primary_feature": "teacher_to_plain_drafter_kl",
        "primary_fit": isotonic,
        "isotonic_fits_by_signal": isotonic_models,
        "fit_rows": len(fit_rows),
        "heldout_rows": len(heldout),
        "run_length_tail_excluded_above": 8,
        "excluded_tail_rows": tail,
        "comparison_parameters": {
            "linear_run_length": run_linear,
            "linear_log_kl": log_kl_linear,
            "saturating": {"rate": saturating_rate, "training_kl_scale": maximum_kl},
        },
        "heldout_scores": {
            name: score(name)
            for name in ("isotonic", "linear_run_length", "linear_log_kl", "saturating")
        },
        "isotonic_heldout_scores_by_signal": {
            name: score_isotonic_feature(name, field)
            for name, field in isotonic_features.items()
        },
    }
