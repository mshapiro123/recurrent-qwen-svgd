"""Pure utilities for the bounded learned-depth selector assessment."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from models.halting import categorical_kl, pondernet_halting_probabilities


SELECTOR_PARAMETER_NAMES = {
    "halt_predictor.proj.weight",
    "halt_predictor.proj.bias",
    "halt_predictor.loop_embedding.weight",
    "halt_predictor.loop_bias",
}


def configure_selector_only(wrapper: torch.nn.Module) -> set[str]:
    """Freeze the mechanism and expose only prompt-driven halt parameters.

    The target-loop router, target embeddings, and target-conditioned biases are
    deliberately excluded. They are oracle-control machinery and would leak the
    gold depth in the supervised depth-reading arm.
    """

    selected: set[str] = set()
    for name, parameter in wrapper.named_parameters():
        trainable = name in SELECTOR_PARAMETER_NAMES
        parameter.requires_grad_(trainable)
        if trainable:
            selected.add(name)
    if selected != SELECTOR_PARAMETER_NAMES:
        missing = sorted(SELECTOR_PARAMETER_NAMES - selected)
        extra = sorted(selected - SELECTOR_PARAMETER_NAMES)
        raise RuntimeError(f"Selector parameter contract mismatch: missing={missing}, extra={extra}")
    return selected


def frozen_parameter_hash(wrapper: torch.nn.Module) -> str:
    """Hash all parameters outside the bounded selector trainable set."""

    digest = hashlib.sha256()
    for name, parameter in sorted(wrapper.named_parameters(), key=lambda item: item[0]):
        if name in SELECTOR_PARAMETER_NAMES:
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_frozen_gradients_zero(wrapper: torch.nn.Module) -> int:
    """Fail if a non-selector parameter receives a nonzero gradient."""

    nonzero: list[str] = []
    observed = 0
    for name, parameter in wrapper.named_parameters():
        if name in SELECTOR_PARAMETER_NAMES or parameter.grad is None:
            continue
        observed += 1
        if bool(parameter.grad.detach().ne(0).any()):
            nonzero.append(name)
    if nonzero:
        raise RuntimeError(f"Frozen parameters received nonzero gradients: {nonzero[:8]}")
    return observed


def assert_active_selector_gradient(wrapper: torch.nn.Module) -> dict[str, float]:
    """Require live gradients on both the halt projection and loop controls."""

    norms: dict[str, float] = {}
    for name, parameter in wrapper.named_parameters():
        if name not in SELECTOR_PARAMETER_NAMES:
            continue
        grad = parameter.grad
        norms[name] = 0.0 if grad is None else float(grad.detach().float().norm().item())
    proj = sum(value for name, value in norms.items() if ".proj." in name)
    loop = sum(value for name, value in norms.items() if ".loop_" in name)
    if not math.isfinite(proj) or not math.isfinite(loop) or proj <= 0.0 or loop <= 0.0:
        raise RuntimeError(f"Selector supervision is not gradient-live: {norms}")
    return norms


def halting_weights_from_features(
    predictor: torch.nn.Module,
    pooled_features: torch.Tensor,
) -> torch.Tensor:
    """Run the existing sequential halt head on cached loop states."""

    if pooled_features.dim() != 3:
        raise ValueError("pooled_features must be [batch, loops, hidden]")
    probabilities = [
        predictor(pooled_features[:, loop_idx], loop_idx=loop_idx).squeeze(-1)
        for loop_idx in range(pooled_features.shape[1])
    ]
    return pondernet_halting_probabilities(torch.stack(probabilities, dim=-1))


def supervised_depth_loss(
    predictor: torch.nn.Module,
    pooled_features: torch.Tensor,
    target_depths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = halting_weights_from_features(predictor, pooled_features)
    loss = F.nll_loss(
        weights.float().clamp_min(1e-8).log(),
        target_depths.to(device=weights.device, dtype=torch.long) - 1,
    )
    return loss, weights


def truncated_geometric_prior(
    *,
    max_loops: int,
    target_mean: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a geometric stopping prior whose truncated mean is target_mean."""

    if max_loops < 2:
        raise ValueError("max_loops must be >= 2")
    if not 1.0 < target_mean < float(max_loops):
        raise ValueError("target_mean must be strictly between 1 and max_loops")

    def distribution(probability: float) -> torch.Tensor:
        p = torch.tensor(probability, dtype=torch.float64)
        survival = torch.tensor(1.0, dtype=torch.float64)
        values = []
        for loop_idx in range(max_loops):
            if loop_idx == max_loops - 1:
                values.append(survival)
            else:
                values.append(survival * p)
                survival = survival * (1.0 - p)
        return torch.stack(values)

    low, high = 1e-8, 1.0 - 1e-8
    loop_ids = torch.arange(1, max_loops + 1, dtype=torch.float64)
    for _ in range(100):
        midpoint = (low + high) / 2.0
        mean = float((distribution(midpoint) * loop_ids).sum().item())
        if mean > target_mean:
            low = midpoint
        else:
            high = midpoint
    prior = distribution((low + high) / 2.0)
    return prior.to(dtype=dtype)


def ponder_outcome_loss(
    predictor: torch.nn.Module,
    pooled_features: torch.Tensor,
    per_loop_outcome_nll: torch.Tensor,
    *,
    prior: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    weights = halting_weights_from_features(predictor, pooled_features)
    nll = (weights.float() * per_loop_outcome_nll.float()).sum(dim=-1).mean()
    expanded_prior = prior.to(device=weights.device, dtype=weights.dtype).view(1, -1).expand_as(weights)
    kl = categorical_kl(weights.float(), expanded_prior.float()).mean()
    loss = nll + float(beta) * kl
    metrics = {"loss": loss.detach(), "outcome_nll": nll.detach(), "kl": kl.detach()}
    return loss, metrics, weights


def _average_ranks(values: Sequence[float | int]) -> list[float]:
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average
        start = end
    return ranks


def spearman_correlation(left: Sequence[float | int], right: Sequence[float | int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Spearman inputs must be nonempty and equally sized")
    x = _average_ranks(left)
    y = _average_ranks(right)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_norm = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return float(numerator / (x_norm * y_norm))


def _accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def summarize_selector_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_depth[int(row["depth"])].append(row)
    per_depth: dict[str, Any] = {}
    for depth, depth_rows in sorted(by_depth.items()):
        total = len(depth_rows)
        selection_correct = sum(int(row["selected_loop"]) == depth for row in depth_rows)
        forced_correct = sum(bool(row["forced_hit"]) for row in depth_rows)
        selected_correct = sum(bool(row["selected_hit"]) for row in depth_rows)
        forced_accuracy = _accuracy(forced_correct, total)
        selected_accuracy = _accuracy(selected_correct, total)
        per_depth[str(depth)] = {
            "rows": total,
            "selection_correct": selection_correct,
            "selection_accuracy": _accuracy(selection_correct, total),
            "forced_correct": forced_correct,
            "forced_accuracy": forced_accuracy,
            "selected_correct": selected_correct,
            "selected_accuracy": selected_accuracy,
            "selected_minus_forced": selected_accuracy - forced_accuracy,
        }
    total = len(materialized)
    selected_correct = sum(bool(row["selected_hit"]) for row in materialized)
    forced_correct = sum(bool(row["forced_hit"]) for row in materialized)
    return {
        "rows": total,
        "by_depth": per_depth,
        "selection_accuracy": _accuracy(
            sum(int(row["selected_loop"]) == int(row["depth"]) for row in materialized),
            total,
        ),
        "selected_answer_accuracy": _accuracy(selected_correct, total),
        "forced_answer_accuracy": _accuracy(forced_correct, total),
        "mean_selected_depth": (
            sum(int(row["selected_loop"]) for row in materialized) / total if total else 0.0
        ),
        "mean_expected_depth": (
            sum(float(row.get("expected_loops", row["selected_loop"])) for row in materialized) / total
            if total
            else 0.0
        ),
        "selected_depth_histogram": {
            str(loop): sum(int(row["selected_loop"]) == loop for row in materialized)
            for loop in range(1, 13)
        },
    }


def evaluate_s1_gate(
    rows: Iterable[dict[str, Any]],
    *,
    min_correct_per_depth: int = 46,
    answer_delta_floor: float = -0.03,
) -> dict[str, Any]:
    summary = summarize_selector_rows(rows)
    per_depth = summary["by_depth"]
    depth_checks = {
        depth: int(values["selection_correct"]) >= min_correct_per_depth
        for depth, values in per_depth.items()
    }
    answer_checks = {
        depth: float(values["selected_minus_forced"]) >= answer_delta_floor
        for depth, values in per_depth.items()
    }
    complete = set(per_depth) == {str(depth) for depth in range(1, 13)}
    summary.update(
        {
            "kind": "bounded_depth_selector_s1_gate",
            "scope": "depth_reading_from_stated_depth_prompt_not_difficulty_inference",
            "min_correct_per_depth": min_correct_per_depth,
            "answer_delta_floor": answer_delta_floor,
            "complete_depths_1_12": complete,
            "selection_gate_by_depth": depth_checks,
            "answer_delta_gate_by_depth": answer_checks,
            "all_depth_selection_gates_pass": complete and all(depth_checks.values()),
            "all_answer_delta_gates_pass": complete and all(answer_checks.values()),
        }
    )
    summary["status"] = (
        "pass"
        if summary["all_depth_selection_gates_pass"] and summary["all_answer_delta_gates_pass"]
        else "blocked"
    )
    return summary


def optimization_trace_gate(trace: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(trace) < 20:
        return {
            "loss_decreased": False,
            "kl_stable": False,
            "reason": "fewer_than_20_logged_steps",
        }
    window = max(10, len(trace) // 10)
    first_loss = sum(float(row["loss"]) for row in trace[:window]) / window
    last_loss = sum(float(row["loss"]) for row in trace[-window:]) / window
    tail_kl = [float(row["kl"]) for row in trace[-window:]]
    half = max(1, len(tail_kl) // 2)
    first_tail = sum(tail_kl[:half]) / len(tail_kl[:half])
    second_tail = sum(tail_kl[half:]) / len(tail_kl[half:])
    denominator = max(abs(first_tail), abs(second_tail), 1e-6)
    relative_kl_drift = abs(second_tail - first_tail) / denominator
    finite = all(math.isfinite(value) for value in (first_loss, last_loss, *tail_kl))
    return {
        "first_window_mean_loss": first_loss,
        "last_window_mean_loss": last_loss,
        "loss_ratio": last_loss / max(abs(first_loss), 1e-8),
        "loss_decreased": finite and last_loss <= 0.9 * first_loss,
        "tail_kl_first_half_mean": first_tail,
        "tail_kl_second_half_mean": second_tail,
        "tail_kl_relative_drift": relative_kl_drift,
        "kl_stable": finite and relative_kl_drift <= 0.25,
    }


def evaluate_s2_gate(
    rows: Iterable[dict[str, Any]],
    *,
    training_trace: Sequence[dict[str, Any]],
    s1_gate: dict[str, Any],
    strong_spearman_floor: float = 0.8,
    partial_spearman_floor: float = 0.3,
    answer_delta_from_s1_floor: float = -0.05,
) -> dict[str, Any]:
    materialized = list(rows)
    summary = summarize_selector_rows(materialized)
    trace_gate = optimization_trace_gate(training_trace)
    correlation = spearman_correlation(
        [int(row["selected_loop"]) for row in materialized],
        [int(row["depth"]) for row in materialized],
    )
    answer_delta = float(summary["selected_answer_accuracy"]) - float(
        s1_gate["selected_answer_accuracy"]
    )
    mean_depth_in_range = 1.5 < float(summary["mean_selected_depth"]) < 11.5
    accuracy_preserved = answer_delta >= answer_delta_from_s1_floor
    base_pass = bool(
        trace_gate["loss_decreased"]
        and trace_gate["kl_stable"]
        and mean_depth_in_range
        and accuracy_preserved
    )
    if base_pass and correlation >= strong_spearman_floor:
        status = "strong"
    elif base_pass and correlation >= partial_spearman_floor:
        status = "partial"
    else:
        status = "collapse"
    summary.update(
        {
            "kind": "bounded_depth_selector_s2_gate",
            "scope": "outcome_only_ponder_depth_discovery_on_stated_depth_random_tables",
            "training_gate": trace_gate,
            "spearman_selected_vs_true": correlation,
            "mean_selected_depth_in_open_interval_1_5_11_5": mean_depth_in_range,
            "selected_answer_minus_s1": answer_delta,
            "answer_delta_from_s1_floor": answer_delta_from_s1_floor,
            "accuracy_preserved": accuracy_preserved,
            "strong_spearman_floor": strong_spearman_floor,
            "partial_spearman_floor": partial_spearman_floor,
            "status": status,
            "collapse_sentence": (
                "Supervised routing works, but unsupervised depth discovery does not train on this family."
                if status == "collapse"
                else None
            ),
        }
    )
    return summary
