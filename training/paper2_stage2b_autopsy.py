"""Lock and analysis primitives for the score-only Stage 2B-A autopsy."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


LOCK_KIND = "paper2_stage2b_autopsy_lock_v1"
DIAGNOSTIC_MODES = {
    "standard",
    "zero_write",
    "constitutive_off",
    "fresh_state_each_loop",
    "inherited_flow_off",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_autopsy_lock(lock: Mapping[str, Any], *, require_signature: bool) -> None:
    if lock.get("kind") != LOCK_KIND:
        raise RuntimeError("wrong Stage 2B-A lock kind")
    if lock.get("optimizer_steps_allowed") != 0 or lock.get("training_authorized") is not False:
        raise RuntimeError("Stage 2B-A must remain score-only")
    sealed = lock.get("sealed_partitions", {})
    if sealed != {"confirm_scored": False, "eval_e_scored": False, "remain_sealed": True}:
        raise RuntimeError("Stage 2B-A sealed-partition contract changed")
    if lock.get("amplitude_response", {}).get("gamma") != [0.0, 0.01, 0.02, 0.05]:
        raise RuntimeError("Stage 2B-A amplitude cells changed")
    modes = set(lock.get("component_attribution", {}).get("diagnostic_modes", []))
    if modes != {"standard", "constitutive_off", "fresh_state_each_loop", "inherited_flow_off"}:
        raise RuntimeError("Stage 2B-A component cells changed")
    if not modes <= DIAGNOSTIC_MODES:
        raise RuntimeError("Stage 2B-A contains an unknown diagnostic mode")
    if lock.get("flow_loops") != [1, 2, 3, 4]:
        raise RuntimeError("Stage 2B-A K sweep changed")
    if require_signature:
        if lock.get("status") != "SIGNED" or lock.get("mark_signed") is not True:
            raise RuntimeError("Stage 2B-A score-only execution remains unsigned")
        if lock.get("locked_before_model_contact") is not True:
            raise RuntimeError("Stage 2B-A lock was not frozen before model contact")
        unresolved = lock.get("unresolved_lock_fields", [])
        if unresolved:
            raise RuntimeError(f"Stage 2B-A lock has unresolved fields: {unresolved}")


def load_and_validate_autopsy_lock(
    path: str | Path, *, require_signature: bool
) -> dict[str, Any]:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_autopsy_lock(lock, require_signature=require_signature)
    return lock


def stable_dev2_subsample(
    rows: Sequence[Mapping[str, Any]], *, size: int = 256, seed: int = 20260819
) -> list[dict[str, Any]]:
    """Select a proportional battery-stratified panel with no model signal."""

    if size < 1 or size > len(rows):
        raise ValueError("invalid Stage 2B-A DEV-2 subsample size")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["battery"])].append(row)
    total = len(rows)
    quotas = {
        battery: max(1, int(math.floor(size * len(values) / total)))
        for battery, values in grouped.items()
    }
    while sum(quotas.values()) > size:
        candidate = max(
            (name for name in quotas if quotas[name] > 1),
            key=lambda name: (quotas[name] - size * len(grouped[name]) / total, name),
        )
        quotas[candidate] -= 1
    remainders = sorted(
        grouped,
        key=lambda name: (size * len(grouped[name]) / total - quotas[name], name),
        reverse=True,
    )
    cursor = 0
    while sum(quotas.values()) < size:
        name = remainders[cursor % len(remainders)]
        if quotas[name] < len(grouped[name]):
            quotas[name] += 1
        cursor += 1

    selected = []
    for battery, values in sorted(grouped.items()):
        ranked = sorted(
            values,
            key=lambda row: hashlib.sha256(
                f"{seed}:{battery}:{row['item_id']}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(dict(row) for row in ranked[: quotas[battery]])
    selected.sort(key=lambda row: str(row["item_id"]))
    if len(selected) != size or len({str(row["item_id"]) for row in selected}) != size:
        raise RuntimeError("Stage 2B-A DEV-2 subsample construction failed")
    return selected


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired correlation requires equal nontrivial vectors")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = sum((value - left_mean) ** 2 for value in left) ** 0.5
    right_norm = sum((value - right_mean) ** 2 for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return numerator / (left_norm * right_norm)


def rank_values(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in range(start, end):
            ranks[indexed[index][0]] = rank
        start = end
    return ranks


def margin_correlation_receipt(
    rows: Sequence[Mapping[str, Any]], *, k_left: int = 1, k_right: int = 4
) -> dict[str, Any]:
    left = [float(row["per_loop_mean_teacher_token_margin"][k_left - 1]) for row in rows]
    right = [float(row["per_loop_mean_teacher_token_margin"][k_right - 1]) for row in rows]
    return {
        "rows": len(rows),
        "k_left": k_left,
        "k_right": k_right,
        "pearson": pearson_correlation(left, right),
        "spearman": pearson_correlation(rank_values(left), rank_values(right)),
        "mean_left": sum(left) / len(left),
        "mean_right": sum(right) / len(right),
    }


def decision_mapping(flags: Mapping[str, bool]) -> list[str]:
    actions = []
    if flags.get("h_b_magnitude"):
        actions.append("radius_control_successor")
    if flags.get("h_c_constitutive"):
        actions.append("gated_additive_constructor_successor")
    if flags.get("h_a_attractor"):
        actions.append("task_preservation_anchor_required")
    return actions


def battery_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["battery"]) for row in rows).items()))
