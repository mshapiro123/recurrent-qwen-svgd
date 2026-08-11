"""Training-inert P3.3 data staging and Tier-1 observatory contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from training.paper2_phase3_p32 import GateLabel, canonical_sha256


P33_TEACHABILITY_THRESHOLD = 0.70
P33_AUDIT_ROWS = 4_096
P33_NEGATIVE_AUDIT_ROWS = 12_288
P33_RETENTION_PANEL_ROWS = 1_024
P33_NEGATIVE_TO_POSITIVE = 3
P33_SPLIT_SEED = 20260810
P33_PROJECTION_SEED = 20260810
P33_GATE_CEILING = 0.02
P33_AUDIT_RADIUS = 0.15
P33_RMS_CAP = 0.550893


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(row: Mapping[str, Any], *, namespace: str, seed: int) -> bytes:
    return hashlib.sha256(
        f"{seed}:{namespace}:{row['record_id']}".encode("utf-8")
    ).digest()


def _teachability_deciles(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    ranked = sorted(rows, key=lambda row: (float(row["teachability"]), str(row["record_id"])))
    return {
        str(row["record_id"]): min(9, int(index * 10 / max(1, len(ranked))))
        for index, row in enumerate(ranked)
    }


def stratified_audit_slice(
    rows: Sequence[Mapping[str, Any]],
    *,
    size: int = P33_AUDIT_ROWS,
    seed: int = P33_SPLIT_SEED,
) -> list[dict[str, Any]]:
    if size <= 0 or size >= len(rows):
        raise ValueError("P3.3 audit size must leave non-audit positives")
    decile = _teachability_deciles(rows)
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["horizon"]), decile[str(row["record_id"])])].append(row)
    exact = {key: size * len(group) / len(rows) for key, group in groups.items()}
    allocation = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = size - sum(allocation.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1
    selected = []
    for key, group in sorted(groups.items()):
        ranked = sorted(group, key=lambda row: _stable_key(row, namespace="audit", seed=seed))
        count = allocation[key]
        if count > len(ranked):
            raise RuntimeError(f"P3.3 audit stratum over-allocation: {key}")
        for row in ranked[:count]:
            selected.append({**dict(row), "teachability_decile": key[1]})
    if len(selected) != size:
        raise RuntimeError(f"P3.3 audit slice size changed: {len(selected)}")
    return sorted(selected, key=lambda row: str(row["record_id"]))


def stratified_retention_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    size: int = P33_RETENTION_PANEL_ROWS,
) -> list[dict[str, Any]]:
    """Select a deterministic, horizon-balanced confident-agreement panel."""

    horizons = (1, 2, 3, 4)
    if size <= 0 or size % len(horizons):
        raise ValueError("P3.3 retention panel must divide evenly across four horizons")
    per_horizon = size // len(horizons)
    selected: list[dict[str, Any]] = []
    for horizon in horizons:
        candidates = [row for row in rows if int(row["horizon"]) == horizon]
        ranked = sorted(
            candidates,
            key=lambda row: (
                -min(
                    float(row["student_top1_probability"]),
                    float(row["teacher_14b_top1_probability"]),
                ),
                str(row["record_id"]),
            ),
        )
        if len(ranked) < per_horizon:
            raise RuntimeError(
                "P3.3 retention panel lacks confident agreements at "
                f"horizon {horizon}: {len(ranked)} < {per_horizon}"
            )
        for rank, row in enumerate(ranked[:per_horizon], start=1):
            selected.append(
                {
                    **dict(row),
                    "retention_horizon_rank": rank,
                    "retention_confidence": min(
                        float(row["student_top1_probability"]),
                        float(row["teacher_14b_top1_probability"]),
                    ),
                }
            )
    return sorted(selected, key=lambda row: str(row["record_id"]))


def prepare_training_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    teachability_threshold: float = P33_TEACHABILITY_THRESHOLD,
    audit_rows: int = P33_AUDIT_ROWS,
    negative_audit_rows: int = P33_NEGATIVE_AUDIT_ROWS,
    retention_panel_rows: int = P33_RETENTION_PANEL_ROWS,
    negative_to_positive: int = P33_NEGATIVE_TO_POSITIVE,
    seed: int = P33_SPLIT_SEED,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    required_fields = {
        "record_id",
        "flip_candidate_14b",
        "cross_scale_consistent",
        "teachability",
        "prediction_position",
        "horizon",
        "student_top1_probability",
        "teacher_14b_top1_probability",
        "teacher_js_divergence",
    }
    for index, row in enumerate(records):
        missing = required_fields - set(row)
        if missing:
            raise ValueError(
                f"P3.3 coverage record {index} lacks fields: {sorted(missing)}"
            )
    ids = [str(row["record_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("P3.3 coverage records must have unique ids")
    threshold_eligible = [
        row
        for row in records
        if bool(row["flip_candidate_14b"])
        and bool(row["cross_scale_consistent"])
        and float(row["teachability"]) >= teachability_threshold
    ]
    positives = [row for row in threshold_eligible if int(row["prediction_position"]) != 0]
    audit = stratified_audit_slice(positives, size=audit_rows, seed=seed)
    audit_ids = {str(row["record_id"]) for row in audit}
    train_positives = [row for row in positives if str(row["record_id"]) not in audit_ids]

    agreements = [
        row
        for row in records
        if not bool(row["flip_candidate_14b"])
        and int(row["prediction_position"]) != 0
        and "student_top1_probability" in row
        and "teacher_14b_top1_probability" in row
    ]
    ranked_negatives = sorted(
        agreements,
        key=lambda row: (
            -min(
                float(row["student_top1_probability"]),
                float(row["teacher_14b_top1_probability"]),
            ),
            str(row["record_id"]),
        ),
    )
    negative_count = negative_to_positive * len(train_positives)
    required_negatives = negative_count + int(negative_audit_rows) + int(retention_panel_rows)
    if len(ranked_negatives) < required_negatives:
        raise RuntimeError(
            "P3.3 negative pool too small for disjoint training and audit cohorts: "
            f"{len(ranked_negatives)} < {required_negatives}"
        )
    negatives = ranked_negatives[:negative_count]
    negative_audit = [
        {**dict(row), "negative_confidence_rank": negative_count + index + 1}
        for index, row in enumerate(
            ranked_negatives[negative_count : negative_count + negative_audit_rows]
        )
    ]
    negative_audit_ids = {str(row["record_id"]) for row in negative_audit}
    retention_pool = ranked_negatives[negative_count + negative_audit_rows :]
    retention_panel = stratified_retention_panel(
        retention_pool,
        size=retention_panel_rows,
    )
    retention_ids = {str(row["record_id"]) for row in retention_panel}
    confidence_cut = min(
        min(
            float(row["student_top1_probability"]),
            float(row["teacher_14b_top1_probability"]),
        )
        for row in negatives
    )
    labels = {
        **{str(row["record_id"]): int(GateLabel.POSITIVE) for row in train_positives},
        **{str(row["record_id"]): int(GateLabel.NEGATIVE) for row in negatives},
    }
    staged = []
    for row in records:
        record_id = str(row["record_id"])
        label = labels.get(record_id, int(GateLabel.IGNORED))
        if int(row["prediction_position"]) == 0:
            label = int(GateLabel.IGNORED)
        staged.append(
            {
                **dict(row),
                "gate_label": label,
                "audit_holdout": (
                    record_id in audit_ids
                    or record_id in negative_audit_ids
                    or record_id in retention_ids
                ),
                "audit_role": (
                    "positive"
                    if record_id in audit_ids
                    else "negative"
                    if record_id in negative_audit_ids
                    else "retention"
                    if record_id in retention_ids
                    else None
                ),
                "training_eligible": label != int(GateLabel.IGNORED),
            }
        )
    positive_count = len(train_positives)
    active = positive_count + negative_count
    class_weights = {
        "positive": active / (2.0 * positive_count),
        "negative": active / (2.0 * negative_count),
        "ignored": 0.0,
    }
    counts = Counter(row["gate_label"] for row in staged)
    receipt = {
        "kind": "paper2_phase3_p33_data_staging_receipt_v3",
        "status": "staged_build_only_training_unauthorized",
        "teachability_threshold": teachability_threshold,
        "strict_concurrent_count_at_threshold_before_position_zero": len(threshold_eligible),
        "position_zero_excluded_from_positive_count": len(threshold_eligible) - len(positives),
        "strict_positive_count_before_audit": len(positives),
        "audit_rows": len(audit),
        "negative_audit_rows": len(negative_audit),
        "retention_panel_rows": len(retention_panel),
        "train_positive_count": positive_count,
        "train_negative_count": negative_count,
        "negative_to_positive_ratio": negative_count / positive_count,
        "negative_confidence_statistic": "min(student_top1_probability,teacher_14b_top1_probability)",
        "teacher_tier_js_field": "teacher_js_divergence",
        "realized_negative_confidence_cut": confidence_cut,
        "inverse_class_weights": class_weights,
        "label_counts_all_records": {
            "positive": counts[int(GateLabel.POSITIVE)],
            "negative": counts[int(GateLabel.NEGATIVE)],
            "ignored": counts[int(GateLabel.IGNORED)],
        },
        "audit_slice_sha256": canonical_sha256(
            [
                {
                    "record_id": row["record_id"],
                    "horizon": row["horizon"],
                    "teachability_decile": row["teachability_decile"],
                }
                for row in audit
            ]
        ),
        "negative_audit_slice_sha256": canonical_sha256(
            [
                {
                    "record_id": row["record_id"],
                    "negative_confidence_rank": row["negative_confidence_rank"],
                }
                for row in negative_audit
            ]
        ),
        "retention_panel_sha256": canonical_sha256(
            [
                {
                    "record_id": row["record_id"],
                    "horizon": row["horizon"],
                    "retention_horizon_rank": row["retention_horizon_rank"],
                    "retention_confidence": row["retention_confidence"],
                }
                for row in retention_panel
            ]
        ),
        "audit_cohorts_disjoint": not bool(
            (audit_ids & negative_audit_ids)
            or (audit_ids & retention_ids)
            or (negative_audit_ids & retention_ids)
        ),
        "negative_audit_excluded_from_training": not bool(
            negative_audit_ids & set(labels)
        ),
        "retention_panel_excluded_from_training": not bool(
            retention_ids & set(labels)
        ),
        "retention_panel_estimand": (
            "fraction of positions where augmented top1 matches frozen base top1"
        ),
        "retention_panel_by_horizon": dict(
            Counter(str(row["horizon"]) for row in retention_panel)
        ),
        "retention_panel_minimum_confidence": min(
            float(row["retention_confidence"]) for row in retention_panel
        ),
        "audit_by_horizon_decile": dict(
            Counter(f"h{row['horizon']}_d{row['teachability_decile']}" for row in audit)
        ),
        "position_zero_labels": dict(
            Counter(
                str(row["gate_label"])
                for row in staged
                if int(row["prediction_position"]) == 0
            )
        ),
        "p33_training_authorized": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }
    return staged, audit, negative_audit, retention_panel, receipt


def fixed_random_projection(
    *,
    input_dim: int = 1_024,
    output_dim: int = 128,
    seed: int = P33_PROJECTION_SEED,
) -> torch.Tensor:
    if output_dim > input_dim:
        raise ValueError("projection output cannot exceed input width")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float64)
    orthogonal, _ = torch.linalg.qr(values, mode="reduced")
    return orthogonal.T.contiguous().to(torch.float32)


def forced_audit_write(
    delta: torch.Tensor,
    h0: torch.Tensor,
    *,
    radius: float = P33_AUDIT_RADIUS,
    rms_cap: float = P33_RMS_CAP,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply the V-series audit radius without changing the deployed bridge."""

    if delta.shape != h0.shape or delta.ndim != 3:
        raise ValueError("forced audit write requires shape-matched hidden tensors")
    if not 0.0 < float(radius) <= 1.0:
        raise ValueError("audit radius must be inside (0, 1]")
    delta_rms = delta.float().square().mean(dim=-1, keepdim=True).sqrt()
    reference = (
        h0.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_max(float(rms_cap))
    )
    normalized = delta.float() / delta_rms.clamp_min(float(eps)) * reference
    return normalized.to(delta.dtype) * float(radius)


def observatory_metrics(
    *,
    states: torch.Tensor,
    writes: torch.Tensor,
    loss_gradient: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Tier-1 scalars for canonical-state trajectories and bridge writes.

    ``states`` is ``[batch, loops+1, slots, latent]``. ``writes`` is
    ``[batch, loops, positions, hidden]``. Only batch and loop axes must align;
    the observatory deliberately does not conflate the carried canonical state
    with the hidden-stream bridge write.
    """

    if states.ndim != 4 or writes.ndim != 4:
        raise ValueError("observatory states/writes must be rank four")
    if states.shape[0] != writes.shape[0]:
        raise ValueError("observatory states/writes batch axes do not align")
    if states.shape[1] != writes.shape[1] + 1:
        raise ValueError("states must include the initial state before loop writes")
    state_before = states[:, :-1].float()
    write = writes.float()
    write_rms = write.flatten(2).square().mean(dim=-1).sqrt()
    state_rms = state_before.flatten(2).square().mean(dim=-1).sqrt().clamp_min(1e-8)
    steps = states[:, 1:].float() - states[:, :-1].float()
    step_norm = steps.flatten(2).norm(dim=-1)
    displacement = (states[:, 1:].float() - states[:, :1].float()).flatten(2).norm(dim=-1)
    tortuosity = step_norm.cumsum(dim=1) / displacement.clamp_min(1e-8)
    if steps.shape[1] > 1:
        turning = F.cosine_similarity(
            steps[:, 1:].flatten(2), steps[:, :-1].flatten(2), dim=-1
        ).clamp(-1.0, 1.0).acos()
        turning = torch.cat([turning.new_zeros((turning.shape[0], 1)), turning], dim=1)
    else:
        turning = step_norm.new_zeros(step_norm.shape)
    flattened = states.float().permute(1, 0, 2, 3).flatten(1, 2)
    effective_rank = []
    participation = []
    for loop_state in flattened:
        centered = loop_state - loop_state.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        energy = singular.square()
        probability = energy / energy.sum().clamp_min(1e-12)
        effective_rank.append((-(probability * probability.clamp_min(1e-30).log()).sum()).exp())
        participation.append(energy.sum().square() / energy.square().sum().clamp_min(1e-12))
    result = {
        "bridge_write_ratio_r_b": write_rms / state_rms,
        "tortuosity": tortuosity,
        "turning_angle_radians": turning,
        "fixed_point_residual": step_norm,
        "effective_rank": torch.stack(effective_rank),
        "participation_ratio": torch.stack(participation),
    }
    if loss_gradient is not None:
        if loss_gradient.shape != writes.shape:
            raise ValueError("loss gradient must match writes")
        result["gradient_dot_write"] = (
            loss_gradient.float() * write
        ).flatten(2).sum(dim=-1)
    return result


def observatory_event_rows(
    *,
    record_ids: Sequence[str],
    metrics: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Convert Tier-1 tensors into the lock-bound prompt-by-loop event grain."""

    required = {
        "bridge_write_ratio_r_b",
        "tortuosity",
        "turning_angle_radians",
        "fixed_point_residual",
        "effective_rank",
        "participation_ratio",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"observatory metrics missing fields: {sorted(missing)}")
    ratio = metrics["bridge_write_ratio_r_b"]
    if ratio.ndim != 2 or ratio.shape[0] != len(record_ids):
        raise ValueError("event rows require [batch, loops] bridge ratios")
    loops = ratio.shape[1]
    rows = []
    for batch_index, record_id in enumerate(record_ids):
        for loop_index in range(loops):
            row = {
                "kind": "paper2_phase3_tier1_observatory_event_v1",
                "record_id": str(record_id),
                "loop_index": loop_index + 1,
                "bridge_write_ratio_r_b": float(ratio[batch_index, loop_index]),
                "tortuosity": float(metrics["tortuosity"][batch_index, loop_index]),
                "turning_angle_radians": float(
                    metrics["turning_angle_radians"][batch_index, loop_index]
                ),
                "fixed_point_residual": float(
                    metrics["fixed_point_residual"][batch_index, loop_index]
                ),
                "state_effective_rank": float(metrics["effective_rank"][loop_index + 1]),
                "state_participation_ratio": float(
                    metrics["participation_ratio"][loop_index + 1]
                ),
            }
            if "gradient_dot_write" in metrics:
                row["gradient_dot_write"] = float(
                    metrics["gradient_dot_write"][batch_index, loop_index]
                )
            rows.append(row)
    return rows


def state_sketches(
    states: torch.Tensor,
    *,
    random_projection: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Preserve raw/normalized canonical state and one fixed random sketch."""

    if states.ndim != 4:
        raise ValueError("state sketches require [batch, loops, slots, latent]")
    flattened = states.float().flatten(2)
    if flattened.shape[-1] != random_projection.shape[-1]:
        raise ValueError("random projection width does not match flattened canonical state")
    normalized = F.normalize(flattened, dim=-1)
    return {
        "canonical_state_raw": flattened,
        "canonical_state_normalized": normalized,
        "fixed_random_projection": flattened @ random_projection.T.float(),
    }


def intervene_state(
    state: torch.Tensor,
    *,
    mode: str,
    seed: int,
    stale_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, bool]:
    def match_norm(candidate: torch.Tensor) -> torch.Tensor:
        target_norm = state.float().flatten(1).norm(dim=1)
        candidate_norm = candidate.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        scale = target_norm / candidate_norm
        return candidate * scale.view(-1, *([1] * (state.ndim - 1))).to(candidate.dtype)

    if mode == "zero":
        return torch.zeros_like(state), False
    if mode == "norm_matched_random":
        generator = torch.Generator(device=state.device).manual_seed(seed)
        random = torch.randn(state.shape, generator=generator, device=state.device, dtype=state.dtype)
        return match_norm(random), False
    if mode == "cross_example":
        if state.shape[0] < 2:
            raise ValueError("cross-example permutation requires batch >= 2")
        generator = torch.Generator(device=state.device).manual_seed(seed)
        shift = int(
            torch.randint(
                1, state.shape[0], (1,), generator=generator, device=state.device
            ).item()
        )
        permutation = torch.arange(state.shape[0], device=state.device).roll(shift)
        return match_norm(state.index_select(0, permutation)), False
    if mode == "stale":
        if stale_state is None or stale_state.shape != state.shape:
            raise ValueError("stale-state intervention requires a shape-matched cached state")
        return match_norm(stale_state), False
    if mode == "bypass":
        return state, True
    raise ValueError(f"unknown A_state intervention mode {mode}")


def paired_astate_execution(
    forward_from_cached: Callable[[torch.Tensor, bool], torch.Tensor],
    *,
    cached_state: torch.Tensor,
    baseline_without_recurrence: torch.Tensor,
    mode: str,
    seed: int,
    stale_state: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Execute baseline/intervention from the same cached pre-intervention state."""

    baseline = forward_from_cached(cached_state, False)
    intervened_state, bypass = intervene_state(
        cached_state, mode=mode, seed=seed, stale_state=stale_state
    )
    intervention = forward_from_cached(intervened_state, bypass)
    numerator = baseline.float() - intervention.float()
    denominator = baseline.float() - baseline_without_recurrence.float()
    ratio = numerator / denominator
    return {
        "mode": mode,
        "numerator": numerator,
        "denominator": denominator,
        "a_state_unclipped": ratio,
        "paired_from_same_cached_state": True,
        "ratio_clipped": False,
    }
