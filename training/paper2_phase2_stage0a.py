"""Pure contracts and metrics for the development-only Phase-2 Stage 0A job."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
DC2_CONSTANTS_PATH = ROOT / "training/paper2_phase2_dc2_constants.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


STAGE0A_CONFIG: dict[str, Any] = {
    "kind": "paper2_phase2_stage0a_config",
    "version": "stage0a_v1_20260803",
    "run_id": "stage5_paper2_phase2_stage0a_20260803",
    "data_partition": "DEV-C",
    "data_sha256": "05bca2ee3ba71421296b2e31a0439746eb9c1b0e15e2cea4471be202ab6ac29d",
    "seed": 20260803,
    "anchor_count": 50_000,
    "anchors_per_stratum": {"general": 25_000, "code": 25_000},
    "boundary_sample_count": 200_000,
    "horizons": [1, 2, 3, 4],
    "top_k": 128,
    "full_logit_audit_fraction": 0.01,
    "selected_layer_ordinals_one_based": [16, 32, 44],
    "teacher_state_model": {
        "key": "teacher_14b",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "hidden_size": 5120,
        "num_hidden_layers": 48,
    },
    "models": {
        "student_0p5b": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "role": "zero_loop_student",
        },
        "teacher_7b": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
            "role": "broad_teacher",
        },
        "teacher_14b": {
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
            "role": "canonical_state_teacher",
        },
        "teacher_32b": {
            "model": "Qwen/Qwen2.5-32B-Instruct",
            "revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
            "role": "cascaded_teacher",
        },
    },
    "cascade": {
        "query_32b_on_7b_14b_argmax_disagreement": True,
        "query_32b_on_verifier_available": True,
        "stable_audit_fraction": 0.01,
    },
    "dc2_constants_sha256": sha256_file(DC2_CONSTANTS_PATH),
    "training_started": False,
    "optimizer_steps": 0,
    "frozen_evaluation_partitions_touched": [],
}


def _stable_hash(*values: object, seed: int) -> str:
    payload = ":".join([str(seed), *[str(value) for value in values]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_id(row: dict[str, Any], row_index: int) -> str:
    return str(row.get("row_id") or row.get("id") or row_index)


def select_stage0a_samples(
    rows: Sequence[dict[str, Any]],
    *,
    anchors_per_stratum: dict[str, int],
    horizons: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    """Select deterministic, stratum-balanced anchors without crossing rows."""

    horizon_values = tuple(sorted({int(value) for value in horizons}))
    if not horizon_values or horizon_values[0] < 1:
        raise ValueError("horizons must be positive")
    if horizon_values != tuple(range(1, max(horizon_values) + 1)):
        raise ValueError("horizons must be contiguous from one")

    document_ids = [str(row.get("document_id", "")) for row in rows]
    if not all(document_ids):
        raise ValueError("Stage 0A input rows must carry a document_id")

    max_horizon = max(horizon_values)
    candidates: dict[str, list[tuple[str, int, int]]] = {
        stratum: [] for stratum in anchors_per_stratum
    }
    for row_index, row in enumerate(rows):
        stratum = str(row.get("stratum") or "")
        if stratum not in candidates:
            continue
        values = list(row.get("input_ids") or [])
        # Prediction position p emits token p+1. A horizon-j teacher state is
        # taken after consuming token p+j, so p+max_horizon must exist.
        for prediction_position in range(max(0, len(values) - max_horizon)):
            candidates[stratum].append(
                (
                    _stable_hash(
                        _row_id(row, row_index), prediction_position, seed=seed
                    ),
                    row_index,
                    prediction_position,
                )
            )

    selected: list[tuple[str, int, int, str]] = []
    for stratum, required in anchors_per_stratum.items():
        available = sorted(candidates.get(stratum, []))
        nonoverlapping: list[tuple[str, int, int]] = []
        occupied_by_row: dict[int, set[int]] = {}
        for candidate in available:
            _selection_hash, row_index, position = candidate
            occupied = occupied_by_row.setdefault(row_index, set())
            span = set(range(position, position + max_horizon))
            if occupied.intersection(span):
                continue
            occupied.update(span)
            nonoverlapping.append(candidate)
            if len(nonoverlapping) == int(required):
                break
        if len(nonoverlapping) < int(required):
            raise ValueError(
                "insufficient eligible anchors for Stage 0A: "
                f"stratum={stratum} required={required} "
                f"nonoverlapping_available={len(nonoverlapping)}"
            )
        selected.extend((*entry, stratum) for entry in nonoverlapping)

    # Row-major ordering minimizes repeated teacher forwards while retaining a
    # stable hash as the selection mechanism.
    selected.sort(key=lambda item: (item[1], item[2], item[3], item[0]))
    anchors: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    counts_by_stratum = {stratum: 0 for stratum in anchors_per_stratum}
    for anchor_index, (selection_hash, row_index, position, stratum) in enumerate(selected):
        row = rows[row_index]
        anchor_key = _stable_hash(
            "anchor", _row_id(row, row_index), position, seed=seed
        )
        anchors.append(
            {
                "anchor_index": anchor_index,
                "anchor_key": anchor_key,
                "selection_hash": selection_hash,
                "row_index": row_index,
                "row_id": _row_id(row, row_index),
                "document_id": str(row["document_id"]),
                "stratum": stratum,
                "anchor_prediction_position": position,
            }
        )
        for horizon in horizon_values:
            prediction_position = position + horizon - 1
            state_position = position + horizon
            sample_key = _stable_hash(
                "sample", _row_id(row, row_index), position, horizon, seed=seed
            )
            samples.append(
                {
                    "sample_index": len(samples),
                    "sample_key": sample_key,
                    "anchor_index": anchor_index,
                    "row_index": row_index,
                    "row_id": _row_id(row, row_index),
                    "document_id": str(row["document_id"]),
                    "stratum": stratum,
                    "horizon": horizon,
                    "prediction_position": prediction_position,
                    "state_position": state_position,
                    "observed_next_token_id": int(row["input_ids"][prediction_position + 1]),
                    "verifier_available": bool(row.get("verifier_label") is not None),
                    "verifier_label": row.get("verifier_label"),
                }
            )
            counts_by_stratum[stratum] += 1

    position_key_sha256 = hashlib.sha256(
        "\n".join(sample["sample_key"] for sample in samples).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "paper2_phase2_stage0a_sample_manifest",
        "seed": int(seed),
        "horizons": list(horizon_values),
        "anchors": anchors,
        "samples": samples,
        "anchor_count": len(anchors),
        "boundary_sample_count": len(samples),
        "counts_by_stratum": dict(sorted(counts_by_stratum.items())),
        "position_key_sha256": position_key_sha256,
        "document_isolated": True,
    }


def post_block_hidden_state_indices(
    *, num_hidden_layers: int, ordinals_one_based: Sequence[int]
) -> tuple[int, ...]:
    """Map one-based post-block ordinals to HF hidden_states tuple indices."""

    result = tuple(int(value) for value in ordinals_one_based)
    if any(value < 1 or value > int(num_hidden_layers) for value in result):
        raise ValueError(
            f"post-block layer ordinal outside 1..{num_hidden_layers}: {result}"
        )
    if len(result) != len(set(result)):
        raise ValueError("post-block layer ordinals must be unique")
    return result


def build_sparse_union(topk_id_tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sorted per-position unions, padded by -1 with a validity mask."""

    if not topk_id_tensors:
        raise ValueError("at least one top-k tensor is required")
    shape = topk_id_tensors[0].shape
    if len(shape) != 2 or any(tensor.shape != shape for tensor in topk_id_tensors):
        raise ValueError("top-k tensors must be rank two with matching shapes")
    rows: list[torch.Tensor] = []
    for row_index in range(shape[0]):
        values = torch.cat([tensor[row_index].long() for tensor in topk_id_tensors])
        values = values[values.ge(0)].unique(sorted=True)
        rows.append(values)
    max_width = max((int(row.numel()) for row in rows), default=0)
    union = torch.full((shape[0], max_width), -1, dtype=torch.long)
    mask = torch.zeros((shape[0], max_width), dtype=torch.bool)
    for row_index, values in enumerate(rows):
        union[row_index, : values.numel()] = values
        mask[row_index, : values.numel()] = True
    return union, mask


def _normalize_log_distribution(log_probs: torch.Tensor) -> torch.Tensor:
    values = log_probs.float()
    if values.ndim != 1:
        raise ValueError("coarse distributions must be rank one")
    return torch.log_softmax(values, dim=0)


def coarse_lattice_metrics(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: Sequence[torch.Tensor],
    student_topk_mask: torch.Tensor,
) -> dict[str, Any]:
    """Compute exact union-plus-tail metrics on one sparse lattice row."""

    if not teacher_log_probs:
        raise ValueError("at least one teacher distribution is required")
    student = _normalize_log_distribution(student_log_probs)
    teachers = [_normalize_log_distribution(value) for value in teacher_log_probs]
    if any(value.shape != student.shape for value in teachers):
        raise ValueError("student and teacher coarse distributions must align")
    mask = student_topk_mask.bool()
    if mask.shape != student.shape:
        raise ValueError("student_topk_mask must match the coarse distribution")

    teacher_stack = torch.stack(teachers)
    teacher_probs = teacher_stack.exp()
    mixture_probs = teacher_probs.mean(dim=0)
    mixture_log = mixture_probs.clamp_min(1e-30).log()
    js = torch.stack(
        [torch.sum(prob * (log_prob - mixture_log)) for prob, log_prob in zip(teacher_probs, teachers)]
    ).mean()
    teacher_count = len(teachers)
    normalized_agreement = (
        1.0 if teacher_count == 1 else 1.0 - float(js) / math.log(teacher_count)
    )
    student_gap = torch.sum(mixture_probs * (mixture_log - student))
    teachability = mixture_probs[mask].sum()
    return {
        "teacher_count": teacher_count,
        "normalized_teacher_agreement": max(0.0, min(1.0, normalized_agreement)),
        "student_gap_coarse_kl": float(student_gap),
        "teachability_student_topk": float(teachability),
        "teacher_tail_mass": float(mixture_probs[-1]),
    }


def stable_fraction(*values: object, seed: int) -> float:
    value = int(_stable_hash(*values, seed=seed)[:16], 16)
    return value / float(16**16 - 1)


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]
