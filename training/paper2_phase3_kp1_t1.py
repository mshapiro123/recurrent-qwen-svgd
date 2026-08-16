"""Deterministic contracts for the KP-1 and amended T1 score-only wave."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch


KP1_SPLIT_SEED = 20260816
KP1_EVAL_FRACTION = 0.30
KP1_RIDGE = 1.0
T1_LAYER_TAPS = (6, 12, 18, 24)
T1_LOOPS = 4
T1_SLOTS = 8
T1_CORE_CELLS = 44
T1_CELL_DIM = 128
T1_CEILINGS = (0.02, 0.05, 0.08, 0.11)


def canonical_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def knowledge_gap_rows(
    panel: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return DEV rows where the pinned 14B is correct and the base is wrong."""

    reference = {str(row["item_id"]): row for row in references}
    if len(reference) != len(references):
        raise ValueError("KP-1 reference rows must have unique item ids")
    selected = []
    for row in panel:
        item_id = str(row["item_id"])
        found = reference.get(item_id)
        if found is None:
            raise ValueError(f"KP-1 reference table lacks panel row {item_id}")
        if found.get("partition") != "dev" or row.get("partition") != "dev":
            raise RuntimeError("KP-1 may read DEV rows only")
        if bool(found["teacher_14b_correct"]) and not bool(found["base_correct"]):
            selected.append(dict(row))
    if not selected:
        raise RuntimeError("KP-1 knowledge-gap population is empty")
    return sorted(selected, key=lambda row: str(row["item_id"]))


def stratified_probe_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = KP1_SPLIT_SEED,
    eval_fraction: float = KP1_EVAL_FRACTION,
) -> dict[str, str]:
    """Create a deterministic row-disjoint split with every battery represented."""

    if not 0.0 < float(eval_fraction) < 1.0:
        raise ValueError("KP-1 evaluation fraction must lie inside (0, 1)")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["battery"])].append(row)
    assignments: dict[str, str] = {}
    for battery, group in sorted(grouped.items()):
        ranked = sorted(
            group,
            key=lambda row: hashlib.sha256(
                f"{seed}:{battery}:{row['item_id']}".encode("utf-8")
            ).digest(),
        )
        eval_rows = max(1, round(len(ranked) * float(eval_fraction)))
        if len(ranked) > 1:
            eval_rows = min(eval_rows, len(ranked) - 1)
        for index, row in enumerate(ranked):
            assignments[str(row["item_id"])] = (
                "probe_eval" if index < eval_rows else "probe_train"
            )
    return assignments


def core_cell_mask(*, loop_count: int, batch: int, device: torch.device) -> torch.Tensor:
    """Mask the fixed 44-cell schema without inventing unavailable future loops."""

    if loop_count < 1 or loop_count > T1_LOOPS:
        raise ValueError("T1 loop count must be in 1..4")
    mask = torch.zeros((batch, T1_CORE_CELLS), dtype=torch.bool, device=device)
    mask[:, : T1_SLOTS] = True
    loop_stop = T1_SLOTS + loop_count * T1_SLOTS
    mask[:, T1_SLOTS:loop_stop] = True
    mask[:, -len(T1_LAYER_TAPS) :] = True
    return mask


def assemble_core_cells(
    prelude: torch.Tensor,
    recurrent: Sequence[torch.Tensor],
    layer_cells: torch.Tensor,
) -> torch.Tensor:
    if prelude.ndim != 3 or prelude.shape[1:] != (T1_SLOTS, T1_CELL_DIM):
        raise ValueError("T1 prelude cells must be [batch, 8, 128]")
    if len(recurrent) != T1_LOOPS:
        raise ValueError("T1 requires four recurrent states")
    for state in recurrent:
        if state.shape != prelude.shape:
            raise ValueError("T1 recurrent cells must match the prelude geometry")
    if layer_cells.shape != (prelude.shape[0], len(T1_LAYER_TAPS), T1_CELL_DIM):
        raise ValueError("T1 layer cells must be [batch, 4, 128]")
    cells = torch.cat([prelude, *recurrent, layer_cells], dim=1)
    if cells.shape[1:] != (T1_CORE_CELLS, T1_CELL_DIM):
        raise RuntimeError("T1 core-cell schema changed")
    return cells


def ridge_embedding_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    eval_features: torch.Tensor,
    *,
    ridge: float = KP1_RIDGE,
) -> torch.Tensor:
    """Fit a deterministic affine ridge map in the sample-space dual."""

    if train_features.ndim != 2 or eval_features.ndim != 2:
        raise ValueError("probe features must be matrices")
    if train_targets.ndim != 2 or train_targets.shape[0] != train_features.shape[0]:
        raise ValueError("probe targets must align with training rows")
    if train_features.shape[1] != eval_features.shape[1]:
        raise ValueError("probe train/eval widths differ")
    if train_features.shape[0] < 2:
        raise ValueError("probe fit requires at least two training rows")
    x = train_features.float()
    y = train_targets.float()
    mean = x.mean(dim=0, keepdim=True)
    scale = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / scale
    z = (eval_features.float() - mean) / scale
    x = torch.cat([x, torch.ones((x.shape[0], 1), device=x.device)], dim=1)
    z = torch.cat([z, torch.ones((z.shape[0], 1), device=z.device)], dim=1)
    gram = x @ x.T
    gram.diagonal().add_(float(ridge))
    coefficients = torch.linalg.solve(gram, y)
    return z @ x.T @ coefficients


def token_ranks(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or target_ids.shape != (logits.shape[0],):
        raise ValueError("token-rank inputs must be [rows,vocab] and [rows]")
    target = logits.gather(1, target_ids[:, None])
    return 1 + (logits > target).sum(dim=1)


def row_reindex(source_ids: Sequence[str], target_ids: Sequence[str]) -> list[int]:
    """Return source indexes that align row-first tensors to a locked target order."""

    if len(source_ids) != len(set(source_ids)) or len(target_ids) != len(set(target_ids)):
        raise ValueError("row alignment ids must be unique")
    if set(source_ids) != set(target_ids):
        raise ValueError("row alignment source and target ids differ")
    source = {item_id: index for index, item_id in enumerate(source_ids)}
    return [source[item_id] for item_id in target_ids]
