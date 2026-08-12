"""Pure contracts for the P3.3 reader and zero-collateral verification."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


P33_CANONICAL_READER_DTYPE = torch.bfloat16
P33_CANONICAL_READER_NAME = "bf16_serving_matmul_v1"
P33_VERIFICATION_AUDIT_RADIUS = 0.15
P33_GATE_OPEN_THRESHOLD = 0.5


def canonical_logits(states: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    """Run the tied output projection in the deployed BF16 serving precision."""

    return states.to(P33_CANONICAL_READER_DTYPE) @ embedding.to(
        P33_CANONICAL_READER_DTYPE
    ).T


def canonical_top1(states: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    return canonical_logits(states, embedding).argmax(dim=-1)


def fixed_pair_margin(
    logits: torch.Tensor, winner: torch.Tensor, runner_up: torch.Tensor
) -> torch.Tensor:
    index = torch.arange(logits.shape[0], device=logits.device)
    return logits[index, winner] - logits[index, runner_up]


def margin_delta_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray([float(row["margin_delta"]) for row in rows], dtype=np.float64)
    opened = [row for row in rows if bool(row["gate_predicted_open"])]
    opened_values = np.asarray(
        [float(row["margin_delta"]) for row in opened], dtype=np.float64
    )
    hidden = np.asarray(
        [float(row["hidden_delta_rms"]) for row in rows], dtype=np.float64
    )

    def quantiles(array: np.ndarray) -> dict[str, float | None]:
        if not array.size:
            return {name: None for name in ("minimum", "q25", "median", "q75", "maximum")}
        return {
            "minimum": float(array.min()),
            "q25": float(np.quantile(array, 0.25)),
            "median": float(np.quantile(array, 0.50)),
            "q75": float(np.quantile(array, 0.75)),
            "maximum": float(array.max()),
        }

    opened_zero = int(np.count_nonzero(opened_values == 0.0))
    return {
        "rows": len(rows),
        "gate_predicted_open_rows": len(opened),
        "margin_delta": quantiles(values),
        "absolute_margin_delta": quantiles(np.abs(values)),
        "hidden_delta_rms": quantiles(hidden),
        "exact_zero_margin_delta_rows": int(np.count_nonzero(values == 0.0)),
        "exact_zero_margin_delta_on_open_rows": opened_zero,
        "nonzero_hidden_delta_rows": int(np.count_nonzero(hidden > 0.0)),
        "passed_nonzero_delta_check": bool(len(opened) > 0 and opened_zero == 0),
    }


def verification_verdict(
    *,
    negative_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    positive_deployed_flips: int,
) -> dict[str, Any]:
    negative = margin_delta_summary(negative_rows)
    retention = margin_delta_summary(retention_rows)
    forced_flips = sum(bool(row["forced_open_collateral_change"]) for row in negative_rows)
    checks = {
        "v1_negative_nonzero_delta": negative["passed_nonzero_delta_check"],
        "v1_retention_nonzero_delta": retention["passed_nonzero_delta_check"],
        "v2_shared_path_registered_positive_flips": int(positive_deployed_flips) > 0,
        "v3_forced_open_negative_flips": forced_flips > 0,
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "negative_margin_delta": negative,
        "retention_margin_delta": retention,
        "positive_deployed_flips_same_path": int(positive_deployed_flips),
        "forced_open_negative_flips": int(forced_flips),
        "forced_open_negative_rows": len(negative_rows),
        "forced_open_negative_collateral_rate": forced_flips / max(1, len(negative_rows)),
    }
