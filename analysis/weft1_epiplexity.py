"""D-EP-1 receipt math that is independent of a future S2 run harness.

The model-side VISIT-KL lives with the forward diagnostics.  This module owns
the report-time prequential-area reduction so a future trainer only has to
emit explicit intervals and losses; it does not infer token intervals from an
underspecified step-zero convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class EvalLossInterval:
    """One chronological eval point and the training-token interval it closes."""

    delta_prediction_tokens: int
    eval_bits_per_prediction_token: float
    executed_k: int
    scored_k: int

    def __post_init__(self) -> None:
        if (
            type(self.delta_prediction_tokens) is not int
            or self.delta_prediction_tokens <= 0
        ):
            raise ValueError("delta_prediction_tokens must be a positive exact integer")
        if isinstance(self.eval_bits_per_prediction_token, bool):
            raise TypeError("eval_bits_per_prediction_token must be a real scalar")
        try:
            loss = float(self.eval_bits_per_prediction_token)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "eval_bits_per_prediction_token must be a real scalar"
            ) from error
        if not math.isfinite(loss) or loss < 0.0:
            raise ValueError(
                "eval_bits_per_prediction_token must be finite and non-negative"
            )
        for name, value in (("executed_k", self.executed_k), ("scored_k", self.scored_k)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.scored_k != self.executed_k:
            raise ValueError(
                "every eval point must be scored at that step's executed K_t"
            )


@dataclass(frozen=True)
class PrequentialAreaReceipt:
    """JSON-safe per-arm D-EP-1 report line and its audit inputs."""

    preq_area: float
    terminal_bits_per_prediction_token: float
    preq_area_units: str
    loss_units: str
    token_interval_semantics: str
    curriculum_scoring: str
    interpretation_constraint: str
    intervals: tuple[EvalLossInterval, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_prequential_area_receipt(
    intervals: Sequence[EvalLossInterval],
) -> PrequentialAreaReceipt:
    """Compute ``sum_e (L_e - L_final) * delta_prediction_tokens_e``.

    The ratified expression is intentionally *not* clamped.  Explicit token
    deltas avoid inventing whether a missing step-zero evaluation closes the
    first interval.  Requiring ``scored_k == executed_k`` makes the K-curriculum
    rule executable rather than a prose-only promise.
    """

    points = tuple(intervals)
    if not points:
        raise ValueError("preq_area requires at least one eval interval")
    if any(not isinstance(point, EvalLossInterval) for point in points):
        raise TypeError("preq_area intervals must be EvalLossInterval values")
    terminal_loss = float(points[-1].eval_bits_per_prediction_token)
    area = math.fsum(
        (float(point.eval_bits_per_prediction_token) - terminal_loss)
        * point.delta_prediction_tokens
        for point in points
    )
    if not math.isfinite(area):
        raise ValueError("preq_area is non-finite")
    return PrequentialAreaReceipt(
        preq_area=area,
        terminal_bits_per_prediction_token=terminal_loss,
        preq_area_units="bits",
        loss_units="bits_per_prediction_token",
        token_interval_semantics="explicit_consumed_prediction_tokens",
        curriculum_scoring="each_eval_at_executed_k_t",
        interpretation_constraint=(
            "conditioning_confounded_do_not_read_as_more_structure_without_seam_audit"
        ),
        intervals=points,
    )
