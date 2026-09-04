"""Trajectory-grounded calibration gates for experimental modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


SYM_COLLAPSE_WINDOW = 1_000
_SYM_COLLAPSE_STATE_KIND = "weft1.sym_collapse_tracker.v1"


def final_to_earlier_visit_kl_bits(
    final_logits: torch.Tensor,
    earlier_logits: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Return ``KL(p_final || p_earlier)`` in bits per valid token.

    D-EP-1 makes the final executed visit the self-teacher for the one earlier
    visit already decoded by D-MC-1.  This is a receipt-only diagnostic: it is
    evaluated in FP32 under ``no_grad`` so logging cannot retain or alter the
    training graph.
    """

    if not isinstance(final_logits, torch.Tensor) or not isinstance(
        earlier_logits, torch.Tensor
    ):
        raise TypeError("visit KL logits must be tensors")
    if final_logits.shape != earlier_logits.shape:
        raise ValueError("final and earlier visit logits must have identical shapes")
    if final_logits.ndim != 3:
        raise ValueError("visit KL logits must have shape [batch, sequence, vocabulary]")
    if final_logits.device != earlier_logits.device:
        raise ValueError("final and earlier visit logits must share a device")
    if not isinstance(valid_token_mask, torch.Tensor):
        raise TypeError("visit KL valid_token_mask must be a tensor")
    if valid_token_mask.shape != final_logits.shape[:2]:
        raise ValueError("visit KL valid_token_mask must match [batch, sequence]")
    if valid_token_mask.device != final_logits.device:
        raise ValueError("visit KL valid_token_mask must share the logits device")
    if valid_token_mask.dtype is not torch.bool:
        raise TypeError("visit KL valid_token_mask must be boolean")
    if not bool(valid_token_mask.any()):
        raise ValueError("visit KL requires at least one valid token")

    with torch.no_grad():
        final_log_prob = F.log_softmax(final_logits.float(), dim=-1)
        earlier_log_prob = F.log_softmax(earlier_logits.float(), dim=-1)
        per_token_nats = (
            final_log_prob.exp() * (final_log_prob - earlier_log_prob)
        ).sum(dim=-1)
        # Clamp only tiny negative round-off after the exact KL sum.  A material
        # negative value indicates non-finite/corrupt inputs and fails closed.
        selected = per_token_nats.masked_select(valid_token_mask)
        if not bool(torch.isfinite(selected).all()):
            raise ValueError("visit KL logits produced a non-finite divergence")
        minimum = float(selected.min().item())
        if minimum < -1e-6:
            raise ValueError("visit KL produced a materially negative divergence")
        return selected.clamp_min(0.0).mean().div(math.log(2.0)).detach()


class SymmetryCollapseBlocked(RuntimeError):
    """Raised on the exact D-HD-1 1,000-step SYM-COLLAPSE boundary."""

    def __init__(
        self,
        *,
        step: int,
        matrix_name: str,
        initial_delta_ratio: float,
        observed_delta_ratio: float,
        consecutive_steps: int,
        receipt: SymmetryCollapseStepReceipt,
    ) -> None:
        self.step = step
        self.matrix_name = matrix_name
        self.initial_delta_ratio = initial_delta_ratio
        self.observed_delta_ratio = observed_delta_ratio
        self.consecutive_steps = consecutive_steps
        self.receipt = receipt
        super().__init__(
            "SYM-COLLAPSE: "
            f"{matrix_name} delta_ratio remained below its initialization value "
            f"for {consecutive_steps} consecutive eligible steps through step {step} "
            f"(init={initial_delta_ratio:.9g}, observed={observed_delta_ratio:.9g})"
        )


@dataclass(frozen=True)
class SymmetryCollapseStepReceipt:
    """Per-step paired-matrix receipt governed by D-HD-1."""

    step: int
    delta_ratio: tuple[tuple[str, float], ...]
    below_initial_consecutive_steps: tuple[tuple[str, int], ...]
    sym_collapse_window: int = SYM_COLLAPSE_WINDOW

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable receipt payload."""

        return asdict(self)


def _ratio_value(value: float | torch.Tensor, *, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} delta_ratio must be scalar")
        value = float(value.detach().float().cpu().item())
    elif isinstance(value, bool):
        raise TypeError(f"{name} delta_ratio must be a real scalar")
    else:
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} delta_ratio must be a real scalar") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} delta_ratio must be finite and non-negative")
    return value


def _ratio_mapping(
    values: Mapping[str, float | torch.Tensor],
    *,
    require_positive: bool,
) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, Mapping):
        raise TypeError("delta_ratio values must be a mapping")
    if not values:
        raise ValueError("delta_ratio values must name at least one SwapLinear")
    normalized: list[tuple[str, float]] = []
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("delta_ratio matrix names must be nonempty strings")
        ratio = _ratio_value(value, name=name)
        if require_positive and ratio <= 0.0:
            raise ValueError("initial delta_ratio values must be strictly positive")
        normalized.append((name, ratio))
    return tuple(sorted(normalized))


class SymmetryCollapseTracker:
    """Track D-HD-1 collapse streaks against the immutable T7 init receipt.

    Construct the tracker from T7's per-``SwapLinear`` initialization values,
    then call :meth:`observe` at every eligible optimizer step.  Missing or
    reordered eligible steps fail closed because they cannot prove a
    consecutive 1,000-step window.
    """

    def __init__(
        self,
        initial_delta_ratio: Mapping[str, float | torch.Tensor],
    ) -> None:
        initial = _ratio_mapping(initial_delta_ratio, require_positive=True)
        self._initial = dict(initial)
        self._below_initial = {name: 0 for name in self._initial}
        self._last_step: int | None = None

    @property
    def initial_delta_ratio(self) -> tuple[tuple[str, float], ...]:
        return tuple(sorted(self._initial.items()))

    @property
    def last_step(self) -> int | None:
        return self._last_step

    def state_dict(self) -> dict[str, Any]:
        """Return the complete JSON-safe state needed for exact run resume."""

        return {
            "kind": _SYM_COLLAPSE_STATE_KIND,
            "sym_collapse_window": SYM_COLLAPSE_WINDOW,
            "initial_delta_ratio": list(self.initial_delta_ratio),
            "below_initial_consecutive_steps": list(
                sorted(self._below_initial.items())
            ),
            "last_step": self._last_step,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> SymmetryCollapseTracker:
        """Restore a tracker without resetting a partially accrued streak."""

        if not isinstance(state, Mapping):
            raise TypeError("SYM-COLLAPSE tracker state must be a mapping")
        expected_fields = {
            "kind",
            "sym_collapse_window",
            "initial_delta_ratio",
            "below_initial_consecutive_steps",
            "last_step",
        }
        if set(state) != expected_fields:
            raise ValueError("SYM-COLLAPSE tracker state fields do not match v1")
        if state["kind"] != _SYM_COLLAPSE_STATE_KIND:
            raise ValueError("SYM-COLLAPSE tracker state kind does not match v1")
        if state["sym_collapse_window"] != SYM_COLLAPSE_WINDOW:
            raise ValueError("SYM-COLLAPSE tracker window must remain exactly 1000")
        try:
            initial = dict(state["initial_delta_ratio"])
            below = dict(state["below_initial_consecutive_steps"])
        except (TypeError, ValueError) as error:
            raise ValueError("SYM-COLLAPSE tracker pairs are malformed") from error
        tracker = cls(initial)
        if tuple(sorted(below)) != tuple(sorted(tracker._initial)):
            raise ValueError("SYM-COLLAPSE tracker state matrix set changed")
        normalized_counts: dict[str, int] = {}
        for name, count in below.items():
            if type(count) is not int or not 0 <= count < SYM_COLLAPSE_WINDOW:
                raise ValueError("SYM-COLLAPSE streak counts must be integers in [0, 999]")
            normalized_counts[name] = count
        last_step = state["last_step"]
        if last_step is not None and (type(last_step) is not int or last_step < 0):
            raise ValueError("SYM-COLLAPSE last_step must be non-negative or None")
        if last_step is None and any(normalized_counts.values()):
            raise ValueError("SYM-COLLAPSE pre-observation state cannot carry streaks")
        tracker._below_initial = normalized_counts
        tracker._last_step = last_step
        return tracker

    def observe(
        self,
        step: int,
        delta_ratio: Mapping[str, float | torch.Tensor],
    ) -> SymmetryCollapseStepReceipt:
        """Record one eligible step or raise on a gap/schema/tripwire hit."""

        if type(step) is not int or step < 0:
            raise ValueError("SYM-COLLAPSE step must be a non-negative exact integer")
        if self._last_step is not None and step != self._last_step + 1:
            raise ValueError(
                "SYM-COLLAPSE eligible steps must be observed exactly once in "
                "strict consecutive order"
            )
        observed = _ratio_mapping(delta_ratio, require_positive=False)
        observed_names = tuple(name for name, _value in observed)
        expected_names = tuple(sorted(self._initial))
        if observed_names != expected_names:
            raise ValueError(
                "SYM-COLLAPSE delta_ratio matrix set must exactly match the T7 init receipt"
            )

        next_counts: dict[str, int] = {}
        observed_by_name = dict(observed)
        for name in expected_names:
            next_counts[name] = (
                self._below_initial[name] + 1
                if observed_by_name[name] < self._initial[name]
                else 0
            )

        receipt = SymmetryCollapseStepReceipt(
            step=step,
            delta_ratio=observed,
            below_initial_consecutive_steps=tuple(sorted(next_counts.items())),
        )
        self._below_initial = next_counts
        self._last_step = step
        for name in expected_names:
            if next_counts[name] >= SYM_COLLAPSE_WINDOW:
                raise SymmetryCollapseBlocked(
                    step=step,
                    matrix_name=name,
                    initial_delta_ratio=self._initial[name],
                    observed_delta_ratio=observed_by_name[name],
                    consecutive_steps=next_counts[name],
                    receipt=receipt,
                )
        return receipt


@dataclass(frozen=True)
class RouterMomentSnapshot:
    """Router logit moments ``(m, s)`` observed at one optimizer step."""

    step: int
    mean: float
    std: float

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("router snapshot step must be a non-negative integer")
        values = torch.tensor((self.mean, self.std), dtype=torch.float64)
        if not bool(torch.isfinite(values).all()) or self.std < 0:
            raise ValueError("router moments must be finite and std must be non-negative")


@dataclass(frozen=True)
class RouterCalibrationDecision:
    """Whether a rolling trajectory, never step zero, can freeze calibration."""

    ready: bool
    reason: str
    mean_relative_drift: float | None
    std_relative_drift: float | None


def router_calibration_stability(
    snapshots: tuple[RouterMomentSnapshot, ...] | list[RouterMomentSnapshot],
    *,
    window: int = 8,
    minimum_step: int = 16,
    relative_tolerance: float = 0.05,
    absolute_floor: float = 1e-6,
) -> RouterCalibrationDecision:
    """Compare two adjacent windows of router ``(m, s)`` trajectory moments.

    The gate cannot pass from initialization data: it requires two full,
    non-overlapping windows ending no earlier than ``minimum_step``. This is a
    freeze-eligibility receipt, not an automatic freeze operation.
    """

    if type(window) is not int or window < 2:
        raise ValueError("window must be an integer of at least two")
    if type(minimum_step) is not int or minimum_step < 1:
        raise ValueError("minimum_step must be a positive integer")
    if (
        not math.isfinite(relative_tolerance)
        or not math.isfinite(absolute_floor)
        or relative_tolerance <= 0
        or absolute_floor <= 0
    ):
        raise ValueError("stability tolerances must be finite and positive")
    if len(snapshots) < 2 * window:
        return RouterCalibrationDecision(False, "insufficient_trajectory_windows", None, None)
    ordered = tuple(sorted(snapshots, key=lambda item: item.step))
    if len({item.step for item in ordered}) != len(ordered):
        raise ValueError("router calibration snapshots must have unique steps")
    if ordered[-1].step < minimum_step:
        return RouterCalibrationDecision(False, "minimum_nonzero_step_not_reached", None, None)
    previous = ordered[-2 * window : -window]
    current = ordered[-window:]
    window_steps = tuple(item.step for item in (*previous, *current))
    expected_steps = tuple(range(window_steps[0], window_steps[0] + 2 * window))
    if window_steps[0] <= 0 or window_steps != expected_steps:
        return RouterCalibrationDecision(False, "nonadjacent_or_step_zero_windows", None, None)

    def average(items: tuple[RouterMomentSnapshot, ...], field: str) -> float:
        return sum(float(getattr(item, field)) for item in items) / len(items)

    previous_mean = average(previous, "mean")
    current_mean = average(current, "mean")
    previous_std = average(previous, "std")
    current_std = average(current, "std")
    mean_drift = abs(current_mean - previous_mean) / max(abs(previous_mean), absolute_floor)
    std_drift = abs(current_std - previous_std) / max(abs(previous_std), absolute_floor)
    ready = mean_drift <= relative_tolerance and std_drift <= relative_tolerance
    return RouterCalibrationDecision(
        ready,
        "stable_trajectory_windows" if ready else "router_moments_still_drifting",
        mean_drift,
        std_drift,
    )
