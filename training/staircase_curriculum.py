"""Dose accounting for stagewise recurrent-loop curricula."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch


def exposure_fractions(rows: Sequence[dict[str, Any]], *, cap: int) -> list[float]:
    """Return the fraction of rows carrying an active label at each loop."""

    if cap < 1:
        raise ValueError("cap must be positive")
    if not rows:
        raise ValueError("rows must not be empty")
    total = float(len(rows))
    fractions = [
        sum(int(int(row["depth"]) >= loop) for row in rows) / total
        for loop in range(1, int(cap) + 1)
    ]
    if any(value <= 0.0 for value in fractions):
        raise ValueError(f"every loop through cap={cap} must have nonzero exposure: {fractions}")
    return fractions


def equalized_loop_weights(
    exposure: Sequence[float],
    *,
    cap: int,
    newest_multiplier: float = 2.0,
) -> list[float]:
    """Equalize expected per-loop mass, then emphasize the newest loop."""

    if len(exposure) < int(cap):
        raise ValueError(f"exposure has {len(exposure)} loops but cap={cap}")
    if newest_multiplier <= 0.0:
        raise ValueError("newest_multiplier must be positive")
    raw = [1.0 / max(float(exposure[index]), 1e-6) for index in range(int(cap))]
    raw[-1] *= float(newest_multiplier)
    normalizer = sum(raw)
    return [value * int(cap) / normalizer for value in raw]


def optimizer_steps_for_weighted_budget(
    *,
    weighted_label_budget: float,
    newest_exposure: float,
    newest_weight: float,
    effective_batch_size: int,
    eval_every: int,
) -> int:
    """Use the last evaluation checkpoint within the newest-loop dose cap.

    One checkpoint interval is the minimum observable stage. It is allowed when
    even the first interval exceeds the requested dose budget.
    """

    mass_per_step = float(newest_exposure) * float(newest_weight) * int(effective_batch_size)
    if weighted_label_budget <= 0.0 or mass_per_step <= 0.0:
        raise ValueError("weighted budget and expected mass per optimizer step must be positive")
    if eval_every <= 0:
        raise ValueError("eval_every must be positive")
    raw_steps = float(weighted_label_budget) / mass_per_step
    checkpoint_intervals = max(1, math.floor(raw_steps / int(eval_every)))
    return int(checkpoint_intervals * int(eval_every))


def mass_equalization_receipt(
    mass: Sequence[float],
    *,
    newest_loop: int,
    newest_multiplier: float,
) -> dict[str, list[float]]:
    adjusted = [float(value) for value in mass]
    adjusted[int(newest_loop) - 1] /= float(newest_multiplier)
    mean = sum(adjusted) / len(adjusted) if adjusted else 0.0
    ratios = [value / mean if mean > 0.0 else 0.0 for value in adjusted]
    return {"adjusted_mass": adjusted, "ratios": ratios}


def assert_mass_equalized(
    mass: Sequence[float],
    *,
    newest_loop: int,
    newest_multiplier: float,
    min_ratio: float = 0.8,
    max_ratio: float = 1.25,
) -> dict[str, list[float]]:
    receipt = mass_equalization_receipt(
        mass,
        newest_loop=newest_loop,
        newest_multiplier=newest_multiplier,
    )
    if any(value < float(min_ratio) or value > float(max_ratio) for value in receipt["ratios"]):
        raise RuntimeError(
            "weighted loop-mass equalization failed: "
            f"ratios={receipt['ratios']}, adjusted_mass={receipt['adjusted_mass']}, "
            f"allowed=[{min_ratio}, {max_ratio}]"
        )
    return receipt


@dataclass
class LoopDoseLedger:
    """Accumulate active-row dose in raw and weighted units."""

    weights: Sequence[float]
    newest_loop: int
    newest_multiplier: float = 2.0
    raw_active_labels: list[int] = field(init=False)
    weighted_active_labels: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.weights = [float(value) for value in self.weights]
        if not self.weights:
            raise ValueError("weights must not be empty")
        if not 1 <= int(self.newest_loop) <= len(self.weights):
            raise ValueError("newest_loop must index the supplied weights")
        if self.newest_multiplier <= 0.0:
            raise ValueError("newest_multiplier must be positive")
        self.raw_active_labels = [0 for _ in self.weights]
        self.weighted_active_labels = [0.0 for _ in self.weights]

    def update(self, loop_labels: torch.Tensor) -> None:
        if loop_labels.dim() != 3:
            raise ValueError("loop_labels must be shaped [batch, loops, seq_len]")
        if loop_labels.shape[1] < len(self.weights):
            raise ValueError("loop_labels has fewer loop columns than the dose ledger")
        active = loop_labels[:, : len(self.weights), :].ne(-100).any(dim=-1).sum(dim=0)
        for index, value in enumerate(active.detach().cpu().tolist()):
            count = int(value)
            self.raw_active_labels[index] += count
            self.weighted_active_labels[index] += count * self.weights[index]

    def equalization_mass(self) -> list[float]:
        return mass_equalization_receipt(
            self.weighted_active_labels,
            newest_loop=self.newest_loop,
            newest_multiplier=self.newest_multiplier,
        )["adjusted_mass"]

    def equalization_ratios(self) -> list[float]:
        return mass_equalization_receipt(
            self.weighted_active_labels,
            newest_loop=self.newest_loop,
            newest_multiplier=self.newest_multiplier,
        )["ratios"]

    def assert_equalized(self, *, min_ratio: float = 0.8, max_ratio: float = 1.25) -> None:
        assert_mass_equalized(
            self.weighted_active_labels,
            newest_loop=self.newest_loop,
            newest_multiplier=self.newest_multiplier,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": list(self.weights),
            "newest_loop": int(self.newest_loop),
            "newest_multiplier": float(self.newest_multiplier),
            "raw_active_labels": list(self.raw_active_labels),
            "weighted_active_labels": list(self.weighted_active_labels),
            "equalization_mass_excluding_newest_multiplier": self.equalization_mass(),
            "equalization_ratios": self.equalization_ratios(),
        }
