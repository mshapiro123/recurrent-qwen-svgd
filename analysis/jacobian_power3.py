"""PF-1.2 rerun of the registered WEFT-1 Jacobian power arithmetic.

The constants are the literal values in ``STRATEGY_JACOBIAN_POWER_20260826``.
PF-1 changes only the depth coordinate and therefore ``Sxx``.  This module
reports the resulting arithmetic without silently deciding whether an
approximately stated frontier is close enough to count as unchanged.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from analysis.weft1_jacobian_panel import REGISTERED_DESIGN_SXX


REGISTERED_N = 512
REFERENCE_SXX_BASE2 = 5.0
SIGMA_W = 0.25
PRIMARY_TARGET_SE = 0.051
PRIMARY_SIGMA_SLOPE_FRONTIER = 1.15
SECONDARY_TARGET_SE = 0.036
SECONDARY_SIGMA_SLOPE_FRONTIER = 0.80


def standard_error(
    *,
    n: int,
    sigma_slope: float,
    sigma_w: float = SIGMA_W,
    sxx: float = REGISTERED_DESIGN_SXX,
) -> float:
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive exact integer")
    if any(
        type(value) not in {float, int} or not math.isfinite(float(value))
        for value in (sigma_slope, sigma_w, sxx)
    ):
        raise TypeError("sigma_slope, sigma_w, and sxx must be finite real values")
    if sigma_slope < 0 or sigma_w < 0 or sxx <= 0:
        raise ValueError("sigmas must be non-negative and sxx must be positive")
    return math.sqrt((sigma_slope**2 + sigma_w**2 / sxx) / n)


def minimum_n(
    *,
    target_se: float,
    sigma_slope: float,
    sigma_w: float = SIGMA_W,
    sxx: float = REGISTERED_DESIGN_SXX,
) -> int:
    if type(target_se) not in {float, int} or not math.isfinite(float(target_se)):
        raise TypeError("target_se must be a finite real value")
    if target_se <= 0:
        raise ValueError("target_se must be positive")
    numerator = sigma_slope**2 + sigma_w**2 / sxx
    return math.ceil(numerator / target_se**2)


def covered_sigma_slope(
    *,
    n: int,
    target_se: float,
    sigma_w: float = SIGMA_W,
    sxx: float = REGISTERED_DESIGN_SXX,
) -> float:
    radicand = n * target_se**2 - sigma_w**2 / sxx
    return math.sqrt(max(radicand, 0.0))


@dataclass(frozen=True)
class JacobianPowerRerun:
    registered_n: int
    sxx: float
    measurement_se_growth_from_base2: float
    primary_target_se: float
    primary_literal_sigma_slope: float
    primary_realized_se_at_registered_n: float
    primary_minimum_n_for_literal_frontier: int
    primary_covered_sigma_slope_at_registered_n: float
    secondary_target_se: float
    secondary_literal_sigma_slope: float
    secondary_realized_se_at_registered_n: float
    secondary_minimum_n_for_literal_frontier: int
    secondary_covered_sigma_slope_at_registered_n: float
    literal_frontiers_both_met: bool
    disposition: str

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def rerun_registered_power() -> JacobianPowerRerun:
    primary_se = standard_error(
        n=REGISTERED_N,
        sigma_slope=PRIMARY_SIGMA_SLOPE_FRONTIER,
    )
    secondary_se = standard_error(
        n=REGISTERED_N,
        sigma_slope=SECONDARY_SIGMA_SLOPE_FRONTIER,
    )
    primary_met = primary_se <= PRIMARY_TARGET_SE
    secondary_met = secondary_se <= SECONDARY_TARGET_SE
    both_met = primary_met and secondary_met
    return JacobianPowerRerun(
        registered_n=REGISTERED_N,
        sxx=REGISTERED_DESIGN_SXX,
        measurement_se_growth_from_base2=math.sqrt(
            REFERENCE_SXX_BASE2 / REGISTERED_DESIGN_SXX
        ),
        primary_target_se=PRIMARY_TARGET_SE,
        primary_literal_sigma_slope=PRIMARY_SIGMA_SLOPE_FRONTIER,
        primary_realized_se_at_registered_n=primary_se,
        primary_minimum_n_for_literal_frontier=minimum_n(
            target_se=PRIMARY_TARGET_SE,
            sigma_slope=PRIMARY_SIGMA_SLOPE_FRONTIER,
        ),
        primary_covered_sigma_slope_at_registered_n=covered_sigma_slope(
            n=REGISTERED_N,
            target_se=PRIMARY_TARGET_SE,
        ),
        secondary_target_se=SECONDARY_TARGET_SE,
        secondary_literal_sigma_slope=SECONDARY_SIGMA_SLOPE_FRONTIER,
        secondary_realized_se_at_registered_n=secondary_se,
        secondary_minimum_n_for_literal_frontier=minimum_n(
            target_se=SECONDARY_TARGET_SE,
            sigma_slope=SECONDARY_SIGMA_SLOPE_FRONTIER,
        ),
        secondary_covered_sigma_slope_at_registered_n=covered_sigma_slope(
            n=REGISTERED_N,
            target_se=SECONDARY_TARGET_SE,
        ),
        literal_frontiers_both_met=both_met,
        disposition=(
            "registered_n_stands_on_literal_frontiers"
            if both_met
            else "return_to_strategy_for_registered_n"
        ),
    )


def main() -> None:
    print(json.dumps(rerun_registered_power().to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

