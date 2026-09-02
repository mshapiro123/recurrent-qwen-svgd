"""Synthetic PRE-FLIGHT B1 calibration for the WEFT-1 Jacobian panel.

The plant is physical: at depth ``T`` and probe direction ``j`` it defines
``lambda_Tj = -c * T**(-(p + delta_j))`` and therefore a diagonal linear
transition with directional Jacobian gain ``exp(T * lambda_Tj)``.  The
response is never constructed from the regression coordinate.  Phase 1 has
zero probe measurement noise; phase 2 uses paired direction-specific exponent
perturbations and the PF-1.3 jackknife.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from analysis.weft1_jacobian_panel import (
    MAIN_PANEL_EXAMPLES,
    MAIN_PANEL_PROBES,
    PILOT_EXAMPLES,
    REGISTERED_DEPTHS,
    build_pilot_diagnostics,
    cluster_bootstrap_ci,
    design_sxx,
    paired_probe_jackknife,
    rejection_conditions,
)


PREFLIGHT_REPLICATES = 20
PREFLIGHT_PLANTED_EXPONENTS = (1.0, 1.5)
PREFLIGHT_BOOTSTRAP_REPLICATES = 10_000
PREFLIGHT_ROOT_SEED = 20260902
PREFLIGHT_BETWEEN_EXAMPLE_SD = 0.25
PREFLIGHT_PROBE_SLOPE_SD = 0.04


@dataclass(frozen=True)
class B1PhaseResult:
    phase: str
    planted_p: float
    replicates: int
    estimates: tuple[float, ...]
    mean_estimate: float
    monte_carlo_se: float
    recovered_within_two_se: bool
    ci_coverage_count: int
    mean_sigma_w_hat: float
    max_sigma_w_hat: float
    coverage_fraction: float
    probe_slope_sd: float
    passed: bool


@dataclass(frozen=True)
class B1PreflightReceipt:
    depth_coordinate: str
    sxx: float
    physical_plant: str
    phases: tuple[B1PhaseResult, ...]
    pilot_n: int
    pilot_admissible_panel_result: bool
    pilot_reporting_status: str
    negative_control_plant: str
    negative_control_lambda_by_depth: tuple[float, ...]
    negative_control_rejection_reasons: tuple[str, ...]
    phase2_is_counting_coverage: bool
    all_gates_passed: bool
    a100_hours: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def physical_power_law_log_gains(
    p_by_example: torch.Tensor,
    c_by_example: torch.Tensor,
    probe_slope_offsets: torch.Tensor,
    *,
    depths: tuple[int, ...] = REGISTERED_DEPTHS,
) -> torch.Tensor:
    """Return exact log gains from the PF-1.1 scalar linear physical plant."""

    if (
        type(p_by_example) is not torch.Tensor
        or type(c_by_example) is not torch.Tensor
        or type(probe_slope_offsets) is not torch.Tensor
    ):
        raise TypeError("plant inputs must be exact tensors")
    if p_by_example.ndim != 1 or c_by_example.shape != p_by_example.shape:
        raise ValueError("p and c must be aligned one-dimensional example vectors")
    if probe_slope_offsets.ndim != 2 or probe_slope_offsets.shape[0] != p_by_example.numel():
        raise ValueError("probe offsets must be [example, probe]")
    if probe_slope_offsets.shape[1] < 2:
        raise ValueError("the physical plant needs at least two paired probes")
    if any(
        not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all())
        for tensor in (p_by_example, c_by_example, probe_slope_offsets)
    ):
        raise ValueError("plant inputs must be finite floating-point tensors")
    if bool(c_by_example.le(0.0).any()):
        raise ValueError("physical plant amplitudes c must be positive")
    depth = torch.tensor(depths, dtype=torch.float64, device=p_by_example.device)
    exponent = p_by_example.double()[:, None, None] + probe_slope_offsets.double()[:, None, :]
    lambda_t = -c_by_example.double()[:, None, None] * depth[None, :, None].pow(-exponent)
    # A diagonal linear transition with directional singular values
    # exp(T * lambda_Tj) has these exact log gains.  The panel recovers lambda
    # by dividing each direction-averaged log gain by T.
    return depth[None, :, None] * lambda_t


def white_noise_recurrence_log_gains(
    *,
    seed: int,
    depths: tuple[int, ...] = REGISTERED_DEPTHS,
    n_probe: int = MAIN_PANEL_PROBES,
) -> torch.Tensor:
    """Plant a deterministic scalar recurrence with no depth power law.

    Each depth receives an independent finite-time Lyapunov value.  Repeating
    its scalar log gain across probes models an exactly measured linear map;
    the rejection inputs are derived from this returned panel, not supplied.
    """

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative exact integer")
    if type(n_probe) is not int or n_probe < 2:
        raise ValueError("n_probe must be an exact integer of at least two")
    generator = np.random.default_rng(seed)
    lambda_t = torch.from_numpy(generator.normal(0.0, 0.5, size=len(depths))).double()
    depth = torch.tensor(depths, dtype=torch.float64)
    return (depth * lambda_t)[:, None].expand(-1, n_probe).clone()


def _negative_control(root_seed: int) -> tuple[tuple[float, ...], tuple[str, ...]]:
    log_gains = white_noise_recurrence_log_gains(seed=root_seed + 8_000_000)
    depth = torch.tensor(REGISTERED_DEPTHS, dtype=torch.float64)
    lambda_hat = log_gains.mean(dim=-1) / depth
    signs = tuple(int(value) for value in torch.sign(lambda_hat).tolist())
    magnitudes = tuple(float(value) for value in lambda_hat.abs().tolist())
    return tuple(float(value) for value in lambda_hat.tolist()), rejection_conditions(
        signs,
        magnitudes,
    )


def _phase_seed(root_seed: int, planted_index: int, noisy: bool, replication: int) -> int:
    return root_seed + 100_000 * planted_index + 10_000 * int(noisy) + replication


def _run_phase(
    *,
    planted_p: float,
    planted_index: int,
    noisy: bool,
    replicates: int,
    bootstrap_replicates: int,
    root_seed: int,
) -> B1PhaseResult:
    estimates: list[float] = []
    sigma_w_values: list[float] = []
    coverage_count = 0
    for replication in range(replicates):
        seed = _phase_seed(root_seed, planted_index, noisy, replication)
        generator = np.random.default_rng(seed)
        p_values = planted_p + generator.normal(
            0.0,
            PREFLIGHT_BETWEEN_EXAMPLE_SD,
            size=MAIN_PANEL_EXAMPLES,
        )
        c_values = np.exp(
            math.log(0.3) + generator.normal(0.0, 0.15, size=MAIN_PANEL_EXAMPLES)
        )
        probe_offsets = (
            generator.normal(
                0.0,
                PREFLIGHT_PROBE_SLOPE_SD,
                size=(MAIN_PANEL_EXAMPLES, MAIN_PANEL_PROBES),
            )
            if noisy
            else np.zeros((MAIN_PANEL_EXAMPLES, MAIN_PANEL_PROBES), dtype=np.float64)
        )
        log_gains = physical_power_law_log_gains(
            torch.from_numpy(p_values),
            torch.from_numpy(c_values),
            torch.from_numpy(probe_offsets),
        )
        jackknife = paired_probe_jackknife(log_gains)
        estimate, ci_lo, ci_hi = cluster_bootstrap_ci(
            jackknife.slopes,
            replicates=bootstrap_replicates,
            seed=seed + 5_000_000,
        )
        estimates.append(estimate)
        sigma_w_values.append(jackknife.sigma_w_hat)
        coverage_count += ci_lo <= planted_p <= ci_hi
    coverage_fraction = coverage_count / replicates
    estimate_array = np.asarray(estimates, dtype=np.float64)
    mean_estimate = float(estimate_array.mean())
    monte_carlo_se = float(estimate_array.std(ddof=1) / math.sqrt(replicates))
    recovered_within_two_se = abs(mean_estimate - planted_p) <= 2.0 * monte_carlo_se
    max_sigma_w = max(sigma_w_values)
    zero_noise_valid = noisy or max_sigma_w <= 1e-12
    passed = (
        zero_noise_valid
        and coverage_fraction >= 0.90
        and recovered_within_two_se
    )
    return B1PhaseResult(
        phase="phase2_noisy" if noisy else "phase1_zero_measurement_noise",
        planted_p=planted_p,
        replicates=replicates,
        estimates=tuple(estimates),
        mean_estimate=mean_estimate,
        monte_carlo_se=monte_carlo_se,
        recovered_within_two_se=recovered_within_two_se,
        ci_coverage_count=coverage_count,
        mean_sigma_w_hat=float(np.mean(sigma_w_values)),
        max_sigma_w_hat=max_sigma_w,
        coverage_fraction=coverage_fraction,
        probe_slope_sd=PREFLIGHT_PROBE_SLOPE_SD if noisy else 0.0,
        passed=passed,
    )


def _pilot_payload(root_seed: int) -> dict[str, object]:
    generator = np.random.default_rng(root_seed + 9_000_000)
    p_values = 1.0 + generator.normal(
        0.0,
        PREFLIGHT_BETWEEN_EXAMPLE_SD,
        size=PILOT_EXAMPLES,
    )
    c_values = np.exp(math.log(0.3) + generator.normal(0.0, 0.15, size=PILOT_EXAMPLES))
    offsets = generator.normal(
        0.0,
        PREFLIGHT_PROBE_SLOPE_SD,
        size=(PILOT_EXAMPLES, MAIN_PANEL_PROBES),
    )
    log_gains = physical_power_law_log_gains(
        torch.from_numpy(p_values),
        torch.from_numpy(c_values),
        torch.from_numpy(offsets),
    )
    jackknife = paired_probe_jackknife(log_gains)
    return build_pilot_diagnostics(
        jackknife.slopes,
        log_gains,
        run_seed=root_seed,
        provenance_sha256="b1" * 32,
        depths=REGISTERED_DEPTHS,
        c_l_by_depth=(0.3, 0.3, 0.3, 0.3),
        lambda_sign_by_depth=(-1, -1, -1, -1),
        fixed_branch_verified=True,
    ).to_dict()


def run_preflight_b1(
    *,
    replicates: int = PREFLIGHT_REPLICATES,
    bootstrap_replicates: int = PREFLIGHT_BOOTSTRAP_REPLICATES,
    root_seed: int = PREFLIGHT_ROOT_SEED,
) -> B1PreflightReceipt:
    if type(replicates) is not int or replicates < 2:
        raise ValueError("replicates must be an exact integer of at least two")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive exact integer")
    phases = tuple(
        _run_phase(
            planted_p=planted_p,
            planted_index=planted_index,
            noisy=noisy,
            replicates=replicates,
            bootstrap_replicates=bootstrap_replicates,
            root_seed=root_seed,
        )
        for planted_index, planted_p in enumerate(PREFLIGHT_PLANTED_EXPONENTS)
        for noisy in (False, True)
    )
    pilot = _pilot_payload(root_seed)
    negative_lambda, negative_reasons = _negative_control(root_seed)
    pilot_valid = (
        pilot["n"] == PILOT_EXAMPLES
        and pilot["admissible_panel_result"] is False
        and pilot["reporting_status"] == "pilot_diagnostic_only"
    )
    all_passed = (
        all(phase.passed for phase in phases)
        and pilot_valid
        and "sign_inconsistency" in negative_reasons
    )
    return B1PreflightReceipt(
        depth_coordinate="natural_log",
        sxx=design_sxx(REGISTERED_DEPTHS),
        physical_plant=(
            "lambda_Tj=-c*T**(-(p+delta_j)); "
            "diagonal_gain_j=exp(T*lambda_Tj)"
        ),
        phases=phases,
        pilot_n=int(pilot["n"]),
        pilot_admissible_panel_result=bool(pilot["admissible_panel_result"]),
        pilot_reporting_status=str(pilot["reporting_status"]),
        negative_control_plant=(
            "seeded independent lambda_T; scalar_gain=exp(T*lambda_T)"
        ),
        negative_control_lambda_by_depth=negative_lambda,
        negative_control_rejection_reasons=negative_reasons,
        phase2_is_counting_coverage=True,
        all_gates_passed=all_passed,
    )


def main() -> None:
    print(json.dumps(run_preflight_b1().to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
