from __future__ import annotations

import math

import pytest
import torch

from analysis.jacobian_power3 import rerun_registered_power
from analysis.weft1_jacobian_panel import (
    REGISTERED_DESIGN_SXX,
    REGISTERED_DEPTHS,
    paired_probe_jackknife,
)
from analysis.weft1_preflight_b1 import (
    PREFLIGHT_PLANTED_EXPONENTS,
    PREFLIGHT_REPLICATES,
    physical_power_law_log_gains,
    run_preflight_b1,
    white_noise_recurrence_log_gains,
)


@pytest.mark.parametrize("planted_p", PREFLIGHT_PLANTED_EXPONENTS)
def test_pf1_physical_power_law_plant_recovers_base_independent_p(
    planted_p: float,
) -> None:
    p_values = torch.full((12,), planted_p, dtype=torch.float64)
    c_values = torch.linspace(0.2, 0.5, 12, dtype=torch.float64)
    probe_offsets = torch.zeros(12, 4, dtype=torch.float64)

    log_gains = physical_power_law_log_gains(
        p_values,
        c_values,
        probe_offsets,
    )
    observed = paired_probe_jackknife(log_gains)

    torch.testing.assert_close(
        observed.slopes,
        torch.full_like(observed.slopes, -planted_p),
        rtol=0,
        atol=1e-12,
    )
    assert observed.sigma_w_hat <= 1e-12


def test_pf1_power_rerun_exposes_literal_n_branch_without_rounding() -> None:
    receipt = rerun_registered_power()

    assert receipt.sxx == pytest.approx(REGISTERED_DESIGN_SXX, rel=0, abs=1e-15)
    assert receipt.measurement_se_growth_from_base2 == pytest.approx(
        1.4426950408889636,
        rel=0,
        abs=1e-15,
    )
    assert receipt.primary_realized_se_at_registered_n == pytest.approx(
        0.051320780244339934
    )
    assert receipt.primary_minimum_n_for_literal_frontier == 519
    assert receipt.secondary_minimum_n_for_literal_frontier == 514
    assert receipt.literal_frontiers_both_met is False
    assert receipt.disposition == "return_to_strategy_for_registered_n"


def test_preflight_b1_two_phase_coverage_and_pilot_semantics() -> None:
    receipt = run_preflight_b1()

    assert receipt.depth_coordinate == "natural_log"
    assert receipt.sxx == pytest.approx(REGISTERED_DESIGN_SXX)
    assert receipt.physical_plant.startswith("lambda_Tj=-c*T**(-(p+delta_j))")
    assert len(receipt.phases) == 4
    assert all(phase.replicates == PREFLIGHT_REPLICATES for phase in receipt.phases)
    assert all(phase.coverage_fraction >= 0.90 for phase in receipt.phases)
    assert all(phase.recovered_within_two_se for phase in receipt.phases)
    assert all(
        abs(phase.mean_estimate - phase.planted_p) <= 2.0 * phase.monte_carlo_se
        for phase in receipt.phases
    )
    assert all(phase.passed for phase in receipt.phases)
    zero_noise = [phase for phase in receipt.phases if phase.phase.startswith("phase1")]
    noisy = [phase for phase in receipt.phases if phase.phase.startswith("phase2")]
    assert all(phase.max_sigma_w_hat <= 1e-12 for phase in zero_noise)
    assert all(phase.mean_sigma_w_hat > 0.0 for phase in noisy)
    assert receipt.phase2_is_counting_coverage is True
    assert receipt.pilot_n == 32
    assert receipt.pilot_admissible_panel_result is False
    assert receipt.pilot_reporting_status == "pilot_diagnostic_only"
    negative_log_gains = white_noise_recurrence_log_gains(seed=20260902 + 8_000_000)
    depth = torch.tensor(REGISTERED_DEPTHS, dtype=torch.float64)
    derived_lambda = tuple(
        float(value) for value in (negative_log_gains.mean(dim=-1) / depth).tolist()
    )
    assert receipt.negative_control_lambda_by_depth == pytest.approx(derived_lambda)
    assert len({math.copysign(1.0, value) for value in derived_lambda}) == 2
    assert receipt.negative_control_plant.startswith("seeded independent lambda_T")
    assert "sign_inconsistency" in receipt.negative_control_rejection_reasons
    assert receipt.all_gates_passed is True
    assert receipt.a100_hours == 0.0
    assert math.isfinite(receipt.sxx)
    assert tuple(REGISTERED_DEPTHS) == (1, 2, 4, 8)
