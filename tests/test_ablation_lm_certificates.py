"""Standalone PF-1 certificates; no test claims production-graph integration."""

from __future__ import annotations

import math

import pytest
import torch

from models.ablation_lm.callosum import PerBandBirkhoffCallosum
from models.ablation_lm.certificates import (
    absent_weft1_adapter_placeholders,
    certify_callosum_factor,
    certify_linear_factor,
    certify_rotor_factor,
    certify_sidecar_factor,
    compose_adapter_certificate,
    estimate_empirical_core_factor,
    make_loop_lipschitz_receipt,
)
from models.ablation_lm.geometry import cl20_rotate, lanes_to_modes


def test_pf1_sidecar_uses_the_pre_gate_mixture_and_gate_exactly_once() -> None:
    one = torch.ones(1, 1, dtype=torch.float64)

    receipt = certify_sidecar_factor((one,), (one,), (1.0,), gate=0.5)

    assert receipt.pre_gate_mixture_norm_bound == pytest.approx(1.0)
    assert receipt.pre_gate_mixture_exact_norm == pytest.approx(1.0)
    assert receipt.absolute_gate == pytest.approx(0.5)
    assert receipt.weight_l1 == pytest.approx(1.0)
    assert receipt.factor.bound == pytest.approx(1.5)
    assert receipt.factor.bound != pytest.approx(1.25)


def test_pf1_sidecar_records_nonnegative_top_k_l1_and_true_factor_bound() -> None:
    generator = torch.Generator().manual_seed(20_260_902)
    left = tuple(torch.randn(8, 2, generator=generator) for _ in range(3))
    right = tuple(torch.randn(8, 2, generator=generator) for _ in range(3))
    weights = torch.tensor([0.6, 0.25, 0.1], dtype=torch.float64)
    gate = -0.4

    receipt = certify_sidecar_factor(left, right, weights, gate=gate)
    mixture = sum(
        float(weight) * a.double() @ b.double().T
        for weight, a, b in zip(weights, left, right, strict=True)
    )
    applied_residual = torch.eye(8, dtype=torch.float64) + gate * mixture

    assert receipt.selected_weights == pytest.approx((0.6, 0.25, 0.1))
    assert receipt.weight_l1 == pytest.approx(0.95)
    assert receipt.pre_gate_mixture_exact_norm <= (
        receipt.pre_gate_mixture_norm_bound + 1e-12
    )
    assert torch.linalg.matrix_norm(applied_residual, ord=2).item() <= (
        receipt.factor.bound + 1e-12
    )
    with pytest.raises(ValueError, match="non-negative"):
        certify_sidecar_factor(left, right, (0.6, -0.1, 0.1), gate=gate)
    with pytest.raises(ValueError, match="top_k_limit"):
        certify_sidecar_factor(left + (left[0],), right + (right[0],), (0.25,) * 4, gate=gate)


def test_pf1_adapter_product_excludes_named_absent_module_placeholders() -> None:
    live = (
        certify_linear_factor(
            torch.diag(torch.tensor([2.0, 1.0])),
            gate=0.25,
            residual=True,
            name="reentry_bridge",
        ),
        certify_linear_factor(
            torch.diag(torch.tensor([0.5, 0.25])),
            gate=0.5,
            residual=True,
            name="scratch_injection",
        ),
    )
    placeholders = absent_weft1_adapter_placeholders()

    receipt = compose_adapter_certificate(live, placeholders=placeholders)

    assert receipt.lambda_adapters == pytest.approx(1.5 * 1.25)
    assert {item.name for item in placeholders} == {
        "integrated_rotor_carrier",
        "per_band_callosum",
        "sidecar",
    }
    assert all(
        item.production_status == "absent_from_integrated_production_graph"
        for item in placeholders
    )


def test_pf1_rotor_and_callosum_bound_sources_are_explicit() -> None:
    exact_rotor = certify_rotor_factor(
        orthogonality_certified_by_construction=True
    )
    generic_rotor = certify_rotor_factor(
        torch.diag(torch.tensor([2.0, 0.5])),
        orthogonality_certified_by_construction=False,
    )

    assert exact_rotor.bound == 1.0
    assert "orthogonality_by_construction" in exact_rotor.bound_source
    assert generic_rotor.bound == pytest.approx(2.0)
    assert "exact_svd" in generic_rotor.bound_source
    with pytest.raises(ValueError, match="conflicts"):
        certify_rotor_factor(
            torch.diag(torch.tensor([2.0, 0.5])),
            orthogonality_certified_by_construction=True,
        )
    assert all(
        certify_callosum_factor(rho).bound == 1.0
        for rho in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        certify_callosum_factor(1.01)


def test_pf1_core_estimate_has_convergence_sequence_and_paired_lower_bound() -> None:
    matrix = torch.diag(torch.tensor([3.0, 1.0], dtype=torch.float64))
    primal = torch.tensor([0.25, -0.75], dtype=torch.float64)

    receipt = estimate_empirical_core_factor(
        lambda value: matrix @ value,
        primal,
        max_iterations=32,
        minimum_iterations=3,
        convergence_tolerance=1e-3,
        randomized_probe_pairs=4,
        seed=41,
    )

    assert receipt.converged
    assert receipt.iterations == len(receipt.rayleigh_quotient_sequence)
    assert receipt.last_relative_change is not None
    assert receipt.last_relative_change < 1e-3
    assert receipt.power_iteration_estimate == pytest.approx(3.0, rel=1e-4)
    assert len(receipt.randomized_probe_pair_gains) == 4
    assert receipt.paired_randomized_lower_bound <= 3.0 + 1e-12
    assert receipt.lambda_hat_core >= receipt.paired_randomized_lower_bound
    assert receipt.semantics == "empirical_local_lower_estimate_not_a_certificate"


def test_pf1_loop_receipt_keeps_certified_and_empirical_numbers_separate() -> None:
    adapter = compose_adapter_certificate(
        (
            certify_linear_factor(
                torch.eye(2), gate=0.2, residual=True, name="reentry_bridge"
            ),
        ),
        placeholders=absent_weft1_adapter_placeholders(),
    )
    core = estimate_empirical_core_factor(
        lambda value: 2.0 * value,
        torch.ones(2, dtype=torch.float64),
        seed=7,
    )

    receipt = make_loop_lipschitz_receipt(adapter, core)

    assert receipt.lambda_adapters == pytest.approx(1.2)
    assert receipt.lambda_hat_core == pytest.approx(2.0)
    assert receipt.alarm_threshold is None
    assert receipt.alarm_fired is None
    assert receipt.production_claim == (
        "standalone_utility_production_integration_not_asserted"
    )


@pytest.mark.parametrize("dtype, tolerance", ((torch.float32, 1e-6), (torch.bfloat16, 1e-2)))
def test_a1_standalone_callosum_properties_across_rho_sweep(
    dtype: torch.dtype,
    tolerance: float,
) -> None:
    """A1 standalone math certificate; this is not production integration."""

    generator = torch.Generator().manual_seed(20_260_903)
    lanes = torch.randn(64, 2, 8, generator=generator).to(dtype)
    before = lanes_to_modes(lanes)
    for rho in (0.0, 1e-4, 0.25, 0.5, 0.75, 1.0):
        mixed = (1.0 - rho) * lanes + rho * lanes.flip(dims=(-2,))
        after = lanes_to_modes(mixed)
        relative_expansion = mixed.float().norm() / lanes.float().norm()

        assert relative_expansion.item() <= 1.0 + tolerance
        torch.testing.assert_close(after.mu, before.mu, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(
            after.delta,
            (1.0 - 2.0 * rho) * before.delta,
            rtol=tolerance,
            atol=tolerance,
        )
        if rho <= 0.5:
            assert 1.0 - 2.0 * rho >= 0.0


def test_a1_per_band_primitive_preserves_orthogonal_hidden_complement() -> None:
    """Audit the standalone per-band primitive, not an integrated graph."""

    callosum = PerBandBirkhoffCallosum(
        8,
        num_bands=2,
        rho_init=0.1,
        stop_gradient_senders=False,
    )
    shared = torch.randn(3, 1, 8)
    consensus_lanes = shared.expand(-1, 2, -1).clone()

    output = callosum(consensus_lanes)

    torch.testing.assert_close(output, consensus_lanes, rtol=1e-6, atol=1e-6)


def test_a2_standalone_cl20_composition_isometry_k_one_through_eight() -> None:
    """A2 covers the tensor primitive; the learned carrier remains absent."""

    generator = torch.Generator().manual_seed(20_260_904)
    initial = torch.randn(512, 2, generator=generator)
    identity = cl20_rotate(initial, torch.tensor(0.0))
    assert torch.equal(identity, initial)
    angles = torch.linspace(-0.7, 0.9, 8)
    rotated = initial
    initial_norm = initial.norm()
    for visit, angle in enumerate(angles, start=1):
        rotated = cl20_rotate(rotated, angle)
        relative_drift = (rotated.norm() / initial_norm - 1.0).abs().item()
        assert relative_drift < 1e-6, f"standalone rotor drift at K={visit}"

    low_precision = initial.bfloat16()
    low_precision_initial_norm = low_precision.float().norm()
    for angle in angles:
        low_precision = cl20_rotate(low_precision, angle)
    bf16_relative_drift = (
        low_precision.float().norm() / low_precision_initial_norm - 1.0
    ).abs()
    assert bf16_relative_drift.item() < 1e-2
    assert "integrated_rotor_carrier" in {
        item.name for item in absent_weft1_adapter_placeholders()
    }


@pytest.mark.parametrize("steps", (1, 2, 4, 8, 16))
def test_a4_mu_r_product_bound_is_a_standing_test(steps: int) -> None:
    """A4 checks the matrix product theorem independently of model wiring."""

    generator = torch.Generator().manual_seed(20_260_905 + steps)
    width = 8
    lipschitz = 1.0
    c = 0.5
    alpha = c / steps
    identity = torch.eye(width, dtype=torch.float64)
    product = identity.clone()
    for _ in range(steps):
        jacobian = torch.randn(width, width, generator=generator, dtype=torch.float64)
        jacobian = jacobian / torch.linalg.matrix_norm(jacobian, ord=2) * lipschitz
        product = (identity + alpha * jacobian) @ product

    observed = torch.linalg.matrix_norm(product, ord=2).item()
    polynomial_bound = (1.0 + c * lipschitz / steps) ** steps
    exponential_bound = math.exp(c * lipschitz)
    adversarial_product = torch.linalg.matrix_power(
        identity + alpha * lipschitz * identity,
        steps,
    )
    adversarial_norm = torch.linalg.matrix_norm(adversarial_product, ord=2).item()

    assert observed <= polynomial_bound + 1e-12
    assert adversarial_norm == pytest.approx(polynomial_bound, rel=1e-12)
    assert polynomial_bound <= exponential_bound + 1e-12
