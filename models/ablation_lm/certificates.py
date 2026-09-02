"""Standalone-only local Lipschitz receipts for WEFT-1 adapter compositions.

This module deliberately separates quantities that have provable upper bounds
from a local core-Jacobian estimate.  A finite power iteration is an empirical
lower estimate of a local operator norm; it is never promoted to a certificate.
Likewise, placeholders for modules that are absent from the production graph do
not participate in ``lambda_adapters``.  This utility is not wired into the
production visit and makes no integrated-production certificate claim.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CertifiedAdapterFactor:
    """One provable spectral-norm factor in an adapter composition."""

    name: str
    bound: float
    bound_source: str
    formula: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("adapter factor name must be non-empty")
        if not math.isfinite(self.bound) or self.bound < 0.0:
            raise ValueError("adapter factor bound must be finite and non-negative")
        if not self.bound_source.strip() or not self.formula.strip():
            raise ValueError("adapter factor source and formula must be non-empty")


@dataclass(frozen=True)
class AdapterFactorPlaceholder:
    """Pre-bound formula for a module absent from the integrated graph."""

    name: str
    formula: str
    required_bound_source: str
    production_status: str = "absent_from_integrated_production_graph"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("placeholder name must be non-empty")
        if not self.formula.strip() or not self.required_bound_source.strip():
            raise ValueError("placeholder formula and source must be non-empty")
        if self.production_status != "absent_from_integrated_production_graph":
            raise ValueError("placeholder production status is fixed and fail-closed")


@dataclass(frozen=True)
class SidecarFactorReceipt:
    """Certified residual factor for a non-negative low-rank expert mixture.

    ``U`` is the pre-gate mixture ``sum_e w_e A_e B_e.T`` and the applied
    update is ``DeltaW = gate * U``.  The gate therefore appears exactly once
    in the residual-map bound.
    """

    factor: CertifiedAdapterFactor
    selected_weights: tuple[float, ...]
    weight_l1: float
    absolute_gate: float
    per_expert_product_bounds: tuple[float, ...]
    pre_gate_mixture_norm_bound: float
    pre_gate_mixture_exact_norm: float

    def __post_init__(self) -> None:
        if not 1 <= len(self.selected_weights) <= 3:
            raise ValueError("sidecar receipt requires one to three selected experts")
        if len(self.per_expert_product_bounds) != len(self.selected_weights):
            raise ValueError("sidecar receipt weights and expert bounds must align")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in self.selected_weights
        ):
            raise ValueError("sidecar receipt weights must be finite and non-negative")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in self.per_expert_product_bounds
        ):
            raise ValueError("sidecar expert product bounds must be finite and non-negative")
        if not math.isfinite(self.absolute_gate) or self.absolute_gate < 0.0:
            raise ValueError("sidecar absolute gate must be finite and non-negative")
        expected_l1 = sum(self.selected_weights)
        if not math.isclose(self.weight_l1, expected_l1, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("sidecar weight_l1 is inconsistent")
        expected_mixture_bound = sum(
            weight * bound
            for weight, bound in zip(
                self.selected_weights,
                self.per_expert_product_bounds,
                strict=True,
            )
        )
        if not math.isclose(
            self.pre_gate_mixture_norm_bound,
            expected_mixture_bound,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("sidecar pre-gate mixture bound is inconsistent")
        if (
            not math.isfinite(self.pre_gate_mixture_exact_norm)
            or self.pre_gate_mixture_exact_norm < 0.0
            or self.pre_gate_mixture_exact_norm
            > self.pre_gate_mixture_norm_bound + 1e-12
        ):
            raise ValueError("sidecar exact mixture norm exceeds its certified bound")
        expected_factor_bound = 1.0 + self.absolute_gate * expected_mixture_bound
        if not math.isclose(
            self.factor.bound,
            expected_factor_bound,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("sidecar residual factor bound is inconsistent")
        if (
            self.factor.bound_source
            != "triangle_inequality_over_exact_factor_svd_norms"
            or self.factor.formula
            != "1 + |g| * sum_e w_e ||A_e||_2 ||B_e||_2"
        ):
            raise ValueError("sidecar factor source or formula is inconsistent")


@dataclass(frozen=True)
class AdapterCertificateReceipt:
    """Product of certified live factors, excluding named absent modules."""

    factors: tuple[CertifiedAdapterFactor, ...]
    placeholders: tuple[AdapterFactorPlaceholder, ...]
    lambda_adapters: float
    scope: str = "certified_live_adapter_factors_only"

    def __post_init__(self) -> None:
        expected = math.prod(factor.bound for factor in self.factors)
        if not math.isclose(self.lambda_adapters, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("lambda_adapters must equal the product of live factors")
        factor_names = [factor.name for factor in self.factors]
        placeholder_names = [placeholder.name for placeholder in self.placeholders]
        if len(factor_names) != len(set(factor_names)):
            raise ValueError("live adapter factor names must be unique")
        if len(placeholder_names) != len(set(placeholder_names)):
            raise ValueError("adapter placeholder names must be unique")
        if set(factor_names).intersection(placeholder_names):
            raise ValueError("a module cannot be both live and an absent placeholder")
        if self.scope != "certified_live_adapter_factors_only":
            raise ValueError("adapter receipt scope is fixed and fail-closed")


@dataclass(frozen=True)
class CoreEstimateReceipt:
    """Convergence evidence for an empirical local core-Jacobian estimate."""

    lambda_hat_core: float
    power_iteration_estimate: float
    rayleigh_quotient_sequence: tuple[float, ...]
    last_relative_change: float | None
    iterations: int
    minimum_iterations: int
    convergence_tolerance: float
    converged: bool
    randomized_probe_pair_gains: tuple[tuple[float, float], ...]
    paired_randomized_lower_bound: float
    seed: int
    semantics: str = "empirical_local_lower_estimate_not_a_certificate"

    def __post_init__(self) -> None:
        scalar_values = (
            self.lambda_hat_core,
            self.power_iteration_estimate,
            self.convergence_tolerance,
            self.paired_randomized_lower_bound,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in scalar_values):
            raise ValueError("core estimate values must be finite and non-negative")
        if self.iterations != len(self.rayleigh_quotient_sequence) or self.iterations < 1:
            raise ValueError("iteration count must match the Rayleigh sequence")
        if not 1 <= self.minimum_iterations <= self.iterations:
            raise ValueError("minimum iteration count is inconsistent")
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence tolerance must be positive")
        if self.last_relative_change is not None and (
            not math.isfinite(self.last_relative_change) or self.last_relative_change < 0.0
        ):
            raise ValueError("last relative change must be finite and non-negative")
        if self.semantics != "empirical_local_lower_estimate_not_a_certificate":
            raise ValueError("core estimate semantics are fixed and fail-closed")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in self.rayleigh_quotient_sequence
        ):
            raise ValueError("Rayleigh quotient sequence must be finite and non-negative")
        expected_power = math.sqrt(self.rayleigh_quotient_sequence[-1])
        if not math.isclose(
            self.power_iteration_estimate,
            expected_power,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("power estimate must equal the final Rayleigh square root")
        if not self.randomized_probe_pair_gains:
            raise ValueError("at least one randomized probe pair is required")
        flat_probe_gains = tuple(
            gain for pair in self.randomized_probe_pair_gains for gain in pair
        )
        if any(len(pair) != 2 for pair in self.randomized_probe_pair_gains) or not all(
            math.isfinite(gain) and gain >= 0.0 for gain in flat_probe_gains
        ):
            raise ValueError("randomized probe gains must be finite non-negative pairs")
        expected_probe_bound = max(flat_probe_gains)
        if not math.isclose(
            self.paired_randomized_lower_bound,
            expected_probe_bound,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("paired randomized lower bound is inconsistent")
        expected_lambda_hat = max(expected_power, expected_probe_bound)
        if not math.isclose(
            self.lambda_hat_core,
            expected_lambda_hat,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("lambda_hat_core must equal the strongest empirical lower estimate")
        if self.iterations >= 2:
            previous = self.rayleigh_quotient_sequence[-2]
            expected_change = abs(
                self.rayleigh_quotient_sequence[-1] - previous
            ) / max(abs(previous), 1e-30)
            if self.last_relative_change is None or not math.isclose(
                self.last_relative_change,
                expected_change,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("last relative change is inconsistent")
        expected_converged = (
            self.iterations >= self.minimum_iterations
            and self.last_relative_change is not None
            and self.last_relative_change < self.convergence_tolerance
            and expected_probe_bound
            <= expected_power * (1.0 + self.convergence_tolerance)
        )
        if self.converged is not expected_converged:
            raise ValueError("core convergence disposition is inconsistent")


_JOINT_STATE_COMPONENT_ORDER = ("h", "lanes", "scratch", "carrier")
_JOINT_STATE_METRIC = "plain_euclidean_concatenation_no_per_block_reweighting"


@dataclass(frozen=True)
class JointStateLayout:
    """Shape receipt for ``z=[h; lanes; scratch; carrier-when-integrated]``.

    Components that are absent from the current graph are omitted, never
    represented by fabricated zeros. Packing is a literal flatten-and-concatenate
    operation, so the packed Euclidean norm is exactly the Euclidean norm on the
    direct-sum state with no component weighting.
    """

    component_names: tuple[str, ...]
    component_shapes: tuple[tuple[int, ...], ...]
    component_numels: tuple[int, ...]
    total_numel: int
    metric: str = _JOINT_STATE_METRIC

    def __post_init__(self) -> None:
        count = len(self.component_names)
        if count < 1 or count != len(self.component_shapes) or count != len(
            self.component_numels
        ):
            raise ValueError("joint-state layout fields must have equal nonzero length")
        if self.component_names[0] != "h":
            raise ValueError("joint-state layout must begin with h")
        if len(set(self.component_names)) != count:
            raise ValueError("joint-state component names must be unique")
        order_indices: list[int] = []
        for name in self.component_names:
            if name not in _JOINT_STATE_COMPONENT_ORDER:
                raise ValueError(f"unknown joint-state component: {name}")
            order_indices.append(_JOINT_STATE_COMPONENT_ORDER.index(name))
        if order_indices != sorted(order_indices):
            raise ValueError("joint-state components must follow the ratified order")
        expected_numels: list[int] = []
        for shape in self.component_shapes:
            if not shape or any(type(size) is not int or size < 1 for size in shape):
                raise ValueError("joint-state component shapes must be positive")
            expected_numels.append(math.prod(shape))
        if tuple(expected_numels) != self.component_numels:
            raise ValueError("joint-state component numels do not match their shapes")
        if self.total_numel != sum(expected_numels):
            raise ValueError("joint-state total numel is inconsistent")
        if self.metric != _JOINT_STATE_METRIC:
            raise ValueError("joint-state metric is fixed to plain Euclidean concatenation")

    def unpack(self, packed: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Unpack one rank-one joint vector without changing any scale."""

        if packed.ndim != 1 or packed.numel() != self.total_numel:
            raise ValueError("packed joint state does not match its layout")
        values: list[torch.Tensor] = []
        offset = 0
        for shape, numel in zip(
            self.component_shapes,
            self.component_numels,
            strict=True,
        ):
            values.append(packed[offset : offset + numel].reshape(shape))
            offset += numel
        return tuple(values)

    def pack(self, values: Sequence[torch.Tensor]) -> torch.Tensor:
        """Pack values against this exact layout with no component weighting."""

        value_tuple = tuple(values)
        if len(value_tuple) != len(self.component_names):
            raise ValueError("joint-state output component count changed")
        reference = value_tuple[0]
        if not reference.is_floating_point():
            raise TypeError("joint-state components must be floating point")
        flattened: list[torch.Tensor] = []
        for value, shape in zip(value_tuple, self.component_shapes, strict=True):
            if tuple(value.shape) != shape:
                raise ValueError("joint-state output component shape changed")
            if (
                not value.is_floating_point()
                or value.device != reference.device
                or value.dtype != reference.dtype
            ):
                raise TypeError("joint-state components must share device and dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("joint-state output components must be finite")
            flattened.append(value.reshape(-1))
        return torch.cat(flattened)


@dataclass(frozen=True)
class JointStateCoreEstimateReceipt:
    """PF-3.3 local estimate on a complete current-graph joint-state visit."""

    layout: JointStateLayout
    core_estimate: CoreEstimateReceipt
    lambda_hat_core: float
    visit_scope: str
    current_graph_components: tuple[str, ...]
    absent_ratified_components: tuple[str, ...]
    legacy_model_diagnostic_status: str
    production_certificate_authorized: bool = False
    production_alarm_authorized: bool = False
    topology_complete: bool = False

    def __post_init__(self) -> None:
        if self.lambda_hat_core != self.core_estimate.lambda_hat_core:
            raise ValueError("joint-state lambda_hat_core must come from the core estimate")
        if self.current_graph_components != self.layout.component_names:
            raise ValueError("joint-state component receipt does not match its layout")
        expected_absent = tuple(
            name
            for name in _JOINT_STATE_COMPONENT_ORDER
            if name not in self.current_graph_components
        )
        if self.absent_ratified_components != expected_absent:
            raise ValueError("joint-state absent-component receipt is inconsistent")
        if self.visit_scope != "full_current_graph_recurrent_visit":
            raise ValueError("joint-state estimate scope is fixed")
        if self.legacy_model_diagnostic_status != (
            "non_governing_pre_pf3_dimension_reweighted_hidden_scratch_probe"
        ):
            raise ValueError("legacy Jacobian diagnostic status is fixed and fail-closed")
        if (
            self.production_certificate_authorized
            or self.production_alarm_authorized
            or self.topology_complete
        ):
            raise ValueError(
                "incomplete PF-3.3 topology cannot mint a production certificate or alarm"
            )


@dataclass(frozen=True)
class JointStateLoopLipschitzReceipt:
    """PF-3.3 two-number line with certified and empirical roles separated."""

    adapter_certificate: AdapterCertificateReceipt
    joint_core_estimate: JointStateCoreEstimateReceipt
    lambda_adapters: float
    lambda_hat_core: float
    two_number_line: tuple[tuple[str, float], ...]
    production_claim: str = "current_graph_measurement_only_topology_incomplete"
    alarm_threshold: float | None = None
    alarm_fired: bool | None = None

    def __post_init__(self) -> None:
        if self.lambda_adapters != self.adapter_certificate.lambda_adapters:
            raise ValueError("joint loop lambda_adapters must come from certified factors")
        if self.lambda_hat_core != self.joint_core_estimate.lambda_hat_core:
            raise ValueError("joint loop lambda_hat_core must come from the joint estimate")
        expected_line = (
            ("Lambda_adapters", self.lambda_adapters),
            ("Lambda_hat_core", self.lambda_hat_core),
        )
        if self.two_number_line != expected_line:
            raise ValueError("joint loop two-number line is inconsistent")
        if self.production_claim != "current_graph_measurement_only_topology_incomplete":
            raise ValueError("joint loop production claim is fixed and fail-closed")
        if self.alarm_threshold is not None or self.alarm_fired is not None:
            raise ValueError("PF-3.3 prohibits an alarm while topology is incomplete")


@dataclass(frozen=True)
class LoopLipschitzReceipt:
    """Two-number receipt; it is not a full production-loop certificate."""

    adapter_certificate: AdapterCertificateReceipt
    core_estimate: CoreEstimateReceipt
    lambda_adapters: float
    lambda_hat_core: float
    alarm_threshold: float | None = None
    alarm_fired: bool | None = None
    production_claim: str = "standalone_utility_production_integration_not_asserted"

    def __post_init__(self) -> None:
        if self.lambda_adapters != self.adapter_certificate.lambda_adapters:
            raise ValueError("lambda_adapters must come from the adapter certificate")
        if self.lambda_hat_core != self.core_estimate.lambda_hat_core:
            raise ValueError("lambda_hat_core must come from the empirical core receipt")
        if self.alarm_threshold is not None or self.alarm_fired is not None:
            raise ValueError("the cL alarm is unratified and must remain unset")
        if self.production_claim != "standalone_utility_production_integration_not_asserted":
            raise ValueError("production integration may not be claimed by this utility")


def _finite_scalar(value: float | torch.Tensor, *, name: str) -> float:
    tensor = torch.as_tensor(value).detach()
    if tensor.numel() != 1 or not tensor.is_floating_point():
        raise TypeError(f"{name} must be one floating-point scalar")
    result = float(tensor.double().cpu().item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _spectral_norm(weight: torch.Tensor, *, name: str) -> float:
    if weight.ndim != 2 or not weight.is_floating_point():
        raise TypeError(f"{name} must be a floating-point matrix")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError(f"{name} must be finite")
    return float(torch.linalg.matrix_norm(weight.detach().double(), ord=2).cpu().item())


def certify_sidecar_factor(
    expert_a: Sequence[torch.Tensor],
    expert_b: Sequence[torch.Tensor],
    weights: torch.Tensor | Sequence[float],
    *,
    gate: float | torch.Tensor,
    name: str = "sidecar",
    top_k_limit: int = 3,
) -> SidecarFactorReceipt:
    """Certify a standalone ``I + gate * sum_e w_e A_e B_e.T`` composition."""

    if len(expert_a) != len(expert_b) or not 1 <= len(expert_a) <= top_k_limit:
        raise ValueError("sidecar requires one-to-top_k_limit paired expert factors")
    weight_tensor = torch.as_tensor(weights, dtype=torch.float64).detach().flatten()
    if weight_tensor.numel() != len(expert_a):
        raise ValueError("one selected weight is required per expert")
    if not bool(torch.isfinite(weight_tensor).all()) or bool((weight_tensor < 0).any()):
        raise ValueError("selected sidecar weights must be finite and non-negative")

    product_bounds: list[float] = []
    products: list[torch.Tensor] = []
    expected_shape: tuple[int, int] | None = None
    expected_device: torch.device | None = None
    for index, (left, right) in enumerate(zip(expert_a, expert_b, strict=True)):
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            raise ValueError("each sidecar expert requires A[out, rank] and B[in, rank]")
        product_shape = (int(left.shape[0]), int(right.shape[0]))
        if product_shape[0] != product_shape[1]:
            raise ValueError("sidecar residual expert products must be square")
        if left.device != right.device:
            raise ValueError("each sidecar expert factor pair must share a device")
        if expected_device is None:
            expected_device = left.device
        elif left.device != expected_device:
            raise ValueError("all sidecar expert factors must share a device")
        if expected_shape is None:
            expected_shape = product_shape
        elif product_shape != expected_shape:
            raise ValueError("all sidecar expert products must have the same shape")
        left_norm = _spectral_norm(left, name=f"expert_a[{index}]")
        right_norm = _spectral_norm(right, name=f"expert_b[{index}]")
        product_bounds.append(left_norm * right_norm)
        products.append(left.detach().double() @ right.detach().double().T)

    selected_weights = tuple(float(value) for value in weight_tensor.cpu().tolist())
    l1 = float(weight_tensor.sum().item())
    mixture_bound = sum(
        weight * bound for weight, bound in zip(selected_weights, product_bounds, strict=True)
    )
    mixture = sum(
        weight * product for weight, product in zip(selected_weights, products, strict=True)
    )
    exact_mixture_norm = _spectral_norm(mixture, name="pre_gate_expert_mixture")
    absolute_gate = abs(_finite_scalar(gate, name="gate"))
    factor_bound = 1.0 + absolute_gate * mixture_bound
    factor = CertifiedAdapterFactor(
        name=name,
        bound=factor_bound,
        bound_source="triangle_inequality_over_exact_factor_svd_norms",
        formula="1 + |g| * sum_e w_e ||A_e||_2 ||B_e||_2",
    )
    return SidecarFactorReceipt(
        factor=factor,
        selected_weights=selected_weights,
        weight_l1=l1,
        absolute_gate=absolute_gate,
        per_expert_product_bounds=tuple(product_bounds),
        pre_gate_mixture_norm_bound=mixture_bound,
        pre_gate_mixture_exact_norm=exact_mixture_norm,
    )


def certify_callosum_factor(
    rho: float | torch.Tensor,
    *,
    name: str = "per_band_callosum",
) -> CertifiedAdapterFactor:
    """Return the exact spectral bound for the two-lane Birkhoff mixer."""

    rho_value = _finite_scalar(rho, name="rho")
    if not 0.0 <= rho_value <= 1.0:
        raise ValueError("rho must lie in [0, 1] for the nonexpansive certificate")
    return CertifiedAdapterFactor(
        name=name,
        bound=max(1.0, abs(1.0 - 2.0 * rho_value)),
        bound_source="analytic_two_lane_birkhoff_eigenvalues",
        formula="max(1, |1 - 2*rho|)",
    )


def certify_rotor_factor(
    operator: torch.Tensor | None = None,
    *,
    orthogonality_certified_by_construction: bool,
    name: str = "rotor",
) -> CertifiedAdapterFactor:
    """Require an explicit operator witness for every rotor certificate."""

    if orthogonality_certified_by_construction:
        if operator is None:
            raise ValueError(
                "an orthogonality-by-construction rotor requires its explicit operator witness"
            )
        _spectral_norm(operator, name="rotor operator")
        if operator.shape[0] != operator.shape[1]:
            raise ValueError("a certified orthogonal rotor operator must be square")
        operator64 = operator.detach().double()
        gram = operator64.T @ operator64
        identity = torch.eye(
            operator64.shape[1],
            device=operator64.device,
            dtype=operator64.dtype,
        )
        if not bool(torch.allclose(gram, identity, rtol=1e-6, atol=1e-6)):
            raise ValueError(
                "operator conflicts with its orthogonality-by-construction claim"
            )
        return CertifiedAdapterFactor(
            name=name,
            bound=1.0,
            bound_source="analytic_cl20_orthogonality_by_construction",
            formula="||R||_2 = 1",
        )
    if operator is None:
        raise ValueError("a non-certified rotor requires its explicit operator")
    return CertifiedAdapterFactor(
        name=name,
        bound=_spectral_norm(operator, name="rotor operator"),
        bound_source="exact_svd_of_structured_rotor_operator",
        formula="||R||_2",
    )


def certify_linear_factor(
    weight: torch.Tensor,
    *,
    gate: float | torch.Tensor = 1.0,
    residual: bool,
    name: str,
) -> CertifiedAdapterFactor:
    """Certify a linear or residual gated-linear map using exact SVD."""

    norm = _spectral_norm(weight, name=f"{name} weight")
    absolute_gate = abs(_finite_scalar(gate, name=f"{name} gate"))
    bound = absolute_gate * norm
    if residual:
        bound = 1.0 + bound
    return CertifiedAdapterFactor(
        name=name,
        bound=bound,
        bound_source="exact_weight_svd_with_gate_and_residual_triangle_bound",
        formula=("1 + |g| * ||W||_2" if residual else "|g| * ||W||_2"),
    )


def certify_state_translation_factor(*, name: str) -> CertifiedAdapterFactor:
    """Certify ``z -> z + c`` when ``c`` is fixed with respect to state."""

    return CertifiedAdapterFactor(
        name=name,
        bound=1.0,
        bound_source="analytic_state_translation_jacobian_identity",
        formula="||d(z + c)/dz||_2 = ||I||_2 = 1",
    )


def absent_weft1_adapter_placeholders() -> tuple[AdapterFactorPlaceholder, ...]:
    """Named PF-1.4 formulas for modules not yet in the production graph."""

    return (
        AdapterFactorPlaceholder(
            name="integrated_rotor_carrier",
            formula="||R||_2 = 1 when orthogonality is certified; else exact ||R||_2",
            required_bound_source=(
                "explicit Cl(2,0) operator construction witness or exact structured SVD"
            ),
        ),
        AdapterFactorPlaceholder(
            name="per_band_callosum",
            formula="max(1, |1 - 2*rho|) = 1 for rho in [0, 1]",
            required_bound_source="analytic two-lane Birkhoff eigenvalues",
        ),
        AdapterFactorPlaceholder(
            name="sidecar",
            formula="1 + |g| * sum_e w_e ||A_e||_2 ||B_e||_2",
            required_bound_source="non-negative top-k weights and exact SVDs of low-rank factors",
        ),
    )


def compose_adapter_certificate(
    factors: Sequence[CertifiedAdapterFactor],
    *,
    placeholders: Sequence[AdapterFactorPlaceholder] = (),
) -> AdapterCertificateReceipt:
    """Compose only certified live factors; placeholders never enter the product."""

    factor_tuple = tuple(factors)
    placeholder_tuple = tuple(placeholders)
    return AdapterCertificateReceipt(
        factors=factor_tuple,
        placeholders=placeholder_tuple,
        lambda_adapters=math.prod(factor.bound for factor in factor_tuple),
    )


def _unit_random_direction(
    primal: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    direction = torch.randn(
        primal.shape,
        generator=generator,
        device=primal.device,
        dtype=primal.dtype,
    )
    norm = direction.double().norm()
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise RuntimeError("failed to draw a finite nonzero probe direction")
    return direction / norm.to(dtype=direction.dtype)


def estimate_empirical_core_factor(
    function: Callable[[torch.Tensor], torch.Tensor],
    primal: torch.Tensor,
    *,
    max_iterations: int = 32,
    minimum_iterations: int = 3,
    convergence_tolerance: float = 1e-3,
    randomized_probe_pairs: int = 4,
    seed: int = 0,
) -> CoreEstimateReceipt:
    """Estimate one local Jacobian norm and return explicit convergence evidence.

    Each randomized probe gain is itself a lower bound on the local spectral
    norm.  Gains are recorded in pairs so paired checks can reuse the exact
    registered directions rather than drawing a second, untracked stream.
    """

    if not primal.is_floating_point() or not bool(torch.isfinite(primal).all()):
        raise ValueError("primal must be a finite floating-point tensor")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if type(minimum_iterations) is not int or not 1 <= minimum_iterations <= max_iterations:
        raise ValueError("minimum_iterations must lie in [1, max_iterations]")
    if not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be finite and positive")
    if type(randomized_probe_pairs) is not int or randomized_probe_pairs < 1:
        raise ValueError("randomized_probe_pairs must be a positive integer")

    generator = torch.Generator(device=primal.device).manual_seed(int(seed))
    direction = _unit_random_direction(primal, generator=generator)
    output, vjp = torch.func.vjp(function, primal)
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise TypeError("core function must return one floating-point tensor")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("core function output must be finite at the primal")

    rayleigh: list[float] = []
    last_relative_change: float | None = None
    converged = False
    for iteration in range(max_iterations):
        _value, tangent = torch.func.jvp(function, (primal,), (direction,))
        quotient = float(tangent.detach().double().square().sum().cpu().item())
        if not math.isfinite(quotient) or quotient < 0.0:
            raise RuntimeError("core power iteration produced an invalid Rayleigh quotient")
        rayleigh.append(quotient)
        if len(rayleigh) >= 2:
            previous = rayleigh[-2]
            last_relative_change = abs(quotient - previous) / max(abs(previous), 1e-30)
        (normal_direction,) = vjp(tangent)
        normal_norm = normal_direction.detach().double().norm()
        if float(normal_norm.cpu().item()) == 0.0:
            converged = iteration + 1 >= minimum_iterations and quotient == 0.0
            if converged:
                last_relative_change = 0.0
                break
            direction = _unit_random_direction(primal, generator=generator)
        else:
            direction = normal_direction.detach() / normal_norm.to(
                dtype=normal_direction.dtype
            )
        if (
            iteration + 1 >= minimum_iterations
            and last_relative_change is not None
            and last_relative_change < convergence_tolerance
        ):
            converged = True
            break

    pair_gains: list[tuple[float, float]] = []
    for _ in range(randomized_probe_pairs):
        gains: list[float] = []
        for _member in range(2):
            probe = _unit_random_direction(primal, generator=generator)
            _value, tangent = torch.func.jvp(function, (primal,), (probe,))
            gains.append(float(tangent.detach().double().norm().cpu().item()))
        pair_gains.append((gains[0], gains[1]))
    randomized_lower_bound = max(gain for pair in pair_gains for gain in pair)
    power_estimate = math.sqrt(rayleigh[-1])
    lambda_hat_core = max(power_estimate, randomized_lower_bound)
    if randomized_lower_bound > power_estimate * (1.0 + convergence_tolerance):
        converged = False

    return CoreEstimateReceipt(
        lambda_hat_core=lambda_hat_core,
        power_iteration_estimate=power_estimate,
        rayleigh_quotient_sequence=tuple(rayleigh),
        last_relative_change=last_relative_change,
        iterations=len(rayleigh),
        minimum_iterations=minimum_iterations,
        convergence_tolerance=convergence_tolerance,
        converged=converged,
        randomized_probe_pair_gains=tuple(pair_gains),
        paired_randomized_lower_bound=randomized_lower_bound,
        seed=int(seed),
    )


def pack_joint_state(
    components: Sequence[tuple[str, torch.Tensor]],
) -> tuple[torch.Tensor, JointStateLayout]:
    """Pack a named PF-3.3 state and return its immutable shape receipt."""

    component_tuple = tuple(components)
    if not component_tuple:
        raise ValueError("joint state requires at least h")
    names = tuple(name for name, _value in component_tuple)
    values = tuple(value for _name, value in component_tuple)
    reference = values[0]
    if not reference.is_floating_point():
        raise TypeError("joint-state components must be floating point")
    for name, value in component_tuple:
        if not isinstance(name, str) or not name:
            raise ValueError("joint-state names must be non-empty strings")
        if (
            not value.is_floating_point()
            or value.device != reference.device
            or value.dtype != reference.dtype
        ):
            raise TypeError("joint-state components must share device and dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("joint-state components must be finite")
    shapes = tuple(tuple(value.shape) for value in values)
    numels = tuple(value.numel() for value in values)
    layout = JointStateLayout(
        component_names=names,
        component_shapes=shapes,
        component_numels=numels,
        total_numel=sum(numels),
    )
    return layout.pack(values), layout


def estimate_empirical_joint_state_core_factor(
    function: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
    components: Sequence[tuple[str, torch.Tensor]],
    *,
    max_iterations: int = 32,
    minimum_iterations: int = 3,
    convergence_tolerance: float = 1e-3,
    randomized_probe_pairs: int = 4,
    seed: int = 0,
) -> JointStateCoreEstimateReceipt:
    """Estimate ``J`` for a full visit on the plain-Euclidean joint state."""

    packed, layout = pack_joint_state(components)

    def packed_transition(state: torch.Tensor) -> torch.Tensor:
        output = function(layout.unpack(state))
        if not isinstance(output, tuple):
            raise TypeError("joint-state transition must return an exact tuple")
        return layout.pack(output)

    estimate = estimate_empirical_core_factor(
        packed_transition,
        packed,
        max_iterations=max_iterations,
        minimum_iterations=minimum_iterations,
        convergence_tolerance=convergence_tolerance,
        randomized_probe_pairs=randomized_probe_pairs,
        seed=seed,
    )
    return JointStateCoreEstimateReceipt(
        layout=layout,
        core_estimate=estimate,
        lambda_hat_core=estimate.lambda_hat_core,
        visit_scope="full_current_graph_recurrent_visit",
        current_graph_components=layout.component_names,
        absent_ratified_components=tuple(
            name for name in _JOINT_STATE_COMPONENT_ORDER if name not in layout.component_names
        ),
        legacy_model_diagnostic_status=(
            "non_governing_pre_pf3_dimension_reweighted_hidden_scratch_probe"
        ),
    )


def make_joint_state_loop_lipschitz_receipt(
    adapter_certificate: AdapterCertificateReceipt,
    joint_core_estimate: JointStateCoreEstimateReceipt,
) -> JointStateLoopLipschitzReceipt:
    """Emit PF-3.3's two numbers without promoting either one's role."""

    return JointStateLoopLipschitzReceipt(
        adapter_certificate=adapter_certificate,
        joint_core_estimate=joint_core_estimate,
        lambda_adapters=adapter_certificate.lambda_adapters,
        lambda_hat_core=joint_core_estimate.lambda_hat_core,
        two_number_line=(
            ("Lambda_adapters", adapter_certificate.lambda_adapters),
            ("Lambda_hat_core", joint_core_estimate.lambda_hat_core),
        ),
    )


def make_loop_lipschitz_receipt(
    adapter_certificate: AdapterCertificateReceipt,
    core_estimate: CoreEstimateReceipt,
) -> LoopLipschitzReceipt:
    """Keep certified and empirical quantities separate in one loggable record."""

    return LoopLipschitzReceipt(
        adapter_certificate=adapter_certificate,
        core_estimate=core_estimate,
        lambda_adapters=adapter_certificate.lambda_adapters,
        lambda_hat_core=core_estimate.lambda_hat_core,
    )
