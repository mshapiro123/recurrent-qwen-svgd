from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from analysis.weft1_jacobian_panel import (
    DIRECTION_CLASSES,
    NORM_RANK_REPORTING_STATUS,
    REGISTERED_DESIGN_SXX,
    REGISTERED_DEPTHS,
    StochasticStateSnapshot,
    build_pilot_diagnostics,
    build_panel_report,
    cluster_bootstrap_ci,
    compare_main_and_norm_tiers,
    derive_example_probe_seed,
    design_sxx,
    draw_example_probe_directions,
    loop_log_gains,
    measure_example_depths,
    operator_norm,
    paired_probe_jackknife,
    p_hat,
    participation_ratio,
    rejection_conditions,
    sigma_slope_hat,
    theil_sen_slopes,
)
from models.ablation_lm.rng import ModuleRNGStream


PROVENANCE_SHA256 = "ab" * 32


class _TinyTwoBlock(nn.Module):
    """Eight-dimensional, two-block transition with a known dense Jacobian."""

    def __init__(self) -> None:
        super().__init__()
        singular_values = torch.tensor(
            [4.0, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2],
            dtype=torch.float32,
        )
        first = singular_values.sqrt()
        self.register_buffer("first", torch.diag(first))
        self.register_buffer("second", torch.diag(first))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.second @ (self.first @ state)


class _RoutedStochasticTransition(nn.Module):
    def __init__(self, run_seed: int = 77) -> None:
        super().__init__()
        self.router_rng = ModuleRNGStream(run_seed, "model.router.noise")

    def with_aux(self, state: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        generator = self.router_rng.next_generator(state.device)
        noise = torch.randn(
            state.shape,
            generator=generator,
            device=state.device,
            dtype=torch.float32,
        )
        experts = noise.gt(0)
        gates = noise.abs().gt(0.5)
        scale = 1.0 + 0.125 * experts.float()
        return scale * state + 0.01 * noise, {"experts": experts, "gates": gates}


class _AnisotropicDepthTransition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "matrix",
            torch.diag(torch.tensor([3.0, 1.5, 1.0, 0.5], dtype=torch.float32)),
        )

    def transition_for_depth(
        self, depth: int
    ) -> Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        scale = float(depth + 1)

        def transition(
            state: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            return scale * (self.matrix @ state), {
                "experts": torch.tensor([depth % 2], dtype=torch.int64)
            }

        return transition


class _IdentityDepthTransition(nn.Module):
    def transition_for_depth(
        self, depth: int
    ) -> Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        def transition(
            state: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            return state, {
                "experts": torch.zeros(depth, dtype=torch.int64, device=state.device)
            }

        return transition


def _dense_jacobian(model: nn.Module, primal: torch.Tensor) -> torch.Tensor:
    return torch.autograd.functional.jacobian(model, primal)


def _report_inputs(
    example_count: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[float, ...]]]:
    p_values = torch.linspace(0.8, 1.2, example_count, dtype=torch.float64)
    c_values = torch.linspace(0.2, 0.4, example_count, dtype=torch.float64)
    depths = torch.tensor(REGISTERED_DEPTHS, dtype=torch.float64)
    lambda_t = -c_values[:, None] * depths[None, :].pow(-p_values[:, None])
    log_gains = (depths[None, :] * lambda_t)[:, :, None].expand(-1, -1, 4).clone()
    slopes = paired_probe_jackknife(log_gains).slopes
    directions = {
        name: tuple(0.1 * (index + 1) for index in range(4))
        for name in DIRECTION_CLASSES
    }
    return slopes, log_gains, directions


def test_pt1_golden_theil_sen_slope() -> None:
    depths = torch.tensor(REGISTERED_DEPTHS, dtype=torch.float64)
    x = torch.log(depths)
    c = torch.linspace(0.2, 0.8, 23, dtype=torch.float64)
    true_p = 1.137
    # PF-1.1 physical plant: lambda_T = -c * T**(-p).  The response
    # comes from the planted law, not from the regression coordinate.
    lambda_t = -c[:, None] * depths[None, :].pow(-true_p)
    y = torch.log(lambda_t.abs())

    slopes = theil_sen_slopes(x, y)

    torch.testing.assert_close(
        slopes,
        torch.full_like(slopes, -true_p),
        rtol=0,
        atol=1e-12,
    )
    assert abs(p_hat(x, y) - true_p) < 1e-6
    assert design_sxx(REGISTERED_DEPTHS) == pytest.approx(
        REGISTERED_DESIGN_SXX,
        rel=0,
        abs=1e-15,
    )


def test_pt2_forward_jvp_matches_dense_jacobian_for_sixteen_directions() -> None:
    model = _TinyTwoBlock()
    primal = torch.linspace(-0.4, 0.6, 8, dtype=torch.float32)
    directions = draw_example_probe_directions(
        primal,
        n_probe=16,
        example_probe_seed=20260826,
    )
    snapshot = StochasticStateSnapshot.capture(model)

    log_gains = loop_log_gains(
        model,
        primal,
        directions,
        stochastic_snapshot=snapshot,
        has_aux=False,
    )
    dense = _dense_jacobian(model, primal)
    expected = torch.stack([(dense @ direction).norm() for direction in directions])
    relative = (log_gains.exp() - expected).abs() / expected

    assert relative.max().item() < 1e-5


def test_identity_has_bit_exact_zero_log_gain_and_rejects_zero_lambda() -> None:
    model = _IdentityDepthTransition()
    primal = torch.linspace(-0.4, 0.6, 17, dtype=torch.float32)
    directions = draw_example_probe_directions(
        primal,
        n_probe=4,
        example_probe_seed=20260826,
    )
    direct = loop_log_gains(
        nn.Identity(),
        primal,
        directions,
        stochastic_snapshot=StochasticStateSnapshot.capture(nn.Identity()),
        has_aux=False,
    )
    assert torch.equal(direct, torch.zeros_like(direct))

    measurement = measure_example_depths(
        model.transition_for_depth,
        primal,
        model=model,
        panel_seed=20260826,
        example_id="sentinel/identity-golden",
        depths=REGISTERED_DEPTHS,
        n_probe=4,
        has_aux=True,
    )

    assert torch.equal(
        measurement.log_gains,
        torch.zeros_like(measurement.log_gains),
    )
    assert torch.equal(
        measurement.lambda_hat,
        torch.zeros_like(measurement.lambda_hat),
    )
    assert bool(torch.isneginf(measurement.log_abs_lambda).all())
    signs = tuple(int(value) for value in measurement.lambda_sign.tolist())
    assert rejection_conditions(signs, (0.0, 0.0, 0.0, 0.0)) == (
        "zero_lambda",
        "conditioning_failure",
    )


def test_public_gain_probe_contract_rejects_scaled_non_unit_directions() -> None:
    model = _TinyTwoBlock()
    primal = torch.linspace(-0.4, 0.6, 8, dtype=torch.float32)
    directions = draw_example_probe_directions(
        primal,
        n_probe=2,
        example_probe_seed=19,
    )

    with pytest.raises(ValueError, match="unit directions"):
        loop_log_gains(
            model,
            primal,
            3.0 * directions,
            stochastic_snapshot=StochasticStateSnapshot.capture(model),
            has_aux=False,
        )


def test_pt3_hutchinson_participation_ratio_matches_dense_svd() -> None:
    model = _TinyTwoBlock()
    primal = torch.linspace(-0.4, 0.6, 8, dtype=torch.float32)
    dense = _dense_jacobian(model, primal)
    singular_squared = torch.linalg.svdvals(dense).square()
    truth = singular_squared.sum().square() / singular_squared.square().sum()

    estimate = participation_ratio(
        model,
        primal,
        model=model,
        n_probe=4096,
        seed=31,
    )

    assert abs(estimate - truth.item()) / truth.item() < 0.10


def test_pt4_power_iteration_matches_dense_operator_norm() -> None:
    model = _TinyTwoBlock()
    primal = torch.linspace(-0.4, 0.6, 8, dtype=torch.float32)
    dense = _dense_jacobian(model, primal)
    truth = torch.linalg.svdvals(dense)[0].item()

    estimate = operator_norm(
        model,
        primal,
        model=model,
        iterations=10,
        seed=37,
    )

    assert abs(estimate - truth) / truth < 1e-4


def test_pt5_probe_reset_branch_identity_and_rng_isolation() -> None:
    model = _RoutedStochasticTransition()
    primal = torch.tensor(
        [-0.7, -0.2, -0.01, 0.03, 0.2, 0.8],
        dtype=torch.float32,
    )
    one_direction = draw_example_probe_directions(
        primal,
        n_probe=1,
        example_probe_seed=41,
    )[0]
    repeated_direction = torch.stack((one_direction, one_direction))
    snapshot = StochasticStateSnapshot.capture(model)
    ambient_before = torch.random.get_rng_state().clone()

    gains = loop_log_gains(
        model.with_aux,
        primal,
        repeated_direction,
        stochastic_snapshot=snapshot,
        has_aux=True,
    )

    torch.testing.assert_close(gains[0], gains[1], rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), ambient_before, rtol=0, atol=0)
    assert model.router_rng.draw_index == 0
    assert snapshot.stream_names == ("router_rng",)


def test_pt5_same_run_and_probe_seeds_are_bit_identical_across_processes() -> None:
    repository = Path(__file__).resolve().parents[1]
    script = r'''
import hashlib
import torch
from torch import nn
from analysis.weft1_jacobian_panel import (
    StochasticStateSnapshot,
    derive_example_probe_seed,
    draw_example_probe_directions,
    loop_log_gains,
)
from models.ablation_lm.rng import ModuleRNGStream

class Transition(nn.Module):
    def __init__(self):
        super().__init__()
        self.rng = ModuleRNGStream(77, "model.router.noise")
    def with_aux(self, state):
        generator = self.rng.next_generator(state.device)
        noise = torch.randn(state.shape, generator=generator, dtype=torch.float32)
        experts = noise.gt(0)
        return (1.0 + 0.125 * experts.float()) * state + 0.01 * noise, experts

model = Transition()
primal = torch.tensor([-0.7, -0.2, -0.01, 0.03, 0.2, 0.8])
seed = derive_example_probe_seed(20260826, "sentinel/example-0042")
directions = draw_example_probe_directions(primal, n_probe=4, example_probe_seed=seed)
gains = loop_log_gains(
    model.with_aux,
    primal,
    directions,
    stochastic_snapshot=StochasticStateSnapshot.capture(model),
    has_aux=True,
)
print(hashlib.sha256(gains.numpy().tobytes()).hexdigest())
'''

    first = subprocess.check_output(
        [sys.executable, "-c", script], cwd=repository, text=True
    ).strip()
    second = subprocess.check_output(
        [sys.executable, "-c", script], cwd=repository, text=True
    ).strip()

    assert first == second
    assert len(first) == hashlib.sha256().digest_size * 2


def test_pt5_p5_reuses_one_example_owned_probe_bank_at_every_depth() -> None:
    model = _AnisotropicDepthTransition()
    primal = torch.tensor([0.2, -0.1, 0.7, -0.4], dtype=torch.float32)

    measurement = measure_example_depths(
        model.transition_for_depth,
        primal,
        model=model,
        panel_seed=20260826,
        example_id="sentinel/example-0017",
        depths=REGISTERED_DEPTHS,
        n_probe=4,
        has_aux=True,
    )

    # For J_T = scalar(T) * M, reusing v_i makes the centered directional
    # sampling pattern exactly shared across depths.  Redrawing by depth would
    # destroy this equality for anisotropic M.
    centered = measurement.log_gains - measurement.log_gains.mean(dim=1, keepdim=True)
    torch.testing.assert_close(
        centered,
        centered[0].expand_as(centered),
        rtol=1e-6,
        atol=1e-6,
    )
    assert measurement.depths == REGISTERED_DEPTHS
    assert measurement.example_probe_seed == derive_example_probe_seed(
        20260826, "sentinel/example-0017"
    )
    assert measurement.fixed_branch_verified is True


def test_pt5_rejects_ambient_rng_as_an_unregistered_stochastic_source() -> None:
    class AmbientTransition(nn.Module):
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return state + 0.01 * torch.randn(state.shape)

    model = AmbientTransition()
    primal = torch.ones(4)
    directions = draw_example_probe_directions(
        primal,
        n_probe=2,
        example_probe_seed=3,
    )

    with pytest.raises(RuntimeError, match="ambient RNG"):
        loop_log_gains(
            model,
            primal,
            directions,
            stochastic_snapshot=StochasticStateSnapshot.capture(model),
            has_aux=False,
        )


def test_pt6_cluster_bootstrap_has_registered_synthetic_coverage() -> None:
    true_p = 1.0
    generator = np.random.default_rng(20260826)
    covered = 0
    replications = 1000
    for replication in range(replications):
        # One slope per example is the cluster passed to the bootstrap.  There
        # is deliberately no probe or depth axis here.
        slopes = -true_p + generator.normal(0.0, 0.35, size=48)
        _estimate, lower, upper = cluster_bootstrap_ci(
            slopes,
            replicates=999,
            seed=17 + replication,
        )
        covered += lower <= true_p <= upper

    coverage = covered / replications
    assert 0.93 <= coverage <= 0.97


def test_pf1_paired_leave_one_probe_out_jackknife_is_exact() -> None:
    depths = torch.tensor(REGISTERED_DEPTHS, dtype=torch.float64)
    p_values = torch.tensor([0.8, 1.0, 1.2], dtype=torch.float64)
    c_values = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)
    probe_offsets = torch.tensor(
        [
            [-0.06, -0.02, 0.02, 0.06],
            [-0.04, -0.01, 0.01, 0.04],
            [-0.08, -0.03, 0.03, 0.08],
        ],
        dtype=torch.float64,
    )
    exponent = p_values[:, None, None] + probe_offsets[:, None, :]
    lambda_t = -c_values[:, None, None] * depths[None, :, None].pow(-exponent)
    log_gains = depths[None, :, None] * lambda_t

    observed = paired_probe_jackknife(log_gains)
    manual_leave_one_out = []
    for omitted in range(4):
        retained = [probe for probe in range(4) if probe != omitted]
        lambda_subset = log_gains[:, :, retained].mean(dim=-1) / depths[None, :]
        manual_leave_one_out.append(
            theil_sen_slopes(torch.log(depths), torch.log(lambda_subset.abs()))
        )
    manual = torch.stack(manual_leave_one_out, dim=1)
    manual_mean = manual.mean(dim=1, keepdim=True)
    manual_variance = 0.75 * (manual - manual_mean).square().sum(dim=1)

    torch.testing.assert_close(observed.leave_one_out_slopes, manual, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        observed.measurement_variance_by_example,
        manual_variance,
        rtol=0,
        atol=1e-12,
    )
    assert observed.sigma_w_hat == pytest.approx(manual_variance.mean().sqrt().item())


def test_autocast_is_rejected_and_probe_path_stays_fp32() -> None:
    model = _TinyTwoBlock()
    primal = torch.linspace(-0.4, 0.6, 8, dtype=torch.float32)
    directions = draw_example_probe_directions(
        primal,
        n_probe=2,
        example_probe_seed=11,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="outside autocast"):
            loop_log_gains(
                model,
                primal,
                directions,
                stochastic_snapshot=StochasticStateSnapshot.capture(model),
                has_aux=False,
            )

    observed = loop_log_gains(
        model,
        primal.to(torch.bfloat16),
        directions,
        stochastic_snapshot=StochasticStateSnapshot.capture(model),
        has_aux=False,
    )
    assert observed.dtype is torch.float32


def test_reporting_contract_and_registered_rejection_conditions() -> None:
    slopes, log_gains, directions = _report_inputs(520)

    report = build_panel_report(
        slopes,
        log_gains,
        run_seed=73,
        tier="main",
        provenance_sha256=PROVENANCE_SHA256,
        depths=REGISTERED_DEPTHS,
        c_l_by_depth=(0.2, 0.1, 0.04, 0.03),
        r_pr_by_depth=(7.0, 6.0, 5.0, 4.0),
        lambda_sign_by_depth=(1, 1, -1, -1),
        direction_class_gains=directions,
        fixed_branch_verified=True,
        bootstrap_seed=79,
    )
    payload = report.to_dict()

    assert payload["p_hat"] == pytest.approx(1.0)
    assert payload["Sxx"] == pytest.approx(REGISTERED_DESIGN_SXX)
    assert payload["instrument_tier"] == 1
    assert payload["tier"] == "main"
    assert payload["n"] == 520
    assert payload["n_probe"] == 4
    assert payload["depths"] == [1, 2, 4, 8]
    assert payload["conditioning_flag"] is True
    assert payload["rejection_reasons"] == [
        "sign_inconsistency",
        "conditioning_failure",
    ]
    assert set(payload["direction_class_gains_by_depth"]) == set(DIRECTION_CLASSES)
    assert payload["differentiation"] == "forward_mode_jvp"
    assert payload["jacobian_semantics"] == "fixed_routing_branch"
    assert payload["routing_flip_derivative_included"] is False
    assert payload["probe_seed_scope"] == "example"
    assert payload["depth_coordinate"] == "natural_log"
    assert payload["sigma_w_semantics"] == "paired_leave_one_probe_out_slope_sd"
    assert payload["provenance_sha256"] == PROVENANCE_SHA256
    assert payload["bootstrap_replicates"] == 10_000
    assert payload["bootstrap_seed"] == 79
    assert payload["admissible_panel_result"] is False
    assert payload["admissibility_blockers"] == ["branch_evidence_unlinked"]
    assert payload["branch_evidence_status"] == "unlinked_caller_assertion"

    with pytest.raises(RuntimeError, match="routing branch evidence"):
        build_panel_report(
            slopes,
            log_gains,
            run_seed=73,
            tier="main",
            provenance_sha256=PROVENANCE_SHA256,
            depths=REGISTERED_DEPTHS,
            c_l_by_depth=(0.2, 0.1, 0.08, 0.07),
            r_pr_by_depth=(7.0, 6.0, 5.0, 4.0),
            lambda_sign_by_depth=(1, 1, 1, 1),
            direction_class_gains=directions,
            fixed_branch_verified=False,
        )


def test_pilot_is_explicitly_non_admissible_and_cannot_be_main() -> None:
    slopes, log_gains, directions = _report_inputs(32)
    with pytest.raises(ValueError, match="n=520"):
        build_panel_report(
            slopes,
            log_gains,
            run_seed=1,
            tier="main",
            provenance_sha256=PROVENANCE_SHA256,
            depths=REGISTERED_DEPTHS,
            c_l_by_depth=(0.2, 0.2, 0.2, 0.2),
            r_pr_by_depth=(4.0, 4.0, 4.0, 4.0),
            lambda_sign_by_depth=(1, 1, 1, 1),
            direction_class_gains=directions,
            fixed_branch_verified=True,
        )

    pilot = build_pilot_diagnostics(
        slopes,
        log_gains,
        run_seed=1,
        provenance_sha256=PROVENANCE_SHA256,
        depths=REGISTERED_DEPTHS,
        c_l_by_depth=(0.2, 0.2, 0.2, 0.2),
        lambda_sign_by_depth=(1, 1, 1, 1),
        fixed_branch_verified=True,
    ).to_dict()

    assert pilot["n"] == 32
    assert pilot["n_probe"] == 4
    assert pilot["admissible_panel_result"] is False
    assert pilot["reporting_status"] == "pilot_diagnostic_only"
    assert "p_diagnostic" in pilot
    assert "p_hat" not in pilot


@pytest.mark.parametrize("tier", ("norm", "rank"))
def test_norm_and_rank_report_assembly_is_fail_closed(tier: str) -> None:
    slopes, log_gains, directions = _report_inputs(520)
    assert NORM_RANK_REPORTING_STATUS == "blocked_pending_shared_subsample_schema"
    with pytest.raises(RuntimeError, match="shared-subsample schema"):
        build_panel_report(
            slopes,
            log_gains,
            run_seed=1,
            tier=tier,  # type: ignore[arg-type]
            provenance_sha256=PROVENANCE_SHA256,
            depths=REGISTERED_DEPTHS,
            c_l_by_depth=(0.2, 0.2, 0.2, 0.2),
            r_pr_by_depth=(4.0, 4.0, 4.0, 4.0),
            lambda_sign_by_depth=(1, 1, 1, 1),
            direction_class_gains=directions,
            fixed_branch_verified=True,
        )


def test_main_bootstrap_contract_is_fixed_and_receipted() -> None:
    slopes, log_gains, directions = _report_inputs(520)
    with pytest.raises(ValueError, match="10,000 bootstrap"):
        build_panel_report(
            slopes,
            log_gains,
            run_seed=1,
            tier="main",
            provenance_sha256=PROVENANCE_SHA256,
            depths=REGISTERED_DEPTHS,
            c_l_by_depth=(0.2, 0.2, 0.2, 0.2),
            r_pr_by_depth=(4.0, 4.0, 4.0, 4.0),
            lambda_sign_by_depth=(1, 1, 1, 1),
            direction_class_gains=directions,
            fixed_branch_verified=True,
            bootstrap_replicates=9999,
        )

    with pytest.raises(ValueError, match="n_probe=4"):
        build_panel_report(
            slopes,
            log_gains[:, :, :3],
            run_seed=1,
            tier="main",
            provenance_sha256=PROVENANCE_SHA256,
            depths=REGISTERED_DEPTHS,
            c_l_by_depth=(0.2, 0.2, 0.2, 0.2),
            r_pr_by_depth=(4.0, 4.0, 4.0, 4.0),
            lambda_sign_by_depth=(1, 1, 1, 1),
            direction_class_gains=directions,
            fixed_branch_verified=True,
        )


def test_clipped_variance_and_tier_comparison_are_fail_closed() -> None:
    sigma, clipped = sigma_slope_hat(torch.ones(8), torch.ones(8))
    assert sigma == 0.0
    assert clipped is True
    assert rejection_conditions((1, 1, 1, 1), (0.2, 0.1, 0.08, 0.07)) == ()

    slopes, log_gains, directions = _report_inputs(520)
    main = build_panel_report(
        slopes,
        log_gains,
        run_seed=1,
        tier="main",
        provenance_sha256=PROVENANCE_SHA256,
        depths=REGISTERED_DEPTHS,
        c_l_by_depth=(0.2, 0.2, 0.2, 0.2),
        r_pr_by_depth=(4.0, 4.0, 4.0, 4.0),
        lambda_sign_by_depth=(1, 1, 1, 1),
        direction_class_gains=directions,
        fixed_branch_verified=True,
    )
    # Even a relabeled dataclass with mutually compatible intervals cannot
    # bypass the unresolved norm/rank evidence schema.
    norm = replace(main, tier="norm", p_hat=main.p_hat, ci_lo=0.8, ci_hi=1.2)
    comparison = compare_main_and_norm_tiers(main, norm)

    assert comparison.outcome == "return_to_strategy"
    assert comparison.main_inside_norm_interval is False
    assert "norm_rank_schema_unresolved" in comparison.comparison_reasons
    assert "main_rejected" in comparison.comparison_reasons

    mismatched = replace(
        norm,
        run_seed=2,
        depths=(1, 2, 4, 16),
        provenance_sha256="cd" * 32,
        jacobian_semantics="different_branch_semantics",
    )
    mismatch = compare_main_and_norm_tiers(main, mismatched)
    assert mismatch.outcome == "return_to_strategy"
    assert set(
        (
            "run_seed_mismatch",
            "depth_mismatch",
            "provenance_mismatch",
            "provenance_semantics_mismatch",
        )
    ).issubset(mismatch.comparison_reasons)

    rejected_norm = replace(
        norm,
        rejection_reasons=("conditioning_failure",),
        admissible_panel_result=False,
    )
    rejected = compare_main_and_norm_tiers(main, rejected_norm)
    assert "norm_rejected" in rejected.comparison_reasons


@pytest.mark.parametrize(
    ("panel_seed", "example_id", "error"),
    [
        (True, "example", TypeError),
        (-1, "example", ValueError),
        (1, "", ValueError),
        (1, 7, TypeError),
    ],
)
def test_example_probe_seed_derivation_is_fail_closed(
    panel_seed: object,
    example_id: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        derive_example_probe_seed(panel_seed, example_id)  # type: ignore[arg-type]
