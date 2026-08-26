from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from models.ablation_lm.callosum import (
    PerBandBirkhoffCallosum,
    empirical_disagreement_retention_receipt,
    hemisphere_gradient_cosine_receipt,
)
from models.ablation_lm.hadamard import wht


def _set_band_rho(
    callosum: PerBandBirkhoffCallosum, values: torch.Tensor
) -> None:
    if values.shape != (callosum.num_bands,):
        raise ValueError("test rho vector has the wrong shape")
    probabilities = 2.0 * values
    raw = torch.log(probabilities / (1.0 - probabilities))
    with torch.no_grad():
        callosum.raw_rho.copy_(raw)


def _band_modes(
    callosum: PerBandBirkhoffCallosum, lanes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    coefficients = callosum.sequency_coefficients(lanes).reshape(
        *lanes.shape[:-1], callosum.num_bands, callosum.band_width
    )
    lane_a, lane_b = coefficients.unbind(dim=-3)
    return (lane_a + lane_b) / 2.0, (lane_a - lane_b) / 2.0


def test_both_rungs_keep_eight_bands_and_use_small_nonzero_static_rho() -> None:
    proxy = PerBandBirkhoffCallosum(512, num_bands=8, rho_init=0.005)
    callosum = PerBandBirkhoffCallosum(1024, num_bands=8, rho_init=0.005)

    assert proxy.num_bands == callosum.num_bands == 8
    assert proxy.band_width == 64
    assert callosum.band_width == 128
    assert callosum.num_bands == 8
    assert callosum.raw_rho.shape == (8,)
    torch.testing.assert_close(
        callosum.band_rho(),
        torch.full((8,), 0.005),
        rtol=1e-6,
        atol=1e-8,
    )
    assert bool(callosum.band_rho().gt(0).all())
    assert bool(callosum.band_rho().le(0.5).all())
    assert tuple(callosum.sequency_order.shape) == (1024,)
    assert not any(name == "gate" for name, _ in callosum.named_parameters())


def test_t16_birkhoff_spectral_bound_and_closed_form_damping() -> None:
    torch.manual_seed(20_260_826)
    callosum = PerBandBirkhoffCallosum(
        32,
        num_bands=8,
        rho_init=0.005,
        stop_gradient_senders=False,
    )
    rho = torch.tensor([0.0001, 0.005, 0.03, 0.08, 0.15, 0.24, 0.37, 0.4999])
    _set_band_rho(callosum, rho)
    matrices = callosum.matrices()
    eigenvalues = callosum.disagreement_eigenvalues().detach()

    torch.testing.assert_close(matrices.sum(dim=-1), torch.ones(8, 2))
    torch.testing.assert_close(matrices.sum(dim=-2), torch.ones(8, 2))
    assert bool(matrices.ge(0).all())
    assert torch.linalg.svdvals(matrices).amax().item() <= 1.0 + 1e-6
    consensus = torch.tensor([1.0, 1.0])
    disagreement = torch.tensor([1.0, -1.0])
    torch.testing.assert_close(
        matrices @ consensus,
        consensus.expand(8, -1),
    )
    torch.testing.assert_close(
        matrices @ disagreement,
        eigenvalues[:, None] * disagreement,
    )

    lanes = torch.randn(2, 3, 2, 32)
    initial_mu, initial_delta = _band_modes(callosum, lanes)
    steps = 5
    carried = lanes
    for _ in range(steps):
        carried = callosum(carried)
    final_mu, final_delta = _band_modes(callosum, carried)
    retention = callosum.disagreement_retention(steps).reshape(1, 1, 8, 1)

    torch.testing.assert_close(final_mu, initial_mu, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        final_delta,
        retention * initial_delta,
        rtol=3e-4,
        atol=3e-5,
    )


def test_sequency_bands_transform_in_fp32_and_restore_input_dtype() -> None:
    torch.manual_seed(11)
    callosum = PerBandBirkhoffCallosum(16, num_bands=4, rho_init=0.02)

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        lanes = torch.randn(2, 5, 2, 16).to(dtype=dtype)
        coefficients = callosum.sequency_coefficients(lanes)
        expected = wht(lanes.float()).index_select(-1, callosum.sequency_order)

        assert coefficients.dtype is torch.float32
        torch.testing.assert_close(coefficients, expected, rtol=0, atol=0)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = callosum(lanes)
        assert output.dtype is dtype
        assert bool(torch.isfinite(output).all())

    differentiable = torch.randn(2, 3, 2, 16, requires_grad=True)
    callosum(differentiable).square().mean().backward()
    assert callosum.raw_rho.grad is not None
    assert callosum.raw_rho.grad.dtype is torch.float32
    assert bool(torch.isfinite(callosum.raw_rho.grad).all())

    converted = PerBandBirkhoffCallosum(16, num_bands=4).to(torch.bfloat16)
    assert converted.raw_rho.dtype is torch.float32
    converted(torch.randn(2, 2, 16, dtype=torch.bfloat16)).float().sum().backward()
    assert converted.raw_rho.grad is not None
    assert converted.raw_rho.grad.dtype is torch.float32


def test_fp32_gate_contract_survives_direct_and_recursive_assign_loading() -> None:
    direct = PerBandBirkhoffCallosum(16, num_bands=4)
    direct_state = direct.state_dict()
    direct_state["raw_rho"] = direct_state["raw_rho"].to(torch.bfloat16)
    direct.load_state_dict(direct_state, assign=True)
    assert direct.raw_rho.dtype is torch.float32

    parent = nn.ModuleDict(
        {"carrier": PerBandBirkhoffCallosum(16, num_bands=4)}
    )
    parent_state = parent.state_dict()
    parent_state["carrier.raw_rho"] = parent_state["carrier.raw_rho"].to(
        torch.bfloat16
    )
    parent.load_state_dict(parent_state, assign=True)
    assert parent["carrier"].raw_rho.dtype is torch.float32

    parent["carrier"](torch.randn(2, 2, 16)).sum().backward()
    assert parent["carrier"].raw_rho.grad is not None
    assert parent["carrier"].raw_rho.grad.dtype is torch.float32


def test_t9_sender_stop_gradient_changes_backward_not_forward() -> None:
    torch.manual_seed(19)
    callosum = PerBandBirkhoffCallosum(16, num_bands=4, rho_init=0.1)
    lane_a = torch.randn(2, 3, 16, requires_grad=True)
    lane_b = torch.randn(2, 3, 16, requires_grad=True)
    lanes = torch.stack((lane_a, lane_b), dim=-2)

    detached_output = callosum(lanes, stop_gradient_senders=True)
    coupled_output = callosum(lanes, stop_gradient_senders=False)
    torch.testing.assert_close(detached_output, coupled_output, rtol=0, atol=0)

    detached_grads = torch.autograd.grad(
        detached_output[..., 0, :].square().sum(),
        (lane_a, lane_b),
        allow_unused=True,
        retain_graph=True,
    )
    assert detached_grads[0] is not None
    assert bool(detached_grads[0].ne(0).any())
    assert detached_grads[1] is None or torch.count_nonzero(detached_grads[1]) == 0

    coupled_grads = torch.autograd.grad(
        coupled_output[..., 0, :].square().sum(),
        (lane_a, lane_b),
        allow_unused=True,
    )
    assert coupled_grads[1] is not None
    assert bool(coupled_grads[1].ne(0).any())


@pytest.mark.parametrize("step_index", (0, 1))
def test_alternating_parity_changes_directional_backward_not_forward(
    step_index: int,
) -> None:
    torch.manual_seed(29)
    callosum = PerBandBirkhoffCallosum(16, num_bands=4, rho_init=0.1)
    lane_a = torch.randn(2, 3, 16, requires_grad=True)
    lane_b = torch.randn(2, 3, 16, requires_grad=True)
    lanes = torch.stack((lane_a, lane_b), dim=-2)

    alternating = callosum(
        lanes,
        coupling_mode="alternating",
        step_index=step_index,
    )
    fully_coupled = callosum(lanes, coupling_mode="full")
    torch.testing.assert_close(alternating, fully_coupled, rtol=0, atol=0)

    detached_receiver = 0 if step_index % 2 == 0 else 1
    detached_sender = lane_b if detached_receiver == 0 else lane_a
    detached_gradient = torch.autograd.grad(
        alternating[..., detached_receiver, :].square().sum(),
        detached_sender,
        allow_unused=True,
        retain_graph=True,
    )[0]
    assert detached_gradient is None or torch.count_nonzero(detached_gradient) == 0

    live_receiver = 1 - detached_receiver
    live_sender = lane_a if live_receiver == 1 else lane_b
    live_gradient = torch.autograd.grad(
        alternating[..., live_receiver, :].square().sum(),
        live_sender,
        allow_unused=True,
    )[0]
    assert live_gradient is not None and bool(live_gradient.ne(0).any())


def test_hemisphere_gradient_cosine_receipt_is_fp32_and_fail_closed() -> None:
    gradient_a = (torch.tensor([1.0, 2.0]), torch.tensor([3.0]))
    gradient_b = (torch.tensor([2.0, 4.0]), torch.tensor([6.0]))
    receipt = hemisphere_gradient_cosine_receipt(gradient_a, gradient_b)

    assert receipt.cosine.dtype is torch.float32
    torch.testing.assert_close(receipt.cosine, torch.tensor(1.0))
    assert receipt.norm_a.item() > 0 and receipt.norm_b.item() > 0
    with pytest.raises(ValueError, match="live gradients"):
        hemisphere_gradient_cosine_receipt(
            gradient_a,
            tuple(torch.zeros_like(gradient) for gradient in gradient_b),
        )
    large = (torch.tensor([3e38], dtype=torch.float32),)
    large_receipt = hemisphere_gradient_cosine_receipt(large, large)
    torch.testing.assert_close(large_receipt.cosine, torch.tensor(1.0))
    assert bool(torch.isfinite(large_receipt.cosine))


def test_empirical_retention_receipt_enforces_the_non_tautological_tripwire() -> None:
    torch.manual_seed(37)
    initial = torch.randn(2, 3, 2, 16)
    gentle = PerBandBirkhoffCallosum(
        16,
        num_bands=4,
        rho_init=0.005,
        stop_gradient_senders=False,
    )
    steps = 4
    final = initial
    for _ in range(steps):
        final = gentle(final)
    receipt = empirical_disagreement_retention_receipt(
        initial,
        final,
        steps=steps,
    )

    assert receipt.retention.item() >= 0.9
    assert receipt.steps == steps
    assert receipt.floor == 0.9
    assert receipt.initial_disagreement_norm.item() > 0
    assert receipt.final_disagreement_norm.item() > 0

    strong = PerBandBirkhoffCallosum(
        16,
        num_bands=4,
        rho_init=0.2,
        stop_gradient_senders=False,
    )
    strong_final = initial
    for _ in range(steps):
        strong_final = strong(strong_final)
    with pytest.raises(RuntimeError, match="tripwire at K=4"):
        empirical_disagreement_retention_receipt(
            initial,
            strong_final,
            steps=steps,
        )


@pytest.mark.parametrize(
    ("args", "kwargs", "error", "match"),
    [
        ((0,), {}, ValueError, "d_model"),
        ((12,), {}, ValueError, "d_model"),
        ((True,), {}, ValueError, "d_model"),
        ((16,), {"num_bands": 3}, ValueError, "num_bands"),
        ((16,), {"num_bands": 32}, ValueError, "num_bands"),
        ((16,), {"num_bands": 4, "rho_init": 0.0}, ValueError, "rho_init"),
        ((16,), {"num_bands": 4, "rho_init": 0.5}, ValueError, "rho_init"),
        ((16,), {"num_bands": 4, "rho_init": math.nan}, ValueError, "rho_init"),
        ((16,), {"num_bands": 4, "rho_init": True}, TypeError, "rho_init"),
        (
            (16,),
            {"num_bands": 4, "stop_gradient_senders": 1},
            TypeError,
            "stop_gradient",
        ),
    ],
)
def test_constructor_rejects_foot_guns(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        PerBandBirkhoffCallosum(*args, **kwargs)


def test_forward_rejects_shape_dtype_device_and_finiteness_foot_guns() -> None:
    callosum = PerBandBirkhoffCallosum(8, num_bands=2)

    with pytest.raises(TypeError, match="tensor"):
        callosum.sequency_coefficients([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="final-but-one"):
        callosum(torch.randn(8))
    with pytest.raises(ValueError, match="final-but-one"):
        callosum(torch.randn(2, 3, 8))
    with pytest.raises(ValueError, match="final width"):
        callosum(torch.randn(2, 2, 16))
    with pytest.raises(TypeError, match="floating-point"):
        callosum(torch.ones(2, 2, 8, dtype=torch.long))
    non_finite = torch.zeros(2, 2, 8)
    non_finite[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite before"):
        callosum(non_finite)
    with pytest.raises(TypeError, match="bool or None"):
        callosum(torch.randn(2, 2, 8), stop_gradient_senders=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="coupling_mode"):
        callosum(torch.randn(2, 2, 8), coupling_mode="unknown")
    with pytest.raises(ValueError, match="step_index"):
        callosum(torch.randn(2, 2, 8), coupling_mode="alternating")
    with pytest.raises(ValueError, match="choose coupling_mode"):
        callosum(
            torch.randn(2, 2, 8),
            coupling_mode="full",
            stop_gradient_senders=False,
        )
    with pytest.raises(ValueError, match="steps"):
        callosum.disagreement_retention(True)
    with pytest.raises(ValueError, match="steps"):
        callosum.disagreement_retention(-1)
