"""Static per-sequency-band Birkhoff transport between two hemispheres."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

from .hadamard import sequency_permutation, wht
from .optim import ParameterRole, tag_optimizer_role


_COUPLING_MODES = ("detached", "alternating", "full")
CALLOSUM_RETENTION_TRIPWIRE = 0.9


@dataclass(frozen=True)
class HemisphereGradientCosineReceipt:
    """FP32 gradient-cosine evidence for the two symmetric lane families."""

    cosine: torch.Tensor
    norm_a: torch.Tensor
    norm_b: torch.Tensor


@dataclass(frozen=True)
class EmpiricalCarrierRetentionReceipt:
    """Observed disagreement-norm retention with its binding tripwire."""

    retention: torch.Tensor
    initial_disagreement_norm: torch.Tensor
    final_disagreement_norm: torch.Tensor
    steps: int
    floor: float


@dataclass(frozen=True)
class DeltaModePredictionReceipt:
    """Per-band callosum-only test of the closed-form disagreement law.

    The amplitude multiplier is ``(1 - 2 rho_b)**K``.  Squared energy has the
    distinct multiplier ``(1 - 2 rho_b)**(2K)``; both are named explicitly so
    the word "energy" cannot silently stand in for an amplitude norm.
    """

    rho: torch.Tensor
    disagreement_eigenvalue: torch.Tensor
    expected_amplitude_retention: torch.Tensor
    observed_amplitude_retention: torch.Tensor
    expected_energy_retention: torch.Tensor
    observed_energy_retention: torch.Tensor
    amplitude_absolute_error: torch.Tensor
    energy_absolute_error: torch.Tensor
    steps: int
    scope: str = "callosum_only_excludes_intervening_core_dynamics"


def _robust_normalized_vector(
    values: torch.Tensor,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a unit FP32 vector and overflow-checked FP32 norm."""

    flat = values.detach().float().reshape(-1)
    scale = flat.abs().amax()
    if not bool(torch.isfinite(scale)) or bool(scale.eq(0)):
        raise ValueError(f"{name} requires a finite nonzero vector")
    scaled = flat / scale
    scaled_norm = torch.linalg.vector_norm(scaled)
    norm = scale * scaled_norm
    if not bool(torch.isfinite(scaled_norm)) or not bool(torch.isfinite(norm)):
        raise ValueError(f"{name} norm overflowed FP32")
    return scaled / scaled_norm, norm


def hemisphere_gradient_cosine_receipt(
    gradients_a: tuple[torch.Tensor, ...],
    gradients_b: tuple[torch.Tensor, ...],
    *,
    eps: float = 1e-12,
) -> HemisphereGradientCosineReceipt:
    """Measure the co-adaptation instrument without constructing an optimizer."""

    if not isinstance(gradients_a, tuple) or not isinstance(gradients_b, tuple):
        raise TypeError("hemisphere gradients must be tuples")
    if not gradients_a or len(gradients_a) != len(gradients_b):
        raise ValueError("hemisphere gradient tuples must be nonempty and aligned")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    for gradient_a, gradient_b in zip(gradients_a, gradients_b, strict=True):
        if not isinstance(gradient_a, torch.Tensor) or not isinstance(
            gradient_b, torch.Tensor
        ):
            raise TypeError("hemisphere gradients must contain tensors")
        if gradient_a.shape != gradient_b.shape:
            raise ValueError("paired hemisphere gradients must have identical shapes")
        if not gradient_a.is_floating_point() or not gradient_b.is_floating_point():
            raise TypeError("hemisphere gradients must be floating point")
        if gradient_a.device != gradient_b.device:
            raise ValueError("paired hemisphere gradients must share a device")
        if not bool(torch.isfinite(gradient_a.detach()).all()) or not bool(
            torch.isfinite(gradient_b.detach()).all()
        ):
            raise ValueError("hemisphere gradients must be finite")
    flat_a = torch.cat(
        tuple(gradient.detach().float().reshape(-1) for gradient in gradients_a)
    )
    flat_b = torch.cat(
        tuple(gradient.detach().float().reshape(-1) for gradient in gradients_b)
    )
    if not bool(flat_a.ne(0).any()) or not bool(flat_b.ne(0).any()):
        raise ValueError("hemisphere gradient cosine requires two live gradients")
    unit_a, norm_a = _robust_normalized_vector(flat_a, name="hemisphere A gradient")
    unit_b, norm_b = _robust_normalized_vector(flat_b, name="hemisphere B gradient")
    if bool(norm_a.le(eps)) or bool(norm_b.le(eps)):
        raise ValueError("hemisphere gradient cosine requires two live gradients")
    cosine = torch.dot(unit_a, unit_b)
    if not bool(torch.isfinite(cosine)):
        raise ValueError("hemisphere gradient cosine is not finite")
    cosine = cosine.clamp(-1.0, 1.0)
    return HemisphereGradientCosineReceipt(
        cosine=cosine,
        norm_a=norm_a,
        norm_b=norm_b,
    )


def empirical_disagreement_retention_receipt(
    initial_lanes: torch.Tensor,
    final_lanes: torch.Tensor,
    *,
    steps: int,
) -> EmpiricalCarrierRetentionReceipt:
    """Measure horizon retention and enforce the binding 0.9 tripwire."""

    if not isinstance(initial_lanes, torch.Tensor) or not isinstance(
        final_lanes, torch.Tensor
    ):
        raise TypeError("carrier retention inputs must be tensors")
    if initial_lanes.shape != final_lanes.shape:
        raise ValueError("carrier retention inputs must have identical shapes")
    if initial_lanes.ndim < 2 or initial_lanes.shape[-2] != 2:
        raise ValueError("carrier retention inputs require a two-lane axis")
    if not initial_lanes.is_floating_point() or not final_lanes.is_floating_point():
        raise TypeError("carrier retention inputs must be floating point")
    if initial_lanes.device != final_lanes.device:
        raise ValueError("carrier retention inputs must share a device")
    if not bool(torch.isfinite(initial_lanes.detach()).all()) or not bool(
        torch.isfinite(final_lanes.detach()).all()
    ):
        raise ValueError("carrier retention inputs must be finite")
    if type(steps) is not int or steps < 1:
        raise ValueError("retention steps must be a positive exact integer")

    initial_delta = initial_lanes[..., 0, :] - initial_lanes[..., 1, :]
    final_delta = final_lanes[..., 0, :] - final_lanes[..., 1, :]
    _initial_unit, initial_norm = _robust_normalized_vector(
        initial_delta,
        name="initial carrier disagreement",
    )
    _final_unit, final_norm = _robust_normalized_vector(
        final_delta,
        name="final carrier disagreement",
    )
    retention = final_norm / initial_norm
    if not bool(torch.isfinite(retention)):
        raise ValueError("empirical carrier retention is not finite")
    if bool(retention.lt(CALLOSUM_RETENTION_TRIPWIRE)):
        raise RuntimeError(
            f"empirical carrier retention {retention.item():.6f} is below "
            f"the {CALLOSUM_RETENTION_TRIPWIRE:.6f} tripwire at K={steps}"
        )
    return EmpiricalCarrierRetentionReceipt(
        retention=retention,
        initial_disagreement_norm=initial_norm,
        final_disagreement_norm=final_norm,
        steps=steps,
        floor=CALLOSUM_RETENTION_TRIPWIRE,
    )


def _positive_power_of_two(value: int, *, name: str) -> int:
    if type(value) is not int or value < 1 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two")
    return value


class PerBandBirkhoffCallosum(nn.Module):
    """Mix two lanes with one static Birkhoff coefficient per WHT band.

    The lane axis is final-but-one and the hidden axis is final.  Hidden
    coordinates are transformed to sequency order before contiguous bands are
    mixed.  For every band ``b`` the exact carrier is

    ``A_b = (1 - rho_b) I + rho_b P``, with ``rho_b in [0, 1/2]``.

    Its consensus eigenvalue is one and its disagreement eigenvalue is
    ``1 - 2 rho_b``.  This module is the active primitive: structural OFF is a
    caller-level construction choice, not a zero gate inside this module.
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_bands: int = 8,
        rho_init: float = 0.005,
        stop_gradient_senders: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = _positive_power_of_two(d_model, name="d_model")
        self.num_bands = _positive_power_of_two(num_bands, name="num_bands")
        if self.num_bands > self.d_model or self.d_model % self.num_bands:
            raise ValueError("num_bands must divide d_model and may not exceed it")
        self.band_width = self.d_model // self.num_bands
        if type(stop_gradient_senders) is not bool:
            raise TypeError("stop_gradient_senders must be a bool")
        if isinstance(rho_init, bool):
            raise TypeError("rho_init must be a finite real scalar")
        try:
            initial_rho = float(rho_init)
        except (TypeError, ValueError) as error:
            raise TypeError("rho_init must be a finite real scalar") from error
        if not math.isfinite(initial_rho) or not 0.0 < initial_rho < 0.5:
            raise ValueError("rho_init must lie strictly between zero and one half")

        self.stop_gradient_senders = stop_gradient_senders
        probability = 2.0 * initial_rho
        raw_initial = math.log(probability / (1.0 - probability))
        self.raw_rho = nn.Parameter(
            torch.full((self.num_bands,), raw_initial, dtype=torch.float32)
        )
        self._restore_parameter_contract()

        order = sequency_permutation(self.d_model)
        self.register_buffer("sequency_order", order)
        self.register_buffer("inverse_sequency_order", torch.argsort(order))
        self._inverse_scale = 2.0 ** -(self.d_model.bit_length() - 1)

    def band_rho(self) -> torch.Tensor:
        """Return the static per-band coefficients in ``[0, 1/2]``."""

        return 0.5 * torch.sigmoid(self.raw_rho)

    def disagreement_eigenvalues(self) -> torch.Tensor:
        """Return the exact disagreement eigenvalue of every band carrier."""

        return 1.0 - 2.0 * self.band_rho()

    def disagreement_retention(self, steps: int) -> torch.Tensor:
        """Return closed-form disagreement retention after ``steps`` passes."""

        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        return self.disagreement_eigenvalues().pow(steps)

    def matrices(self) -> torch.Tensor:
        """Return the ``[bands, 2, 2]`` doubly-stochastic carriers."""

        rho = self.band_rho()
        one = torch.ones_like(rho)
        return torch.stack((one - rho, rho, rho, one - rho), dim=-1).view(
            self.num_bands, 2, 2
        )

    def sequency_coefficients(self, lanes: torch.Tensor) -> torch.Tensor:
        """Transform lanes to sequency order, always using FP32 arithmetic."""

        self._validate_lanes(lanes)
        coefficients = wht(lanes.float()).index_select(-1, self.sequency_order)
        if coefficients.dtype is not torch.float32:
            raise RuntimeError("WHT coefficients must remain FP32")
        if not bool(torch.isfinite(coefficients.detach()).all()):
            raise FloatingPointError("WHT produced non-finite coefficients before mixing")
        return coefficients

    def forward(
        self,
        lanes: torch.Tensor,
        *,
        stop_gradient_senders: bool | None = None,
        coupling_mode: str | None = None,
        step_index: int | None = None,
    ) -> torch.Tensor:
        """Apply per-band transport and restore the input dtype.

        When sender stop-gradient is active, the cross-lane value is unchanged
        in the forward pass, but each receiving lane cannot backpropagate into
        its sender through this transport edge.
        """

        if stop_gradient_senders is not None and type(stop_gradient_senders) is not bool:
            raise TypeError("stop_gradient_senders must be a bool or None")
        if coupling_mode is not None and stop_gradient_senders is not None:
            raise ValueError("choose coupling_mode or stop_gradient_senders, not both")
        if coupling_mode is None:
            detach_senders = (
                self.stop_gradient_senders
                if stop_gradient_senders is None
                else stop_gradient_senders
            )
            mode = "detached" if detach_senders else "full"
        else:
            if coupling_mode not in _COUPLING_MODES:
                raise ValueError(f"coupling_mode must be one of {_COUPLING_MODES!r}")
            mode = coupling_mode
        if mode == "alternating":
            if type(step_index) is not int or step_index < 0:
                raise ValueError(
                    "alternating coupling requires a non-negative integer step_index"
                )
        elif step_index is not None:
            raise ValueError("step_index is only valid for alternating coupling")

        coefficients = self.sequency_coefficients(lanes)
        banded = coefficients.reshape(
            *coefficients.shape[:-1], self.num_bands, self.band_width
        )
        rho_shape = (1,) * (banded.ndim - 2) + (self.num_bands, 1)
        rho = self.band_rho().reshape(rho_shape)
        senders = banded.flip(dims=(-3,))
        if mode == "detached":
            senders = senders.detach()
        elif mode == "alternating":
            assert step_index is not None
            live_receiver = 1 if step_index % 2 == 0 else 0
            sender_lanes = senders.unbind(dim=-3)
            senders = torch.stack(
                tuple(
                    sender if receiver == live_receiver else sender.detach()
                    for receiver, sender in enumerate(sender_lanes)
                ),
                dim=-3,
            )
        mixed = (1.0 - rho) * banded + rho * senders
        if not bool(torch.isfinite(mixed.detach()).all()):
            raise FloatingPointError("callosum mixing produced non-finite coefficients")

        mixed_sequency = mixed.flatten(-2)
        mixed_natural = mixed_sequency.index_select(
            -1, self.inverse_sequency_order
        )
        restored_fp32 = wht(mixed_natural) * self._inverse_scale
        if restored_fp32.dtype is not torch.float32:
            raise RuntimeError("inverse WHT must remain FP32")
        if not bool(torch.isfinite(restored_fp32.detach()).all()):
            raise FloatingPointError("inverse WHT produced non-finite lanes")
        restored = restored_fp32.to(dtype=lanes.dtype)
        if not bool(torch.isfinite(restored.detach()).all()):
            raise FloatingPointError("dtype restoration produced non-finite lanes")
        return restored

    def _validate_lanes(self, lanes: torch.Tensor) -> None:
        if not isinstance(lanes, torch.Tensor):
            raise TypeError("lanes must be a tensor")
        if lanes.ndim < 2 or lanes.shape[-2] != 2:
            raise ValueError("lanes must have a final-but-one axis of length two")
        if lanes.shape[-1] != self.d_model:
            raise ValueError(f"lanes must have final width {self.d_model}")
        if not lanes.is_floating_point():
            raise TypeError("lanes must be a real floating-point tensor")
        if lanes.device.type == "meta":
            raise ValueError("meta tensors cannot be transported")
        if lanes.device != self.raw_rho.device:
            raise ValueError("lanes and callosum parameters must be on the same device")
        if not bool(torch.isfinite(lanes.detach()).all()):
            raise ValueError("lanes must be finite before callosum mixing")

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_bands={self.num_bands}, "
            f"band_width={self.band_width}, "
            f"stop_gradient_senders={self.stop_gradient_senders}"
        )

    def _apply(self, fn, recurse: bool = True) -> "PerBandBirkhoffCallosum":
        """Move the module while keeping the carrier coefficient in FP32."""

        super()._apply(fn, recurse=recurse)
        self._restore_parameter_contract()
        return self

    def _restore_parameter_contract(self) -> None:
        if not isinstance(self.raw_rho, nn.Parameter):
            raise RuntimeError("raw_rho must remain a direct Parameter")
        if tuple(self.raw_rho.shape) != (self.num_bands,):
            raise RuntimeError("raw_rho shape changed from the configured band count")
        if self.raw_rho.device.type != "meta" and self.raw_rho.dtype is not torch.float32:
            with torch.no_grad():
                self.raw_rho.data = self.raw_rho.data.float()
                if self.raw_rho.grad is not None:
                    self.raw_rho.grad.data = self.raw_rho.grad.data.float()
        tag_optimizer_role(self, "raw_rho", ParameterRole.GATE)

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._restore_parameter_contract()

    def load_state_dict(
        self,
        state_dict: Any,
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        finally:
            self._restore_parameter_contract()

    def _load_from_state_dict(
        self,
        state_dict: Any,
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        try:
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
        finally:
            self._restore_parameter_contract()


def delta_mode_prediction_receipt(
    callosum: PerBandBirkhoffCallosum,
    initial_lanes: torch.Tensor,
    final_lanes: torch.Tensor,
    *,
    steps: int,
) -> DeltaModePredictionReceipt:
    """Compare a pure callosum carry with its exact per-band prediction.

    ``final_lanes`` must be obtained by applying only ``callosum`` for the
    declared number of steps.  Core blocks create new disagreement, so this
    receipt deliberately makes no claim about an entire recurrent transition.
    """

    if not isinstance(callosum, PerBandBirkhoffCallosum):
        raise TypeError("callosum must be a PerBandBirkhoffCallosum")
    if not isinstance(initial_lanes, torch.Tensor) or not isinstance(
        final_lanes, torch.Tensor
    ):
        raise TypeError("delta-mode receipt inputs must be tensors")
    if initial_lanes.shape != final_lanes.shape:
        raise ValueError("delta-mode receipt inputs must have identical shapes")
    if initial_lanes.device != final_lanes.device:
        raise ValueError("delta-mode receipt inputs must share a device")
    if initial_lanes.dtype != final_lanes.dtype:
        raise TypeError("delta-mode receipt inputs must share a dtype")
    if type(steps) is not int or steps < 1:
        raise ValueError("delta-mode receipt steps must be a positive exact integer")

    def band_energy(lanes: torch.Tensor) -> torch.Tensor:
        coefficients = callosum.sequency_coefficients(lanes.detach()).reshape(
            *lanes.shape[:-1], callosum.num_bands, callosum.band_width
        )
        lane_a, lane_b = coefficients.unbind(dim=-3)
        delta = (lane_a - lane_b) / 2.0
        by_band = delta.movedim(-2, 0).reshape(callosum.num_bands, -1)
        energy = by_band.square().mean(dim=-1)
        if not bool(torch.isfinite(energy).all()):
            raise ValueError("delta-mode band energy must be finite")
        return energy

    with torch.no_grad():
        initial_energy = band_energy(initial_lanes)
        final_energy = band_energy(final_lanes)
        if bool(initial_energy.le(0).any()):
            raise ValueError(
                "every measured callosum band needs nonzero initial disagreement"
            )
        observed_energy = final_energy / initial_energy
        if not bool(torch.isfinite(observed_energy).all()):
            raise ValueError("observed delta-mode energy retention must be finite")
        observed_amplitude = observed_energy.sqrt()
        eigenvalue = callosum.disagreement_eigenvalues().detach().float()
        rho = callosum.band_rho().detach().float()
        expected_amplitude = eigenvalue.pow(steps)
        expected_energy = eigenvalue.pow(2 * steps)
        amplitude_error = (observed_amplitude - expected_amplitude).abs()
        energy_error = (observed_energy - expected_energy).abs()
        named_values = {
            "rho": rho,
            "disagreement eigenvalue": eigenvalue,
            "expected amplitude retention": expected_amplitude,
            "observed amplitude retention": observed_amplitude,
            "expected energy retention": expected_energy,
            "observed energy retention": observed_energy,
            "amplitude absolute error": amplitude_error,
            "energy absolute error": energy_error,
        }
        for name, value in named_values.items():
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"delta-mode {name} must be finite")
    return DeltaModePredictionReceipt(
        rho=rho,
        disagreement_eigenvalue=eigenvalue,
        expected_amplitude_retention=expected_amplitude,
        observed_amplitude_retention=observed_amplitude,
        expected_energy_retention=expected_energy,
        observed_energy_retention=observed_energy,
        amplitude_absolute_error=amplitude_error,
        energy_absolute_error=energy_error,
        steps=steps,
    )
