"""S-2 per-sequency-band unit-circle read of two terminal hemispheres."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .hadamard import sequency_permutation, wht
from .optim import ParameterRole, tag_optimizer_role


def _positive_power_of_two(value: int, *, name: str) -> int:
    if type(value) is not int or value < 1 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two")
    return value


class PerBandUnitCircleCombiner(nn.Module):
    """Read consensus and disagreement with a unit-norm pair per WHT band.

    In sequency band ``b`` the output is

    ``y_b = cos(theta_b) * mu_b + sin(theta_b) * delta_b``.

    ``theta=0`` initializes the read at consensus, while disagreement gives the
    angle a live gradient. The unit-circle parameterization prevents the read
    from amplifying the joint ``(mu, delta)`` state in any band.
    """

    def __init__(self, d_model: int, *, num_bands: int = 8) -> None:
        super().__init__()
        self.d_model = _positive_power_of_two(d_model, name="d_model")
        self.num_bands = _positive_power_of_two(num_bands, name="num_bands")
        if self.num_bands > self.d_model or self.d_model % self.num_bands:
            raise ValueError("num_bands must divide d_model and may not exceed it")
        self.band_width = self.d_model // self.num_bands
        self.theta = nn.Parameter(torch.zeros(self.num_bands, dtype=torch.float32))
        order = sequency_permutation(self.d_model)
        self.register_buffer("sequency_order", order)
        self.register_buffer("inverse_sequency_order", torch.argsort(order))
        self._inverse_scale = 2.0 ** -(self.d_model.bit_length() - 1)
        self._restore_parameter_contract()

    def unit_circle_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact per-band ``(cos(theta), sin(theta))`` pair."""

        return self.theta.cos(), self.theta.sin()

    def lateralization_index(self) -> torch.Tensor:
        """Return the D-HD-1 per-band index ``sin(2 * theta_b)``.

        At the registered initialization, zero is the consensus read; at the
        principal points ``theta=+/-pi/4``, ``+/-1`` identify the pure
        hemisphere-A/B directions under the S-2 coefficient convention.  The
        raw index is periodic, so interpreting other trained angles requires a
        separately governed principal-domain or canonicalization rule.
        """

        index = torch.sin(2.0 * self.theta)
        if not bool(torch.isfinite(index.detach()).all()):
            raise ValueError("lateralization_index requires finite combiner angles")
        return index

    def _validate_pair(self, h_a: torch.Tensor, h_b: torch.Tensor) -> None:
        if not isinstance(h_a, torch.Tensor) or not isinstance(h_b, torch.Tensor):
            raise TypeError("h_a and h_b must be tensors")
        if h_a.shape != h_b.shape:
            raise ValueError("h_a and h_b must have identical shapes")
        if h_a.ndim < 1 or h_a.shape[-1] != self.d_model:
            raise ValueError(f"hemisphere states must have final width {self.d_model}")
        if not h_a.is_floating_point() or not h_b.is_floating_point():
            raise TypeError("hemisphere states must be real floating-point tensors")
        if h_a.dtype != h_b.dtype or h_a.device != h_b.device:
            raise ValueError("h_a and h_b must share one dtype and device")
        if h_a.device.type == "meta":
            raise ValueError("meta hemisphere states cannot be combined")
        if h_a.device != self.theta.device:
            raise ValueError("hemisphere states and combiner must share a device")
        if not bool(torch.isfinite(h_a.detach()).all()) or not bool(
            torch.isfinite(h_b.detach()).all()
        ):
            raise ValueError("hemisphere states must be finite")

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        self._validate_pair(h_a, h_b)
        consensus = (h_a.float() + h_b.float()) * 0.5
        disagreement = (h_a.float() - h_b.float()) * 0.5
        consensus_coefficients = wht(consensus).index_select(-1, self.sequency_order)
        disagreement_coefficients = wht(disagreement).index_select(
            -1, self.sequency_order
        )
        band_shape = (*consensus_coefficients.shape[:-1], self.num_bands, self.band_width)
        consensus_bands = consensus_coefficients.reshape(band_shape)
        disagreement_bands = disagreement_coefficients.reshape(band_shape)
        coefficient_shape = (1,) * (consensus_bands.ndim - 2) + (self.num_bands, 1)
        cosine, sine = self.unit_circle_coefficients()
        combined_bands = (
            cosine.reshape(coefficient_shape) * consensus_bands
            + sine.reshape(coefficient_shape) * disagreement_bands
        )
        natural_order = combined_bands.flatten(-2).index_select(
            -1,
            self.inverse_sequency_order,
        )
        combined_fp32 = wht(natural_order) * self._inverse_scale
        if not bool(torch.isfinite(combined_fp32.detach()).all()):
            raise FloatingPointError("S-2 combination produced non-finite values")
        return combined_fp32.to(dtype=h_a.dtype)

    def _restore_parameter_contract(self) -> None:
        if not isinstance(self.theta, nn.Parameter):
            raise RuntimeError("theta must remain a direct Parameter")
        if tuple(self.theta.shape) != (self.num_bands,):
            raise RuntimeError("theta shape changed from the configured band count")
        if self.theta.device.type != "meta" and self.theta.dtype is not torch.float32:
            with torch.no_grad():
                self.theta.data = self.theta.data.float()
                if self.theta.grad is not None:
                    self.theta.grad.data = self.theta.grad.data.float()
        tag_optimizer_role(self, "theta", ParameterRole.GATE)

    def _apply(self, fn: Any, recurse: bool = True) -> "PerBandUnitCircleCombiner":
        super()._apply(fn, recurse=recurse)
        self._restore_parameter_contract()
        return self

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
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

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_bands={self.num_bands}, "
            f"band_width={self.band_width}"
        )
