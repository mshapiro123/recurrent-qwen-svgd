"""Bicameral linear weights stored directly in swap-eigenmode coordinates."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .optim import (
    OptimizerTarget,
    ParameterRole,
    RANK_ONLY_MUON_PROHIBITED_ATTR,
    partition_optimizer_parameters,
    tag_optimizer_role,
)


def _positive_integer(value: int, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class SwapLinear(nn.Module):
    """A pair of linear maps stored as a mean plus factored disagreement.

    The physical hemisphere matrices are never parameters.  For hemisphere
    sign ``h`` in ``{+1, -1}``, the forward map is exactly

    ``F.linear(x, mu) + h * ((x @ dV) @ dU.T)``.

    ``mu``, ``dU``, and ``dV`` form one semantic coupled-mode family and are
    therefore kept together on AdamW.  Rank-only Muon splitting is prohibited
    for every stored tensor, including after deepcopy, dtype/device conversion,
    and assign-style state loading.
    """

    _COUPLED_PARAMETER_NAMES = ("mu", "dU", "dV")

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        rank: int = 32,
        sigma_delta0: float = 0.02,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_in = _positive_integer(d_in, name="d_in")
        self.d_out = _positive_integer(d_out, name="d_out")
        self.rank = _positive_integer(rank, name="rank")
        if self.rank > min(self.d_in, self.d_out):
            raise ValueError("rank may not exceed min(d_in, d_out)")
        if isinstance(sigma_delta0, bool):
            raise TypeError("sigma_delta0 must be a finite positive real scalar")
        try:
            sigma = float(sigma_delta0)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "sigma_delta0 must be a finite positive real scalar"
            ) from error
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_delta0 must be finite and strictly positive")
        if type(seed) is not int or seed < 0 or seed >= 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if dtype is not None and not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("SwapLinear parameters require a real floating dtype")

        self.sigma_delta0 = sigma
        self.initialization_seed = seed
        factory_kwargs = {"device": device, "dtype": dtype}
        self.mu = nn.Parameter(torch.empty(self.d_out, self.d_in, **factory_kwargs))
        self.dU = nn.Parameter(torch.empty(self.d_out, self.rank, **factory_kwargs))
        self.dV = nn.Parameter(torch.empty(self.d_in, self.rank, **factory_kwargs))
        self.reset_parameters()
        self._restore_parameter_contract()

    def reset_parameters(self) -> None:
        """Recreate the registered deterministic asymmetric initialization."""

        if self.mu.device.type == "meta":
            return
        generator = torch.Generator(device="cpu").manual_seed(
            self.initialization_seed
        )
        source_dtype = torch.float64 if self.mu.dtype is torch.float64 else torch.float32
        mu = torch.randn(
            self.d_out,
            self.d_in,
            generator=generator,
            dtype=source_dtype,
        ) * self.d_in**-0.5
        delta_u = torch.randn(
            self.d_out,
            self.rank,
            generator=generator,
            dtype=source_dtype,
        ) * self.sigma_delta0
        delta_v = torch.randn(
            self.d_in,
            self.rank,
            generator=generator,
            dtype=source_dtype,
        ) * self.sigma_delta0
        with torch.no_grad():
            self.mu.copy_(mu.to(device=self.mu.device, dtype=self.mu.dtype))
            self.dU.copy_(delta_u.to(device=self.dU.device, dtype=self.dU.dtype))
            self.dV.copy_(delta_v.to(device=self.dV.device, dtype=self.dV.dtype))
        if not bool(torch.isfinite(self.mu.detach()).all()):
            raise FloatingPointError("mu initialization is not finite in the requested dtype")
        for name, parameter in (("dU", self.dU), ("dV", self.dV)):
            if not bool(torch.isfinite(parameter.detach()).all()):
                raise FloatingPointError(
                    f"{name} initialization is not finite in the requested dtype"
                )
            if not bool(parameter.detach().ne(0).any()):
                raise FloatingPointError(
                    f"{name} initialization underflowed to an all-zero factor"
                )

    def forward(self, inputs: torch.Tensor, hemi: int) -> torch.Tensor:
        """Apply one hemisphere without constructing either dense matrix."""

        self._validate_parameter_contract()
        if type(hemi) is not int or hemi not in (-1, 1):
            raise ValueError("hemi must be the exact integer +1 or -1")
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a tensor")
        if inputs.ndim < 1 or inputs.shape[-1] != self.d_in:
            raise ValueError(f"inputs must have final width {self.d_in}")
        if not inputs.is_floating_point():
            raise TypeError("inputs must be a real floating-point tensor")
        if inputs.device.type == "meta":
            raise ValueError("meta tensors cannot be evaluated")
        if inputs.device != self.mu.device:
            raise ValueError("inputs and SwapLinear parameters must share a device")
        if inputs.dtype != self.mu.dtype:
            raise TypeError("inputs and SwapLinear parameters must share a dtype")
        if not bool(torch.isfinite(inputs.detach()).all()):
            raise ValueError("inputs must be finite")

        return F.linear(inputs, self.mu) + hemi * (
            (inputs @ self.dV) @ self.dU.T
        )

    def _validate_parameter_contract(self) -> None:
        expected_shapes = {
            "mu": (self.d_out, self.d_in),
            "dU": (self.d_out, self.rank),
            "dV": (self.d_in, self.rank),
        }
        parameters = {name: getattr(self, name) for name in self._COUPLED_PARAMETER_NAMES}
        for name, expected in expected_shapes.items():
            parameter = parameters[name]
            if not isinstance(parameter, nn.Parameter):
                raise RuntimeError(f"{name} must remain a direct Parameter")
            if tuple(parameter.shape) != expected:
                raise RuntimeError(f"{name} shape changed from {expected}")
        dtypes = {parameter.dtype for parameter in parameters.values()}
        devices = {parameter.device for parameter in parameters.values()}
        if len(dtypes) != 1 or len(devices) != 1:
            raise RuntimeError("mu, dU, and dV must share one dtype and device")

    def _restore_parameter_contract(self) -> None:
        self._validate_parameter_contract()
        for name in self._COUPLED_PARAMETER_NAMES:
            tag_optimizer_role(self, name, ParameterRole.COUPLED_MODE)
            setattr(getattr(self, name), RANK_ONLY_MUON_PROHIBITED_ATTR, True)

    def optimizer_provenance(self) -> dict[str, str]:
        """Return a fail-closed audit of the coupled optimizer assignment."""

        partition = partition_optimizer_parameters(self)
        provenance: dict[str, str] = {}
        for name in self._COUPLED_PARAMETER_NAMES:
            assignment = partition.assignment_for(name)
            if assignment.role is not ParameterRole.COUPLED_MODE:
                raise RuntimeError(f"{name} lost coupled-mode optimizer provenance")
            if assignment.target is not OptimizerTarget.AUXILIARY_ADAMW:
                raise RuntimeError(f"{name} escaped the shared AdamW partition")
            provenance[name] = assignment.target.value
        return provenance

    def adamw_weight_decay_families(
        self,
    ) -> dict[str, tuple[nn.Parameter, ...]]:
        """Name the two unpriced AdamW decay families without building an optimizer.

        All stored tensors remain under the same AdamW learning rule.  A later
        preregistration may bind distinct ``lambda_mu`` and ``lambda_delta``
        decay values; this build-only method provides the complete, disjoint
        tensor inventory but intentionally supplies no numerical settings.
        """

        self._validate_parameter_contract()
        return {
            "lambda_mu": (self.mu,),
            "lambda_delta": (self.dU, self.dV),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore provenance attributes that Parameter deepcopy can drop."""

        super().__setstate__(state)
        self._restore_parameter_contract()

    def _apply(self, fn: Any, recurse: bool = True) -> "SwapLinear":
        """Preserve coupled provenance across device and dtype transforms."""

        super()._apply(fn, recurse=recurse)
        self._restore_parameter_contract()
        return self

    def load_state_dict(
        self,
        state_dict: Any,
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Restore provenance after ordinary or assign-style state loading."""

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
        """Restore provenance when an owning parent performs recursive loading."""

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
            f"d_in={self.d_in}, d_out={self.d_out}, rank={self.rank}, "
            f"sigma_delta0={self.sigma_delta0:g}, seed={self.initialization_seed}"
        )
