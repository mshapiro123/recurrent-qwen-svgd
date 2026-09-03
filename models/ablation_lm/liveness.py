"""Eligibility-aware gradient-liveness receipts for the materialized WEFT graph.

PF-1.5 requires liveness to follow execution eligibility.  In particular, the
re-entry bridge is structurally absent from visit zero, so a ``K=1`` backward
must not pretend that its ``None`` gradients are a defect.  This module keeps a
closed parameter-to-visit contract and reports that absence explicitly; an
unknown trainable parameter is an error until its execution rule is authored.

The integrated rotor carrier, per-band callosum, and sidecar are not part of
``AblationLM`` at this checkpoint.  They are disclosed below rather than
invented as liveness rows.  Their standalone primitive tests do not imply that
the production graph has passed this instrument.

Step 2 does execute the position-aligned lane update, but the lanes do not yet
write back to either hemisphere.  Those tensors are typed as gradient-deferred
and keep an integrated T2 receipt from passing until the Step-3 path exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .model import AblationLM


PF1_NOT_MATERIALIZED_INTEGRATIONS = (
    "integrated_rotor_carrier",
    "per_band_callosum",
    "sidecar",
)

_NON_RECURRENT_MODULES = frozenset(
    {
        "token_embedding",
        "front_hadamard",
        "prelude_blocks",
        "engram",
        "long_term_memory",
        "coda_blocks",
        "final_norm",
    }
)


class GradientLivenessBlocked(RuntimeError):
    """Raised when a PF-1.5 liveness or inverse-liveness gate fails."""


@dataclass(frozen=True)
class ParameterEligibility:
    """One explicit ``(parameter, module, K, visit)`` eligibility row.

    ``visit_index=None`` denotes a fixed, once-per-forward phase outside the
    recurrent loop.  It is not a wildcard over visits.
    """

    parameter_name: str
    module_name: str
    recurrent_steps: int
    visit_index: int | None
    eligible: bool
    executed: bool
    deferred: bool
    reason: str


@dataclass(frozen=True)
class IneligibleParameter:
    """A trainable tensor with no eligible execution in one probed run."""

    parameter_name: str
    module_name: str
    deferred: bool
    reason: str


@dataclass(frozen=True)
class ModuleGradientMinimum:
    """Minimum L2 gradient norm across eligible tensors in one module."""

    module_name: str
    eligible_parameter_count: int
    minimum_gradient_norm: float


@dataclass(frozen=True)
class GradientLivenessReceipt:
    """Measured liveness for one recurrent depth after a completed backward."""

    recurrent_steps: int
    eligibility_matrix: tuple[ParameterEligibility, ...]
    module_minimums: tuple[ModuleGradientMinimum, ...]
    ineligible_parameters: tuple[IneligibleParameter, ...]
    deferred_parameters: tuple[IneligibleParameter, ...]
    eligible_missing_gradients: tuple[str, ...]
    eligible_zero_gradients: tuple[str, ...]
    eligible_nonfinite_gradients: tuple[str, ...]
    not_materialized_integrations: tuple[str, ...]

    @property
    def eligible_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    row.parameter_name
                    for row in self.eligibility_matrix
                    if row.eligible and row.executed
                }
            )
        )

    @property
    def live_parameter_names(self) -> tuple[str, ...]:
        failed = {
            *self.eligible_missing_gradients,
            *self.eligible_zero_gradients,
            *self.eligible_nonfinite_gradients,
        }
        return tuple(name for name in self.eligible_parameter_names if name not in failed)

    @property
    def passed(self) -> bool:
        return not (
            self.deferred_parameters
            or self.eligible_missing_gradients
            or self.eligible_zero_gradients
            or self.eligible_nonfinite_gradients
        )

    def require_passed(self) -> None:
        """Fail closed when any eligible tensor is missing, zero, or non-finite."""

        if self.passed:
            return
        raise GradientLivenessBlocked(
            "eligible gradient liveness failed: "
            f"deferred={tuple(item.parameter_name for item in self.deferred_parameters)}, "
            f"missing={self.eligible_missing_gradients}, "
            f"zero={self.eligible_zero_gradients}, "
            f"nonfinite={self.eligible_nonfinite_gradients}"
        )


@dataclass(frozen=True)
class InverseLivenessReceipt:
    """Whether every K=1-ineligible tensor becomes eligible and live at K=4."""

    k1_ineligible_parameter_names: tuple[str, ...]
    activated_and_live_at_k4: tuple[str, ...]
    not_eligible_at_k4: tuple[str, ...]
    not_live_at_k4: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.not_eligible_at_k4 and not self.not_live_at_k4

    def require_passed(self) -> None:
        if self.passed:
            return
        raise GradientLivenessBlocked(
            "K=1-ineligible tensors did not become eligible and live at K=4: "
            f"not_eligible={self.not_eligible_at_k4}, "
            f"not_live={self.not_live_at_k4}"
        )


def _module_name(parameter_name: str) -> str:
    return parameter_name.split(".", maxsplit=1)[0]


def _eligibility_rows_for_parameter(
    model: AblationLM,
    parameter_name: str,
    *,
    recurrent_steps: int,
) -> tuple[ParameterEligibility, ...]:
    module_name = _module_name(parameter_name)

    def row(
        visit_index: int | None,
        *,
        eligible: bool,
        executed: bool,
        deferred: bool = False,
        reason: str,
    ) -> ParameterEligibility:
        return ParameterEligibility(
            parameter_name=parameter_name,
            module_name=module_name,
            recurrent_steps=recurrent_steps,
            visit_index=visit_index,
            eligible=eligible,
            executed=executed,
            deferred=deferred,
            reason=reason,
        )

    if module_name in _NON_RECURRENT_MODULES:
        return (
            row(
                None,
                eligible=True,
                executed=True,
                reason="fixed once-per-forward phase outside the recurrent loop",
            ),
        )

    if module_name == "core_blocks":
        legacy_static_projection = model.config.use_static_kv_core and (
            ".attention.k_proj." in parameter_name
            or ".attention.v_proj." in parameter_name
        )
        if legacy_static_projection:
            return (
                row(
                    None,
                    eligible=True,
                    executed=True,
                    reason="static K/V anchor projection executes once before recurrence",
                ),
            )
        bicameral_kv_projection = model.config.use_bicameral_core and (
            parameter_name.endswith(".k_proj.weight")
            or parameter_name.endswith(".v_proj.weight")
            or parameter_name.endswith(".key_norm.weight")
        )
        if bicameral_kv_projection and model.config.kv_policy == "static":
            return (
                row(
                    None,
                    eligible=True,
                    executed=True,
                    reason=(
                        "bicameral static K/V projects the prelude anchor once "
                        "for this core block"
                    ),
                ),
            )
        if bicameral_kv_projection and model.config.kv_policy == "midpoint":
            rows = [
                row(
                    None,
                    eligible=True,
                    executed=True,
                    reason=(
                        "bicameral midpoint K/V projects the prelude anchor once "
                        "for this core block"
                    ),
                )
            ]
            if recurrent_steps >= 2:
                rows.append(
                    row(
                        recurrent_steps // 2,
                        eligible=True,
                        executed=True,
                        reason="bicameral midpoint K/V refresh executes at floor(K/2)",
                    )
                )
            return tuple(rows)
        return tuple(
            row(
                visit,
                eligible=True,
                executed=True,
                reason="shared recurrent core executes at every requested visit",
            )
            for visit in range(recurrent_steps)
        )

    if module_name == "bicameral_combiner":
        return (
            row(
                None,
                eligible=True,
                executed=True,
                reason="terminal S-2 combiner executes once after the final visit",
            ),
        )

    if module_name == "loop_embedding":
        return tuple(
            row(
                visit,
                eligible=True,
                executed=True,
                reason="the indexed loop embedding executes at this visit",
            )
            for visit in range(recurrent_steps)
        )

    if module_name == "reentry_bridge":
        return tuple(
            row(
                visit,
                eligible=visit > 0,
                executed=visit > 0,
                reason=(
                    "anchored re-entry executes only after visit zero"
                    if visit == 0
                    else "anchored re-entry executes at every additional visit"
                ),
            )
            for visit in range(recurrent_steps)
        )

    if module_name == "scratch":
        if model.config.use_bicameral_core:
            deferred_reason = (
                "Step-2 lanes are updated but do not yet reach either hemisphere "
                "or the logits; gradient liveness is deferred until the Step-3 "
                "carrier/write path is integrated"
            )
            if parameter_name.startswith("scratch.initializer."):
                return (
                    row(
                        None,
                        eligible=False,
                        executed=True,
                        deferred=True,
                        reason=deferred_reason,
                    ),
                )
            if parameter_name in ("scratch.layer_scale",) or parameter_name.startswith(
                "scratch.readout."
            ):
                return (
                    row(
                        None,
                        eligible=False,
                        executed=False,
                        deferred=True,
                        reason=(
                            "legacy scratch injection is not executed by the Step-2 "
                            "bicameral graph; the Step-3 write path is absent"
                        ),
                    ),
                )
            rows: list[ParameterEligibility] = []
            if parameter_name.startswith("scratch.hidden_norm."):
                rows.append(
                    row(
                        None,
                        eligible=False,
                        executed=True,
                        deferred=True,
                        reason=deferred_reason,
                    )
                )
            rows.extend(
                row(
                    visit,
                    eligible=False,
                    executed=True,
                    deferred=True,
                    reason=deferred_reason,
                )
                for visit in range(recurrent_steps)
            )
            return tuple(rows)
        if parameter_name.startswith("scratch.initializer."):
            return (
                row(
                    None,
                    eligible=True,
                    executed=True,
                    reason="scratch initializer executes once before recurrence",
                ),
            )
        rows: list[ParameterEligibility] = []
        if parameter_name.startswith("scratch.hidden_norm."):
            rows.append(
                row(
                    None,
                    eligible=True,
                    executed=True,
                    reason="scratch hidden norm executes during initialization",
                )
            )
        rows.extend(
            row(
                visit,
                eligible=True,
                executed=True,
                reason="scratch update, carrier, or injection executes at this visit",
            )
            for visit in range(recurrent_steps)
        )
        return tuple(rows)

    raise GradientLivenessBlocked(
        f"no PF-1.5 eligibility rule is authored for trainable parameter {parameter_name!r}"
    )


def parameter_eligibility_matrix(
    model: AblationLM,
    *,
    recurrent_steps: int,
) -> tuple[ParameterEligibility, ...]:
    """Author the closed eligibility matrix for the currently materialized graph."""

    if type(recurrent_steps) is not int:
        raise TypeError("recurrent_steps must be an exact integer")
    if recurrent_steps < 1 or recurrent_steps > model.config.max_recurrent_steps:
        raise ValueError("recurrent_steps must lie within the configured recurrence cap")
    if not model.config.use_recurrence and recurrent_steps != 1:
        raise ValueError("K > 1 requires structural recurrence")

    rows = tuple(
        row
        for parameter_name, parameter in model.named_parameters()
        if parameter.requires_grad
        for row in _eligibility_rows_for_parameter(
            model,
            parameter_name,
            recurrent_steps=recurrent_steps,
        )
    )
    if not rows:
        raise GradientLivenessBlocked("the liveness matrix contains no trainable tensors")
    return rows


def _gradient_values(gradient: torch.Tensor) -> torch.Tensor:
    if gradient.is_sparse:
        return gradient.detach().coalesce().values()
    return gradient.detach()


def gradient_liveness_receipt(
    model: AblationLM,
    *,
    recurrent_steps: int,
) -> GradientLivenessReceipt:
    """Measure PF-1.5 after the caller runs the corresponding backward pass."""

    matrix = parameter_eligibility_matrix(model, recurrent_steps=recurrent_steps)
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    eligible_names = tuple(
        sorted(
            {
                row.parameter_name
                for row in matrix
                if row.eligible and row.executed
            }
        )
    )
    ineligible: list[IneligibleParameter] = []
    for name in sorted(parameters):
        rows = tuple(row for row in matrix if row.parameter_name == name)
        if any(row.eligible and row.executed for row in rows):
            continue
        reasons = tuple(dict.fromkeys(row.reason for row in rows))
        deferred = bool(rows) and all(row.deferred for row in rows)
        ineligible.append(
            IneligibleParameter(
                parameter_name=name,
                module_name=_module_name(name),
                deferred=deferred,
                reason="; ".join(reasons),
            )
        )

    missing: list[str] = []
    zero: list[str] = []
    nonfinite: list[str] = []
    gradient_norms: dict[str, float] = {}
    for name in eligible_names:
        gradient = parameters[name].grad
        if gradient is None:
            missing.append(name)
            gradient_norms[name] = 0.0
            continue
        values = _gradient_values(gradient)
        if not bool(torch.isfinite(values).all()):
            nonfinite.append(name)
            gradient_norms[name] = 0.0
            continue
        norm = float(torch.linalg.vector_norm(values.double()).cpu().item())
        if not math.isfinite(norm):
            nonfinite.append(name)
            gradient_norms[name] = 0.0
        elif norm == 0.0:
            zero.append(name)
            gradient_norms[name] = 0.0
        else:
            gradient_norms[name] = norm

    module_names = sorted({_module_name(name) for name in eligible_names})
    module_minimums = tuple(
        ModuleGradientMinimum(
            module_name=module_name,
            eligible_parameter_count=sum(
                _module_name(name) == module_name for name in eligible_names
            ),
            minimum_gradient_norm=min(
                gradient_norms[name]
                for name in eligible_names
                if _module_name(name) == module_name
            ),
        )
        for module_name in module_names
    )
    return GradientLivenessReceipt(
        recurrent_steps=recurrent_steps,
        eligibility_matrix=matrix,
        module_minimums=module_minimums,
        ineligible_parameters=tuple(ineligible),
        deferred_parameters=tuple(item for item in ineligible if item.deferred),
        eligible_missing_gradients=tuple(missing),
        eligible_zero_gradients=tuple(zero),
        eligible_nonfinite_gradients=tuple(nonfinite),
        not_materialized_integrations=PF1_NOT_MATERIALIZED_INTEGRATIONS,
    )


def inverse_k1_k4_liveness(
    k1_receipt: GradientLivenessReceipt,
    k4_receipt: GradientLivenessReceipt,
) -> InverseLivenessReceipt:
    """Check that every structurally ineligible K=1 tensor wakes by K=4."""

    if k1_receipt.recurrent_steps != 1 or k4_receipt.recurrent_steps != 4:
        raise ValueError("inverse liveness requires K=1 and K=4 receipts")
    k1_ineligible = tuple(
        sorted(
            item.parameter_name
            for item in k1_receipt.ineligible_parameters
            if not item.deferred
        )
    )
    k4_eligible = set(k4_receipt.eligible_parameter_names)
    k4_live = set(k4_receipt.live_parameter_names)
    not_eligible = tuple(name for name in k1_ineligible if name not in k4_eligible)
    not_live = tuple(
        name for name in k1_ineligible if name in k4_eligible and name not in k4_live
    )
    activated = tuple(
        name for name in k1_ineligible if name in k4_eligible and name in k4_live
    )
    return InverseLivenessReceipt(
        k1_ineligible_parameter_names=k1_ineligible,
        activated_and_live_at_k4=activated,
        not_eligible_at_k4=not_eligible,
        not_live_at_k4=not_live,
    )


__all__ = [
    "GradientLivenessBlocked",
    "GradientLivenessReceipt",
    "IneligibleParameter",
    "InverseLivenessReceipt",
    "ModuleGradientMinimum",
    "PF1_NOT_MATERIALIZED_INTEGRATIONS",
    "ParameterEligibility",
    "gradient_liveness_receipt",
    "inverse_k1_k4_liveness",
    "parameter_eligibility_matrix",
]
